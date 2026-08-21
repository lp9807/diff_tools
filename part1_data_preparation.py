"""
part1_data_preparation.py
数据准备 + 交互式标注
自动预标记 + 多种显示模式

功能：
1. 从 before/after 文件夹匹配成对图片
2. 采样供标注
3. 自动标记差异 ≤ 0.01% 的图片对为 "匹配"
4. 交互式标注差异 > 0.01% 的图片对（6种标记类型）
5. 支持 Split/Animation/Triple 三种视图模式
6. 鼠标拖拽分割线、滚轮缩放、平移
7. 划分训练集和验证集
8. 生成标注统计报告

依赖: view_renderer.py, config_manager.py
"""

import os
import shutil
import json
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys
import time

# ============================================================
# 导入配置管理模块
# ============================================================

from config_manager import load_config

# ============================================================
# 导入三视图渲染器
# ============================================================

from view_renderer import (
    ViewRenderer, ViewState, MouseHandler, KeyboardHandler,
    setup_viewer, ALL_CLASSES, LABEL_DISPLAY, LABEL_COLORS
)

# ============================================================
# 加载配置
# ============================================================

CONFIG = load_config('config.json')

BEFORE_DIR = Path(CONFIG['folders']['before'])
AFTER_DIR = Path(CONFIG['folders']['after'])
PROJECT_DIR = Path(CONFIG['folders']['project'])
EXTENSIONS = CONFIG['pairing']['extensions']
SAMPLE_SIZE = CONFIG['training']['sample_size']

# ============================================================
# 自动推导子目录
# ============================================================

TO_ANNOTATE_DIR = PROJECT_DIR / "01_to_annotate"
ANNOTATED_DIR = PROJECT_DIR / "02_annotated"
TRAIN_DIR = PROJECT_DIR / "03_train_data"
VAL_DIR = PROJECT_DIR / "04_val_data"
OUTPUT_DIR = PROJECT_DIR / "06_final_output"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"


# ============================================================
# 像素比对工具
# ============================================================

class PixelComparator:
    """像素比对工具 - 计算两张图片的像素差异"""
    
    DIFF_THRESHOLD = 10
    AUTO_MATCH_THRESHOLD = 0.01
    
    @staticmethod
    def compute_diff(img1, img2):
        """
        计算两张图片的像素差异
        
        Args:
            img1: 第一张图片 (BGR)
            img2: 第二张图片 (BGR)
        
        Returns:
            dict: 包含差异热力图、差异百分比、MSE等
        """
        # 确保两张图片尺寸一致
        if img1.shape != img2.shape:
            h = min(img1.shape[0], img2.shape[0])
            w = min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (w, h))
            img2 = cv2.resize(img2, (w, h))
        
        # 转换为浮点数计算
        img1_float = img1.astype(np.float32)
        img2_float = img2.astype(np.float32)
        
        # 计算绝对差异
        diff_float = np.abs(img1_float - img2_float)
        
        # 均方误差
        mse = np.mean(diff_float ** 2)
        
        # 差异像素：像素值变化超过阈值视为有差异
        diff_mask = diff_float > PixelComparator.DIFF_THRESHOLD
        diff_pixel_count = np.sum(diff_mask)
        total_pixels = diff_mask.size
        diff_percentage = (diff_pixel_count / total_pixels) * 100
        
        # 生成差异热力图
        max_diff = diff_float.max()
        if max_diff > 0:
            diff_normalized = np.clip(diff_float / max_diff * 255, 0, 255)
        else:
            diff_normalized = np.zeros_like(diff_float)
        diff_heatmap = diff_normalized.astype(np.uint8)
        diff_colored = cv2.applyColorMap(diff_heatmap, cv2.COLORMAP_JET)
        
        return {
            'heatmap': diff_colored,
            'diff_percentage': diff_percentage,
            'diff_pixel_count': int(diff_pixel_count),
            'total_pixels': int(total_pixels),
            'mse': mse,
            'max_diff': max_diff,
            'mean_diff': np.mean(diff_float),
            'diff_mask': diff_mask
        }
    
    @staticmethod
    def should_auto_match(diff_percentage):
        """判断是否应该自动标记为匹配"""
        return diff_percentage <= PixelComparator.AUTO_MATCH_THRESHOLD
    
    @staticmethod
    def get_status_info(diff_percentage):
        """获取状态信息用于UI显示"""
        if PixelComparator.should_auto_match(diff_percentage):
            return {'status': 'AUTO_MATCH', 'color': (0, 255, 0), 'label': '自动匹配'}
        else:
            return {'status': 'NEED_REVIEW', 'color': (255, 165, 0), 'label': '需人工标注'}


# ============================================================
# 标注工具
# ============================================================

class PairAnnotator:
    """
    交互式标注工具
    
    功能：
    1. 差异像素 ≤ 0.01% → 自动标记为匹配
    2. 差异像素 > 0.01% → 人工标注（6种类型）
    3. 三种显示模式：Split/Animation/Triple
    4. 鼠标拖拽分割线、滚轮缩放、平移
    5. 进度保存与恢复
    """
    
    def __init__(self):
        self.annotate_dir = TO_ANNOTATE_DIR
        self.output_dir = ANNOTATED_DIR
        self.pairs = [d for d in self.annotate_dir.iterdir() if d.is_dir()]
        self.annotations = {}
        self.current_idx = 0
        self.progress_file = self.annotate_dir / 'annotation_progress.json'
        self.comparator = PixelComparator()
        
        # 初始化视图组件
        self.view_state, self.mouse_handler, self.keyboard_handler = setup_viewer()
        
        # 统计
        self.stats = {
            'auto_matched': 0,
            'quality_improved': 0,
            'quality_degradation': 0,
            'render_failed': 0,
            'uncertain_difference': 0,
            'skipped': 0
        }
        
        print(f"找到 {len(self.pairs)} 对待标注")
        print(f"\n📊 自动处理规则:")
        print(f"  差异像素 ≤ 0.01% → 自动标记为 '匹配'")
        print(f"  差异像素 > 0.01% → 需要人工标注")
    
    def start(self):
        """开始标注流程"""
        self.load_progress()
        
        print("\n" + "="*70)
        print("成对图像标注工具（6种标记类型）")
        print("="*70)
        print("📌 差异像素 ≤ 0.01% → 自动标记为匹配")
        print("📌 差异像素 > 0.01% → 需要人工判断")
        print("\n鼠标操作:")
        print("  拖拽分割线 → 调整左右视图比例 (Split模式)")
        print("  滚轮 → 放大/缩小")
        print("  拖拽画面 → 平移 (缩放状态下)")
        print("\n快捷键:")
        print("  1 → ✅ 质量提升 (渲染后更好)")
        print("  2 → ⚠️ 质量退化 (渲染后更差)")
        print("  3 → ❌ 渲染失败 (元素缺失/严重异常)")
        print("  4 → ❓ 不确定差异 (难以判断)")
        print("  m → 🔄 标记为匹配 (几乎相同)")
        print("  v → 🔄 切换显示模式")
        print("  ← → →  微调分割线位置")
        print("  ↑ → ↓  调整动画间隔")
        print("  r → 🔄 重置缩放和平移")
        print("  s → ⏭️  跳过")
        print("  b → ⏪ 回退")
        print("  q → 🚪 退出保存（需确认）")
        print("="*70)
        
        for idx in range(self.current_idx, len(self.pairs)):
            pair_dir = self.pairs[idx]
            
            before_path = pair_dir / 'before.png'
            after_path = pair_dir / 'after.png'
            
            if not before_path.exists() or not after_path.exists():
                print(f"⚠️ 跳过 {pair_dir.name}: 缺少图片")
                continue
            
            before_img = cv2.imread(str(before_path))
            after_img = cv2.imread(str(after_path))
            
            if before_img is None or after_img is None:
                print(f"⚠️ 跳过 {pair_dir.name}: 无法读取")
                continue
            
            # 计算像素差异
            diff_result = self.comparator.compute_diff(before_img, after_img)
            diff_pct = diff_result['diff_percentage']
            
            # ============================================================
            # 自动匹配：差异像素 ≤ 0.01%
            # ============================================================
            if self.comparator.should_auto_match(diff_pct):
                dest_dir = self.output_dir / 'matched' / pair_dir.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(pair_dir, dest_dir)
                self.stats['auto_matched'] += 1
                print(f"🤖 [{idx+1}/{len(self.pairs)}] {pair_dir.name} → 自动标记为匹配 "
                      f"(差异像素: {diff_pct:.4f}% ≤ 0.01%)")
                self.current_idx = idx + 1
                self.save_progress()
                continue
            
            # ============================================================
            # 差异 > 0.01%：人工标注
            # ============================================================
            label = self.manual_annotate(before_img, after_img, diff_result, pair_dir, idx)
            
            if label == 'skipped':
                self.stats['skipped'] += 1
                print(f"⏭️  [{idx+1}/{len(self.pairs)}] {pair_dir.name} → 跳过")
            elif label:
                dest_dir = self.output_dir / label / pair_dir.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(pair_dir, dest_dir)
                self.annotations[str(pair_dir)] = label
                self.stats[label] = self.stats.get(label, 0) + 1
                
                label_display = LABEL_DISPLAY.get(label, label)
                print(f"✅ [{idx+1}/{len(self.pairs)}] {pair_dir.name} → {label_display} "
                      f"(差异像素: {diff_pct:.3f}%)")
            
            self.current_idx = idx + 1
            self.save_progress()
        
        cv2.destroyAllWindows()
        print("\n🎉 标注完成！")
        self.report()
    
    def manual_annotate(self, before_img, after_img, diff_result, pair_dir, idx):
        """
        人工标注界面
        
        Returns:
            str: 'quality_improved', 'quality_degradation', 'render_failed',
                 'uncertain_difference', 'matched', 'skipped', None
        """
        diff_pct = diff_result['diff_percentage']
        status_info = self.comparator.get_status_info(diff_pct)
        
        # 重置视图状态
        self.view_state.reset()
        
        window_name = f'标注 - {pair_dir.name}'
        cv2.namedWindow(window_name)
        
        # 构建样本信息
        sample = {
            'name': pair_dir.name,
            'pred_label': 'unknown',
            'confidence': 0,
            'auto_matched': False,
            'ssim_score': diff_pct / 100
        }
        
        # 自定义快捷键标签
        custom_labels = [
            '1:质量提升  2:质量退化  3:渲染失败',
            '4:不确定  5:匹配  v:切换视图',
            's:跳过  b:回退  q:退出'
        ]
        
        while True:
            # 使用共享视图渲染器
            display = ViewRenderer.create_full_view(
                before_img, after_img,
                sample=sample,
                idx=idx,
                total_samples=len(self.pairs),
                view_state=self.view_state,
                show_shortcuts=False,
                custom_labels=custom_labels
            )
            
            # 设置鼠标回调
            cv2.setMouseCallback(window_name, self.mouse_handler.callback)
            cv2.imshow(window_name, display)
            
            # 处理键盘事件
            key = cv2.waitKey(50) & 0xFF
            
            # ============================================================
            # 标记类型选择
            # ============================================================
            if key == ord('1'):
                cv2.destroyAllWindows()
                return 'quality_improved'
            elif key == ord('2'):
                cv2.destroyAllWindows()
                return 'quality_degradation'
            elif key == ord('3'):
                cv2.destroyAllWindows()
                return 'render_failed'
            elif key == ord('4'):
                cv2.destroyAllWindows()
                return 'uncertain_difference'
            elif key == ord('m'):
                cv2.destroyAllWindows()
                return 'matched'
            
            # ============================================================
            # 视图控制
            # ============================================================
            elif key == ord('v'):
                self.view_state.toggle_mode()
            elif key == ord('r'):
                self.view_state.reset()
            elif key == 81:  # 左箭头
                self.view_state.adjust_split(-0.02)
            elif key == 83:  # 右箭头
                self.view_state.adjust_split(0.02)
            elif key == 82:  # 上箭头
                self.view_state.adjust_animation_interval(50)
            elif key == 84:  # 下箭头
                self.view_state.adjust_animation_interval(-50)
            
            # ============================================================
            # 导航控制
            # ============================================================
            elif key == ord('s'):
                cv2.destroyAllWindows()
                return 'skipped'
            elif key == ord('b'):
                cv2.destroyAllWindows()
                return 'back'
            elif key == ord('q'):
                cv2.destroyAllWindows()
                self.save_progress()
                print("\n⚠️ 已保存进度")
                return None
            
            # ============================================================
            # 动画逻辑
            # ============================================================
            if self.view_state.display_mode == 'animation':
                self.view_state.toggle_animation_frame()
                time.sleep(self.view_state.animation_interval / 1000.0)
    
    def load_progress(self):
        """加载保存的进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.current_idx = data.get('current_idx', 0)
                self.annotations = data.get('annotations', {})
                self.stats = data.get('stats', {
                    'auto_matched': 0,
                    'quality_improved': 0,
                    'quality_degradation': 0,
                    'render_failed': 0,
                    'uncertain_difference': 0,
                    'skipped': 0
                })
                print(f"📂 恢复进度: 已标注 {len(self.annotations)} 对")
    
    def save_progress(self):
        """保存当前进度"""
        data = {
            'current_idx': self.current_idx,
            'annotations': self.annotations,
            'stats': self.stats,
            'total': len(self.pairs),
            'timestamp': datetime.now().isoformat()
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def report(self):
        """生成并打印标注统计报告"""
        total = len(self.pairs)
        
        print("\n" + "="*60)
        print("📊 标注统计报告")
        print("="*60)
        print(f"  总pair: {total}")
        print(f"\n  自动处理 (差异 ≤ 0.01%):")
        print(f"    - 自动标记匹配: {self.stats['auto_matched']} "
              f"({self.stats['auto_matched']/total*100:.1f}%)")
        print(f"\n  人工标注 (差异 > 0.01%):")
        
        manual_labels = [
            ('quality_improved', '质量提升', '✅'),
            ('quality_degradation', '质量退化', '⚠️'),
            ('render_failed', '渲染失败', '❌'),
            ('uncertain_difference', '不确定差异', '❓'),
            ('matched', '匹配', '🔄'),
        ]
        
        for key, display, emoji in manual_labels:
            count = self.stats.get(key, 0)
            if count > 0:
                print(f"    - {emoji} {display}: {count}")
        
        print(f"    - ⏭️  跳过: {self.stats['skipped']}")
        
        total_manual = sum(self.stats.get(key, 0) for key, _, _ in manual_labels)
        print(f"\n  人工标注总数: {total_manual}")
        print(f"  自动化率: {self.stats['auto_matched'] / total * 100:.1f}%")
        
        # 保存报告
        report = {
            'total_pairs': total,
            'auto_matched': self.stats['auto_matched'],
            'manual_labels': {
                'quality_improved': self.stats.get('quality_improved', 0),
                'quality_degradation': self.stats.get('quality_degradation', 0),
                'render_failed': self.stats.get('render_failed', 0),
                'uncertain_difference': self.stats.get('uncertain_difference', 0),
                'matched': self.stats.get('matched', 0),
                'skipped': self.stats.get('skipped', 0)
            },
            'auto_match_threshold': PixelComparator.AUTO_MATCH_THRESHOLD,
            'timestamp': datetime.now().isoformat()
        }
        report_file = self.annotate_dir / 'annotation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 报告已保存: {report_file}")


# ============================================================
# 工具函数
# ============================================================

def setup_directories():
    """创建所有需要的目录（6种类型）"""
    dirs = [
        TO_ANNOTATE_DIR,
        ANNOTATED_DIR / 'quality_improved',
        ANNOTATED_DIR / 'quality_degradation',
        ANNOTATED_DIR / 'render_failed',
        ANNOTATED_DIR / 'uncertain_difference',
        ANNOTATED_DIR / 'matched',
        TRAIN_DIR / 'quality_improved',
        TRAIN_DIR / 'quality_degradation',
        TRAIN_DIR / 'render_failed',
        TRAIN_DIR / 'uncertain_difference',
        TRAIN_DIR / 'matched',
        VAL_DIR / 'quality_improved',
        VAL_DIR / 'quality_degradation',
        VAL_DIR / 'render_failed',
        VAL_DIR / 'uncertain_difference',
        VAL_DIR / 'matched',
        OUTPUT_DIR / 'quality_improved',
        OUTPUT_DIR / 'quality_degradation',
        OUTPUT_DIR / 'render_failed',
        OUTPUT_DIR / 'uncertain_difference',
        OUTPUT_DIR / 'matched',
        OUTPUT_DIR / 'to_review',
        CHECKPOINT_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✅ 项目目录创建完成: {PROJECT_DIR}")


def get_image_files(directory):
    """获取目录中所有图片文件"""
    files = []
    for ext in EXTENSIONS:
        files.extend(directory.glob(f'*{ext}'))
        files.extend(directory.glob(f'*{ext.upper()}'))
    return {f.stem: f for f in files}


def match_pairs():
    """匹配 before 和 after 中的同名文件"""
    before_files = get_image_files(BEFORE_DIR)
    after_files = get_image_files(AFTER_DIR)
    
    print(f"\n📁 扫描目录:")
    print(f"  Before: {BEFORE_DIR} -> {len(before_files)} 张")
    print(f"  After:  {AFTER_DIR} -> {len(after_files)} 张")
    
    common_names = set(before_files.keys()) & set(after_files.keys())
    pairs = []
    for name in common_names:
        pairs.append({
            'name': name,
            'before': before_files[name],
            'after': after_files[name]
        })
    
    print(f"✅ 匹配到 {len(pairs)} 对图片")
    return pairs


def sample_pairs(pairs):
    """采样供标注"""
    n_samples = min(SAMPLE_SIZE, len(pairs))
    if len(pairs) <= n_samples:
        selected = pairs
    else:
        sorted_pairs = sorted(pairs, key=lambda x: x['name'])
        step = len(sorted_pairs) // n_samples
        selected = sorted_pairs[::step][:n_samples]
    
    for pair in tqdm(selected, desc="复制采样数据"):
        pair_dir = TO_ANNOTATE_DIR / pair['name']
        pair_dir.mkdir(exist_ok=True)
        shutil.copy2(pair['before'], pair_dir / 'before.png')
        shutil.copy2(pair['after'], pair_dir / 'after.png')
    
    print(f"✅ 已采样 {len(selected)} 对")
    return selected


def prepare_dataset(val_ratio=0.2):
    """划分训练集和验证集（5种训练类型）"""
    train_classes = ['quality_improved', 'quality_degradation', 'render_failed', 'uncertain_difference', 'matched']
    
    for cls in train_classes:
        src_dir = ANNOTATED_DIR / cls
        if not src_dir.exists():
            continue
        pairs = [d for d in src_dir.iterdir() if d.is_dir()]
        if not pairs:
            continue
        random.shuffle(pairs)
        split = int(len(pairs) * (1 - val_ratio))
        for pair in pairs[:split]:
            dest = TRAIN_DIR / cls / pair.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(pair, dest)
        for pair in pairs[split:]:
            dest = VAL_DIR / cls / pair.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(pair, dest)
        print(f"✅ {cls}: 训练={split}, 验证={len(pairs)-split}")


# ============================================================
# 主程序
# ============================================================

def main():
    """主程序入口"""
    print("="*60)
    print("🚀 数据准备 + 标注（6种标记类型）")
    print("="*60)
    
    # 1. 创建目录
    setup_directories()
    
    # 2. 匹配成对文件
    pairs = match_pairs()
    if not pairs:
        print("❌ 没有匹配到任何成对图片！")
        print("请检查:")
        print(f"  - before目录: {BEFORE_DIR}")
        print(f"  - after目录: {AFTER_DIR}")
        return
    
    # 3. 采样
    sample_pairs(pairs)
    
    # 4. 标注
    print("\n📌 启动标注工具...")
    annotator = PairAnnotator()
    annotator.start()
    
    # 5. 划分数据集
    print("\n📌 划分数据集...")
    prepare_dataset()
    
    print("\n" + "="*60)
    print("🎉 数据准备完成！")
    print("="*60)
    print(f"训练数据: {TRAIN_DIR}")
    print(f"验证数据: {VAL_DIR}")
    print(f"下一步: python part2_training.py")


if __name__ == "__main__":
    main()