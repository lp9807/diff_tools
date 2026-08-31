"""
part1_data_preparation.py
数据准备 + 交互式标注
自动预标记 + 多种显示模式

依赖: view_renderer.py, config_manager.py, model_config.py
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
    setup_viewer, ALL_CLASSES, 
    ViewController, get_class_shortcut_display
)

# ============================================================
# 导入共享配置
# ============================================================

from model_config import (
    TRAIN_CLASSES, SSIM_THRESHOLD, SSIMComparator, ClassType,
    LABEL_DISPLAY, LABEL_COLORS
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
# 标注工具
# ============================================================

class PairAnnotator:
    """交互式标注工具"""
    
    def __init__(self):
        self.annotate_dir = TO_ANNOTATE_DIR
        self.output_dir = ANNOTATED_DIR
        self.pairs = [d for d in self.annotate_dir.iterdir() if d.is_dir()]
        self.annotations = {}
        self.current_idx = 0
        self.progress_file = self.annotate_dir / 'annotation_progress.json'
        
        # 初始化视图组件
        self.view_state, self.mouse_handler, self.keyboard_handler = setup_viewer()
        
        # 统计 - 使用 TRAIN_CLASSES (5类)
        self.stats = {
            'auto_matched': 0,
            'skipped': 0
        }
        for cls in TRAIN_CLASSES:
            self.stats[cls] = 0
        
        # 退出标志
        self.should_exit = False
        
        print(f"找到 {len(self.pairs)} 对待标注")
        print(f"\n📊 自动处理规则:")
        print(f"  SSIM ≥ {SSIM_THRESHOLD} → 自动标记为 'matched' (匹配)")
        print(f"  SSIM < {SSIM_THRESHOLD} → 需要人工标注 (5类)")
        print(f"    1: 质量提升  2: 质量退化  3: 渲染失败")
        print(f"    4: 不确定差异  m: 微小差异(trivial)")
    
    def start(self):
        """开始标注流程"""
        self.load_progress()
        
        # 检查是否已完成所有标注
        if self.current_idx >= len(self.pairs):
            print("\n✅ 所有图片已标注完成！")
            self.report()
            return
        
        # 使用 Enum 生成快捷键显示
        shortcut_lines = get_class_shortcut_display('annotation')
        
        print("\n" + "="*70)
        print("成对图像标注工具")
        print("="*70)
        print(f"📌 SSIM ≥ {SSIM_THRESHOLD} → 自动标记为 'matched' (匹配)")
        print(f"📌 SSIM < {SSIM_THRESHOLD} → 需要人工判断")
        print("\n鼠标操作:")
        print("  拖拽分割线 → 调整左右视图比例 (Split模式)")
        print("  滚轮 → 放大/缩小")
        print("  拖拽画面 → 平移 (缩放状态下)")
        print("\n快捷键:")
        # 动态生成快捷键显示
        for item in shortcut_lines:
            print(f"  {item}")
        print("  v → 🔄 切换显示模式")
        print("  ← → →  微调分割线位置")
        print("  ↑ → ↓  调整动画间隔")
        print("  r → 🔄 重置缩放和平移")
        print("  s → ⏭️  跳过")
        print("  b → ⏪ 回退")
        print("  q → 🚪 退出保存（需确认）")
        print("="*70)
        
        idx = self.current_idx  # 使用本地变量控制循环
        
        while idx < len(self.pairs):
            # 检查退出标志
            if self.should_exit:
                print("\n⚠️ 已退出标注，进度已保存")
                break
            
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
            
            # 计算 SSIM 相似度
            ssim_score = SSIMComparator.compute_ssim(before_path, after_path)
            
            # SSIM ≥ 阈值：自动匹配 (保存到 matched 目录)
            if ssim_score >= SSIM_THRESHOLD:
                dest_dir = self.output_dir / 'matched' / pair_dir.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(pair_dir, dest_dir)
                self.stats['auto_matched'] += 1
                print(f"🤖 [{idx+1}/{len(self.pairs)}] {pair_dir.name} → 自动标记为 matched "
                      f"(SSIM: {ssim_score:.6f} ≥ {SSIM_THRESHOLD})")
                idx += 1
                self.current_idx = idx
                self.save_progress()
                continue
            
            # SSIM < 阈值：人工标注 (5类: TRAIN_CLASSES)
            result = self.manual_annotate(before_img, after_img, ssim_score, pair_dir, idx)
            
            # ============================================================
            # 处理返回结果
            # ============================================================
            if result == 'quit':
                self.should_exit = True
                self.current_idx = idx
                self.save_progress()
                break
            
            elif result == 'skipped':
                self.stats['skipped'] += 1
                idx += 1
                self.current_idx = idx
                self.save_progress()
            
            elif result == 'back':
                # ============================================================
                # 回退：idx 减1，而不是修改 current_idx
                # ============================================================
                if idx <= 0:
                    print("⚠️ 已经是第一张，无法回退")
                    continue
                
                target_idx = -1
                for i in range(idx - 1, -1, -1):
                    prev_pair = self.pairs[i]
                    prev_name = prev_pair.name
                    
                    print(f"check index{i}: {prev_pair}")
                    
                    # ============================================================
                    # 在 annotations 中 = 人工标注 → 找到目标
                    # 不在 annotations 中 = 自动匹配 → 跳过
                    # ============================================================
                    if prev_name in self.annotations:
                        target_idx = i
                        break
                
                if target_idx == -1:
                    print("⚠️ 没有找到可回退的人工标注样本")
                    continue
                
                # 删除人工标注记录
                prev_pair = self.pairs[target_idx]
                prev_name = prev_pair.name
                
                if prev_name in self.annotations:
                    prev_label = self.annotations[prev_name]
                    del self.annotations[prev_name]
                    
                    if prev_label in self.stats:
                        self.stats[prev_label] = max(0, self.stats[prev_label] - 1)
                    
                    dest_dir = self.output_dir / prev_label / prev_name
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                        print(f"  🗑️ 已删除: {dest_dir}")
                
                # 回退
                idx = target_idx
                self.current_idx = idx
                self.save_progress()
                print(f"⏪ 回退到 {idx+1}/{len(self.pairs)} ({prev_name})")
            
            elif result and result in TRAIN_CLASSES:
                # 保存标注
                dest_dir = self.output_dir / result / pair_dir.name
                shutil.copytree(pair_dir, dest_dir)
                self.annotations[pair_dir.name] = result
                self.stats[result] = self.stats.get(result, 0) + 1
                
                idx += 1
                self.current_idx = idx
                self.save_progress()
            
            elif result is None:
                # 异常情况，跳过
                idx += 1
        
        cv2.destroyAllWindows()
        
        # 检查是否全部完成
        if not self.should_exit and self.current_idx >= len(self.pairs):
            print("\n🎉 标注完成！")
        else:
            print("\n⏸️ 标注暂停，进度已保存")
        
        self.report()
    
    def manual_annotate(self, before_img, after_img, ssim_score, pair_dir, idx):
        """人工标注界面"""
        self.view_state.reset()
        
        window_name = f'标注 - {pair_dir.name}'
        
        sample = {
            'name': pair_dir.name,
            'pred_label': 'unknown',
            'confidence': ssim_score,
            'auto_matched': False,
            'ssim_score': ssim_score
        }
        
        # 使用 Enum 生成快捷键标签
        shortcut_lines = get_class_shortcut_display('annotation')
        custom_labels = [
            '  '.join(shortcut_lines),
            'v:切换视图  ←→:分割  ↑↓:间隔  r:重置',
            's:跳过  b:回退  q:退出'
        ]
        
        # 使用共享 ViewController
        controller = ViewController(self.view_state, self.mouse_handler)
        
        # ============================================================
        # on_action 回调处理
        # ============================================================
        result_holder = {'value': None}
        
        def on_action(action):
            if action == 'skipped':
                print(f"⏭️ 跳过 {pair_dir.name}")
                result_holder['value'] = 'skipped'
            elif action == 'back':
                print(f"⏪ 回退")
                result_holder['value'] = 'back'
            elif action == 'quit':
                print("\n⚠️ 保存进度并退出...")
                result_holder['value'] = 'quit'
                controller.stop()
        
        # 运行控制器
        result = controller.run(
            before_img, after_img, sample, idx, len(self.pairs),
            window_name,
            mode='annotation',
            custom_labels=custom_labels,
            show_shortcuts=False,
            on_action=on_action
        )
        
        # 如果控制器返回了结果，使用它；否则使用回调设置的值
        if result is not None:
            return result
        elif result_holder['value'] is not None:
            return result_holder['value']
        
        return None
    
    def load_progress(self):
        """加载保存的进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.current_idx = data.get('current_idx', 0)
                self.annotations = data.get('annotations', {})
                
                loaded_stats = data.get('stats', {})
                self.stats = {
                    'auto_matched': loaded_stats.get('auto_matched', 0),
                    'skipped': loaded_stats.get('skipped', 0)
                }
                for cls in TRAIN_CLASSES:
                    self.stats[cls] = loaded_stats.get(cls, 0)
                
                print(f"📂 恢复进度: 已标注 {len(self.annotations)} 对，当前索引 {self.current_idx}")
    
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
        print(f"  已处理: {self.current_idx}")
        print(f"\n  自动处理 (SSIM ≥ {SSIM_THRESHOLD}):")
        print(f"    - 自动标记匹配 (matched): {self.stats['auto_matched']} "
              f"({self.stats['auto_matched']/total*100:.1f}%)")
        print(f"\n  人工标注 (SSIM < {SSIM_THRESHOLD}):")
        
        total_manual = 0
        for cls in TRAIN_CLASSES:
            count = self.stats.get(cls, 0)
            if count > 0:
                class_enum = ClassType.from_value(cls)
                display = class_enum.display_name if class_enum else cls
                print(f"    - {display}: {count}")
                total_manual += count
        
        print(f"    - ⏭️  跳过: {self.stats['skipped']}")
        
        print(f"\n  人工标注总数: {total_manual}")
        if total > 0:
            print(f"  自动化率: {self.stats['auto_matched'] / total * 100:.1f}%")
        
        # 保存报告
        manual_labels = {}
        for cls in TRAIN_CLASSES:
            manual_labels[cls] = self.stats.get(cls, 0)
        
        report = {
            'total_pairs': total,
            'processed': self.current_idx,
            'auto_matched': self.stats['auto_matched'],
            'manual_labels': manual_labels,
            'skipped': self.stats['skipped'],
            'ssim_threshold': SSIM_THRESHOLD,
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
    """
    使用 ALL_CLASSES 和 TRAIN_CLASSES 动态创建目录
    
    ALL_CLASSES (7类): TRAIN_CLASSES + matched + to_review
    TRAIN_CLASSES (5类): quality_improved, quality_degradation, 
                         render_failed, uncertain_difference, trivial
    """
    dirs = [
        TO_ANNOTATE_DIR,
        CHECKPOINT_DIR,
    ]
    
    # 02_annotated: ALL_CLASSES (7个目录)
    for cls in ALL_CLASSES:
        dirs.append(ANNOTATED_DIR / cls)
    
    # 03_train_data / 04_val_data: TRAIN_CLASSES (5个目录)
    for cls in TRAIN_CLASSES:
        dirs.append(TRAIN_DIR / cls)
        dirs.append(VAL_DIR / cls)
    
    # 06_final_output: ALL_CLASSES (7个目录)
    for cls in ALL_CLASSES:
        dirs.append(OUTPUT_DIR / cls)
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ 项目目录创建完成: {PROJECT_DIR}")
    print(f"  ALL_CLASSES ({len(ALL_CLASSES)}类): {', '.join(ALL_CLASSES)}")
    print(f"  TRAIN_CLASSES ({len(TRAIN_CLASSES)}类): {', '.join(TRAIN_CLASSES)}")


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
    """
    划分训练集和验证集
    只划分 TRAIN_CLASSES (5类) 的目录
    """
    for cls in TRAIN_CLASSES:
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
        
        class_enum = ClassType.from_value(cls)
        display = class_enum.display_name if class_enum else cls
        print(f"✅ {display}: 训练={split}, 验证={len(pairs)-split}")


# ============================================================
# 主程序
# ============================================================

def main():
    print("="*60)
    print("🚀 数据准备 + 标注")
    print("="*60)
    print(f"SSIM 阈值: {SSIM_THRESHOLD}")
    print(f"SSIM ≥ {SSIM_THRESHOLD} → 自动匹配 (matched)")
    print(f"SSIM < {SSIM_THRESHOLD} → 人工标注 (5类)")
    print(f"  1: 质量提升  2: 质量退化  3: 渲染失败")
    print(f"  4: 不确定差异  m: 微小差异")
    print("="*60)
    
    setup_directories()
    
    pairs = match_pairs()
    if not pairs:
        print("❌ 没有匹配到任何成对图片！")
        print("请检查:")
        print(f"  - before目录: {BEFORE_DIR}")
        print(f"  - after目录: {AFTER_DIR}")
        return
    
    sample_pairs(pairs)
    
    print("\n📌 启动标注工具...")
    annotator = PairAnnotator()
    annotator.start()
    
    # 检查是否因为退出而中断
    if annotator.should_exit:
        print("\n⏸️ 标注已中断，进度已保存")
        print(f"   当前进度: {annotator.current_idx}/{len(pairs)}")
        print(f"   重新运行继续标注")
    else:
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