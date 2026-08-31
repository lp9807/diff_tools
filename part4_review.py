"""
part4_review.py
预测后审核比对工具（从 final_output 读取结果）

模式：
    - 审核模式 (python part4_review.py -i)：
        1. 选择子类别
        2. 选择采样策略（随机/低置信度/均衡/全部）
        3. 浏览和审核样本
        4. 可修改分类（只能修改为 TRAIN_CLASSES 中的 5 类）
        5. 导出到标注集
        6. 生成HTML报告
    
    - 报告模式 (python part4_review.py)：
        使用 ReportGenerator 生成包含多数据源分类的报告

依赖: view_renderer.py, config_manager.py, report_generator.py, model_config.py
"""

import os
import shutil
import json
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys
import webbrowser
from collections import defaultdict

# ============================================================
# 导入模块
# ============================================================

from config_manager import load_config

from view_renderer import (
    ViewRenderer, ViewState, MouseHandler, KeyboardHandler,
    setup_viewer, ViewController, get_class_shortcut_display
)

from model_config import (
    ALL_CLASSES， TRAIN_CLASSES, CLASS_TO_IDX, SSIM_THRESHOLD,
    LABEL_DISPLAY, LABEL_COLORS,
    load_model, find_model_file, ClassType
)

from report_generator import ReportGenerator

# ============================================================
# 加载配置
# ============================================================

CONFIG = load_config('config.json')

PROJECT_DIR = Path(CONFIG['folders']['project'])

OUTPUT_DIR = PROJECT_DIR / "06_final_output"
REVIEW_DIR = PROJECT_DIR / "07_review"
ANNOTATED_DIR = PROJECT_DIR / "02_annotated"
TRAIN_DIR = PROJECT_DIR / "03_train_data"
VAL_DIR = PROJECT_DIR / "04_val_data"


# ============================================================
# 审核器
# ============================================================

class PredictionReviewer:
    """预测后审核比对工具"""
    
    def __init__(self, interactive=True):
        self.interactive = interactive
        self.project_dir = PROJECT_DIR
        
        # 检查 final_output 是否存在
        if not OUTPUT_DIR.exists():
            print(f"❌ 预测结果目录不存在: {OUTPUT_DIR}")
            print("请先运行 part3_pair_prediction.py 进行预测")
            sys.exit(1)
        
        # 加载预测结果
        self.load_predictions()
        
        # 审核结果
        self.review_samples = []
        self.review_results = []
        self.review_history = []
        self.current_idx = 0
        self.review_progress_file = REVIEW_DIR / 'review_progress.json'
        self.review_history_file = REVIEW_DIR / 'review_history.json'
        
        # 创建审核目录
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        
        # 加载已保存的进度和修改历史
        self.load_progress()
        self.load_history()
        
        # 初始化视图组件
        self.view_state, self.mouse_handler, self.keyboard_handler = setup_viewer()
    
    def load_predictions(self):
        """从 final_output 加载预测结果"""
        self.predictions = []
        self.all_pairs = {}
        
        for label in ALL_CLASSES:
            label_dir = OUTPUT_DIR / label
            if not label_dir.exists():
                continue
            
            for pair_dir in label_dir.iterdir():
                if not pair_dir.is_dir():
                    continue
                
                before_path = pair_dir / 'before.png'
                after_path = pair_dir / 'after.png'
                
                if not before_path.exists():
                    before_files = list(pair_dir.glob('before.*'))
                    before_path = before_files[0] if before_files else None
                if not after_path.exists():
                    after_files = list(pair_dir.glob('after.*'))
                    after_path = after_files[0] if after_files else None
                
                if before_path and after_path:
                    info = {}
                    info_file = pair_dir / 'prediction_info.json'
                    if info_file.exists():
                        try:
                            with open(info_file, 'r') as f:
                                info = json.load(f)
                        except:
                            pass
                    
                    self.predictions.append({
                        'name': pair_dir.name,
                        'before_path': str(before_path),
                        'after_path': str(after_path),
                        'pred_label': label,
                        'confidence': info.get('confidence', 0.95),
                        'auto_matched': info.get('auto_matched', False),
                        'ssim_score': info.get('ssim_score', 0),
                        'source_dir': str(pair_dir)
                    })
                    
                    self.all_pairs[pair_dir.name] = {
                        'before_path': str(before_path),
                        'after_path': str(after_path),
                        'pred_label': label,
                        'source_dir': str(pair_dir)
                    }
        
        print(f"📁 从 {OUTPUT_DIR} 加载了 {len(self.predictions)} 个预测结果")
        
        class_counts = defaultdict(int)
        for p in self.predictions:
            class_counts[p['pred_label']] += 1
        
        print("类别分布:")
        for cls in ALL_CLASSES:
            count = class_counts.get(cls, 0)
            if count > 0:
                class_enum = ClassType.from_value(cls)
                display = class_enum.display_name if class_enum else cls
                print(f"  - {display}: {count}")
        
        return self.predictions
    
    def load_progress(self):
        """加载审核进度"""
        if self.review_progress_file.exists():
            try:
                with open(self.review_progress_file, 'r') as f:
                    data = json.load(f)
                    self.review_results = data.get('review_results', [])
                    self.current_idx = data.get('current_idx', 0)
                    print(f"📂 恢复审核进度: {len(self.review_results)} 个已审核")
            except:
                self.review_results = []
                self.current_idx = 0
        else:
            self.review_results = []
            self.current_idx = 0
    
    def load_history(self):
        """加载修改历史"""
        if self.review_history_file.exists():
            try:
                with open(self.review_history_file, 'r') as f:
                    self.review_history = json.load(f)
                    print(f"📂 加载修改历史: {len(self.review_history)} 条记录")
            except:
                self.review_history = []
        else:
            self.review_history = []
    
    def save_progress(self):
        """保存审核进度"""
        data = {
            'review_results': self.review_results,
            'current_idx': self.current_idx,
            'total_samples': len(self.review_samples) if self.review_samples else 0,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.review_progress_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_history(self):
        """保存修改历史"""
        with open(self.review_history_file, 'w') as f:
            json.dump(self.review_history, f, indent=2)
    
    def update_prediction(self, name, new_label, reason=""):
        """
        更新预测结果
        
        Args:
            name: 样本名称
            new_label: 新标签 (必须是 TRAIN_CLASSES 中的类别)
            reason: 修改原因
        """
        # 验证新标签是否在 TRAIN_CLASSES 中
        if new_label not in TRAIN_CLASSES:
            print(f"⚠️ 无效标签: {new_label}，必须是 TRAIN_CLASSES 中的类别")
            return False
        
        sample = None
        for p in self.predictions:
            if p['name'] == name:
                sample = p
                break
        
        if not sample:
            print(f"⚠️ 未找到样本: {name}")
            return False
        
        old_label = sample['pred_label']
        
        if old_label == new_label:
            print(f"⏭️ 标签未变化: {name} → {new_label}")
            return True
        
        # 记录修改历史
        history_entry = {
            'name': name,
            'old_label': old_label,
            'new_label': new_label,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        self.review_history.append(history_entry)
        
        # 移动目录
        src_dir = OUTPUT_DIR / old_label / name
        dest_dir = OUTPUT_DIR / new_label / name
        
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        
        if src_dir.exists():
            shutil.move(str(src_dir), str(dest_dir))
            print(f"📁 移动目录: {old_label}/{name} → {new_label}/{name}")
        
        # 更新预测信息
        info_file = dest_dir / 'prediction_info.json'
        if info_file.exists():
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                info['predicted_class'] = new_label
                info['reviewed'] = True
                info['review_time'] = datetime.now().isoformat()
                info['previous_class'] = old_label
                with open(info_file, 'w') as f:
                    json.dump(info, f, indent=2)
            except:
                pass
        
        sample['pred_label'] = new_label
        sample['source_dir'] = str(dest_dir)
        
        if name in self.all_pairs:
            self.all_pairs[name]['pred_label'] = new_label
            self.all_pairs[name]['source_dir'] = str(dest_dir)
        
        self.save_history()
        print(f"✅ 已更新: {name} → {new_label}")
        return True
    
    def refresh_statistics(self):
        """刷新统计信息"""
        self.load_predictions()
        
        class_counts = defaultdict(int)
        for p in self.predictions:
            class_counts[p['pred_label']] += 1
        
        summary_path = OUTPUT_DIR / 'prediction_summary.json'
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                summary['class_distribution'] = dict(class_counts)
                summary['last_updated'] = datetime.now().isoformat()
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
            except:
                pass
        
        return class_counts
    
    def select_subcategories(self):
        """选择要审核的子类别（从 ALL_CLASSES 中选择）"""
        available = sorted(set(p['pred_label'] for p in self.predictions))
        
        # 确保 to_review 排在前面
        if 'to_review' in available:
            available.remove('to_review')
            available = ['to_review'] + available
        
        print("\n" + "="*60)
        print("📋 选择要审核的子类别")
        print("="*60)
        print("可用类别:")
        for i, cls in enumerate(available, 1):
            count = sum(1 for p in self.predictions if p['pred_label'] == cls)
            class_enum = ClassType.from_value(cls)
            display = class_enum.display_name if class_enum else cls
            print(f"  {i}. {display} ({count} 个)")
        print(f"  a. 全部类别")
        print(f"  q. 退出")
        
        choice = input("\n请输入编号 (默认: a): ").strip() or "a"
        
        if choice.lower() == 'q':
            print("退出审核")
            sys.exit(0)
        
        if choice.lower() == 'a':
            return available
        else:
            try:
                idx = int(choice) - 1
                return [available[idx]]
            except:
                print("无效选择，使用全部类别")
                return available
    
    def select_sampling_strategy(self):
        """选择采样策略"""
        print("\n" + "="*60)
        print("📋 选择采样策略")
        print("="*60)
        print("  1. 全部样本（浏览所有预测结果）")
        print("  2. 随机采样")
        print("  3. 低置信度优先")
        print("  4. 按类别均衡采样")
        print("  q. 退出")
        
        choice = input("\n请选择 (默认: 1): ").strip() or "1"
        
        if choice.lower() == 'q':
            print("退出审核")
            sys.exit(0)
        
        strategy_map = {
            '1': 'all',
            '2': 'random',
            '3': 'low_confidence',
            '4': 'balanced'
        }
        
        strategy = strategy_map.get(choice, 'all')
        
        if strategy == 'all':
            sample_size = len(self.predictions)
            print(f"✅ 选择全部样本: {sample_size} 个")
        else:
            sample_size = int(input("请输入采样数量 (默认: 50): ").strip() or "50")
            print(f"✅ 选择 {strategy} 采样: {sample_size} 个")
        
        return strategy, sample_size
    
    def sample_for_review(self, selected_classes=None, strategy='all', sample_size=50):
        """采样进行人工审核"""
        import random
        
        # 按类别筛选
        if selected_classes:
            candidates = [p for p in self.predictions if p['pred_label'] in selected_classes]
        else:
            candidates = self.predictions
        
        if not candidates:
            print("❌ 没有匹配的样本")
            return
        
        print(f"候选样本: {len(candidates)}")
        
        # 按策略采样
        if strategy == 'all':
            sampled = candidates.copy()
            print(f"📋 选择全部 {len(sampled)} 个样本")
        
        elif strategy == 'low_confidence':
            sorted_preds = sorted(candidates, key=lambda x: x['confidence'])
            sampled = sorted_preds[:min(sample_size, len(candidates))]
            print(f"📋 低置信度采样: {len(sampled)} 个样本")
        
        elif strategy == 'balanced':
            by_class = defaultdict(list)
            for pred in candidates:
                by_class[pred['pred_label']].append(pred)
            
            sampled = []
            classes = list(by_class.keys())
            per_class = max(1, sample_size // len(classes))
            
            for cls in classes:
                class_samples = by_class[cls]
                if len(class_samples) <= per_class:
                    sampled.extend(class_samples)
                else:
                    sampled.extend(random.sample(class_samples, per_class))
            
            if len(sampled) < sample_size and len(candidates) > len(sampled):
                remaining = [p for p in candidates if p not in sampled]
                needed = sample_size - len(sampled)
                sampled.extend(random.sample(remaining, min(needed, len(remaining))))
            
            print(f"📋 均衡采样: {len(sampled)} 个样本")
        
        else:  # random
            sampled = random.sample(candidates, min(sample_size, len(candidates)))
            print(f"📋 随机采样: {len(sampled)} 个样本")
        
        self.review_samples = sampled
        
        # 打印类别分布
        class_counts = defaultdict(int)
        for s in sampled:
            class_counts[s['pred_label']] += 1
        
        print("\n采样类别分布:")
        for cls, count in class_counts.items():
            class_enum = ClassType.from_value(cls)
            display = class_enum.display_name if class_enum else cls
            print(f"  - {display}: {count}")
        
        return self.review_samples
    
    def launch_review_ui(self):
        """启动审核UI"""
        if not self.review_samples:
            print("❌ 请先运行 sample_for_review()")
            return
        
        # 过滤已审核的样本
        reviewed_names = set(r['name'] for r in self.review_results)
        remaining = [s for s in self.review_samples if s['name'] not in reviewed_names]
        
        if not remaining:
            print("✅ 所有样本已审核完成！")
            self.generate_html_report()
            return
        
        total_remaining = len(remaining)
        total_samples = len(self.review_samples)
        
        print(f"\n📌 开始审核 {total_remaining} 个样本（共 {total_samples} 个）...")
        print("="*70)
        print("预测审核工具 - 全部样本浏览模式")
        print("="*70)
        print(f"进度: {len(self.review_results)}/{total_samples} 已审核")
        print("\n📌 审核分类（必须选择 TRAIN_CLASSES 中的 5 类）:")
        
        # 使用 TRAIN_CLASSES 生成快捷键显示
        shortcut_lines = self._get_review_shortcut_display()
        for item in shortcut_lines:
            print(f"  {item}")
        
        print("\n📌 操作说明:")
        print("  v → 切换视图模式  ↑↓ → 调整动画间隔")
        print("  r → 重置视图  s → 跳过  q → 退出")
        print("  n → 下一个  p → 上一个")
        print("="*70)
        
        for idx, sample in enumerate(remaining):
            self.current_idx = len(self.review_results)
            
            before_img = cv2.imread(sample['before_path'])
            after_img = cv2.imread(sample['after_path'])
            
            if before_img is None or after_img is None:
                print(f"⚠️ 跳过 {sample['name']}: 无法读取图片")
                continue
            
            self.view_state.reset()
            
            window_name = f'审核 - {sample["name"]}'
            
            # 构建样本信息
            sample_info = {
                'name': sample['name'],
                'pred_label': sample['pred_label'],
                'confidence': sample['confidence'],
                'auto_matched': sample.get('auto_matched', False),
                'ssim_score': sample.get('ssim_score', 0)
            }
            
            # 审核快捷键标签（只显示 TRAIN_CLASSES）
            custom_labels = self._get_review_custom_labels()
            
            # 使用共享 ViewController
            controller = ViewController(self.view_state, self.mouse_handler)
            
            # 自定义 action 处理
            def on_action(action):
                if action == 'skipped':
                    print(f"⏭️ 跳过 {sample['name']}")
                    self.review_results.append({
                        'name': sample['name'],
                        'action': 'skipped',
                        'timestamp': datetime.now().isoformat()
                    })
                    self.save_progress()
                elif action == 'next':
                    if sample['name'] not in [r['name'] for r in self.review_results]:
                        self.review_results.append({
                            'name': sample['name'],
                            'action': 'skipped',
                            'timestamp': datetime.now().isoformat()
                        })
                        self.save_progress()
                        print(f"⏭️ 自动跳过 {sample['name']}")
                elif action == 'prev':
                    if self.review_results and self.review_results[-1]['name'] == sample['name']:
                        self.review_results.pop()
                        self.save_progress()
                        print(f"⏪ 回退 {sample['name']}")
                elif action == 'quit':
                    self.save_progress()
                    self.generate_html_report()
                    print("\n✅ 审核进度已保存")
                    sys.exit(0)
            
            result = controller.run(
                before_img, after_img, sample_info, idx, len(remaining),
                window_name,
                mode='review',
                custom_labels=custom_labels,
                show_shortcuts=True,
                on_action=on_action
            )
            
            if result == 'quit':
                self.save_progress()
                self.generate_html_report()
                print("\n✅ 审核进度已保存")
                return
            elif result == 'skipped':
                pass
            elif result in TRAIN_CLASSES:
                self.apply_review(sample, result)
            
            if result == 'back':
                if self.review_results and self.review_results[-1]['name'] == sample['name']:
                    self.review_results.pop()
                    self.save_progress()
                continue
        
        cv2.destroyAllWindows()
        print("\n🎉 审核完成！")
        self.refresh_statistics()
        self.generate_html_report()
    
    def _get_review_shortcut_display(self):
        """获取审核模式的快捷键显示（使用 TRAIN_CLASSES）"""
        key_map = {
            'quality_improved': '1',
            'quality_degradation': '2',
            'render_failed': '3',
            'uncertain_difference': '4',
            'trivial': '5'
        }
        lines = []
        for cls in TRAIN_CLASSES:
            key = key_map.get(cls, '?')
            display = LABEL_DISPLAY.get(cls, cls)
            lines.append(f"  {key} → {display}")
        return lines
    
    def _get_review_custom_labels(self):
        """获取审核自定义标签（使用 TRAIN_CLASSES）"""
        key_map = {
            'quality_improved': '1',
            'quality_degradation': '2',
            'render_failed': '3',
            'uncertain_difference': '4',
            'trivial': '5'
        }
        label_parts = []
        for cls in TRAIN_CLASSES:
            key = key_map.get(cls, '?')
            display = LABEL_DISPLAY.get(cls, cls)
            label_parts.append(f"{key}:{display}")
        
        return [
            '  '.join(label_parts),
            'v:切换视图  ←→:分割  ↑↓:间隔  r:重置',
            's:跳过  n:下一个  p:上一个  q:退出'
        ]
    
    def apply_review(self, sample, new_label, reason=""):
        """
        应用审核结果
        
        Args:
            sample: 样本信息
            new_label: 新标签（必须是 TRAIN_CLASSES 中的类别）
            reason: 修改原因
        """
        # 验证新标签是否在 TRAIN_CLASSES 中
        if new_label not in TRAIN_CLASSES:
            print(f"⚠️ 无效标签: {new_label}，必须是 TRAIN_CLASSES 中的类别")
            return
        
        name = sample['name']
        old_label = sample['pred_label']
        
        # 检查是否已有审核记录
        for r in self.review_results:
            if r['name'] == name:
                r['old_label'] = old_label
                r['new_label'] = new_label
                r['action'] = 'modified' if old_label != new_label else 'confirmed'
                r['timestamp'] = datetime.now().isoformat()
                self.save_progress()
                if old_label != new_label:
                    print(f"🔄 更新: {name} → {new_label}")
                else:
                    print(f"✅ 确认: {name} → {new_label}")
                return
        
        # 新记录
        if old_label == new_label:
            action = 'confirmed'
            print(f"✅ 确认: {name} → {new_label}")
        else:
            action = 'modified'
            self.update_prediction(name, new_label, reason)
            print(f"🔄 修改: {name} → {new_label}")
        
        review_entry = {
            'name': name,
            'old_label': old_label,
            'new_label': new_label,
            'action': action,
            'confidence': sample.get('confidence', 0),
            'ssim_score': sample.get('ssim_score', 0),
            'timestamp': datetime.now().isoformat()
        }
        self.review_results.append(review_entry)
        self.save_progress()
    
    def export_to_annotated(self):
        """导出审核结果到标注集"""
        print("\n📦 导出审核结果到标注集...")
        
        if not self.review_results:
            print("❌ 没有审核结果可导出")
            return
        
        # 创建标注目录 (ALL_CLASSES, 7类)
        for cls in ALL_CLASSES:
            (ANNOTATED_DIR / cls).mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        for r in self.review_results:
            if r.get('action') == 'skipped':
                continue
            
            name = r.get('name')
            new_label = r.get('new_label')
            if not name or not new_label:
                continue
            
            # 验证新标签是否在 TRAIN_CLASSES 中
            if new_label not in TRAIN_CLASSES:
                print(f"⚠️ 跳过 {name}: 标签 {new_label} 不在 TRAIN_CLASSES 中")
                continue
            
            # 在 final_output 中查找样本
            source_dir = None
            for cls in ALL_CLASSES:
                possible_dir = OUTPUT_DIR / cls / name
                if possible_dir.exists():
                    source_dir = possible_dir
                    break
            
            if not source_dir:
                print(f"⚠️ 未找到样本: {name}")
                continue
            
            # 目标目录
            dest_dir = ANNOTATED_DIR / new_label / name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(source_dir, dest_dir)
            
            # 添加审核标记
            info_file = dest_dir / 'review_info.json'
            with open(info_file, 'w') as f:
                json.dump({
                    'name': name,
                    'label': new_label,
                    'source': 'review_export',
                    'reviewed_at': datetime.now().isoformat(),
                    'original_data': r
                }, f, indent=2)
            
            exported_count += 1
        
        print(f"✅ 导出完成: {exported_count} 个样本到 {ANNOTATED_DIR}")
    
    def generate_html_report(self):
        """生成HTML报告 - 使用 ReportGenerator"""
        print("\n📄 生成分类结果报告...")
        
        generator = ReportGenerator(self.project_dir)
        generator.collect_samples()
        generator.generate_html_report()
        
        print(f"\n✅ 报告已生成: {REVIEW_DIR / 'review_report.html'}")


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='预测审核工具')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='交互式审核模式')
    args = parser.parse_args()
    
    print("="*60)
    if args.interactive:
        print("🔍 预测审核工具 - 交互式审核模式")
    else:
        print("📄 预测审核工具 - 报告生成模式")
    print("="*60)
    
    reviewer = PredictionReviewer(interactive=args.interactive)
    
    if not args.interactive:
        # 报告模式：使用 ReportGenerator
        reviewer.generate_html_report()
        return
    
    # ============================================================
    # 交互式审核模式
    # ============================================================
    
    # 1. 选择子类别
    selected_classes = reviewer.select_subcategories()
    
    # 2. 选择采样策略
    strategy, sample_size = reviewer.select_sampling_strategy()
    
    # 3. 采样
    reviewer.sample_for_review(
        selected_classes=selected_classes,
        strategy=strategy,
        sample_size=sample_size
    )
    
    # 4. 启动审核UI
    reviewer.launch_review_ui()
    
    # 5. 询问是否导出到标注集
    print("\n" + "="*60)
    print("📦 导出选项")
    print("="*60)
    print("是否将审核结果导出到标注集?")
    print("  1. 导出到 02_annotated")
    print("  2. 跳过")
    choice = input("\n请选择 (默认: 1): ").strip() or "1"
    
    if choice == '1':
        reviewer.export_to_annotated()
        print(f"\n💡 标注数据已保存到: {ANNOTATED_DIR}")
        print(f"   运行 part2_training.py 重新训练模型")
    
    # 6. 生成报告
    print("\n📄 生成最终报告...")
    reviewer.generate_html_report()
    
    print("\n" + "="*60)
    print("✅ 审核完成!")
    print("="*60)
    print(f"HTML报告: {REVIEW_DIR / 'review_report.html'}")


if __name__ == "__main__":
    main()