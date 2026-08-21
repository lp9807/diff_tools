"""
part4_review.py
预测后审核比对工具（从 final_output 读取结果）

模式：
    - 审核模式 (python part4_review.py -i)：
        1. 选择子类别
        2. 选择采样策略（随机/低置信度/均衡/全部）
        3. 浏览和审核样本
        4. 可修改分类
        5. 导出到标注集
        6. 生成HTML报告
    
    - 报告模式 (python part4_review.py)：
        直接从现有审核结果生成HTML报告
        包含所有样本，按分类显示，鼠标悬停显示缩略图

功能：
1. 从 06_final_output 读取预测结果
2. 按子类别筛选审核样本
3. 支持全部样本浏览模式
4. 审核UI支持三种视图：Split/Animation/Triple
5. 可编辑审核结果并保存进度
6. 修改分类后自动更新 final_output 和统计
7. 生成可展开缩略图的HTML报告（所有样本，按分类显示）
8. 导出审核结果到标注集

依赖: view_renderer.py, config_manager.py
"""

import os
import shutil
import json
import cv2
import numpy as np
import base64
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import sys
import webbrowser
from collections import defaultdict

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

PROJECT_DIR = Path(CONFIG['folders']['project'])

# 目录路径
OUTPUT_DIR = PROJECT_DIR / "06_final_output"
REVIEW_DIR = PROJECT_DIR / "07_review"
ANNOTATED_DIR = PROJECT_DIR / "02_annotated"
TRAIN_DIR = PROJECT_DIR / "03_train_data"
VAL_DIR = PROJECT_DIR / "04_val_data"


# ============================================================
# 审核器
# ============================================================

class PredictionReviewer:
    """
    预测后审核比对工具
    从 06_final_output 读取预测结果
    """
    
    def __init__(self, interactive=True):
        self.interactive = interactive
        
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
                display = LABEL_DISPLAY.get(cls, cls)
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
        """更新预测结果"""
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
        """选择要审核的子类别（包含 to_review）"""
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
            display = LABEL_DISPLAY.get(cls, cls)
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
        """
        采样进行人工审核
        
        Args:
            selected_classes: 选中的类别列表
            strategy: 'all', 'random', 'low_confidence', 'balanced'
            sample_size: 采样数量
        """
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
            display = LABEL_DISPLAY.get(cls, cls)
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
        print("\n快捷键:")
        for i, cls in enumerate(ALL_CLASSES, 1):
            print(f"  {i} → {LABEL_DISPLAY.get(cls, cls)}")
        print("  v → 切换视图模式  ↑↓ → 调整动画间隔")
        print("  r → 重置视图  s → 跳过  q → 退出")
        print("  n → 下一个  p → 上一个")
        print("="*70)
        
        # 从当前进度开始
        for idx, sample in enumerate(remaining):
            # 更新进度
            self.current_idx = len(self.review_results)
            
            before_img = cv2.imread(sample['before_path'])
            after_img = cv2.imread(sample['after_path'])
            
            if before_img is None or after_img is None:
                print(f"⚠️ 跳过 {sample['name']}: 无法读取图片")
                continue
            
            # 重置视图状态
            self.view_state.reset()
            
            window_name = f'审核 - {sample["name"]}'
            cv2.namedWindow(window_name)
            
            while True:
                # 使用共享视图渲染器
                display = ViewRenderer.create_full_view(
                    before_img, after_img,
                    sample=sample,
                    idx=idx,
                    total_samples=total_remaining,
                    view_state=self.view_state,
                    show_shortcuts=True
                )
                
                # 添加进度信息到视图底部
                h = display.shape[0]
                cv2.rectangle(display, (0, h-30), (display.shape[1], h), (0, 0, 0), -1)
                progress_text = f"进度: {len(self.review_results)}/{total_samples}  |  当前: {idx+1}/{total_remaining}  |  [n]下一个  [p]上一个"
                cv2.putText(display, progress_text, (10, h-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.setMouseCallback(window_name, self.mouse_handler.callback)
                cv2.imshow(window_name, display)
                
                key = cv2.waitKey(50) & 0xFF
                
                # ============================================================
                # 标记类型选择
                # ============================================================
                if key == ord('1'):
                    cv2.destroyAllWindows()
                    self.apply_review(sample, ALL_CLASSES[0])
                    break
                elif key == ord('2'):
                    cv2.destroyAllWindows()
                    self.apply_review(sample, ALL_CLASSES[1])
                    break
                elif key == ord('3'):
                    cv2.destroyAllWindows()
                    self.apply_review(sample, ALL_CLASSES[2])
                    break
                elif key == ord('4'):
                    cv2.destroyAllWindows()
                    self.apply_review(sample, ALL_CLASSES[3])
                    break
                elif key == ord('5'):
                    cv2.destroyAllWindows()
                    self.apply_review(sample, ALL_CLASSES[4])
                    break
                
                # ============================================================
                # 导航
                # ============================================================
                elif key == ord('n'):
                    # 下一个 - 保存当前并退出循环
                    cv2.destroyAllWindows()
                    # 如果没有审核就跳过
                    if sample['name'] not in [r['name'] for r in self.review_results]:
                        self.review_results.append({
                            'name': sample['name'],
                            'action': 'skipped',
                            'timestamp': datetime.now().isoformat()
                        })
                        self.save_progress()
                        print(f"⏭️ 跳过 {sample['name']}")
                    break
                elif key == ord('p'):
                    # 上一个 - 回退一个样本
                    cv2.destroyAllWindows()
                    # 移除最后一个审核结果（如果有）
                    if self.review_results and self.review_results[-1]['name'] == sample['name']:
                        self.review_results.pop()
                        self.save_progress()
                    print(f"⏪ 回退到上一个")
                    # 重新获取剩余样本
                    reviewed_names = set(r['name'] for r in self.review_results)
                    remaining_updated = [s for s in self.review_samples if s['name'] not in reviewed_names]
                    if remaining_updated:
                        # 继续循环会处理下一个
                        pass
                    break
                
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
                # 其他
                # ============================================================
                elif key == ord('s'):
                    cv2.destroyAllWindows()
                    self.review_results.append({
                        'name': sample['name'],
                        'action': 'skipped',
                        'timestamp': datetime.now().isoformat()
                    })
                    self.save_progress()
                    print(f"⏭️ 跳过 {sample['name']}")
                    break
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    self.save_progress()
                    print("\n✅ 审核进度已保存")
                    self.generate_html_report()
                    return
                
                # ============================================================
                # 动画逻辑
                # ============================================================
                if self.view_state.display_mode == 'animation':
                    self.view_state.toggle_animation_frame()
                    time.sleep(self.view_state.animation_interval / 1000.0)
        
        cv2.destroyAllWindows()
        print("\n🎉 审核完成！")
        self.refresh_statistics()
        self.generate_html_report()
    
    def apply_review(self, sample, new_label, reason=""):
        """应用审核结果"""
        name = sample['name']
        old_label = sample['pred_label']
        
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
        """导出审核结果到标注集（跳过 to_review）"""
        print("\n📦 导出审核结果到标注集...")
        
        if not self.review_results:
            print("❌ 没有审核结果可导出")
            return
        
        # 创建标注目录（不包含 to_review，因为它不是正式标注类别）
        train_classes = ['quality_improved', 'quality_degradation', 
                         'render_failed', 'uncertain_difference', 'matched']
        for cls in train_classes:
            (ANNOTATED_DIR / cls).mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        for r in self.review_results:
            if r.get('action') == 'skipped':
                continue
            
            name = r.get('name')
            new_label = r.get('new_label')
            if not name or not new_label:
                continue
            
            # 跳过 to_review（它不是正式标注类别）
            if new_label == 'to_review':
                print(f"⏭️ 跳过 {name}: 仍为待复核状态")
                continue
            
            # 在 final_output 中查找样本
            source_dir = None
            all_dirs = ['to_review'] + train_classes
            for cls in all_dirs:
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
     
        # ============================================================
        # HTML 报告生成（包含所有样本，按分类显示，鼠标悬停显示缩略图）
        # ============================================================

        # ============================================================
        # HTML 报告生成（使用图片文件引用，避免内存溢出）
        # ============================================================
        
    def generate_html_report(self):
        """
        生成可视化HTML报告
        
        使用图片文件引用方式，避免内存溢出
        """
        if not self.predictions:
            print("\n⚠️ 没有预测结果，无法生成报告")
            return
        
        print("\n📄 生成可视化HTML报告...")
        
        # 创建报告资源目录
        report_dir = REVIEW_DIR / 'report_files'
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 按分类分组
        grouped = defaultdict(list)
        for p in self.predictions:
            grouped[p['pred_label']].append(p)
        
        # 统计信息
        total = len(self.predictions)
        class_counts = {cls: len(items) for cls, items in grouped.items()}
        
        # 审核统计
        reviewed_names = set(r['name'] for r in self.review_results)
        modified_names = set(r['name'] for r in self.review_results if r.get('action') == 'modified')
        
        # 构建HTML
        html = self.build_html_report_with_files(
            grouped, class_counts, total, 
            reviewed_names, modified_names,
            report_dir
        )
        
        html_path = REVIEW_DIR / 'review_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ HTML报告已生成: {html_path}")
        print(f"📁 图片资源: {report_dir}")
        
        try:
            webbrowser.open(str(html_path.absolute()))
            print("🌐 已在浏览器中打开报告")
        except:
            print(f"请手动打开: {html_path}")
    
    def build_html_report_with_files(self, grouped, class_counts, total, 
                                      reviewed_names, modified_names,
                                      report_dir):
        """构建HTML报告（使用图片文件引用）"""
        
        # 清空图片目录
        for f in report_dir.glob('*'):
            if f.is_file():
                f.unlink()
        
        # 生成每个类别的HTML
        class_sections = []
        display_order = ['to_review', 'render_failed', 'quality_degradation', 
                 'uncertain_difference', 'quality_improved', 'matched']

        
        for cls in display_order:
            if cls not in grouped:
                continue
            
            items = grouped[cls]
            display_name = LABEL_DISPLAY.get(cls, cls)
            color = LABEL_COLORS.get(cls, '#3498db')
            
            rows = []
            for idx, item in enumerate(items):
                name = item['name']
                confidence = item.get('confidence', 0)
                auto_matched = item.get('auto_matched', False)
                ssim_score = item.get('ssim_score', 0)
                
                # 生成缩略图文件
                thumb_files = self.generate_thumb_files(item, report_dir, idx)
                
                # 审核状态
                if name in modified_names:
                    status = '🔄 已修改'
                    status_class = 'modified'
                elif name in reviewed_names:
                    status = '✅ 已确认'
                    status_class = 'confirmed'
                else:
                    status = '⏳ 待审核'
                    status_class = 'pending'
                
                conf_display = f"{confidence:.3f}"
                if auto_matched:
                    conf_display += " (自动匹配)"
                elif confidence < 0.6:
                    conf_display += " ⚠️"
                
                # 构建缩略图悬停HTML（使用文件路径）
                thumb_html = self.build_thumb_html_with_files(thumb_files, name)
                
                row_html = '''
                <tr class="{status_class}">
                    <td class="sample-name">{name}</td>
                    <td class="thumb-cell">
                        {thumb_html}
                    </td>
                    <td>{conf_display}</td>
                    <td>{ssim_display}</td>
                    <td class="status-{status_class}">{status}</td>
                </tr>
                '''.format(
                    status_class=status_class,
                    name=name,
                    thumb_html=thumb_html,
                    conf_display=conf_display,
                    ssim_display=f"{ssim_score:.6f}" if ssim_score else '-',
                    status=status
                )
                rows.append(row_html)
            
            is_default_expanded = cls == 'render_failed'
            icon_symbol = '▼' if is_default_expanded else '▶'
            display_style = 'block' if is_default_expanded else 'none'
            
            section_html = '''
            <div class="class-section">
                <div class="class-header" onclick="toggleClass('{cls}')" 
                     style="background: {color}22; border-left: 4px solid {color};">
                    <span class="class-name">{display_name}</span>
                    <span class="class-count">{count} 个样本</span>
                    <span class="toggle-icon" id="icon-{cls}">
                        {icon_symbol}
                    </span>
                </div>
                <div class="class-body" id="body-{cls}" 
                     style="display: {display_style};">
                    <table class="details-table">
                        <thead>
                            <tr>
                                <th>样本名</th>
                                <th>图片预览 (悬停查看)</th>
                                <th>置信度</th>
                                <th>SSIM</th>
                                <th>状态</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows}
                        </tbody>
                    </table>
                </div>
            </div>
            '''.format(
                cls=cls,
                color=color,
                display_name=display_name,
                count=len(items),
                icon_symbol=icon_symbol,
                display_style=display_style,
                rows=''.join(rows)
            )
            class_sections.append(section_html)
        
        stat_cards = self._build_class_stats(class_counts)
        
        html = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>预测结果审核报告</title>
    <style>
        * {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; box-sizing: border-box; }}
        body {{ 
            background: #f0f2f5; 
            margin: 0; 
            padding: 20px;
            color: #2c3e50;
        }}
        .container {{ 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        .subtitle {{
            color: #7f8c8d;
            margin-bottom: 25px;
            font-size: 14px;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin: 20px 0 30px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 12px 16px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        .stat-card .number {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-card .label {{
            font-size: 13px;
            color: #7f8c8d;
        }}
        .stat-card.green {{ border-color: #2ecc71; }}
        .stat-card.orange {{ border-color: #f39c12; }}
        .stat-card.red {{ border-color: #e74c3c; }}
        .stat-card.blue {{ border-color: #3498db; }}
        .stat-card.purple {{ border-color: #9b59b6; }}
        
        .class-section {{
            margin: 15px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        .class-header {{
            padding: 12px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
            transition: background 0.2s;
        }}
        .class-header:hover {{
            filter: brightness(0.95);
        }}
        .class-name {{
            font-size: 16px;
            font-weight: 600;
        }}
        .class-count {{
            font-size: 14px;
            color: #7f8c8d;
            margin-left: auto;
            margin-right: 15px;
        }}
        .toggle-icon {{
            font-size: 16px;
            font-weight: bold;
            color: #7f8c8d;
            transition: transform 0.3s;
        }}
        .class-body {{
            padding: 0;
            overflow-x: auto;
        }}
        
        .details-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .details-table th {{
            background: #34495e;
            color: white;
            padding: 10px 12px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .details-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #ecf0f1;
            vertical-align: middle;
        }}
        .details-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .status-confirmed {{ color: #27ae60; }}
        .status-modified {{ color: #e67e22; }}
        .status-pending {{ color: #95a5a6; }}
        
        .details-table tr.confirmed {{
            border-left: 3px solid #2ecc71;
        }}
        .details-table tr.modified {{
            border-left: 3px solid #f39c12;
        }}
        .details-table tr.pending {{
            border-left: 3px solid #bdc3c7;
        }}
        
        .thumb-cell {{
            position: relative;
            width: 120px;
            min-width: 120px;
        }}
        .thumb-trigger {{
            color: #3498db;
            cursor: pointer;
            font-size: 13px;
            padding: 4px 10px;
            border: 1px dashed #3498db;
            border-radius: 4px;
            display: inline-block;
            background: #f0f7ff;
            transition: all 0.2s;
        }}
        .thumb-trigger:hover {{
            background: #3498db;
            color: white;
        }}
        
        .thumb-popup {{
            display: none;
            position: fixed;
            z-index: 1000;
            background: white;
            border-radius: 10px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.3);
            padding: 12px;
            max-width: 700px;
            min-width: 400px;
            pointer-events: none;
            border: 2px solid #e0e0e0;
        }}
        .thumb-popup.show {{
            display: block;
        }}
        .thumb-popup .thumb-row {{
            display: flex;
            gap: 8px;
            align-items: center;
            justify-content: center;
        }}
        .thumb-popup .thumb-img {{
            max-width: 200px;
            max-height: 150px;
            border-radius: 6px;
            border: 1px solid #ddd;
            object-fit: contain;
        }}
        .thumb-popup .thumb-label {{
            text-align: center;
            font-size: 11px;
            color: #7f8c8d;
            margin-top: 2px;
        }}
        .thumb-popup .thumb-arrow {{
            font-size: 20px;
            color: #95a5a6;
            padding: 0 4px;
        }}
        .thumb-popup .sample-name-popup {{
            text-align: center;
            font-weight: 600;
            font-size: 12px;
            color: #2c3e50;
            margin-bottom: 6px;
            background: #f0f2f5;
            padding: 4px 10px;
            border-radius: 4px;
        }}
        
        .footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #ecf0f1;
            color: #95a5a6;
            font-size: 13px;
            text-align: center;
        }}
        
        @media (max-width: 768px) {{
            .stats {{ grid-template-columns: repeat(2, 1fr); }}
            .details-table {{ font-size: 12px; }}
            .thumb-popup {{
                max-width: 300px;
                min-width: 200px;
            }}
            .thumb-popup .thumb-img {{
                max-width: 100px;
                max-height: 80px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 预测结果审核报告</h1>
        <div class="subtitle">
            生成时间: {timestamp} &nbsp;|&nbsp; 
            总计: {total} 个样本
        </div>
        
        <div class="stats">
            <div class="stat-card blue">
                <div class="number">{total}</div>
                <div class="label">📊 总样本</div>
            </div>
            <div class="stat-card green">
                <div class="number">{reviewed_count}</div>
                <div class="label">✅ 已审核</div>
            </div>
            <div class="stat-card orange">
                <div class="number">{modified_count}</div>
                <div class="label">🔄 已修改</div>
            </div>
            {stat_cards}
        </div>
        
        <div style="margin: 15px 0 20px 0; font-size: 13px; display: flex; gap: 20px; flex-wrap: wrap;">
            <span><span style="display:inline-block;width:12px;height:12px;background:#2ecc71;border-radius:2px;"></span> 已确认</span>
            <span><span style="display:inline-block;width:12px;height:12px;background:#f39c12;border-radius:2px;"></span> 已修改</span>
            <span><span style="display:inline-block;width:12px;height:12px;background:#bdc3c7;border-radius:2px;"></span> 待审核</span>
            <span style="color:#7f8c8d;">💡 点击类别标题展开/收起</span>
            <span style="color:#7f8c8d;">🖱️ 悬停 "查看图片" 预览 before/after/diff</span>
        </div>
        
        {class_sections}
        
        <div class="footer">
            报告由 Review Tool 自动生成 &nbsp;|&nbsp; 
            <span style="color:#e74c3c;">render_failed</span> 类别默认展开
        </div>
    </div>
    
    <div class="thumb-popup" id="thumbPopup">
        <div class="sample-name-popup" id="popupName">样本名</div>
        <div class="thumb-row">
            <div>
                <div class="thumb-label">BEFORE</div>
                <img class="thumb-img" id="popupBefore" src="" alt="before" />
            </div>
            <span class="thumb-arrow">→</span>
            <div>
                <div class="thumb-label">AFTER</div>
                <img class="thumb-img" id="popupAfter" src="" alt="after" />
            </div>
            <span class="thumb-arrow">→</span>
            <div>
                <div class="thumb-label">DIFF</div>
                <img class="thumb-img" id="popupDiff" src="" alt="diff" />
            </div>
        </div>
    </div>
    
    <script>
        function toggleClass(cls) {{
            var body = document.getElementById('body-' + cls);
            var icon = document.getElementById('icon-' + cls);
            if (body.style.display === 'none') {{
                body.style.display = 'block';
                icon.textContent = '▼';
            }} else {{
                body.style.display = 'none';
                icon.textContent = '▶';
            }}
        }}
        
        var popup = document.getElementById('thumbPopup');
        var popupBefore = document.getElementById('popupBefore');
        var popupAfter = document.getElementById('popupAfter');
        var popupDiff = document.getElementById('popupDiff');
        var popupName = document.getElementById('popupName');
        var hideTimeout = null;
        
        function showThumbPopup(event, name, beforeSrc, afterSrc, diffSrc) {{
            if (hideTimeout) {{
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }}
            
            popupName.textContent = name;
            popupBefore.src = beforeSrc || '';
            popupAfter.src = afterSrc || '';
            popupDiff.src = diffSrc || '';
            
            var rect = event.target.getBoundingClientRect();
            var left = rect.left + rect.width / 2 - 200;
            var top = rect.bottom + 10;
            
            if (left < 10) left = 10;
            if (left + 400 > window.innerWidth - 10) left = window.innerWidth - 410;
            if (top + 200 > window.innerHeight - 10) {{
                top = rect.top - 220;
            }}
            
            popup.style.left = left + 'px';
            popup.style.top = top + 'px';
            popup.classList.add('show');
        }}
        
        function hideThumbPopup() {{
            hideTimeout = setTimeout(function() {{
                popup.classList.remove('show');
                hideTimeout = null;
            }}, 200);
        }}
        
        popup.addEventListener('mouseenter', function() {{
            if (hideTimeout) {{
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }}
        }});
        
        popup.addEventListener('mouseleave', function() {{
            hideThumbPopup();
        }});
    </script>
</body>
</html>
        '''.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total=total,
            reviewed_count=len(reviewed_names),
            modified_count=len(modified_names),
            stat_cards=stat_cards,
            class_sections=''.join(class_sections)
        )
        
        return html
    
    def generate_thumb_files(self, item, report_dir, idx):
        """
        生成缩略图文件（before, after, diff）
        
        Returns:
            dict: {'before': 'path', 'after': 'path', 'diff': 'path'}
        """
        name = item['name']
        thumb_files = {}
        
        try:
            # Before 图片
            before_path = Path(item['before_path'])
            if before_path.exists():
                before_dest = report_dir / f"{name}_before.png"
                shutil.copy2(before_path, before_dest)
                thumb_files['before'] = f"report_files/{name}_before.png"
            
            # After 图片
            after_path = Path(item['after_path'])
            if after_path.exists():
                after_dest = report_dir / f"{name}_after.png"
                shutil.copy2(after_path, after_dest)
                thumb_files['after'] = f"report_files/{name}_after.png"
            
            # Diff 图片（生成）
            diff_data = self.compute_diff_image_data(item['before_path'], item['after_path'])
            if diff_data:
                diff_dest = report_dir / f"{name}_diff.png"
                with open(diff_dest, 'wb') as f:
                    f.write(diff_data)
                thumb_files['diff'] = f"report_files/{name}_diff.png"
        
        except Exception as e:
            print(f"⚠️ 生成缩略图失败 {name}: {e}")
        
        return thumb_files
    
    def compute_diff_image_data(self, before_path, after_path):
        """计算差异图并返回PNG数据"""
        try:
            before = cv2.imread(before_path)
            after = cv2.imread(after_path)
            
            if before is None or after is None:
                return None
            
            if before.shape != after.shape:
                h = min(before.shape[0], after.shape[0])
                w = min(before.shape[1], after.shape[1])
                before = cv2.resize(before, (w, h))
                after = cv2.resize(after, (w, h))
            
            diff = cv2.absdiff(before, after)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
            
            _, buffer = cv2.imencode('.png', diff_colored)
            return buffer.tobytes()
        except Exception as e:
            print(f"⚠️ 计算差异图失败: {e}")
            return None
    
    def build_thumb_html_with_files(self, thumb_files, name):
        """构建缩略图悬停HTML（使用文件路径）"""
        before_src = thumb_files.get('before', '')
        after_src = thumb_files.get('after', '')
        diff_src = thumb_files.get('diff', '')
        
        name_escaped = name.replace("'", "\\'")
        
        return '''
        <span class="thumb-trigger" 
              onmouseenter="showThumbPopup(event, '{name}', '{before}', '{after}', '{diff}')"
              onmouseleave="hideThumbPopup()">
            🖼️ 查看图片
        </span>
        '''.format(
            name=name_escaped,
            before=before_src,
            after=after_src,
            diff=diff_src
        )
    
    def _build_class_stats(self, class_counts):
        """构建类别统计卡片"""
        cards = []
        for cls in ALL_CLASSES:
            count = class_counts.get(cls, 0)
            if count == 0:
                continue
            color = LABEL_COLORS.get(cls, '#3498db')
            display = LABEL_DISPLAY.get(cls, cls)
            cards.append(f'''
            <div class="stat-card" style="border-color: {color};">
                <div class="number">{count}</div>
                <div class="label">{display}</div>
            </div>
            ''')
        return ''.join(cards)
    
   
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
        # 报告模式：直接从现有结果生成HTML
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
    
    print("\n" + "="*60)
    print("✅ 审核完成!")
    print("="*60)
    print(f"HTML报告: {REVIEW_DIR / 'review_report.html'}")


if __name__ == "__main__":
    main()