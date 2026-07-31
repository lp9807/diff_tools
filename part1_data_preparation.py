"""
part1_data_preparation.py
数据准备和标注工具
"""

import os
import shutil
import random
import json
import cv2
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

class DataPreparation:
    """数据准备和标注管理"""
    
    def __init__(self, raw_dir, project_dir):
        self.raw_dir = Path(raw_dir)
        self.project_dir = Path(project_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """创建项目目录结构"""
        dirs = {
            'raw': self.raw_dir,
            'to_annotate': self.project_dir / '01_to_annotate',
            'annotated': self.project_dir / '02_annotated',
            'train': self.project_dir / '03_train_data',
            'val': self.project_dir / '04_val_data',
            'predict': self.project_dir / '05_predict',
            'output': self.project_dir / '06_final_output',
            'checkpoints': self.project_dir / 'checkpoints',
            'logs': self.project_dir / 'logs'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 创建分类子目录
        for cls in ['quality_improved', 'render_failed']:
            (dirs['annotated'] / cls).mkdir(exist_ok=True)
            (dirs['train'] / cls).mkdir(parents=True, exist_ok=True)
            (dirs['val'] / cls).mkdir(parents=True, exist_ok=True)
            (dirs['output'] / cls).mkdir(parents=True, exist_ok=True)
        
        self.dirs = dirs
        print(f"✅ 项目目录创建完成: {self.project_dir}")
    
    def get_all_images(self):
        """获取所有图像"""
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(self.raw_dir.glob(ext))
        return images
    
    def sample_for_annotation(self, n_samples=80, strategy='diverse'):
        """
        采样图片进行标注
        
        Args:
            n_samples: 采样数量
            strategy: 'random' 或 'diverse'
        """
        images = self.get_all_images()
        
        # 安全检查
        if not images:
            print(f"\n❌ 错误：在 {self.raw_dir} 中没有找到任何图片！")
            print(f"支持的格式: .jpg, .jpeg, .png, .bmp")
            print(f"\n请检查以下事项：")
            print(f"  1. 目录是否存在: {self.raw_dir.exists()}")
            print(f"  2. 图片是否直接放在 {self.raw_dir} 下（不能有子文件夹）")
            print(f"  3. 图片扩展名是否在支持列表中")
            print(f"\n当前目录内容：")
            if self.raw_dir.exists():
                all_items = list(self.raw_dir.glob('*'))
                for item in all_items[:10]:
                    print(f"  - {item.name} ({'文件夹' if item.is_dir() else '文件'})")
                if len(all_items) > 10:
                    print(f"  ... 还有 {len(all_items)-10} 个项目")
            return []
        
        print(f"✅ 找到 {len(images)} 张图片")
        
        
        if strategy == 'random':
            selected = random.sample(images, min(n_samples, len(images)))
        else:  # diverse
            # 按文件名排序后均匀采样
            images = sorted(images)
            if len(images) > n_samples:
                step = len(images) // n_samples
                selected = images[::step][:n_samples]
            else:
                selected = images
        
        # 复制到待标注目录
        for img in selected:
            shutil.copy2(img, self.dirs['to_annotate'] / img.name)
        
        # 保存采样记录
        sample_info = {
            'total_images': len(images),
            'sampled_count': len(selected),
            'sampled_files': [str(img.name) for img in selected],
            'strategy': strategy,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.dirs['to_annotate'] / 'sample_info.json', 'w') as f:
            json.dump(sample_info, f, indent=2)
        
        print(f"✅ 已采样 {len(selected)} 张图片到: {self.dirs['to_annotate']}")
        return selected

class InteractiveAnnotator:
    """交互式标注工具"""
    
    def __init__(self, annotate_dir, output_dir):
        self.annotate_dir = Path(annotate_dir)
        self.output_dir = Path(output_dir)
        self.images = list(self.annotate_dir.glob('*.*'))
        self.images = [f for f in self.images if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        self.annotations = {}
        self.current_idx = 0
        self.progress_file = self.annotate_dir / 'annotation_progress.json'
        
    def start_annotation(self):
        """开始标注"""
        self.load_progress()
        
        if not self.images:
            print("\n❌ 错误：没有找到任何待标注的图片！")
            print(f"待标注目录: {self.annotate_dir}")
            print("\n可能原因：")
            print("  1. 采样步骤失败，没有生成图片")
            print("  2. 原始图像目录 {self.annotate_dir} 为空")
            print("  3. 图片格式不被支持")
            print("\n建议：")
            print("  1. 检查 raw_images 文件夹是否有图片")
            print("  2. 重新运行数据准备步骤")
            return
        
        print(f"\n✅ 找到 {len(self.images)} 张图片待标注")
        
        print("\n" + "="*60)
        print("交互式标注工具")
        print("="*60)
        print("快捷键说明:")
        print("  1 → 质量提升 (quality_improved)")
        print("  2 → 渲染失败 (render_failed)")
        print("  s → 跳过此图")
        print("  b → 回退上一张")
        print("  q → 退出保存")
        print("="*60)
        
        for idx in range(self.current_idx, len(self.images)):
            img_path = self.images[idx]
            
            # 如果已经标注过，跳过
            if str(img_path) in self.annotations:
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"❌ 无法读取: {img_path.name}")
                continue
            
            # 显示图片
            display_img = cv2.resize(img, (800, 600))
            cv2.imshow(f'标注 ({idx+1}/{len(self.images)})', display_img)
            
            label = None
            while True:
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('1'):
                    label = 'quality_improved'
                    break
                elif key == ord('2'):
                    label = 'render_failed'
                    break
                elif key == ord('s'):
                    label = 'skipped'
                    break
                elif key == ord('b'):
                    if idx > 0:
                        idx -= 2  # 回退
                        break
                elif key == ord('q'):
                    self.save_progress()
                    cv2.destroyAllWindows()
                    print("\n⚠️ 已保存进度，退出标注")
                    return
                else:
                    print("  按 1=质量提升, 2=渲染失败, s=跳过, b=回退, q=退出")
            
            cv2.destroyAllWindows()
            
            if label and label != 'skipped':
                # 保存到标注目录
                dest_dir = self.output_dir / label
                shutil.copy2(str(img_path), str(dest_dir / img_path.name))
                self.annotations[str(img_path)] = label
                print(f"✅ [{idx+1}/{len(self.images)}] {img_path.name} → {label}")
            elif label == 'skipped':
                print(f"⏭️  [{idx+1}/{len(self.images)}] {img_path.name} → 跳过")
            
            self.current_idx = idx + 1
            self.save_progress()
        
        cv2.destroyAllWindows()
        print("\n✅ 标注完成！")
        self.generate_report()
    
    def load_progress(self):
        """加载进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                self.current_idx = data.get('current_idx', 0)
                self.annotations = data.get('annotations', {})
                print(f"📂 恢复进度: 已标注 {len(self.annotations)} 张")
    
    def save_progress(self):
        """保存进度"""
        data = {
            'current_idx': self.current_idx,
            'annotations': self.annotations,
            'total': len(self.images)
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_report(self):
        """生成标注报告"""
        # 统计标注结果
        if len(self.images) == 0:
            print("⚠️ 警告：没有图片数据，跳过报告生成")
            return
        
        label_counts = {}
        for label in self.annotations.values():
            label_counts[label] = label_counts.get(label, 0) + 1
        
        report = {
            'total_images': len(self.images),
            'annotated_count': len(self.annotations),
            'label_counts': label_counts,
            'annotation_ratio': len(self.annotations) / len(self.images),
            'timestamp': datetime.now().isoformat()
        }
        
        report_file = self.annotate_dir / 'annotation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "="*60)
        print("📊 标注统计报告")
        print("="*60)
        print(f"总图片: {report['total_images']}")
        print(f"已标注: {report['annotated_count']}")
        for label, count in label_counts.items():
            print(f"  - {label}: {count} 张 ({count/report['annotated_count']*100:.1f}%)")
        print(f"\n报告保存至: {report_file}")

def prepare_dataset(annotated_dir, train_dir, val_dir, val_ratio=0.2):
    """
    将标注数据划分为训练集和验证集
    """
    for cls in ['quality_improved', 'render_failed']:
        src_dir = Path(annotated_dir) / cls
        if not src_dir.exists():
            continue
        
        images = list(src_dir.glob('*.*'))
        random.shuffle(images)
        
        split_idx = int(len(images) * (1 - val_ratio))
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # 复制到训练集
        for img in train_images:
            dest = Path(train_dir) / cls / img.name
            shutil.copy2(img, dest)
        
        # 复制到验证集
        for img in val_images:
            dest = Path(val_dir) / cls / img.name
            shutil.copy2(img, dest)
        
        print(f"{cls}: 训练集={len(train_images)}, 验证集={len(val_images)}")

# ============================================================
# 主程序
# ============================================================

def main_prepare():
    """数据准备主程序"""
    print("="*60)
    print("第1步：数据准备和标注")
    print("="*60)
    
    # 1. 初始化数据准备
    data_prep = DataPreparation(
        raw_dir='./raw_images',  # 你的原始图像文件夹
        project_dir='./project'
    )
    
    # 2. 采样图片进行标注
    print("\n📌 开始采样...")
    data_prep.sample_for_annotation(n_samples=80, strategy='diverse')
    
    # 3. 启动交互式标注
    print("\n📌 开始标注...")
    annotator = InteractiveAnnotator(
        annotate_dir='./project/01_to_annotate',
        output_dir='./project/02_annotated'
    )
    annotator.start_annotation()
    
    # 4. 划分数据集
    print("\n📌 划分数据集...")
    prepare_dataset(
        annotated_dir='./project/02_annotated',
        train_dir='./project/03_train_data',
        val_dir='./project/04_val_data',
        val_ratio=0.2
    )
    
    print("\n✅ 数据准备完成！")
    print(f"训练数据: ./project/03_train_data")
    print(f"验证数据: ./project/04_val_data")

if __name__ == "__main__":
    main_prepare()