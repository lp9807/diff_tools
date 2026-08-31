"""
part3_pair_prediction.py
自动分类所有成对图片 - 基于你的 config.json

流程：
1. 先计算 SSIM 相似度
2. SSIM ≥ 阈值(默认0.999) → 自动标记为 matched
3. SSIM < 阈值 → 用模型预测
    - 置信度 ≥ 阈值 → 归入对应类别
    - 置信度 < 阈值 → 放入 to_review（待人工复核）

依赖: scikit-image
安装: pip install scikit-image
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict

# ============================================================
# 导入共享配置
# ============================================================

from model_config import (
    TRAIN_CLASSES, CLASS_TO_IDX, LABEL_DISPLAY,
    load_model, find_model_file,
    SSIM_THRESHOLD, SSIMComparator
)

# ============================================================
# 加载配置
# ============================================================

with open('config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

BEFORE_DIR = Path(CONFIG['folders']['before'])
AFTER_DIR = Path(CONFIG['folders']['after'])
PROJECT_DIR = Path(CONFIG['folders']['project'])
OUTPUT_DIR = PROJECT_DIR / "06_final_output"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
THRESHOLD = CONFIG['classification']['confidence_threshold']
EXTENSIONS = CONFIG['pairing']['extensions']

# ============================================================
# 预测数据集
# ============================================================

class PairPredictionDataset(Dataset):
    def __init__(self, before_dir, after_dir, transform=None):
        self.before_dir = Path(before_dir)
        self.after_dir = Path(after_dir)
        self.transform = transform
        self.pairs = self.match_pairs()
        print(f"找到 {len(self.pairs)} 对待预测")
    
    def match_pairs(self):
        def get_files(directory):
            files = []
            for ext in EXTENSIONS:
                files.extend(directory.glob(f'*{ext}'))
                files.extend(directory.glob(f'*{ext.upper()}'))
            return {f.stem: f for f in files}
        
        before_files = get_files(self.before_dir)
        after_files = get_files(self.after_dir)
        
        common_names = set(before_files.keys()) & set(after_files.keys())
        pairs = []
        for name in common_names:
            pairs.append({
                'name': name,
                'before': before_files[name],
                'after': after_files[name]
            })
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        before = Image.open(pair['before']).convert('RGB')
        after = Image.open(pair['after']).convert('RGB')
        if self.transform:
            before = self.transform(before)
            after = self.transform(after)
        return before, after, pair['name'], str(pair['before']), str(pair['after'])


# ============================================================
# 预测主程序
# ============================================================

def main():
    print("="*60)
    print("🚀 成对图片自动分类 (SSIM + 模型预测)")
    print("="*60)
    print(f"SSIM 阈值: {SSIM_THRESHOLD}")
    print(f"  SSIM ≥ {SSIM_THRESHOLD} → 自动匹配 (matched)")
    print(f"  SSIM < {SSIM_THRESHOLD} → 模型预测")
    print(f"    置信度 ≥ {THRESHOLD} → 归入预测类别")
    print(f"    置信度 < {THRESHOLD} → 放入 to_review (待人工复核)")
    print(f"训练类别: {', '.join(TRAIN_CLASSES)}")
    print("="*60)
    
    # 查找模型文件
    model_path = find_model_file(CHECKPOINT_DIR)
    
    if model_path is None:
        print(f"❌ 模型不存在: {CHECKPOINT_DIR}")
        print("请先运行 part2_pair_training.py")
        return
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, num_classes, saved_classes = load_model(model_path, device)
    
    if model is None:
        print("❌ 模型加载失败")
        return
    
    print(f"✅ 模型加载完成 (类别数: {num_classes})")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 准备数据
    dataset = PairPredictionDataset(BEFORE_DIR, AFTER_DIR, transform)
    if len(dataset) == 0:
        print("❌ 没有匹配到任何成对图片！")
        return
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # 创建输出目录
    to_review_dir = OUTPUT_DIR / 'to_review'
    matched_dir = OUTPUT_DIR / 'matched'
    
    for cls in TRAIN_CLASSES:
        (OUTPUT_DIR / cls).mkdir(parents=True, exist_ok=True)
    to_review_dir.mkdir(parents=True, exist_ok=True)
    
    # 清空旧结果
    for cls in TRAIN_CLASSES:
        for item in (OUTPUT_DIR / cls).iterdir():
            if item.is_dir():
                shutil.rmtree(item)
    for item in to_review_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
    
    results = []
    auto_matched_count = 0
    model_predicted_count = 0
    low_confidence_count = 0
    
    print(f"\n🔮 开始处理 {len(dataset)} 对图片...\n")
    
    with torch.no_grad():
        for before, after, names, before_paths, after_paths in tqdm(dataloader, desc="处理中"):
            for i, name in enumerate(names):
                # 计算 SSIM 相似度
                ssim_score = SSIMComparator.compute_ssim(before_paths[i], after_paths[i])
                
                # ============================================================
                # SSIM ≥ 阈值：自动匹配
                # ============================================================
                if ssim_score >= SSIM_THRESHOLD:
                    pred_class = 'matched'
                    confidence = ssim_score
                    auto_matched_count += 1
                    
                    dest_dir = matched_dir / name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(before_paths[i], dest_dir / 'before.png')
                    shutil.copy2(after_paths[i], dest_dir / 'after.png')
                    
                    with open(dest_dir / 'prediction_info.json', 'w') as f:
                        json.dump({
                            'name': name,
                            'predicted_class': pred_class,
                            'confidence': confidence,
                            'ssim_score': ssim_score,
                            'auto_matched': True
                        }, f, indent=2)
                    
                    results.append({
                        'name': name,
                        'predicted_class': pred_class,
                        'confidence': confidence,
                        'auto_matched': True,
                        'ssim_score': ssim_score
                    })
                    
                    continue
                
                # ============================================================
                # SSIM < 阈值：模型预测
                # ============================================================
                before_tensor = before[i:i+1].to(device)
                after_tensor = after[i:i+1].to(device)
                
                outputs = model(before_tensor, after_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, dim=1)
                
                pred_idx = pred.item()
                if pred_idx < len(saved_classes):
                    pred_class = saved_classes[pred_idx]
                else:
                    pred_class = 'matched'
                
                conf = confidence.item()
                model_predicted_count += 1
                
                # ============================================================
                # 判断置信度是否达标
                # ============================================================
                if conf >= THRESHOLD:
                    # 高置信度：归入预测类别
                    dest_dir = OUTPUT_DIR / pred_class / name
                    status_text = f"✅ 归入 {LABEL_DISPLAY.get(pred_class, pred_class)}"
                else:
                    # 低置信度：放入 to_review
                    dest_dir = to_review_dir / name
                    low_confidence_count += 1
                    status_text = f"⚠️ 置信度 {conf:.3f} < {THRESHOLD}，放入 to_review"
                
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(before_paths[i], dest_dir / 'before.png')
                shutil.copy2(after_paths[i], dest_dir / 'after.png')
                
                with open(dest_dir / 'prediction_info.json', 'w') as f:
                    json.dump({
                        'name': name,
                        'predicted_class': pred_class,
                        'confidence': conf,
                        'ssim_score': ssim_score,
                        'auto_matched': False,
                        'low_confidence': conf < THRESHOLD,
                        'all_probs': probs[0].cpu().numpy().tolist()
                    }, f, indent=2)
                
                results.append({
                    'name': name,
                    'predicted_class': pred_class if conf >= THRESHOLD else 'to_review',
                    'confidence': conf,
                    'auto_matched': False,
                    'ssim_score': ssim_score,
                    'low_confidence': conf < THRESHOLD
                })
                
                # 打印详细状态
                print(f"  {status_text}: {name} (SSIM: {ssim_score:.6f})")
    
    # 统计结果
    total = len(results)
    class_counts = defaultdict(int)
    for r in results:
        class_counts[r['predicted_class']] += 1
    
    print("\n" + "="*60)
    print("📊 预测完成报告")
    print("="*60)
    print(f"总成对数: {total}")
    print(f"自动匹配 (SSIM ≥ {SSIM_THRESHOLD}): {auto_matched_count} ({auto_matched_count/total*100:.1f}%)")
    print(f"模型预测 (SSIM < {SSIM_THRESHOLD}): {model_predicted_count} ({model_predicted_count/total*100:.1f}%)")
    print(f"  其中高置信度 (≥ {THRESHOLD}): {model_predicted_count - low_confidence_count}")
    print(f"  其中低置信度 (< {THRESHOLD}): {low_confidence_count} (放入 to_review)")
    print(f"\n类别分布:")
    for cls, count in class_counts.items():
        if cls == 'to_review':
            display = '📌 待复核 (to_review)'
        else:
            display = LABEL_DISPLAY.get(cls, cls)
        print(f"  - {display}: {count} 对 ({count/total*100:.1f}%)")
    
    # 保存结果摘要
    summary = {
        'total_pairs': total,
        'auto_matched': auto_matched_count,
        'model_predicted': model_predicted_count,
        'high_confidence': model_predicted_count - low_confidence_count,
        'low_confidence': low_confidence_count,
        'class_distribution': dict(class_counts),
        'ssim_threshold': SSIM_THRESHOLD,
        'model_confidence_threshold': THRESHOLD,
        'classes': saved_classes,
        'timestamp': datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / 'prediction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    to_review_count = len(list(to_review_dir.iterdir()))
    print(f"\n🎉 分类完成！结果保存在: {OUTPUT_DIR}")
    for cls in TRAIN_CLASSES:
        count = class_counts.get(cls, 0)
        if count > 0:
            display = LABEL_DISPLAY.get(cls, cls)
            print(f"  - {display}: {count} 对")
    print(f"  - 📌 待复核 (to_review): {to_review_count} 个样本 (置信度 < {THRESHOLD})")
    print(f"\n💡 提示: to_review 中的样本需要人工复核确认分类")
    print(f"   运行 python part4_review.py -i 进行审核")


if __name__ == "__main__":
    main()