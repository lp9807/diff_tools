"""
part2_pair_training.py
成对数据训练 - 基于你的 config.json

支持所有训练类别：
    - quality_improved       : 质量提升
    - quality_degradation    : 质量退化
    - render_failed          : 渲染失败
    - uncertain_difference   : 不确定差异
    - matched                : 匹配

共享配置: model_config.py
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# ============================================================
# 导入共享配置
# ============================================================

from model_config import (
    TRAIN_CLASSES, CLASS_TO_IDX, LABEL_DISPLAY,
    SiameseNetwork, create_model, SSIM_THRESHOLD
)

# ============================================================
# 加载配置
# ============================================================

with open('config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

PROJECT_DIR = Path(CONFIG['folders']['project'])
TRAIN_DIR = PROJECT_DIR / "03_train_data"
VAL_DIR = PROJECT_DIR / "04_val_data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

BATCH_SIZE = CONFIG['training']['batch_size']
EPOCHS = CONFIG['training']['epochs']
LR = CONFIG['training']['learning_rate']

# ============================================================
# 数据集
# ============================================================

class PairDataset(Dataset):
    """成对数据集 - 只读取训练类别"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.pairs = []
        self.labels = []
        self.class_names = []
        
        print(f"\n📂 加载数据集: {self.root_dir}")
        
        for cls_name in TRAIN_CLASSES:
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                print(f"  ⚠️ 类别目录不存在: {cls_name}")
                continue
            
            cls_idx = CLASS_TO_IDX[cls_name]
            pair_dirs = [d for d in cls_dir.iterdir() if d.is_dir()]
            
            for pair_dir in pair_dirs:
                before_path = pair_dir / 'before.png'
                after_path = pair_dir / 'after.png'
                
                if not before_path.exists():
                    before_files = list(pair_dir.glob('before.*'))
                    before_path = before_files[0] if before_files else None
                if not after_path.exists():
                    after_files = list(pair_dir.glob('after.*'))
                    after_path = after_files[0] if after_files else None
                
                if before_path and after_path:
                    self.pairs.append((str(before_path), str(after_path)))
                    self.labels.append(cls_idx)
                    self.class_names.append(cls_name)
        
        print(f"  ✅ 加载完成: {len(self.pairs)} 对")
        
        label_counts = {}
        for cls_name in TRAIN_CLASSES:
            count = self.class_names.count(cls_name)
            if count > 0:
                label_counts[cls_name] = count
                display = LABEL_DISPLAY.get(cls_name, cls_name)
                print(f"  - {display}: {count} 对")
        
        if not self.pairs:
            print("  ❌ 没有找到任何数据！")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        before_path, after_path = self.pairs[idx]
        before = Image.open(before_path).convert('RGB')
        after = Image.open(after_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            before = self.transform(before)
            after = self.transform(after)
        
        return before, after, label


# ============================================================
# 训练
# ============================================================

def main():
    print("="*60)
    print("🚀 成对数据训练")
    print("="*60)
    print(f"类别: {', '.join(TRAIN_CLASSES)}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据
    train_dataset = PairDataset(TRAIN_DIR, train_transform)
    val_dataset = PairDataset(VAL_DIR, val_transform)
    
    if len(train_dataset) == 0:
        print("\n❌ 训练数据为空！请先运行 part1_data_preparation.py")
        return
    
    # 如果验证集为空，从训练集划分
    if len(val_dataset) == 0 and len(train_dataset) > 5:
        print("\n⚠️ 验证集为空，从训练集划分20%")
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        print(f"  训练集: {train_size} 对")
        print(f"  验证集: {val_size} 对")
    
    # 创建 DataLoader
    batch_size = min(BATCH_SIZE, len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\n✅ 训练批次: {len(train_loader)}")
    print(f"✅ 验证批次: {len(val_loader)}")
    
    # 创建模型
    num_classes = len(TRAIN_CLASSES)
    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 开始训练 {EPOCHS} 轮...\n")
    
    best_val_acc = 0
    best_model_path = None
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for before, after, labels in progress_bar:
            before, after, labels = before.to(device), after.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(before, after)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{total_loss/(len(train_loader)):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        if len(val_loader) > 0:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for before, after, labels in val_loader:
                    before, after, labels = before.to(device), after.to(device), labels.to(device)
                    outputs = model(before, after)
                    _, preds = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (preds == labels).sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            scheduler.step(avg_loss)
        else:
            val_acc = 0
        
        print(f'\n📊 Epoch {epoch+1}:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}')
        if len(val_loader) > 0:
            print(f'  Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = CHECKPOINT_DIR / f'best_model_epoch{epoch+1}_acc{val_acc:.4f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'classes': TRAIN_CLASSES,
                'class_to_idx': CLASS_TO_IDX
            }, best_model_path)
            print(f'  ✅ 保存最佳模型 (acc={val_acc:.4f})')
    
    # 保存最终模型
    final_model_path = CHECKPOINT_DIR / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'num_classes': num_classes,
        'classes': TRAIN_CLASSES,
        'class_to_idx': CLASS_TO_IDX
    }, final_model_path)
    
    print(f"\n💾 最终模型已保存: {final_model_path}")
    
    # 保存训练信息
    train_info = {
        'timestamp': datetime.now().isoformat(),
        'classes': TRAIN_CLASSES,
        'num_classes': len(TRAIN_CLASSES),
        'best_val_acc': best_val_acc,
        'epochs_trained': EPOCHS,
        'batch_size': batch_size,
        'learning_rate': LR,
        'ssim_threshold': SSIM_THRESHOLD
    }
    with open(CHECKPOINT_DIR / 'training_info.json', 'w') as f:
        json.dump(train_info, f, indent=2)
    
    print("\n" + "="*60)
    print("🎉 训练完成！")
    print("="*60)
    print(f"类别数: {len(TRAIN_CLASSES)}")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"模型保存: {CHECKPOINT_DIR}")
    print(f"\n下一步: python part3_pair_prediction.py")


if __name__ == "__main__":
    main()