"""
part2_training.py
模型训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from datetime import datetime

class ImageDataset(Dataset):
    """自定义图像数据集"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['quality_improved', 'render_failed']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"加载数据集: {self.root_dir}")
        print(f"  总样本数: {len(self.images)}")
        for cls in self.classes:
            count = sum(1 for l in self.labels if l == self.class_to_idx[cls])
            print(f"  {cls}: {count} 张")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path.name)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, model_name='resnet18', num_classes=2, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_classes = num_classes
        
        # 创建模型
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            # 冻结所有层
            for param in self.model.parameters():
                param.requires_grad = False
            # 替换分类头
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        self.model = self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_one_epoch(self, train_loader, epoch, total_epochs):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}')
        
        for batch_idx, (data, labels, _) in enumerate(progress_bar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels, _ in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def train(self, train_loader, val_loader, epochs=20, save_dir='./checkpoints'):
        """完整训练流程"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0
        best_model_path = None
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch, epochs)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step(val_loss)
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f'\nVal: Loss={val_loss:.4f}, Acc={val_acc:.2f}%')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = save_dir / f'best_model_epoch{epoch+1}_acc{val_acc:.2f}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'model_name': self.model_name
                }, best_model_path)
                print(f'✅ 保存最佳模型: {best_model_path}')
        
        # 保存最后模型
        final_model_path = save_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'model_name': self.model_name
        }, final_model_path)
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc,
            'epochs': epochs
        }
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
        
        return best_val_acc, best_model_path
    
    def plot_training_curves(self, save_dir):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy曲线
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300)
        plt.show()

def get_data_transforms():
    """获取数据增强"""
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
    
    return train_transform, val_transform

def main_train():
    """训练主程序"""
    print("="*60)
    print("第2步：模型训练")
    print("="*60)
    
    # 1. 准备数据
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = ImageDataset(
        root_dir='./project/03_train_data',
        transform=train_transform
    )
    val_dataset = ImageDataset(
        root_dir='./project/04_val_data',
        transform=val_transform
    )
    
    # 1.1 检查数据集是否为空
    if len(train_dataset) == 0:
        print("\n❌ 错误：训练数据集为空！")
        print("\n可能原因：")
        print("  1. 尚未完成标注（请先运行 part1_data_preparation.py）")
        print("  2. 标注数据没有正确划分到训练集")
        print("  3. 图片格式不被支持（仅支持 .jpg, .jpeg, .png）")
        print("\n请检查以下目录：")
        print("  - ./project/02_annotated  (标注数据)")
        print("  - ./project/03_train_data  (训练集)")
        print("  - ./project/04_val_data    (验证集)")
        return
    
    if len(val_dataset) == 0:
        print("\n⚠️ 警告：验证数据集为空！")
        print("将使用训练数据作为验证集...")
        val_dataset = train_dataset
    
    print(f"\n✅ 训练集大小: {len(train_dataset)} 张")
    print(f"✅ 验证集大小: {len(val_dataset)} 张")
    
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"\n训练批次: {len(train_loader)}")
    print(f"验证批次: {len(val_loader)}")
    
    # 2. 训练模型
    trainer = ModelTrainer(model_name='resnet18', num_classes=2)
    best_acc, model_path = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        save_dir='./project/checkpoints'
    )
    
    print(f"\n✅ 训练完成！")
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存路径: {model_path}")

if __name__ == "__main__":
    main_train()