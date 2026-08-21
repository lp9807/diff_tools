"""
model_config.py
模型共享配置 - 供 part2_pair_training 和 part3_prediction 使用
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

# ============================================================
# 训练类别（5分类）
# ============================================================

TRAIN_CLASSES = [
    'quality_improved',      # 0: 质量提升
    'quality_degradation',   # 1: 质量退化
    'render_failed',         # 2: 渲染失败
    'uncertain_difference',  # 3: 不确定差异
    'matched'                # 4: 匹配
]

CLASS_TO_IDX = {cls: i for i, cls in enumerate(TRAIN_CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}

# ============================================================
# 所有类别（包含 to_review，用于显示）
# ============================================================

ALL_CLASSES = [
    'to_review',
    'quality_improved',
    'quality_degradation',
    'render_failed',
    'uncertain_difference',
    'matched'
]

LABEL_DISPLAY = {
    'to_review': '📌 待复核',
    'quality_improved': '✅ 质量提升',
    'quality_degradation': '⚠️ 质量退化',
    'render_failed': '❌ 渲染失败',
    'uncertain_difference': '❓ 不确定差异',
    'matched': '🔄 匹配'
}

LABEL_COLORS = {
    'to_review': '#e67e22',
    'quality_improved': '#2ecc71',
    'quality_degradation': '#f39c12',
    'render_failed': '#e74c3c',
    'uncertain_difference': '#f1c40f',
    'matched': '#3498db'
}

# ============================================================
# SSIM 阈值
# ============================================================

SSIM_THRESHOLD = 0.999

# ============================================================
# 模型定义
# ============================================================

class SiameseNetwork(nn.Module):
    """Siamese网络 - 与 part2_training 完全一致"""
    
    def __init__(self, num_classes=5):
        super(SiameseNetwork, self).__init__()
        
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.encoder = models.resnet18(weights=weights)
        self.encoder.fc = nn.Identity()
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        self.num_classes = num_classes
    
    def forward(self, before, after):
        feat_before = self.encoder(before)
        feat_after = self.encoder(after)
        combined = torch.cat([feat_before, feat_after], dim=1)
        return self.classifier(combined)


# ============================================================
# 模型加载（简化版 - 只处理 model_state_dict）
# ============================================================

def load_model(model_path, device):
    """
    加载模型 - 专门处理 part2_training 保存的格式
    
    Returns:
        (model, num_classes, saved_classes) 或 (None, 5, TRAIN_CLASSES)
    """
    print(f"📂 加载模型: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"❌ 无法加载模型文件: {e}")
        return None, 5, TRAIN_CLASSES
    
    # ============================================================
    # part2_training 保存的格式: {'model_state_dict': ..., 'num_classes': ..., 'classes': ...}
    # ============================================================
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        num_classes = checkpoint.get('num_classes', 5)
        saved_classes = checkpoint.get('classes', TRAIN_CLASSES)
        
        print(f"  ✅ 找到 model_state_dict")
        print(f"  类别数: {num_classes}")
        print(f"  类别: {saved_classes}")
        
        try:
            model = SiameseNetwork(num_classes=num_classes).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"  ✅ 模型加载成功")
            return model, num_classes, saved_classes
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            return None, 5, TRAIN_CLASSES
    
    # 其他格式（兼容旧版本）
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        num_classes = checkpoint.get('num_classes', 5)
        saved_classes = checkpoint.get('classes', TRAIN_CLASSES)
        print(f"  ✅ 找到 state_dict (备用格式)")
        print(f"  类别数: {num_classes}")
        
        try:
            model = SiameseNetwork(num_classes=num_classes).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"  ✅ 模型加载成功")
            return model, num_classes, saved_classes
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            return None, 5, TRAIN_CLASSES
    
    # 直接是 state_dict
    elif isinstance(checkpoint, dict):
        first_key = list(checkpoint.keys())[0] if checkpoint else None
        if first_key and ('weight' in first_key or 'bias' in first_key or 'encoder' in first_key):
            print(f"  ✅ 直接 state_dict 格式")
            try:
                model = SiameseNetwork(num_classes=5).to(device)
                model.load_state_dict(checkpoint)
                model.eval()
                print(f"  ✅ 模型加载成功")
                return model, 5, TRAIN_CLASSES
            except Exception as e:
                print(f"  ❌ 加载失败: {e}")
                return None, 5, TRAIN_CLASSES
    
    print(f"❌ 无法识别的模型格式")
    if isinstance(checkpoint, dict):
        print(f"  可用键: {list(checkpoint.keys())}")
    return None, 5, TRAIN_CLASSES


def create_model(num_classes=5):
    """创建模型（用于训练）"""
    return SiameseNetwork(num_classes=num_classes)


def find_model_file(checkpoint_dir):
    """查找模型文件"""
    checkpoint_dir = Path(checkpoint_dir)
    
    for name in ['best_model.pth', 'final_model.pth']:
        path = checkpoint_dir / name
        if path.exists():
            return path
    
    model_files = list(checkpoint_dir.glob('*.pth'))
    if model_files:
        return max(model_files, key=lambda f: f.stat().st_mtime)
    
    return None