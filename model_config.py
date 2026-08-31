"""
model_config.py
模型共享配置

类别说明：
    - TRAIN_CLASSES: 训练类别 (5类) 
        quality_improved, quality_degradation, render_failed, uncertain_difference, trivial
    - ALL_CLASSES: 所有类别 (7类)
        TRAIN_CLASSES + matched + to_review
    - matched: SSIM 自动匹配结果 (不训练)
    - trivial: 人工标注/模型预测的"微小差异" (训练)

使用场景：
    - 目录创建: 使用 ALL_CLASSES (7个目录)
    - 模型训练: 使用 TRAIN_CLASSES (5类)
    - 模型预测: 输出 TRAIN_CLASSES (5类) + SSIM 自动匹配
    - 用户界面: 使用 LABEL_DISPLAY 显示
    - 标注工具: 使用 TRAIN_CLASSES (5类，用户标注)
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from enum import Enum

# ============================================================
# 类别枚举
# ============================================================

class ClassType(Enum):
    """所有类别枚举"""
    
    # ============================================================
    # 训练类别 (5类)
    # ============================================================
    QUALITY_IMPROVED = 'quality_improved'
    QUALITY_DEGRADATION = 'quality_degradation'
    RENDER_FAILED = 'render_failed'
    UNCERTAIN_DIFFERENCE = 'uncertain_difference'
    TRIVIAL = 'trivial'              # 人工标注/模型预测的"微小差异"
    
    # ============================================================
    # 显示/流程控制类别 (不训练)
    # ============================================================
    MATCHED = 'matched'              # SSIM 自动匹配 (不训练)
    TO_REVIEW = 'to_review'          # 低置信度待复核
    
    @property
    def display_name(self):
        """用户界面显示名称"""
        display_map = {
            'quality_improved': '✅ 质量提升',
            'quality_degradation': '⚠️ 质量退化',
            'render_failed': '❌ 渲染失败',
            'uncertain_difference': '❓ 不确定差异',
            'trivial': '🔄 微小差异',        # 用户标注/模型预测
            'matched': '🔄 匹配 (自动)',      # SSIM 自动匹配
            'to_review': '📌 待复核'
        }
        return display_map.get(self.value, self.value)
    
    @property
    def color(self):
        """显示颜色"""
        color_map = {
            'quality_improved': '#2ecc71',
            'quality_degradation': '#f39c12',
            'render_failed': '#e74c3c',
            'uncertain_difference': '#f1c40f',
            'trivial': '#3498db',
            'matched': '#3498db',
            'to_review': '#e67e22'
        }
        return color_map.get(self.value, '#95a5a6')
    
    @property
    def is_training_class(self):
        """是否为训练类别 (matched 和 to_review 不训练)"""
        return self not in [ClassType.MATCHED, ClassType.TO_REVIEW]
    
    @classmethod
    def training_classes(cls):
        """获取训练类别 (5类)"""
        return [c for c in cls if c.is_training_class]
    
    @classmethod
    def all_classes(cls):
        """获取所有类别 (7类)"""
        return list(cls)
    
    @classmethod
    def from_value(cls, value):
        for member in cls:
            if member.value == value:
                return member
        return None
    
    @classmethod
    def get_display_map(cls):
        return {c.value: c.display_name for c in cls}
    
    @classmethod
    def get_color_map(cls):
        return {c.value: c.color for c in cls}


# ============================================================
# 兼容旧代码的列表
# ============================================================

# 训练类别 (5类: 包含 trivial)
TRAIN_CLASSES = [c.value for c in ClassType.training_classes()]

# 所有类别 (7类: TRAIN_CLASSES + matched + to_review)
ALL_CLASSES = [c.value for c in ClassType.all_classes()]

# 索引映射 (只针对训练类别)
CLASS_TO_IDX = {c.value: idx for idx, c in enumerate(ClassType.training_classes())}
IDX_TO_CLASS = {idx: c.value for idx, c in enumerate(ClassType.training_classes())}

# 显示映射
LABEL_DISPLAY = ClassType.get_display_map()
LABEL_COLORS = ClassType.get_color_map()


# ============================================================
# 类别映射工具
# ============================================================

def to_display_name(class_value):
    """获取类别的显示名称"""
    return LABEL_DISPLAY.get(class_value, class_value)


def is_training_class(class_value):
    """判断是否为训练类别"""
    return class_value in TRAIN_CLASSES


# ============================================================
# 快捷键配置 (5类训练类别)
# ============================================================

def get_class_keys(mode='annotation'):
    """
    生成分类快捷键映射 (5类)
    
    Args:
        mode: 'annotation' 或 'review'
    
    Returns:
        dict: 按键码 -> 类别值 (ClassType 或 str)
    """
    training = ClassType.training_classes()
    
    if mode == 'annotation':
        key_list = ['1', '2', '3', '4', 'm']
    else:  # review
        key_list = ['1', '2', '3', '4', '5']
    
    keys = {}
    for i, cls in enumerate(training):
        if i < len(key_list):
            keys[ord(key_list[i])] = cls
    
    return keys


def get_nav_keys(mode='annotation'):
    """生成导航键映射"""
    if mode == 'annotation':
        return {
            ord('s'): 'skipped',
            ord('b'): 'back',
        }
    else:  # review
        return {
            ord('s'): 'skipped',
            ord('n'): 'next',
            ord('p'): 'prev',
        }


def get_class_shortcut_display(mode='annotation'):
    """生成快捷键显示文本 (使用显示名称)"""
    key_map = get_class_keys(mode)
    lines = []
    for key_code, cls in key_map.items():
        key_char = chr(key_code) if 32 < key_code < 127 else '?'
        # 获取显示名称
        if hasattr(cls, 'value'):
            display = LABEL_DISPLAY.get(cls.value, cls.value)
        else:
            display = LABEL_DISPLAY.get(cls, cls)
        lines.append(f"{key_char}:{display}")
    return lines


# ============================================================
# SSIM 阈值和计算工具
# ============================================================

SSIM_THRESHOLD = 0.999


class SSIMComparator:
    """SSIM 相似度计算工具"""
    
    @staticmethod
    def compute_ssim(img1_path, img2_path):
        """
        计算两张图片的 SSIM 相似度
        
        Returns:
            float: SSIM 相似度 (0-1)，1表示完全相同
        """
        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                return 0.0
            
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            if img1_gray.shape != img2_gray.shape:
                h = min(img1_gray.shape[0], img2_gray.shape[0])
                w = min(img1_gray.shape[1], img2_gray.shape[1])
                img1_gray = cv2.resize(img1_gray, (w, h))
                img2_gray = cv2.resize(img2_gray, (w, h))
            
            score, _ = ssim(img1_gray, img2_gray, full=True)
            return score
        except Exception as e:
            print(f"⚠️ SSIM 计算失败: {e}")
            return 0.0
    
    @staticmethod
    def get_status_info(ssim_score):
        """获取状态信息用于UI显示"""
        if ssim_score >= SSIM_THRESHOLD:
            return {
                'status': 'AUTO_MATCHED',
                'color': (0, 255, 0),
                'label': f'自动匹配 (SSIM: {ssim_score:.6f})'
            }
        else:
            return {
                'status': 'NEED_REVIEW',
                'color': (255, 165, 0),
                'label': '需人工标注'
            }


# ============================================================
# 模型定义 (5分类)
# ============================================================

class SiameseNetwork(nn.Module):
    """Siamese网络 - 5分类 (包含 trivial)"""
    
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
# 模型加载
# ============================================================

def load_model(model_path, device):
    """加载模型"""
    print(f"📂 加载模型: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"❌ 无法加载模型文件: {e}")
        return None, 5, TRAIN_CLASSES
    
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
    
    # 优先查找 best_model
    for name in ['best_model.pth', 'final_model.pth']:
        path = checkpoint_dir / name
        if path.exists():
            return path
    
    # 查找任意 .pth 文件
    model_files = list(checkpoint_dir.glob('*.pth'))
    if model_files:
        return max(model_files, key=lambda f: f.stat().st_mtime)
    
    return None


# ============================================================
# 便捷工具
# ============================================================

def get_class_count_dict():
    """获取所有类别的计数字典模板"""
    return {cls: 0 for cls in ALL_CLASSES}


def get_train_class_count_dict():
    """获取训练类别的计数字典模板"""
    return {cls: 0 for cls in TRAIN_CLASSES}


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("📊 类别配置验证")
    print("="*60)
    
    print(f"\nTRAIN_CLASSES ({len(TRAIN_CLASSES)}类):")
    for cls in TRAIN_CLASSES:
        print(f"  - {cls}: {LABEL_DISPLAY.get(cls, cls)}")
    
    print(f"\nALL_CLASSES ({len(ALL_CLASSES)}类):")
    for cls in ALL_CLASSES:
        is_train = "✅ 训练" if cls in TRAIN_CLASSES else "❌ 不训练"
        print(f"  - {cls}: {LABEL_DISPLAY.get(cls, cls)} ({is_train})")
    
    print("\n快捷键映射 (标注模式):")
    for key_code, cls in get_class_keys('annotation').items():
        key_char = chr(key_code)
        display = LABEL_DISPLAY.get(cls.value if hasattr(cls, 'value') else cls, cls)
        print(f"  {key_char} → {display}")
    
    print(f"\nSSIM 阈值: {SSIM_THRESHOLD}")
    print("  SSIM ≥ 阈值 → matched (自动匹配)")
    print("  SSIM < 阈值 → 人工标注/模型预测")