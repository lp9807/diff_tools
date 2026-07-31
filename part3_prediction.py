"""
part3_prediction.py
预测和自动分类
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import shutil
from datetime import datetime

class Predictor:
    """预测器"""
    
    def __init__(self, model_path, model_name='resnet18', num_classes=2, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['quality_improved', 'render_failed']
        
        # 加载模型
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=False)
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
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 模型加载完成: {model_path}")
        
        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_single(self, image_path):
        """预测单张图像"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        label = self.classes[predicted.item()]
        confidence_score = confidence.item()
        
        return label, confidence_score
    
    def predict_batch(self, image_paths, batch_size=32):
        """批量预测"""
        results = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc='预测中'):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            
            for img_path in batch_paths:
                image = Image.open(img_path).convert('RGB')
                image_tensor = self.transform(image)
                batch_images.append(image_tensor)
            
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predictions = torch.max(probabilities, dim=1)
            
            for img_path, pred, conf in zip(batch_paths, predictions, confidence):
                results.append({
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'predicted_class': self.classes[pred.item()],
                    'confidence': conf.item(),
                    'is_high_confidence': conf.item() > 0.85
                })
        
        return results

class AutoClassifier:
    """自动分类器"""
    
    def __init__(self, model_path, input_dir, output_dir):
        self.predictor = Predictor(model_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.results = []
        
    def classify_all(self, threshold=0.85):
        """分类所有图像"""
        # 获取所有图像
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            images.extend(self.input_dir.glob(ext))
        
        print(f"待分类图像: {len(images)} 张")
        
        # 批量预测
        self.results = self.predictor.predict_batch(images)
        
        # 分类并保存
        high_conf_indices = []
        low_conf_indices = []
        
        for idx, result in enumerate(self.results):
            image_path = Path(result['image_path'])
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            if result['is_high_confidence']:
                # 高置信度：自动分类
                dest_dir = self.output_dir / predicted_class
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, dest_dir / image_path.name)
                high_conf_indices.append(idx)
            else:
                # 低置信度：需要人工复核
                dest_dir = self.output_dir / 'to_review'
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, dest_dir / image_path.name)
                low_conf_indices.append(idx)
        
        # 生成报告
        self.generate_report(high_conf_indices, low_conf_indices)
        
        return self.results
    
    def generate_report(self, high_indices, low_indices):
        """生成分类报告"""
        # 统计
        total = len(self.results)
        auto_count = len(high_indices)
        review_count = len(low_indices)
        automation_rate = auto_count / total * 100
        
        # 类别统计
        class_counts = {}
        for result in self.results:
            cls = result['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # 置信度分布
        confidences = [r['confidence'] for r in self.results]
        
        report = {
            'total_images': total,
            'auto_classified': auto_count,
            'needs_review': review_count,
            'automation_rate': automation_rate,
            'class_distribution': class_counts,
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'threshold_used': 0.85,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存报告
        report_file = self.output_dir / 'classification_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 保存详细结果
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / 'classification_details.csv', index=False)
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 分类报告")
        print("="*60)
        print(f"总图像数: {total}")
        print(f"自动分类: {auto_count} ({automation_rate:.1f}%)")
        print(f"需要复核: {review_count} ({100-automation_rate:.1f}%)")
        print(f"\n类别分布:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} 张 ({count/total*100:.1f}%)")
        print(f"\n置信度统计:")
        for key, value in report['confidence_stats'].items():
            print(f"  {key}: {value:.3f}")
        print(f"\n✅ 报告保存至: {report_file}")
        print(f"详细结果: {self.output_dir / 'classification_details.csv'}")

def find_best_model(checkpoints_dir='./project/checkpoints'):
    """
    自动查找最佳的模型文件
    优先级: best_model_*.pth > final_model.pth > 任何 .pth 文件
    """
    checkpoints_path = Path(checkpoints_dir)
    
    if not checkpoints_path.exists():
        print(f"❌ 模型目录不存在: {checkpoints_path}")
        return None
    
    # 查找所有模型文件
    all_models = list(checkpoints_path.glob('*.pth'))
    
    if not all_models:
        print(f"❌ 在 {checkpoints_path} 中没有找到任何模型文件")
        return None
    
    # 优先选择 best_model 开头的文件
    best_models = [f for f in all_models if f.name.startswith('best_model')]
    if best_models:
        # 按修改时间排序，选择最新的
        best_models.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        selected = best_models[0]
        print(f"✅ 选择最佳模型: {selected.name}")
        return str(selected)
    
    # 其次选择 final_model
    final_models = [f for f in all_models if f.name == 'final_model.pth']
    if final_models:
        selected = final_models[0]
        print(f"✅ 选择最终模型: {selected.name}")
        return str(selected)
    
    # 最后选择任意模型
    selected = all_models[0]
    print(f"✅ 选择模型: {selected.name}")
    return str(selected)

def main_predict():
    """预测主程序"""
    print("="*60)
    print("第3步：自动分类")
    print("="*60)
    
    # ============================================================
    # 自动查找最佳模型
    # ============================================================
    model_path = find_best_model('./project/checkpoints')
    
    if model_path is None:
        print("\n❌ 错误：没有找到训练好的模型！")
        print("请先运行 part2_training.py 完成模型训练")
        return
    
    # 检查输入目录是否存在
    input_dir = Path('./raw_images_all')
    if not input_dir.exists():
        print(f"\n❌ 错误：输入目录不存在: {input_dir}")
        print("请创建该目录并将待分类的图片放入其中")
        return
    
    # 创建分类器
    classifier = AutoClassifier(
        model_path=model_path,  # 替换为你的模型路径
        input_dir='./raw_images_all',  # 所有待分类的图像
        output_dir='./project/06_final_output'
    )
    
    # 执行分类
    results = classifier.classify_all(threshold=0.85)
    
    print("\n✅ 分类完成！")
    print(f"结果目录: ./project/06_final_output")
    print("  - quality_improved/: 自动分类为质量提升的图像")
    print("  - render_failed/: 自动分类为渲染失败的图像")
    print("  - to_review/: 需要人工复核的低置信度图像")

if __name__ == "__main__":
    main_predict()