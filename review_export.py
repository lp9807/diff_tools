"""
review_export.py
读取审核进度文件，按类别导出样本列表

用法:
    python review_export.py -t quality_improved
    python review_export.py -t to_review
    python review_export.py -t all
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_review_progress(project_dir="./project"):
    """加载审核进度文件"""
    review_file = Path(project_dir) / "07_review" / "review_progress.json"
    
    if not review_file.exists():
        print(f"❌ 审核进度文件不存在: {review_file}")
        return None
    
    with open(review_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def load_review_history(project_dir="./project"):
    """加载审核历史"""
    history_file = Path(project_dir) / "07_review" / "review_history.json"
    
    if not history_file.exists():
        return []
    
    with open(history_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_predictions(project_dir="./project"):
    """从 final_output 加载预测结果"""
    output_dir = Path(project_dir) / "06_final_output"
    if not output_dir.exists():
        return {}
    
    predictions = {}
    for cls_dir in output_dir.iterdir():
        if cls_dir.is_dir():
            for pair_dir in cls_dir.iterdir():
                if pair_dir.is_dir():
                    predictions[pair_dir.name] = cls_dir.name
    
    return predictions


def get_samples_by_category(project_dir="./project", target=None):
    """
    获取按类别分组的样本列表
    
    优先级：审核结果 > 预测结果
    """
    # 加载数据
    progress = load_review_progress(project_dir)
    history = load_review_history(project_dir)
    predictions = load_predictions(project_dir)
    
    # 构建审核结果映射 (最新审核为准)
    reviewed = {}
    if history:
        for entry in history:
            name = entry.get('name')
            if name:
                reviewed[name] = entry.get('new_label')
    
    # 如果有 progress，也加入
    if progress:
        for result in progress.get('review_results', []):
            name = result.get('name')
            if name:
                label = result.get('new_label')
                if label:
                    reviewed[name] = label
    
    # 按类别分组
    categories = defaultdict(list)
    
    # 优先使用审核结果，否则使用预测结果
    all_samples = set(predictions.keys()) | set(reviewed.keys())
    
    for name in all_samples:
        if name in reviewed:
            label = reviewed[name]
        elif name in predictions:
            label = predictions[name]
        else:
            label = 'unknown'
        
        categories[label].append(name)
    
    # 按类别排序
    for label in categories:
        categories[label].sort()
    
    return categories


def print_samples(categories, target=None):
    """打印样本列表"""
    if target is None or target == 'all':
        labels = sorted(categories.keys())
    else:
        labels = [target]
    
    total_count = 0
    
    for label in labels:
        if label not in categories:
            print(f"\n⚠️ 类别 '{label}' 不存在")
            print(f"   可用类别: {', '.join(sorted(categories.keys()))}")
            continue
        
        samples = categories[label]
        total_count += len(samples)
        
        # 显示名称映射
        label_display = {
            'quality_improved': '✅ 质量提升',
            'quality_degradation': '⚠️ 质量退化',
            'render_failed': '❌ 渲染失败',
            'uncertain_difference': '❓ 不确定差异',
            'trivial': '🔄 微小差异',
            'matched': '🔄 匹配 (自动)',
            'to_review': '📌 待复核',
            'unknown': '❓ 未知'
        }.get(label, label)
        
        print(f"\n{'='*60}")
        print(f"📊 {label_display} ({len(samples)} 个样本)")
        print(f"{'='*60}")
        
        for i, name in enumerate(samples, 1):
            print(f"{name}")
    
    print(f"\n{'='*60}")
    print(f"📊 总计: {total_count} 个样本")


def main():
    parser = argparse.ArgumentParser(description='读取审核进度并导出样本列表')
    parser.add_argument('-t', '--target', type=str, default='all',
                        help='目标类别 (如 quality_improved, to_review, all)')
    parser.add_argument('-p', '--project', type=str, default='./project',
                        help='项目目录路径 (默认: ./project)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出文件路径 (可选，不指定则打印到控制台)')
    
    args = parser.parse_args()
    
    # 获取分类数据
    categories = get_samples_by_category(args.project, args.target)
    
    if not categories:
        print("❌ 没有找到任何样本")
        return
    
    # 如果指定了输出文件
    if args.output:
        import sys
        from io import StringIO
        
        # 捕获打印输出
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        print_samples(categories, args.target)
        content = sys.stdout.getvalue()
        
        sys.stdout = old_stdout
        
        # 写入文件
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 已导出到: {output_path}")
    else:
        # 打印到控制台
        print_samples(categories, args.target)


if __name__ == "__main__":
    main()