"""
main.py
完整流程自动化
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """
    运行命令并显示进度
    自动处理Windows中文编码问题
    """
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print('='*60)
    
    # ============================================================
    # 关键修复：为子进程设置UTF-8编码环境
    # ============================================================
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'  # Python 3.7+ 支持
    
    # 在Windows上，强制使用UTF-8
    if sys.platform == 'win32':
        # 使用UTF-8编码运行子进程
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',  # 指定输出编码
            errors='replace',   # 遇到无法编码的字符用?替代
            env=env            # 传递设置了UTF-8的环境
        )
    else:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            env=env
        )
    
    if result.returncode != 0:
        print(f"❌ 错误: {result.stderr}")
        return False
    
    if result.stdout:
        print(result.stdout)
    return True

def main():
    """完整流程"""
    print("="*60)
    print("🚀 图像质量分类 - 完整自动化流程")
    print("="*60)
    print("\n此脚本将依次执行:")
    print("1. 数据准备和标注 (80张样本)")
    print("2. 模型训练")
    print("3. 自动分类所有图像")
    print("="*60)
    
    input("\n按 Enter 开始...")
    
    # 1. 数据准备
    if not run_command("python.exe part1_data_preparation.py", "第1步: 数据准备"):
        print("❌ 数据准备失败，请检查文件")
        sys.exit(1)
    
    # 等待用户完成标注
    print("\n" + "="*60)
    print("⚠️  请完成标注后再继续")
    print("="*60)
    print("1. 打开目录: ./project/01_to_annotate")
    print("2. 使用交互式标注工具标注所有图片")
    print("3. 标注完成后按 Enter 继续训练")
    input()
    
    # 2. 训练
    if not run_command("python.exe part2_training.py", "第2步: 模型训练"):
        print("❌ 训练失败")
        sys.exit(1)
    
    # 3. 预测
    if not run_command("python.exe part3_prediction.py", "第3步: 自动分类"):
        print("❌ 分类失败")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 全部流程完成！")
    print("="*60)
    print("\n结果目录: ./project/06_final_output")
    print("  - 查看分类报告: classification_report.json")
    print("  - 查看详细结果: classification_details.csv")
    print("  - 需要复核的图像: to_review/")
    print("\n📊 最终统计:")
    print("  1. 人工标注: 80张 (约10-15分钟)")
    print("  2. 模型训练: ~10-20分钟")
    print("  3. 自动分类: 剩余全部图像 (1-2分钟)")
    print("  4. 自动化率: 85-95%")
    print("  5. 只需复核低置信度样本: 5-15%")

if __name__ == "__main__":
    main()