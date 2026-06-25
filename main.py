"""
机器学习期末作业 - 主程序
基于Dry Bean Dataset的多分类算法对比实验
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# ============ 重要：设置matplotlib后端（避免GUI线程问题）============
import matplotlib
matplotlib.use('Agg')
# ================================================================

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.analyzer import DataAnalyzer

# ============ 重要：先创建日志目录 ============
log_dir = Path('results/logs')
log_dir.mkdir(parents=True, exist_ok=True)
# ============================================

# 设置日志 - 同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('results/logs/training_log.txt', encoding='utf-8', mode='w')
    ]
)
logger = logging.getLogger(__name__)

logging.getLogger('src').setLevel(logging.INFO)
logging.getLogger('src.trainer').setLevel(logging.INFO)
logging.getLogger('src.evaluator').setLevel(logging.INFO)
logging.getLogger('src.analyzer').setLevel(logging.INFO)


def check_data_files(data_dir='data'):
    """检查数据文件是否存在"""
    data_path = Path(data_dir)
    
    possible_files = [
        'Dry_Bean_Dataset_Dirty_train.csv',
        'Dry_Bean_Dataset_Dirty_test.csv',
        'Dry_Bean_Dataset_Dirty_val.csv',
        'train.csv',
        'test.csv',
        'val.csv',
        'Dry_Bean_Dataset.csv'
    ]
    
    found_files = []
    for f in possible_files:
        if (data_path / f).exists():
            found_files.append(f)
    
    return found_files


def inspect_data(df, name="数据"):
    """检查数据的基本信息"""
    print(f"\n{'='*50}")
    print(f"  {name} 数据信息")
    print(f"{'='*50}")
    print(f"形状: {df.shape}")
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n数据类型:")
    print(df.dtypes)
    print(f"\n前5行:")
    print(df.head())
    print(f"\n缺失值统计:")
    print(df.isnull().sum())
    print(f"\n唯一值统计:")
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20:
            print(f"  {col}: {df[col].nunique()} 个唯一值")
            if df[col].nunique() < 10:
                print(f"    值: {df[col].unique().tolist()}")
    print(f"{'='*50}\n")


def generate_course_summary(results, trainer):
    """生成课程总结"""
    print("\n" + "=" * 70)
    print("  课程总结")
    print("=" * 70)
    print()
    
    print("【学习收获】")
    print()
    print("1. 机器学习全流程理解")
    print("   - 从数据加载到模型部署的完整流程")
    print("   - 工程化项目结构和代码组织")
    print("   - 命令行工具的开发和测试")
    print()
    print("2. 数据处理能力提升")
    print("   - 数据清洗和特征工程")
    print("   - 数据标准化和编码")
    print("   - 噪声添加和鲁棒性测试")
    print()
    print("3. 多算法深度理解")
    print("   - 实现了5种多分类算法: RandomForest, XGBoost, LightGBM, CatBoost, ANN")
    print("   - 理解集成学习方法(Bagging/Boosting)的优劣对比")
    print("   - 掌握超参数调优技巧")
    print()
    print("4. 评估与分析能力")
    print("   - 多维度模型评估(准确率、F1、AUC等)")
    print("   - 可视化报告生成")
    print("   - 过拟合和鲁棒性分析")
    print()
    
    print("【课程评价】")
    print()
    print("优点:")
    print("  ✅ 实践性强，理论结合实际")
    print("  ✅ 项目驱动，提升综合能力")
    print("  ✅ 开放要求，鼓励自主探索")
    print()
    print("建议:")
    print("  💡 可增加更多前沿算法的介绍")
    print("  💡 可引入深度学习相关内容")
    print("  💡 可增加模型部署环节")
    print("  💡 可增加更多工业界案例")
    print()
    
    # 保存课程总结到文件
    summary_path = Path('results/reports/course_summary.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 课程总结\n\n")
        f.write("## 学习收获\n\n")
        f.write("### 1. 机器学习全流程理解\n")
        f.write("- 从数据加载到模型部署的完整流程\n")
        f.write("- 工程化项目结构和代码组织\n")
        f.write("- 命令行工具的开发和测试\n\n")
        f.write("### 2. 数据处理能力提升\n")
        f