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


def main():
    """主函数"""
    print("=" * 70)
    print("  机器学习期末作业 - Dry Bean Dataset 多分类实验")
    print("=" * 70)
    print()

    # 记录开始时间
    logger.info("=" * 70)
    logger.info("机器学习期末作业 - Dry Bean Dataset 多分类实验")
    logger.info("=" * 70)

    # 1. 检查数据文件
    print("【步骤0】检查数据文件...")
    logger.info("步骤0: 检查数据文件")
    found_files = check_data_files('data')

    if not found_files:
        print("❌ 错误: 在 data/ 目录下未找到数据文件!")
        print("   请确保数据文件在 data/ 目录下")
        logger.error("在 data/ 目录下未找到数据文件")
        return

    print(f"✅ 找到数据文件: {found_files}")
    logger.info(f"找到数据文件: {found_files}")

    # 确定训练集和测试集文件名
    train_file = None
    test_file = None
    val_file = None

    if 'Dry_Bean_Dataset_Dirty_train.csv' in found_files:
        train_file = 'Dry_Bean_Dataset_Dirty_train.csv'
    elif 'train.csv' in found_files:
        train_file = 'train.csv'

    if 'Dry_Bean_Dataset_Dirty_test.csv' in found_files:
        test_file = 'Dry_Bean_Dataset_Dirty_test.csv'
    elif 'test.csv' in found_files:
        test_file = 'test.csv'

    if 'Dry_Bean_Dataset_Dirty_val.csv' in found_files:
        val_file = 'Dry_Bean_Dataset_Dirty_val.csv'
    elif 'val.csv' in found_files:
        val_file = 'val.csv'

    print(f"   训练集: {train_file}")
    print(f"   测试集: {test_file}")
    if val_file:
        print(f"   验证集: {val_file}")
    logger.info(f"训练集: {train_file}, 测试集: {test_file}")
    print()

    # 2. 创建训练器
    print("【步骤1】初始化训练器...")
    logger.info("步骤1: 初始化训练器")
    trainer = Trainer(data_dir='data', results_dir='results', random_state=42)

    # 3. 加载数据
    print("【步骤2】加载数据...")
    logger.info("步骤2: 加载数据")

    data_loader = DataLoader('data')

    # 加载训练集
    train_path = data_loader.data_dir / train_file
    train_df = pd.read_csv(train_path)

    # 加载测试集
    test_path = data_loader.data_dir / test_file
    test_df = pd.read_csv(test_path)

    # 显示数据信息
    inspect_data(train_df, "训练集")
    inspect_data(test_df, "测试集")

    # 检查目标列名
    target_col = None
    possible_targets = ['Class', 'class', 'target', 'label', 'Type', 'type']
    for col in train_df.columns:
        if col in possible_targets:
            target_col = col
            break
        if 'class' in col.lower() or 'type' in col.lower():
            target_col = col
            break

    if target_col is None:
        target_col = train_df.columns[-1]
        print(f"   ⚠️ 未找到明确的目标列，使用最后一列: {target_col}")
        logger.warning(f"未找到明确的目标列，使用最后一列: {target_col}")

    # 提取特征和标签
    X_train_raw = train_df.drop(columns=[target_col])
    y_train_raw = train_df[target_col]

    X_test_raw = test_df.drop(columns=[target_col])
    y_test_raw = test_df[target_col]

    print(f"   训练集大小: {X_train_raw.shape}")
    print(f"   测试集大小: {X_test_raw.shape}")
    print(f"   特征数量: {X_train_raw.shape[1]}")
    print(f"   目标列名: {target_col}")
    logger.info(f"训练集大小: {X_train_raw.shape}, 测试集大小: {X_test_raw.shape}")

    # 显示类别分布
    print(f"\n   训练集类别分布:")
    print(y_train_raw.value_counts())
    print(f"\n   测试集类别分布:")
    print(y_test_raw.value_counts())
    print()

    # 4. 数据预处理
    print("【步骤2.1】数据预处理...")
    logger.info("步骤2.1: 数据预处理")

    preprocessor = Preprocessor()
    preprocessor.feature_names = X_train_raw.columns.tolist()

    X_train, y_train = preprocessor.fit_transform(X_train_raw, y_train_raw)
    X_test, y_test = preprocessor.transform(X_test_raw, y_test_raw)

    trainer.X_train = X_train
    trainer.y_train = y_train
    trainer.X_test = X_test
    trainer.y_test = y_test
    trainer.preprocessor = preprocessor

    print(f"   预处理完成，训练集大小: {X_train.shape}")
    print(f"   标签编码: {dict(enumerate(preprocessor.label_encoder.classes_))}")
    logger.info(f"预处理完成，训练集大小: {X_train.shape}")
    print()

    # 5. 训练模型
    print("【步骤3】训练模型...")
    logger.info("步骤3: 训练模型")
    trainer.train_models()

    # 6. 评估模型
    print("【步骤4】评估模型...")
    logger.info("步骤4: 评估模型")
    trainer.evaluate_models()

    # 7. 生成对比图表
    if trainer.results:
        print("【步骤4.1】生成对比图表...")
        logger.info("步骤4.1: 生成对比图表")
        trainer.evaluator.plot_comparison(
            trainer.results,
            trainer.training_times,
            trainer.prediction_times
        )

    # 8. 生成结果摘要
    summary = trainer.evaluator.get_results_summary(
        trainer.results,
        trainer.training_times,
        trainer.prediction_times
    )

    print("\n" + "=" * 70)
    print("  模型性能对比结果")
    print("=" * 70)
    print(summary.to_string(index=False))
    logger.info("\n模型性能对比结果:\n" + summary.to_string(index=False))

    # 9. 鲁棒性测试
    print("\n【步骤5】执行鲁棒性测试...")
    logger.info("步骤5: 执行鲁棒性测试")
    robustness_results = trainer.robustness_test(
        noise_levels=[0.05, 0.1, 0.2, 0.3]
    )
    print("\n鲁棒性测试完成!")
    logger.info("鲁棒性测试完成")

    # 10. 过拟合分析
    print("\n【步骤6】过拟合分析...")
    print("=" * 70)
    logger.info("步骤6: 过拟合分析")

    for name, model in trainer.trained_models.items():
        y_train_pred = model.predict(trainer.X_train)
        train_acc = np.mean(y_train_pred == trainer.y_train)

        y_test_pred = model.predict(trainer.X_test)
        test_acc = np.mean(y_test_pred == trainer.y_test)

        gap = train_acc - test_acc
        if gap < 0.02:
            status = "✅ 泛化良好"
        elif gap < 0.08:
            status = "⚠️ 轻微过拟合"
        else:
            status = "❌ 严重过拟合"
        msg = f"{name}: 训练集精度={train_acc:.4f}, 测试集精度={test_acc:.4f}, 差距={gap:.4f} {status}"
        print(f"  {msg}")
        logger.info(msg)

    # 11. 输出总结
    print("\n" + "=" * 70)
    print("  实验总结")
    print("=" * 70)
    logger.info("实验总结")

    if not summary.empty:
        best_acc = summary['Accuracy'].astype(float).max()
        best_model = summary[summary['Accuracy'].astype(float) == best_acc]['Model'].values[0]
        print(f"  🏆 最佳模型: {best_model}")
        print(f"  📊 最高准确率: {best_acc:.4f}")
        logger.info(f"最佳模型: {best_model}, 最高准确率: {best_acc:.4f}")

    print(f"  📁 所有结果已保存到: results/ 目录")
    print(f"     - 图表: results/figures/")
    print(f"     - 指标: results/metrics/model_comparison.csv")
    print(f"     - 日志: results/logs/training_log.txt")
    logger.info("所有结果已保存到 results/ 目录")

    print("\n✅ 实验完成!")
    logger.info("实验完成!")


if __name__ == '__main__':
    main()