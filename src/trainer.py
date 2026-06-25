"""
训练模块 - 完整版
整合数据加载、预处理和模型训练
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import json
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from collections import Counter

from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from src.models.lightgbm import LightGBMModel
from src.models.catboost import CatBoostModel
from src.models.ann import ANNModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器类，整合整个训练流程"""
    
    def __init__(self, data_dir='data', results_dir='results', random_state=42):
        self.data_dir = data_dir
        self.results_dir = Path(results_dir)
        self.random_state = random_state
        
        # 创建结果目录
        self._create_dirs()
        
        self.preprocessor = None
        self.evaluator = Evaluator(results_dir)
        self.analyzer = DataAnalyzer(results_dir)  # 新增
        
        # 初始化模型 - 5种算法
        self.models = {
            'RandomForest': RandomForestModel(random_state),
            'XGBoost': XGBoostModel(random_state),
            'LightGBM': LightGBMModel(random_state),
            'CatBoost': CatBoostModel(random_state),
            'ANN': ANNModel(random_state)
        }
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.trained_models = {}
        self.results = {}
        self.training_times = {}
        self.prediction_times = {}
        self.loss_curves = {}
        self.data_analysis_results = None
        
    def _create_dirs(self):
        """创建结果目录"""
        dirs = [
            self.results_dir,
            self.results_dir / 'figures',
            self.results_dir / 'figures' / 'confusion_matrices',
            self.results_dir / 'figures' / 'feature_importance',
            self.results_dir / 'figures' / 'roc_curves',
            self.results_dir / 'figures' / 'data_analysis',
            self.results_dir / 'metrics',
            self.results_dir / 'logs',
            self.results_dir / 'reports'
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def load_and_analyze(self, train_df, test_df, target_col='Class'):
        """
        加载并分析数据
        
        Args:
            train_df: 训练集DataFrame
            test_df: 测试集DataFrame
            target_col: 目标列名
        """
        logger.info("\n" + "=" * 60)
        logger.info("步骤1: 数据分析")
        logger.info("=" * 60)
        
        # 执行数据分析
        self.data_analysis_results = self.analyzer.analyze_data(
            train_df, test_df, target_col
        )
        
        logger.info("数据分析完成")
        
        # 提取特征和标签
        X_train_raw = train_df.drop(columns=[target_col])
        y_train_raw = train_df[target_col]
        
        X_test_raw = test_df.drop(columns=[target_col])
        y_test_raw = test_df[target_col]
        
        return X_train_raw, y_train_raw, X_test_raw, y_test_raw
    
    def train_models(self):
        """训练所有模型"""
        logger.info("\n" + "=" * 60)
        logger.info("步骤3: 训练模型")
        logger.info("=" * 60)
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("请先加载和预处理数据")
        
        # 检查每个类别的样本数
        class_counts = Counter(self.y_train)
        logger.info(f"训练集类别分布: {dict(class_counts)}")
        
        # 找出样本数少于2的类别
        small_classes = [cls for cls, count in class_counts.items() if count < 2]
        
        # 划分验证集
        if small_classes:
            logger.warning(f"以下类别样本数少于2: {small_classes}，将不使用分层抽样")
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_train, self.y_train, 
                test_size=0.2, 
                random_state=self.random_state
            )
        else:
            min_count = min(class_counts.values())
            if min_count < 2:
                logger.warning(f"最小类别样本数为 {min_count}，无法进行分层抽样，将使用普通抽样")
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X_train, self.y_train, 
                    test_size=0.2, 
                    random_state=self.random_state
                )
            else:
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        self.X_train, self.y_train, 
                        test_size=0.2, 
                        random_state=self.random_state, 
                        stratify=self.y_train
                    )
                except ValueError as e:
                    logger.warning(f"分层抽样失败: {e}，将使用普通抽样")
                    X_train, X_val, y_train, y_val = train_test_split(
                        self.X_train, self.y_train, 
                        test_size=0.2, 
                        random_state=self.random_state
                    )
        
        logger.info(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        for name, model in self.models.items():
            logger.info(f"\n训练 {name}...")
            start_time = time.time()
            
            try:
                model.build_model()
                
                # 训练模型
                if name == 'ANN':
                    model.train(X_train, y_train)
                else:
                    model.train(X_train, y_train, X_val, y_val)
                
                self.training_times[name] = time.time() - start_time
                self.trained_models[name] = model
                logger.info(f"{name} 训练完成，耗时: {self.training_times[name]:.3f}s")
                
                # 获取loss曲线
                try:
                    loss_curve = model.get_loss_curve()
                    if loss_curve is not None and len(loss_curve) > 0:
                        self.loss_curves[name] = loss_curve
                        logger.info(f"{name} loss曲线已记录，共{len(loss_curve)}个点")
                except Exception as e:
                    logger.warning(f"{name} loss曲线获取失败: {e}")
                
            except Exception as e:
                logger.error(f"{name} 训练失败: {e}")
                # 如果是CatBoost的问题，尝试不使用验证集
                if 'CatBoost' in name:
                    try:
                        logger.info(f"尝试不使用验证集重新训练 {name}...")
                        model.build_model()
                        model.train(X_train, y_train, None, None)
                        self.training_times[name] = time.time() - start_time
                        self.trained_models[name] = model
                        logger.info(f"{name} 训练完成，耗时: {self.training_times[name]:.3f}s")
                    except Exception as e2:
                        logger.error(f"{name} 再次训练失败: {e2}")
        
        # 绘制loss曲线对比
        if self.loss_curves:
            self.evaluator.plot_loss_curves_comparison(self.loss_curves)
        
        return self.trained_models
    
    def evaluate_models(self):
        """评估所有模型"""
        logger.info("\n" + "=" * 60)
        logger.info("步骤4: 评估模型")
        logger.info("=" * 60)
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("请先加载和预处理数据")
        
        for name, model in self.trained_models.items():
            logger.info(f"\n评估 {name}...")
            start_time = time.time()
            
            try:
                # 预测
                y_pred = model.predict(self.X_test)
                self.prediction_times[name] = time.time() - start_time
                
                # 获取预测概率
                try:
                    y_pred_proba = model.predict_proba(self.X_test)
                except:
                    y_pred_proba = None
                
                # 评估
                result = self.evaluator.evaluate(
                    self.y_test, y_pred,
                    model_name=name,
                    y_pred_proba=y_pred_proba
                )
                
                # 获取特征重要性
                try:
                    importance = model.get_feature_importance()
                    if importance is not None and len(importance) > 0:
                        self.evaluator.plot_feature_importance(
                            importance, self.preprocessor.feature_names, name
                        )
                except Exception as e:
                    logger.warning(f"{name} 特征重要性获取失败: {e}")
                
                self.results[name] = result
                logger.info(f"{name} 准确率: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")
                
            except Exception as e:
                logger.error(f"{name} 评估失败: {e}")
        
        return self.results
    
    def robustness_test(self, noise_levels=[0.05, 0.1, 0.2, 0.3]):
        """
        鲁棒性测试：在不同噪声级别下评估模型
        """
        logger.info("\n" + "=" * 60)
        logger.info("鲁棒性测试")
        logger.info("=" * 60)
        
        robustness_results = {}
        
        if not self.trained_models:
            logger.warning("没有已训练的模型，跳过鲁棒性测试")
            return robustness_results
        
        for noise_type in ['gaussian', 'uniform']:
            robustness_results[noise_type] = {}
            
            for level in noise_levels:
                logger.info(f"\n测试噪声: {noise_type}, 强度: {level}")
                
                X_test_noisy = self.preprocessor.add_noise(
                    self.X_test, noise_type=noise_type, intensity=level
                )
                
                level_results = {}
                for name, model in self.trained_models.items():
                    try:
                        y_pred = model.predict(X_test_noisy)
                        acc = self.evaluator.calculate_accuracy(self.y_test, y_pred)
                        level_results[name] = acc
                        logger.info(f"  {name}: {acc:.4f}")
                    except Exception as e:
                        logger.warning(f"  {name} 测试失败: {e}")
                        level_results[name] = 0.0
                
                robustness_results[noise_type][level] = level_results
        
        if robustness_results:
            self.evaluator.plot_robustness_heatmap(robustness_results)
        
        return robustness_results


if __name__ == '__main__':
    print("Trainer 模块加载成功")