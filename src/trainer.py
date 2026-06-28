"""
训练模块 - 完整版
整合数据加载、预处理和模型训练
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
import logging
from sklearn.model_selection import train_test_split
from collections import Counter

from src.evaluator import Evaluator
from src.analyzer import DataAnalyzer
from src.models.random_forest import RandomForestModel
from src.models.xgboost import XGBoostModel
from src.models.lightgbm import LightGBMModel
from src.models.catboost import CatBoostModel
from src.models.ann import ANNModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器类"""
    
    def __init__(self, data_dir='data', results_dir='results', random_state=42):
        self.data_dir = data_dir
        self.results_dir = Path(results_dir)
        self.random_state = random_state
        
        self._create_dirs()
        
        self.preprocessor = None
        self.evaluator = Evaluator(results_dir)
        self.analyzer = DataAnalyzer(results_dir)
        
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
        """加载并分析数据"""
        logger.info("=" * 60)
        logger.info("步骤1: 数据分析")
        logger.info("=" * 60)
        
        self.data_analysis_results = self.analyzer.analyze_data(
            train_df, test_df, target_col
        )
        
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
        
        if self.X_train is None:
            raise ValueError("请先加载数据")
        
        # 检查类别分布
        class_counts = Counter(self.y_train)
        logger.info(f"训练集类别分布: {dict(class_counts)}")
        
        # 划分验证集
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_train, self.y_train,
                test_size=0.2,
                random_state=self.random_state,
                stratify=self.y_train
            )
        except ValueError:
            logger.warning("分层抽样失败，使用普通抽样")
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_train, self.y_train,
                test_size=0.2,
                random_state=self.random_state
            )
        
        # 检查验证集类别是否在训练集中
        train_classes = set(y_train)
        val_classes = set(y_val)
        missing = val_classes - train_classes
        
        if missing:
            logger.warning(f"验证集包含训练集没有的类别: {missing}")
            # 将缺失类别的样本移回训练集
            mask = np.isin(y_val, list(missing))
            if np.any(mask):
                X_train = np.vstack([X_train, X_val[mask]])
                y_train = np.concatenate([y_train, y_val[mask]])
                X_val = X_val[~mask]
                y_val = y_val[~mask]
                logger.info(f"已将 {np.sum(mask)} 个样本移回训练集")
        
        logger.info(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")
        logger.info(f"验证集类别分布: {dict(Counter(y_val))}")
        
        for name, model in self.models.items():
            logger.info(f"\n{'='*40}")
            logger.info(f"训练 {name}...")
            logger.info(f"{'='*40}")
            
            start_time = time.time()
            
            try:
                model.build_model()
                
                if name == 'ANN':
                    model.train(X_train, y_train)
                else:
                    model.train(X_train, y_train, X_val, y_val)
                
                self.training_times[name] = time.time() - start_time
                self.trained_models[name] = model
                logger.info(f"✅ {name} 完成，耗时: {self.training_times[name]:.3f}s")
                
                # 获取loss曲线
                loss_curve = model.get_loss_curve()
                if loss_curve is not None and len(loss_curve) > 0:
                    self.loss_curves[name] = loss_curve
                    logger.info(f"✅ {name} loss曲线已记录，共{len(loss_curve)}个点")
                else:
                    logger.warning(f"⚠️ {name} loss曲线为空")
                    
            except Exception as e:
                logger.error(f"❌ {name} 训练失败: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"\n收集到的loss曲线: {list(self.loss_curves.keys())}")
        
        if self.loss_curves:
            self.evaluator.plot_loss_curves_comparison(self.loss_curves)
        
        return self.trained_models
    
    def evaluate_models(self):
        """评估所有模型"""
        logger.info("\n" + "=" * 60)
        logger.info("步骤4: 评估模型")
        logger.info("=" * 60)
        
        for name, model in self.trained_models.items():
            logger.info(f"\n评估 {name}...")
            
            try:
                y_pred = model.predict(self.X_test)
                
                result = self.evaluator.evaluate(
                    self.y_test, y_pred,
                    model_name=name
                )
                
                self.results[name] = result
                logger.info(f"{name} 准确率: {result['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"{name} 评估失败: {e}")
        
        return self.results
    
    def robustness_test(self, noise_levels=[0.05, 0.1, 0.2, 0.3]):
        """鲁棒性测试"""
        logger.info("\n" + "=" * 60)
        logger.info("鲁棒性测试")
        logger.info("=" * 60)
        
        robustness_results = {}
        
        if not self.trained_models:
            return robustness_results
        
        for noise_type in ['gaussian', 'uniform']:
            robustness_results[noise_type] = {}
            
            for level in noise_levels:
                logger.info(f"\n噪声: {noise_type}, 强度: {level}")
                
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
                    except:
                        level_results[name] = 0.0
                
                robustness_results[noise_type][level] = level_results
        
        if robustness_results:
            self.evaluator.plot_robustness_heatmap(robustness_results)
        
        return robustness_results


if __name__ == '__main__':
    print("Trainer 模块加载成功")