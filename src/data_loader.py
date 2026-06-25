"""
数据加载模块
负责加载训练集和测试集数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.test_data = None
        self.feature_names = None
        self.target_name = 'Class'
        
    def load_data(self, train_file='train.csv', test_file='test.csv'):
        """
        加载训练集和测试集数据
        
        Args:
            train_file: 训练集文件名
            test_file: 测试集文件名
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        train_path = self.data_dir / train_file
        test_path = self.data_dir / test_file
        
        logger.info(f"加载训练数据: {train_path}")
        self.train_data = pd.read_csv(train_path)
        
        logger.info(f"加载测试数据: {test_path}")
        self.test_data = pd.read_csv(test_path)
        
        # 提取特征和标签
        X_train = self.train_data.drop(columns=[self.target_name])
        y_train = self.train_data[self.target_name]
        
        X_test = self.test_data.drop(columns=[self.target_name])
        y_test = self.test_data[self.target_name]
        
        self.feature_names = X_train.columns.tolist()
        
        logger.info(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def get_data_info(self):
        """获取数据基本信息"""
        if self.train_data is None:
            return None
            
        info = {
            'train_shape': self.train_data.shape,
            'test_shape': self.test_data.shape if self.test_data is not None else None,
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'target_classes': self.train_data[self.target_name].unique().tolist(),
            'num_classes': len(self.train_data[self.target_name].unique()),
            'class_distribution': self.train_data[self.target_name].value_counts().to_dict(),
            'missing_values': self.train_data.isnull().sum().sum(),
            'data_types': self.train_data.dtypes.to_dict()
        }
        return info


if __name__ == '__main__':
    # 测试数据加载
    loader = DataLoader('../data')
    X_train, y_train, X_test, y_test = loader.load_data()
    info = loader.get_data_info()
    print("\n数据信息:")
    for key, val in info.items():
        print(f"  {key}: {val}")