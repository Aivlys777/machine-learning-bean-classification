"""
基础模型抽象类
定义所有模型必须实现的接口
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """所有模型的基类"""
    
    def __init__(self, name, random_state=42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def build_model(self, **kwargs):
        """构建模型"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """预测"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """预测概率"""
        pass
    
    @abstractmethod
    def get_feature_importance(self):
        """获取特征重要性"""
        pass
    
    def get_loss_curve(self):
        """
        获取训练损失曲线
        非训练型算法（如RandomForest）返回None
        """
        return None
    
    def get_params(self):
        """获取模型参数"""
        if self.model is not None:
            return self.model.get_params()
        return {}
    
    def __str__(self):
        return f"{self.name} (trained={self.is_trained})"