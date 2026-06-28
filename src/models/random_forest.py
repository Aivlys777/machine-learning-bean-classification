"""
随机森林分类器
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.base_model import BaseModel


class RandomForestModel(BaseModel):
    """随机森林模型"""
    
    def __init__(self, random_state=42):
        super().__init__("RandomForest", random_state)
        self.n_estimators = 100
        self.max_depth = 10
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        
    def build_model(self, **kwargs):
        """构建随机森林模型"""
        params = {
            'n_estimators': kwargs.get('n_estimators', self.n_estimators),
            'max_depth': kwargs.get('max_depth', self.max_depth),
            'min_samples_split': kwargs.get('min_samples_split', self.min_samples_split),
            'min_samples_leaf': kwargs.get('min_samples_leaf', self.min_samples_leaf),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        self.model = RandomForestClassifier(**params)
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self.model
    
    def predict(self, X):
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.feature_importances_
    
    def get_loss_curve(self):
        """随机森林不支持loss曲线"""
        return None