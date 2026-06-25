"""
ANN（人工神经网络）分类器
使用多层感知机（MLP）实现
课堂没讲过的算法 - 深度学习基础
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from src.base_model import BaseModel
import warnings
warnings.filterwarnings('ignore')


class ANNModel(BaseModel):
    """ANN人工神经网络模型（基于MLP）"""
    
    def __init__(self, random_state=42):
        super().__init__("ANN", random_state)
        self.hidden_layer_sizes = (100, 50)
        self.activation = 'relu'
        self.solver = 'adam'
        self.alpha = 0.0001
        self.batch_size = 'auto'
        self.learning_rate = 'adaptive'
        self.max_iter = 500
        self.early_stopping = True
        self.validation_fraction = 0.1
        self.n_iter_no_change = 10
        self.loss_curve = None
        
    def build_model(self, **kwargs):
        """构建ANN模型"""
        params = {
            'hidden_layer_sizes': kwargs.get('hidden_layer_sizes', self.hidden_layer_sizes),
            'activation': kwargs.get('activation', self.activation),
            'solver': kwargs.get('solver', self.solver),
            'alpha': kwargs.get('alpha', self.alpha),
            'batch_size': kwargs.get('batch_size', self.batch_size),
            'learning_rate': kwargs.get('learning_rate', self.learning_rate),
            'max_iter': kwargs.get('max_iter', self.max_iter),
            'early_stopping': kwargs.get('early_stopping', self.early_stopping),
            'validation_fraction': kwargs.get('validation_fraction', self.validation_fraction),
            'n_iter_no_change': kwargs.get('n_iter_no_change', self.n_iter_no_change),
            'random_state': self.random_state,
            'verbose': False
        }
        
        self.model = MLPClassifier(**params)
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # 获取loss曲线
        if hasattr(self.model, 'loss_curve_'):
            self.loss_curve = self.model.loss_curve_
        
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
        if hasattr(self.model, 'coefs_') and len(self.model.coefs_) > 0:
            first_layer_weights = np.abs(self.model.coefs_[0])
            importance = np.mean(first_layer_weights, axis=1)
            return importance
        return None
    
    def get_loss_curve(self):
        """获取loss曲线"""
        return self.loss_curve


if __name__ == '__main__':
    model = ANNModel()
    print("ANN模型加载成功")