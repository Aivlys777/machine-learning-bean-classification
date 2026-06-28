"""
CatBoost分类器
"""
import numpy as np
from catboost import CatBoostClassifier
from src.base_model import BaseModel


class CatBoostModel(BaseModel):
    """CatBoost模型"""
    
    def __init__(self, random_state=42):
        super().__init__("CatBoost", random_state)
        self.iterations = 100
        self.depth = 6
        self.learning_rate = 0.1
        self.l2_leaf_reg = 3
        self.border_count = 254
        self.loss_curve = None
        
    def build_model(self, **kwargs):
        """构建CatBoost模型"""
        params = {
            'iterations': kwargs.get('iterations', self.iterations),
            'depth': kwargs.get('depth', self.depth),
            'learning_rate': kwargs.get('learning_rate', self.learning_rate),
            'l2_leaf_reg': kwargs.get('l2_leaf_reg', self.l2_leaf_reg),
            'border_count': kwargs.get('border_count', self.border_count),
            'random_seed': self.random_state,
            'verbose': False
        }
        
        self.model = CatBoostClassifier(**params)
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        # 如果有验证集，使用early stopping并记录loss
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10,
                verbose=False
            )
            # 获取loss曲线 - 取验证集loss
            if hasattr(self.model, 'evals_result_'):
                results = self.model.evals_result_
                if 'validation' in results and 'MultiClass' in results['validation']:
                    self.loss_curve = results['validation']['MultiClass']
                    print(f"✅ CatBoost loss曲线已记录，共{len(self.loss_curve)}个点")
        else:
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
        return self.model.get_feature_importance()
    
    def get_loss_curve(self):
        """获取loss曲线"""
        return self.loss_curve