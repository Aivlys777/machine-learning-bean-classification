"""
LightGBM分类器
"""
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from src.base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM模型"""
    
    def __init__(self, random_state=42):
        super().__init__("LightGBM", random_state)
        self.n_estimators = 100
        self.max_depth = 6
        self.learning_rate = 0.1
        self.num_leaves = 31
        self.subsample = 0.8
        self.colsample_bytree = 0.8
        self.loss_curve = None
        
    def build_model(self, **kwargs):
        """构建LightGBM模型"""
        params = {
            'n_estimators': kwargs.get('n_estimators', self.n_estimators),
            'max_depth': kwargs.get('max_depth', self.max_depth),
            'learning_rate': kwargs.get('learning_rate', self.learning_rate),
            'num_leaves': kwargs.get('num_leaves', self.num_leaves),
            'subsample': kwargs.get('subsample', self.subsample),
            'colsample_bytree': kwargs.get('colsample_bytree', self.colsample_bytree),
            'random_state': self.random_state,
            'verbose': -1
        }
        
        self.model = LGBMClassifier(**params)
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        # 如果有验证集，使用early stopping并记录loss
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]
            # 注意：early_stopping 在 callbacks 中
            callbacks = [lgb.early_stopping(10)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=callbacks
            )
            # 获取loss曲线 - 取验证集loss
            if hasattr(self.model, 'evals_result_'):
                results = self.model.evals_result_
                if 'valid_0' in results and 'multi_logloss' in results['valid_0']:
                    self.loss_curve = results['valid_0']['multi_logloss']
                    print(f"✅ LightGBM loss曲线已记录，共{len(self.loss_curve)}个点")
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
        return self.model.feature_importances_
    
    def get_loss_curve(self):
        """获取loss曲线"""
        return self.loss_curve