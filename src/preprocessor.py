"""
数据预处理模块
包含数据清洗、标准化、特征工程等
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Preprocessor:
    """数据预处理器类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = None
        self.pca = None
        self.is_fitted = False
        self.feature_names = None
        
    def _clean_data(self, X):
        """
        清洗数据：将 '?' 替换为 NaN，并转换数据类型
        
        Args:
            X: 输入数据（DataFrame或数组）
            
        Returns:
            清洗后的数据
        """
        # 如果是DataFrame，复制一份
        if isinstance(X, pd.DataFrame):
            X_clean = X.copy()
        else:
            X_clean = pd.DataFrame(X)
        
        # 将 '?' 替换为 NaN
        X_clean = X_clean.replace('?', np.nan)
        
        # 尝试将所有列转换为数值类型
        for col in X_clean.columns:
            try:
                X_clean[col] = pd.to_numeric(X_clean[col])
            except (ValueError, TypeError):
                # 如果无法转换，尝试去除特殊字符
                try:
                    X_clean[col] = X_clean[col].astype(str).str.replace(',', '').str.strip()
                    X_clean[col] = pd.to_numeric(X_clean[col])
                except:
                    logger.warning(f"列 '{col}' 包含非数值数据，将使用中位数填充")
                    # 保留为字符串，后续处理
        
        return X_clean
    
    def fit(self, X, y=None):
        """
        拟合预处理器
        
        Args:
            X: 特征数据
            y: 标签数据（用于编码）
        """
        logger.info("拟合预处理器...")
        
        # 清洗数据
        X_clean = self._clean_data(X)
        
        # 检查并处理非数值列
        numeric_cols = []
        non_numeric_cols = []
        
        for col in X_clean.columns:
            if pd.api.types.is_numeric_dtype(X_clean[col]):
                numeric_cols.append(col)
            else:
                non_numeric_cols.append(col)
                logger.warning(f"列 '{col}' 包含非数值数据，将尝试转换")
                # 尝试强制转换为数值
                try:
                    X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
                    if X_clean[col].isna().all():
                        logger.warning(f"列 '{col}' 无法转换为数值，将删除")
                    else:
                        numeric_cols.append(col)
                except:
                    logger.warning(f"列 '{col}' 无法转换为数值，将删除")
        
        # 如果存在非数值列，删除或转换
        if non_numeric_cols:
            logger.info(f"删除非数值列: {non_numeric_cols}")
            X_clean = X_clean.drop(columns=non_numeric_cols)
        
        # 确保所有数据都是数值类型
        for col in X_clean.columns:
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        
        # 处理缺失值
        logger.info("处理缺失值...")
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X_clean)
        
        # 标准化
        logger.info("标准化数据...")
        self.scaler.fit(X_imputed)
        
        # 标签编码
        if y is not None:
            self.label_encoder.fit(y)
            logger.info(f"类别编码: {dict(enumerate(self.label_encoder.classes_))}")
        
        # 保存特征名称
        self.feature_names = X_clean.columns.tolist()
        
        self.is_fitted = True
        logger.info(f"预处理器拟合完成，保留 {len(self.feature_names)} 个特征")
        return self
    
    def transform(self, X, y=None):
        """
        转换数据
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            转换后的特征和标签
        """
        if not self.is_fitted:
            raise ValueError("请先调用fit方法")
        
        # 清洗数据
        X_clean = self._clean_data(X)
        
        # 只保留拟合时使用的列
        if hasattr(self, 'feature_names') and self.feature_names:
            # 检查哪些列存在
            existing_cols = [col for col in self.feature_names if col in X_clean.columns]
            missing_cols = [col for col in self.feature_names if col not in X_clean.columns]
            
            if missing_cols:
                logger.warning(f"缺失列: {missing_cols}")
            
            if existing_cols:
                X_clean = X_clean[existing_cols]
            else:
                raise ValueError("没有找到匹配的特征列")
        
        # 确保所有数据都是数值类型
        for col in X_clean.columns:
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        
        # 处理缺失值
        X_imputed = self.imputer.transform(X_clean)
        
        # 标准化
        X_scaled = self.scaler.transform(X_imputed)
        
        # 标签编码
        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        """拟合并转换数据"""
        self.fit(X, y)
        return self.transform(X, y)
    
    def add_noise(self, X, noise_type='gaussian', intensity=0.1):
        """
        向数据添加噪声（用于鲁棒性测试）
        
        Args:
            X: 输入数据
            noise_type: 噪声类型 ('gaussian', 'uniform', 'salt_pepper')
            intensity: 噪声强度
            
        Returns:
            添加噪声后的数据
        """
        X_noisy = X.copy()
        n_samples, n_features = X.shape
        
        if noise_type == 'gaussian':
            noise = np.random.normal(0, intensity, X_noisy.shape)
            X_noisy = X_noisy + noise
            
        elif noise_type == 'uniform':
            noise = np.random.uniform(-intensity, intensity, X_noisy.shape)
            X_noisy = X_noisy + noise
            
        elif noise_type == 'salt_pepper':
            mask = np.random.random(X_noisy.shape) < intensity
            salt_pepper = np.random.choice([-5, 5], size=X_noisy.shape)
            X_noisy[mask] = salt_pepper[mask]
            
        return X_noisy
    
    def get_feature_importance_ranking(self, X, y, model=None):
        """获取特征重要性排名（使用随机森林）"""
        from sklearn.ensemble import RandomForestClassifier
        
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        importances = model.feature_importances_
        
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(X.shape[1])]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df


if __name__ == '__main__':
    preprocessor = Preprocessor()
    print("预处理器模块加载成功")