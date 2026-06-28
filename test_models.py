import numpy as np
from sklearn.datasets import make_classification
from src.models.xgboost import XGBoostModel
from src.models.lightgbm import LightGBMModel
from src.models.catboost import CatBoostModel

# 生成模拟数据 - 修复参数
X, y = make_classification(
    n_samples=200, 
    n_features=10, 
    n_informative=8,  # 增加信息特征数
    n_classes=3, 
    n_clusters_per_class=1,  # 改为1
    random_state=42
)

X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

models = [
    ('XGBoost', XGBoostModel()),
    ('LightGBM', LightGBMModel()),
    ('CatBoost', CatBoostModel())
]

for name, model in models:
    print(f"\n测试 {name}...")
    try:
        model.build_model()
        model.train(X_train, y_train, X_test, y_test)
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        print(f"✅ {name} 训练成功，准确率: {acc:.4f}")
    except Exception as e:
        print(f"❌ {name} 训练失败: {e}")
        import traceback
        traceback.print_exc()