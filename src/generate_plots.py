import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 加载数据
train_df = pd.read_csv('data/Dry_Bean_Dataset_Dirty_train.csv')

# 1. 目标分布图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

target_col = train_df.columns[-1]  # 假设最后一列是目标
counts = train_df[target_col].value_counts()
colors = plt.cm.Set3(range(len(counts)))

axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black')
axes[0].set_title('Target Class Distribution', fontsize=14)
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha='center')

axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors)
axes[1].set_title('Target Class Distribution (Percentage)', fontsize=14)

plt.tight_layout()
plt.savefig('results/figures/data_analysis/target_distribution.png', dpi=200)
plt.close()
print("✅ target_distribution.png 已生成")