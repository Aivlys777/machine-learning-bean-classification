"""
数据分析模块
生成数据分析和处理报告
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """数据分析器类 - 生成数据分析和处理报告"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.fig_dir = self.results_dir / 'figures'
        self.metrics_dir = self.results_dir / 'metrics'
        self.report_dir = self.results_dir / 'reports'
        
        # 创建目录
        self._create_dirs()
    
    def _create_dirs(self):
        """创建必要的目录"""
        dirs = [
            self.fig_dir / 'data_analysis',
            self.metrics_dir,
            self.report_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def analyze_data(self, train_df, test_df, target_col='Class'):
        """
        完整的数据分析
        
        Args:
            train_df: 训练集DataFrame
            test_df: 测试集DataFrame
            target_col: 目标列名
            
        Returns:
            分析结果字典
        """
        results = {
            'data_overview': self._get_data_overview(train_df, test_df),
            'data_quality': self._get_data_quality(train_df),
            'target_distribution': self._get_target_distribution(train_df, target_col),
            'feature_stats': self._get_feature_stats(train_df),
            'correlation': self._get_correlation(train_df),
            'data_contamination': self._get_data_contamination(train_df)
        }
        
        # 生成图表
        self._plot_target_distribution(train_df, target_col)
        self._plot_feature_distributions(train_df)
        self._plot_correlation_heatmap(train_df)
        self._plot_feature_boxplots(train_df)
        
        # 保存报告
        self._save_report(results)
        
        return results
    
    def _get_data_overview(self, train_df, test_df):
        """获取数据概述"""
        return {
            'train_shape': train_df.shape,
            'test_shape': test_df.shape,
            'features': train_df.columns.tolist(),
            'num_features': len(train_df.columns) - 1,  # 减去目标列
            'num_classes': train_df.iloc[:, -1].nunique() if train_df.shape[1] > 0 else 0
        }
    
    def _get_data_quality(self, df):
        """获取数据质量信息"""
        quality = {
            'missing_values': df.isnull().sum().to_dict(),
            'total_missing': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # 检查特殊字符
        special_chars = {}
        for col in df.columns:
            if df[col].dtype == 'object':
                # 检查是否有 '?' 或其他特殊字符
                unique_vals = df[col].unique()
                has_question = '?' in unique_vals
                special_chars[col] = {
                    'has_question_mark': has_question,
                    'unique_values': len(unique_vals),
                    'sample_values': unique_vals[:5].tolist()
                }
        
        quality['special_chars'] = special_chars
        return quality
    
    def _get_target_distribution(self, df, target_col):
        """获取目标分布"""
        if target_col not in df.columns:
            target_col = df.columns[-1]
        
        distribution = df[target_col].value_counts()
        return {
            'counts': distribution.to_dict(),
            'percentages': (distribution / len(df) * 100).to_dict(),
            'num_classes': len(distribution)
        }
    
    def _get_feature_stats(self, df):
        """获取特征统计信息"""
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q50': df[col].quantile(0.50),
                    'q75': df[col].quantile(0.75),
                    'skew': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                }
        return stats
    
    def _get_correlation(self, df):
        """获取特征相关性"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            return corr_matrix.to_dict()
        return {}
    
    def _get_data_contamination(self, df):
        """检测数据污染情况"""
        contamination = {
            'issues': [],
            'warnings': []
        }
        
        # 检查缺失值
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            contamination['issues'].append({
                'type': 'missing_values',
                'columns': missing_cols,
                'description': f'以下列存在缺失值: {missing_cols}'
            })
        
        # 检查非数值数据
        non_numeric_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # 检查是否包含 '?'
                if '?' in df[col].values:
                    contamination['issues'].append({
                        'type': 'special_character',
                        'column': col,
                        'description': f"列 '{col}' 包含 '?' 占位符"
                    })
                else:
                    non_numeric_cols.append(col)
        
        if non_numeric_cols:
            contamination['warnings'].append({
                'type': 'non_numeric',
                'columns': non_numeric_cols,
                'description': f'以下列为非数值类型，可能需要编码: {non_numeric_cols}'
            })
        
        # 检查异常值（使用IQR方法）
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                contamination['warnings'].append({
                    'type': 'outliers',
                    'column': col,
                    'outlier_count': len(outliers),
                    'outlier_ratio': len(outliers) / len(df),
                    'description': f"列 '{col}' 检测到 {len(outliers)} 个异常值 ({len(outliers)/len(df)*100:.2f}%)"
                })
        
        return contamination
    
    def _plot_target_distribution(self, df, target_col):
        """绘制目标分布"""
        if target_col not in df.columns:
            target_col = df.columns[-1]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 柱状图
        counts = df[target_col].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black')
        axes[0].set_title('Target Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        for i, (label, count) in enumerate(counts.items()):
            axes[0].text(i, count + 5, str(count), ha='center', fontsize=10)
        
        # 饼图
        axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
        axes[1].set_title('Target Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.fig_dir / 'data_analysis' / 'target_distribution.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"目标分布图已保存: {save_path}")
    
    def _plot_feature_distributions(self, df):
        """绘制特征分布"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 8:
            numeric_cols = numeric_cols[:8]
        
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row][col_idx] if n_rows > 1 else axes[col_idx]
            
            ax.hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(labelsize=7)
        
        # 隐藏多余的子图
        for idx in range(len(numeric_cols), n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            if n_rows > 1:
                axes[row][col_idx].set_visible(False)
            else:
                axes[col_idx].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = self.fig_dir / 'data_analysis' / 'feature_distributions.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"特征分布图已保存: {save_path}")
    
    def _plot_correlation_heatmap(self, df):
        """绘制相关性热图"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # 生成mask只显示下三角
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True,
                       linewidths=0.5, cbar_kws={'shrink': 0.8},
                       ax=ax)
            
            ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            save_path = self.fig_dir / 'data_analysis' / 'correlation_heatmap.png'
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
            logger.info(f"相关性热图已保存: {save_path}")
    
    def _plot_feature_boxplots(self, df):
        """绘制特征箱线图（检测异常值）"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 8:
            numeric_cols = numeric_cols[:8]
        
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = [axes]
        if n_cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx, col in enumerate(numeric_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            ax = axes[row][col_idx] if n_rows > 1 else axes[col_idx]
            
            ax.boxplot(df[col].dropna(), vert=True)
            ax.set_title(col, fontsize=10)
            ax.set_ylabel('Value', fontsize=8)
            ax.tick_params(labelsize=7)
        
        # 隐藏多余的子图
        for idx in range(len(numeric_cols), n_rows * n_cols):
            row = idx // n_cols
            col_idx = idx % n_cols
            if n_rows > 1:
                axes[row][col_idx].set_visible(False)
            else:
                axes[col_idx].set_visible(False)
        
        plt.suptitle('Feature Boxplots (Outlier Detection)', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = self.fig_dir / 'data_analysis' / 'feature_boxplots.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"特征箱线图已保存: {save_path}")
    
    def _save_report(self, results):
        """保存分析报告到JSON"""
        report_path = self.report_dir / 'data_analysis_report.json'
        
        # 转换numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        results_clean = convert_numpy(results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据分析报告已保存: {report_path}")
        
        # 同时生成Markdown报告
        self._generate_markdown_report(results)
    
    def _generate_markdown_report(self, results):
        """生成Markdown格式的数据分析报告"""
        md_path = self.report_dir / 'data_analysis_report.md'
        
        lines = []
        lines.append("# 数据分析报告\n")
        lines.append("## 1. 数据概述\n")
        
        overview = results.get('data_overview', {})
        lines.append(f"- **训练集大小**: {overview.get('train_shape', 'N/A')}")
        lines.append(f"- **测试集大小**: {overview.get('test_shape', 'N/A')}")
        lines.append(f"- **特征数量**: {overview.get('num_features', 'N/A')}")
        lines.append(f"- **类别数量**: {overview.get('num_classes', 'N/A')}")
        lines.append(f"- **特征列表**: {', '.join(overview.get('features', [])[:10])}...")
        lines.append("")
        
        lines.append("## 2. 数据污染情况\n")
        contamination = results.get('data_contamination', {})
        
        lines.append("### 2.1 存在的问题\n")
        issues = contamination.get('issues', [])
        if issues:
            for issue in issues:
                lines.append(f"- **{issue.get('type', 'unknown')}**: {issue.get('description', '')}")
        else:
            lines.append("✅ 未发现严重数据污染问题")
        lines.append("")
        
        lines.append("### 2.2 警告\n")
        warnings_list = contamination.get('warnings', [])
        if warnings_list:
            for warn in warnings_list:
                lines.append(f"- **{warn.get('type', 'unknown')}**: {warn.get('description', '')}")
        else:
            lines.append("✅ 未发现数据质量警告")
        lines.append("")
        
        lines.append("## 3. 数据质量\n")
        quality = results.get('data_quality', {})
        lines.append(f"- **缺失值总数**: {quality.get('total_missing', 0)}")
        lines.append(f"- **重复行数**: {quality.get('duplicates', 0)}")
        lines.append("")
        
        lines.append("## 4. 目标分布\n")
        target = results.get('target_distribution', {})
        lines.append("| 类别 | 数量 | 占比 |")
        lines.append("|------|------|------|")
        counts = target.get('counts', {})
        percentages = target.get('percentages', {})
        for cls, count in counts.items():
            pct = percentages.get(cls, 0)
            lines.append(f"| {cls} | {count} | {pct:.1f}% |")
        lines.append("")
        
        lines.append("## 5. 特征统计\n")
        stats = results.get('feature_stats', {})
        lines.append("| 特征 | 均值 | 标准差 | 最小值 | 最大值 | 偏度 |")
        lines.append("|------|------|--------|--------|--------|------|")
        for name, stat in stats.items():
            if name != 'Class':
                lines.append(f"| {name} | {stat.get('mean', 0):.3f} | {stat.get('std', 0):.3f} | {stat.get('min', 0):.3f} | {stat.get('max', 0):.3f} | {stat.get('skew', 0):.3f} |")
        lines.append("")
        
        lines.append("## 6. 图表说明\n")
        lines.append("以下图表已生成并保存到 `results/figures/data_analysis/` 目录：")
        lines.append("")
        lines.append("| 图表 | 说明 |")
        lines.append("|------|------|")
        lines.append("| `target_distribution.png` | 目标类别分布（柱状图+饼图） |")
        lines.append("| `feature_distributions.png` | 特征分布直方图 |")
        lines.append("| `correlation_heatmap.png` | 特征相关性热图 |")
        lines.append("| `feature_boxplots.png` | 特征箱线图（异常值检测） |")
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Markdown分析报告已保存: {md_path}")


if __name__ == '__main__':
    print("DataAnalyzer 模块加载成功")