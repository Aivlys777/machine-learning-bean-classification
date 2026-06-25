"""
评估模块
包含模型评估、可视化等功能
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import logging
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """评估器类 - 用于模型评估和可视化"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.fig_dir = self.results_dir / 'figures'
        self.metrics_dir = self.results_dir / 'metrics'
        self.log_dir = self.results_dir / 'logs'
        
        # 创建必要的目录
        self._create_dirs()
        
    def _create_dirs(self):
        """创建必要的目录"""
        dirs = [
            self.fig_dir / 'confusion_matrices',
            self.fig_dir / 'feature_importance',
            self.fig_dir / 'roc_curves',
            self.metrics_dir,
            self.log_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def calculate_accuracy(self, y_true, y_pred):
        """计算准确率"""
        return accuracy_score(y_true, y_pred)
    
    def evaluate(self, y_true, y_pred, model_name='', y_pred_proba=None, class_names=None):
        """
        全面评估模型
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称
            y_pred_proba: 预测概率（用于ROC/AUC）
            class_names: 类别名称列表
            
        Returns:
            评估结果字典
        """
        result = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # 计算每类的指标
        result['per_class'] = {
            'precision': precision_score(y_true, y_pred, average=None, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=None, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=None, zero_division=0)
        }
        
        # 如果有概率预测，计算AUC
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                if y_pred_proba.shape[1] == n_classes:
                    result['auc'] = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
            except:
                result['auc'] = None
                logger.warning(f"{model_name}: 无法计算AUC")
        else:
            result['auc'] = None
        
        # 保存分类报告
        result['classification_report'] = classification_report(y_true, y_pred, zero_division=0)
        
        # 保存混淆矩阵图
        if model_name:
            self.plot_confusion_matrix(result['confusion_matrix'], model_name)
            
            # 如果有概率预测，绘制ROC曲线
            if y_pred_proba is not None and result.get('auc') is not None:
                self.plot_roc_curve(y_true, y_pred_proba, model_name)
        
        return result
    
    def plot_confusion_matrix(self, conf_matrix, model_name):
        """绘制混淆矩阵"""
        fig, ax = plt.subplots(figsize=(9, 7))
        
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            cbar_kws={'label': 'Count'},
            square=True,
            ax=ax
        )
        
        ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        save_path = self.fig_dir / 'confusion_matrices' / f'{model_name}_confusion.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"混淆矩阵已保存: {save_path}")
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """
        绘制ROC曲线（多分类）
        
        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            model_name: 模型名称
        """
        n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        fig, ax = plt.subplots(figsize=(9, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                   label=f'Class {i} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{model_name} - ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.fig_dir / 'roc_curves' / f'{model_name}_roc.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC曲线已保存: {save_path}")
    
    def plot_feature_importance(self, importance, feature_names, model_name, top_n=20):
        """
        绘制特征重要性
        
        Args:
            importance: 特征重要性数组
            feature_names: 特征名称列表
            model_name: 模型名称
            top_n: 显示前N个特征
        """
        if importance is None or len(importance) == 0:
            logger.warning(f"{model_name}: 没有特征重要性数据")
            return
        
        # 确保长度匹配
        if len(importance) > len(feature_names):
            importance = importance[:len(feature_names)]
        elif len(importance) < len(feature_names):
            feature_names = feature_names[:len(importance)]
        
        # 创建DataFrame并排序
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # 只取前N个
        if len(importance_df) > top_n:
            importance_df = importance_df.tail(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))[::-1]
        bars = ax.barh(importance_df['feature'], importance_df['importance'], color=colors)
        
        ax.set_title(f'{model_name} - Feature Importance (Top {len(importance_df)})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.grid(True, axis='x', alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars, importance_df['importance']):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        save_path = self.fig_dir / 'feature_importance' / f'{model_name}_importance.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"特征重要性图已保存: {save_path}")
        
        # 保存重要性数据到CSV
        csv_path = self.metrics_dir / f'{model_name}_feature_importance.csv'
        importance_df.to_csv(csv_path, index=False)
        
        return importance_df
    
    def plot_comparison(self, results, training_times, prediction_times):
        """
        绘制模型对比图表
        
        Args:
            results: 评估结果字典 {model_name: result_dict}
            training_times: 训练时间字典 {model_name: time}
            prediction_times: 预测时间字典 {model_name: time}
        """
        if not results:
            logger.warning("没有结果数据，跳过对比图表")
            return
        
        models = list(results.keys())
        n_models = len(models)
        
        # 提取指标
        accuracies = [results[m]['accuracy'] for m in models]
        f1_scores = [results[m]['f1_score'] for m in models]
        precisions = [results[m]['precision'] for m in models]
        recalls = [results[m]['recall'] for m in models]
        train_times = [training_times.get(m, 0) for m in models]
        pred_times = [prediction_times.get(m, 0) for m in models]
        
        # 提取AUC（如果有）
        aucs = []
        for m in models:
            auc_val = results[m].get('auc')
            aucs.append(auc_val if auc_val is not None else 0)
        
        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. 准确率和F1对比
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 准确率对比
        ax = axes[0, 0]
        colors = ['#2E86AB' if a == max(accuracies) else '#A0C4E2' for a in accuracies]
        bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_ylim(0.5, 1.0)
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # F1-Score对比
        ax = axes[0, 1]
        colors = ['#A23B72' if f == max(f1_scores) else '#E8A0BF' for f in f1_scores]
        bars = ax.bar(models, f1_scores, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title('Model F1-Score Comparison', fontsize=13, fontweight='bold')
        ax.set_ylabel('F1-Score', fontsize=11)
        ax.set_ylim(0.5, 1.0)
        for bar, val in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # AUC对比（如果有）
        ax = axes[0, 2]
        if any(a > 0 for a in aucs):
            colors = ['#F18F01' if a == max(aucs) else '#FADCA5' for a in aucs]
            bars = ax.bar(models, aucs, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_title('Model AUC Comparison', fontsize=13, fontweight='bold')
            ax.set_ylabel('AUC (Weighted)', fontsize=11)
            ax.set_ylim(0.5, 1.0)
            for bar, val in zip(bars, aucs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'AUC not available\n(no probability predictions)',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Model AUC Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 训练时间对比
        ax = axes[1, 0]
        colors = ['#2C3E50' if t == min(train_times) else '#95A5A6' for t in train_times]
        bars = ax.bar(models, train_times, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title('Model Training Time Comparison', fontsize=13, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=11)
        for bar, val in zip(bars, train_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}s', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 预测时间对比
        ax = axes[1, 1]
        colors = ['#27AE60' if t == min(pred_times) else '#82E0AA' for t in pred_times]
        bars = ax.bar(models, pred_times, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title('Model Prediction Time Comparison', fontsize=13, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=11)
        for bar, val in zip(bars, pred_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.4f}s', ha='center', va='bottom', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 综合雷达图（只有2个以上模型时才绘制）
        ax = axes[1, 2]
        if n_models >= 2:
            # 归一化指标（使用4个指标：准确率、F1、训练速度、预测速度）
            max_train_time = max(train_times) if max(train_times) > 0 else 1
            max_pred_time = max(pred_times) if max(pred_times) > 0 else 1
            
            metrics_data = np.array([
                accuracies,
                f1_scores,
                [1 - t/max_train_time for t in train_times],
                [1 - t/max_pred_time for t in pred_times]
            ])
            
            metrics_data = metrics_data.T
            
            angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
            angles += angles[:1]
            
            for i, model in enumerate(models):
                values = metrics_data[i].tolist()
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=model)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(['Accuracy', 'F1-Score', 'Train Speed', 'Predict Speed'])
            ax.set_ylim(0, 1)
            ax.set_title('Model Comprehensive Comparison', fontsize=13, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Need at least 2 models\nfor radar chart',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Model Comprehensive Comparison', fontsize=13, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('Multi-Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = self.fig_dir / 'comprehensive_comparison.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"综合对比图已保存: {save_path}")
    
    def plot_robustness_heatmap(self, robustness_results):
        """
        绘制鲁棒性测试热图
        
        Args:
            robustness_results: 鲁棒性测试结果
                {
                    'gaussian': {0.05: {'RF': 0.95, ...}, ...},
                    'uniform': {0.05: {'RF': 0.95, ...}, ...}
                }
        """
        if not robustness_results:
            logger.warning("没有鲁棒性测试结果")
            return
        
        # 整理数据
        noise_types = []
        noise_levels = []
        model_names = []
        accuracies = []
        
        for noise_type, levels in robustness_results.items():
            for level, model_results in levels.items():
                for model_name, acc in model_results.items():
                    noise_types.append(noise_type)
                    noise_levels.append(level)
                    model_names.append(model_name)
                    accuracies.append(acc)
        
        # 分组绘制
        n_types = len(robustness_results)
        fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5))
        
        if n_types == 1:
            axes = [axes]
        
        for idx, (noise_type, levels) in enumerate(robustness_results.items()):
            ax = axes[idx]
            
            # 准备热图数据
            model_names = list(list(levels.values())[0].keys())
            level_values = sorted(levels.keys())
            
            data = []
            for level in level_values:
                row = [levels[level].get(model, 0) for model in model_names]
                data.append(row)
            
            # 绘制热图
            sns.heatmap(
                data,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0.5,
                vmax=1.0,
                xticklabels=model_names,
                yticklabels=[f'{l:.2f}' for l in level_values],
                ax=ax,
                cbar_kws={'label': 'Accuracy'}
            )
            
            ax.set_title(f'{noise_type.upper()} Noise Robustness', fontsize=13, fontweight='bold')
            ax.set_xlabel('Models', fontsize=11)
            ax.set_ylabel('Noise Level', fontsize=11)
        
        plt.suptitle('Model Robustness Against Different Types of Noise', 
                    fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        save_path = self.fig_dir / 'robustness_heatmap.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"鲁棒性热图已保存: {save_path}")
    
    def plot_learning_curves(self, train_scores, val_scores, model_name):
        """
        绘制学习曲线
        
        Args:
            train_scores: 训练集分数列表（或损失值列表）
            val_scores: 验证集分数列表（可选）
            model_name: 模型名称
        """
        if train_scores is None or len(train_scores) == 0:
            logger.warning(f"{model_name}: 没有学习曲线数据")
            return
        
        fig, ax = plt.subplots(figsize=(9, 6))
        
        epochs = range(1, len(train_scores) + 1)
        
        # 判断是损失值还是分数
        is_loss = np.mean(train_scores) < 1.0 and np.min(train_scores) >= 0
        
        ax.plot(epochs, train_scores, 'b-', 
                label='Training Loss' if is_loss else 'Training Score', 
                linewidth=2)
        
        if val_scores is not None and len(val_scores) > 0:
            ax.plot(epochs[:len(val_scores)], val_scores, 'r-', 
                    label='Validation Loss' if is_loss else 'Validation Score', 
                    linewidth=2)
            ax.fill_between(epochs[:len(val_scores)], 
                           train_scores[:len(val_scores)], val_scores, 
                           alpha=0.2, color='purple')
        
        ax.set_title(f'{model_name} - Learning Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs / Iterations', fontsize=12)
        ax.set_ylabel('Loss' if is_loss else 'Score', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 标注过拟合区域
        if val_scores is not None and len(val_scores) > 5:
            gap = [t - v for t, v in zip(train_scores[:len(val_scores)], val_scores)]
            max_gap_idx = np.argmax(gap)
            if gap[max_gap_idx] > 0.05:
                ax.axvline(x=max_gap_idx, color='gray', linestyle='--', alpha=0.5)
                ax.text(max_gap_idx, 0.5, 'Overfitting starts', rotation=90, fontsize=9, color='gray')
        
        plt.tight_layout()
        save_path = self.fig_dir / f'{model_name}_learning_curves.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"学习曲线已保存: {save_path}")
    
    def plot_loss_curves_comparison(self, loss_curves):
        """
        绘制所有模型的Loss曲线对比
        
        Args:
            loss_curves: {model_name: [loss_values]}
        """
        if not loss_curves:
            logger.warning("没有Loss曲线数据，跳过绘制")
            return
        
        # 过滤掉None或空列表
        valid_curves = {name: curves for name, curves in loss_curves.items() 
                       if curves is not None and len(curves) > 0}
        
        if not valid_curves:
            logger.warning("没有有效的Loss曲线数据")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid_curves)))
        
        for idx, (model_name, losses) in enumerate(valid_curves.items()):
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, '-', color=colors[idx], 
                   label=f'{model_name}', linewidth=2, markersize=3)
            # 标记最后一点
            ax.plot(epochs[-1], losses[-1], 'o', color=colors[idx], markersize=8)
            # 添加最终loss值标注
            ax.annotate(f'{losses[-1]:.4f}', 
                       xy=(epochs[-1], losses[-1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, color=colors[idx])
        
        ax.set_title('Model Loss Curves Comparison', fontsize=15, fontweight='bold')
        ax.set_xlabel('Epochs / Iterations', fontsize=12)
        ax.set_ylabel('Loss (Log Loss / Cross Entropy)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        # 如果有多个模型，找到最小loss值设置y轴下限
        min_loss = min([min(curves) for curves in valid_curves.values()])
        max_loss = max([max(curves) for curves in valid_curves.values()])
        y_margin = (max_loss - min_loss) * 0.1
        ax.set_ylim(max(0, min_loss - y_margin), max_loss + y_margin)
        
        plt.tight_layout()
        save_path = self.fig_dir / 'loss_curves_comparison.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Loss曲线对比图已保存: {save_path}")
        
        # 保存loss曲线数据到CSV
        # 将不同长度的列表对齐
        max_len = max([len(curves) for curves in valid_curves.values()])
        loss_dict = {}
        for name, curves in valid_curves.items():
            # 填充NaN使长度一致
            padded = list(curves) + [np.nan] * (max_len - len(curves))
            loss_dict[name] = padded
        
        loss_df = pd.DataFrame(loss_dict)
        loss_path = self.metrics_dir / 'loss_curves.csv'
        loss_df.to_csv(loss_path, index=False)
        logger.info(f"Loss曲线数据已保存: {loss_path}")
    
    def get_results_summary(self, results, training_times, prediction_times):
        """
        获取结果摘要DataFrame
        
        Args:
            results: 评估结果字典
            training_times: 训练时间字典
            prediction_times: 预测时间字典
            
        Returns:
            DataFrame格式的结果摘要
        """
        if not results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, result in results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'Training Time (s)': f"{training_times.get(model_name, 0):.3f}",
                'Prediction Time (s)': f"{prediction_times.get(model_name, 0):.4f}"
            }
            # 添加AUC（如果有）
            if result.get('auc') is not None:
                row['AUC'] = f"{result['auc']:.4f}"
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        
        # 保存到CSV
        csv_path = self.metrics_dir / 'model_comparison.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"结果摘要已保存: {csv_path}")
        
        return df
    
    def get_best_model(self, results, metric='accuracy'):
        """
        获取最佳模型
        
        Args:
            results: 评估结果字典
            metric: 评估指标 ('accuracy', 'f1_score', 'precision', 'recall', 'auc')
            
        Returns:
            最佳模型名称和分数
        """
        if not results:
            return None, 0
        
        best_model = None
        best_score = -1
        
        for model_name, result in results.items():
            score = result.get(metric, 0)
            if score is not None and score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score


if __name__ == '__main__':
    # 测试评估器
    evaluator = Evaluator()
    print("评估器模块加载成功")
    print(f"结果目录: {evaluator.results_dir}")