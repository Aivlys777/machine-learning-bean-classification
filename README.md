# 机器学习期末作业 - Dry Bean Dataset 多分类实验

> 基于 Dry Bean Dataset 的机器学习全流程项目，包含数据分析、数据处理、多算法对比实验和系统展示

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.0-yellow.svg)](https://catboost.ai/)

---

## 📋 目录

- [项目概述](#项目概述)
- [数据集说明](#数据集说明)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [功能模块](#功能模块)
- [实验对比](#实验对比)
- [结果展示](#结果展示)
- [课程总结](#课程总结)
- [贡献者](#贡献者)

---

## 📖 项目概述

本项目是一个完整的机器学习工程项目，基于 **Dry Bean Dataset**（干豆数据集）进行多分类任务。项目涵盖了从数据分析、数据预处理、多算法实验到系统集成的全流程。

### 🎯 项目目标

- 完成干豆品种的多分类任务（7个类别）
- 实现并对比4种主流分类算法
- 进行全面的模型评估和鲁棒性分析
- 构建工程化的项目结构和可视化报告

### 📊 实现算法

| 算法 | 类型 | 来源 |
|------|------|------|
| **Random Forest** | 集成学习 - Bagging | 课堂讲解 |
| **XGBoost** | 梯度提升 - Boosting | 课堂讲解 |
| **LightGBM** | 梯度提升 - Boosting | 课外自学 |
| **CatBoost** | 梯度提升 - Boosting | 课外自学 |
| **ANN (MLP)** | 神经网络 | 课外自学 |

---

## 📦 数据集说明

### 数据集来源
[Dry Bean Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset) 来自 UCI Machine Learning Repository

### 数据概况

| 属性 | 说明 |
|------|------|
| **样本数量** | 13,611 条 |
| **特征维度** | 16 维 |
| **目标类别** | 7 种干豆品种 |
| **数据完整性** | 无缺失值 |

### 特征描述

| 特征名 | 描述 | 类型 |
|--------|------|------|
| Area | 豆粒面积 | 数值型 |
| Perimeter | 豆粒周长 | 数值型 |
| MajorAxisLength | 长轴长度 | 数值型 |
| MinorAxisLength | 短轴长度 | 数值型 |
| AspectRation | 长宽比 | 数值型 |
| Eccentricity | 离心率 | 数值型 |
| ConvexArea | 凸包面积 | 数值型 |
| EquivDiameter | 等效直径 | 数值型 |
| Extent | 范围 | 数值型 |
| Solidity | 坚实度 | 数值型 |
| roundness | 圆度 | 数值型 |
| Compactness | 紧凑度 | 数值型 |
| ShapeFactor1 | 形状因子1 | 数值型 |
| ShapeFactor2 | 形状因子2 | 数值型 |
| ShapeFactor3 | 形状因子3 | 数值型 |
| ShapeFactor4 | 形状因子4 | 数值型 |
| **Class** | **豆类品种（目标）** | **类别型** |

### 目标类别
- `BARBUNYA` - 巴布尼亚豆
- `BOMBAY` - 孟买豆
- `CALI` - 卡利豆
- `DERMASON` - 德马森豆
- `HOROZ` - 霍罗兹豆
- `SEKER` - 塞克尔豆
- `SIRMALI` - 西尔马里豆

---

## 🛠 技术栈

### 核心依赖

```txt
numpy==1.24.3          # 数值计算
pandas==2.0.3          # 数据处理
scikit-learn==1.3.0    # 机器学习
matplotlib==3.7.2      # 可视化
seaborn==0.12.2        # 统计可视化
xgboost==1.7.6         # XGBoost算法
lightgbm==4.0.0        # LightGBM算法
catboost==1.2.0        # CatBoost算法
joblib==1.3.1          # 模型持久化
openpyxl==3.1.2        # Excel文件支持