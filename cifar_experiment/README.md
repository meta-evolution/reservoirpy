# CIFAR-10 Reservoir Computing 实验文件夹

本文件夹包含了使用Reservoir Computing进行CIFAR-10图像分类的完整实验内容。

## 文件清单

### 主要程序
- `cifar10_reservoir_classification.py` - CIFAR-10图像分类的Reservoir Computing主程序

### 实验文档
- `CIFAR10_Reservoir_Computing_Experiment_Report.md` - 完整的实验报告，包含系统性的实验结果分析
- `CIFAR10_Program_Documentation.md` - 程序使用说明文档，包含详细的函数说明和使用指南

### 实验结果图表
- `cifar10_reservoir_results.png` - 基础网络配置的实验结果图表
- `cifar10_enhanced_reservoir_results.png` - 增强网络配置的实验结果图表

## 实验概述

本实验系统性地评估了Reservoir Computing在CIFAR-10图像分类任务上的性能，通过调整以下关键参数进行了全面的性能分析：

- **网络架构**: 输入层大小、Reservoir神经元数量
- **数据规模**: 5000样本 vs 50000样本
- **输入表示**: 96特征×128时间步 vs 256特征×48时间步
- **时序策略**: 重复增强时序长度

## 主要发现

1. **数据规模的决定性作用**: 完整数据集相比小数据集，测试准确率提升9.46个百分点
2. **时序长度的重要性**: 128时间步配置比48时间步配置性能更优
3. **最优配置**: 96输入特征 + 1000神经元Reservoir + 128时间步，测试准确率46.66%

## 使用方法

1. 确保已安装所需依赖包（numpy, matplotlib, scikit-learn, tensorflow, reservoirpy）
2. 运行主程序：`python cifar10_reservoir_classification.py`
3. 查看生成的结果图表和控制台输出

## 实验环境

- Python 3.11
- ReservoirPy
- TensorFlow/Keras (用于CIFAR-10数据加载)
- scikit-learn (用于评估指标)

## 实验日期

2025年8月5日

## 技术特点

- 图像到时间序列的转换策略
- 系统性的超参数对比实验
- 详细的类别级别性能分析
- 过拟合程度的量化评估
- 可视化结果展示
