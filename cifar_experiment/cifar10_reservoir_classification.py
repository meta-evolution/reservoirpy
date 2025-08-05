#!/usr/bin/env python
# coding: utf-8

"""
CIFAR-10 Classification with Reservoir Computing
模仿5-Classification-with-RC.py，使用reservoir网络对CIFAR-10数据集进行分类
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy import set_seed, verbosity

# 设置随机种子和静默模式
set_seed(42)
verbosity(0)

def load_and_preprocess_cifar10():
    """加载并预处理真实的CIFAR-10数据集（从本地文件）"""
    import tarfile
    import pickle
    import os
    
    print("Loading CIFAR-10 dataset from local file...")
    
    cifar_path = "/Users/lijin/Documents/cifar-10-python.tar.gz"
    
    try:
        # 检查文件是否存在
        if not os.path.exists(cifar_path):
            raise FileNotFoundError(f"CIFAR-10 file not found at {cifar_path}")
        
        print(f"Found CIFAR-10 file: {cifar_path}")
        
        # 解压并读取数据
        with tarfile.open(cifar_path, 'r:gz') as tar:
            # 读取训练数据
            X_train = []
            y_train = []
            
            # 训练数据分为5个batch
            for i in range(1, 6):
                batch_name = f"cifar-10-batches-py/data_batch_{i}"
                batch_file = tar.extractfile(batch_name)
                if batch_file:
                    batch_data = pickle.load(batch_file, encoding='bytes')
                    X_train.append(batch_data[b'data'])
                    y_train.extend(batch_data[b'labels'])
            
            # 读取测试数据
            test_batch_name = "cifar-10-batches-py/test_batch"
            test_file = tar.extractfile(test_batch_name)
            if test_file:
                test_data = pickle.load(test_file, encoding='bytes')
                X_test = test_data[b'data']
                y_test = test_data[b'labels']
        
        # 合并训练数据
        X_train = np.vstack(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # 重塑数据：从(N, 3072)到(N, 32, 32, 3)
        X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # 归一化到[-1, 1]范围
        X_train = X_train.astype('float32') / 127.5 - 1.0
        X_test = X_test.astype('float32') / 127.5 - 1.0
        
        # 转换标签为one-hot编码
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        print(f"Successfully loaded CIFAR-10 dataset from local file!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Testing labels shape: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"Failed to load CIFAR-10 dataset from local file: {e}")
        print("Falling back to tensorflow keras download...")
        
        try:
            # 尝试使用tensorflow下载
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            
            # 归一化到[-1, 1]范围
            X_train = X_train.astype('float32') / 127.5 - 1.0
            X_test = X_test.astype('float32') / 127.5 - 1.0
            
            # 转换标签为one-hot编码
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)
            
            print(f"Successfully loaded CIFAR-10 via tensorflow!")
            print(f"Training data shape: {X_train.shape}")
            print(f"Testing data shape: {X_test.shape}")
            print(f"Training labels shape: {y_train.shape}")
            print(f"Testing labels shape: {y_test.shape}")
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e2:
            print(f"Tensorflow download also failed: {e2}")
            raise Exception("无法加载CIFAR-10数据集")

def images_to_sequences(images, input_features=256, repeat_times=4):
    """
    将图像转换为时间序列，支持重复策略
    将32x32x3的图像重塑为可配置的输入特征数，并可选择重复多次
    
    Parameters:
    - images: 图像数据 (N, 32, 32, 3)
    - input_features: 每个时间步的特征数 (默认256)
    - repeat_times: 重复次数 (默认4，用于增加时序长度)
    """
    n_samples, height, width, channels = images.shape
    total_pixels = height * width * channels  # 32*32*3 = 3072
    
    # 计算基础时间步数：确保能整除
    if total_pixels % input_features == 0:
        base_timesteps = total_pixels // input_features
    else:
        # 如果不能整除，需要padding
        base_timesteps = (total_pixels + input_features - 1) // input_features
        padding_needed = base_timesteps * input_features - total_pixels
    
    sequences = []
    
    for img in images:
        # 将图像展平为一维
        flat_img = img.flatten()  # shape: (3072,)
        
        # 如果需要padding，添加零
        if total_pixels % input_features != 0:
            flat_img = np.pad(flat_img, (0, padding_needed), mode='constant', constant_values=0)
        
        # 重塑为基础时间序列：(base_timesteps, input_features)
        base_sequence = flat_img.reshape(base_timesteps, input_features)
        
        # 重复序列 repeat_times 次
        if repeat_times > 1:
            repeated_sequence = np.tile(base_sequence, (repeat_times, 1))
        else:
            repeated_sequence = base_sequence
        
        sequences.append(repeated_sequence)
    
    return sequences

def create_reservoir_model(reservoir_size=1500, spectral_radius=0.9, leak_rate=0.1, ridge_param=1e-6):
    """创建reservoir分类模型"""
    print(f"Creating reservoir model with {reservoir_size} neurons...")
    
    source = Input()
    reservoir = Reservoir(reservoir_size, sr=spectral_radius, lr=leak_rate)
    readout = Ridge(ridge=ridge_param)
    
    model = source >> reservoir >> readout
    
    return source, reservoir, readout, model

def train_reservoir_classifier(reservoir, readout, X_train_seq, y_train):
    """训练reservoir分类器"""
    print("Training reservoir classifier...")
    
    # 收集所有训练序列的最后状态
    states_train = []
    for i, x in enumerate(X_train_seq):
        if i % 1000 == 0:
            print(f"Processing training sample {i}/{len(X_train_seq)}")
        
        states = reservoir.run(x, reset=True)
        states_train.append(states[-1])  # 只取最后状态，不添加额外维度
    
    # 将状态列表转换为numpy数组
    states_train = np.array(states_train)
    
    # 训练读出层
    print("Training readout layer...")
    print(f"States shape: {states_train.shape}, Labels shape: {y_train.shape}")
    readout.fit(states_train, y_train)
    
    return states_train

def predict_reservoir_classifier(reservoir, readout, X_test_seq):
    """使用reservoir分类器进行预测"""
    print("Making predictions...")
    
    Y_pred = []
    for i, x in enumerate(X_test_seq):
        if i % 1000 == 0:
            print(f"Processing test sample {i}/{len(X_test_seq)}")
            
        states = reservoir.run(x, reset=True)
        y = readout.run(states[-1, np.newaxis])  # 只用最后状态预测
        Y_pred.append(y)
    
    return Y_pred

def evaluate_model(Y_pred, y_test):
    """评估模型性能"""
    Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
    y_test_class = [np.argmax(y_t) for y_t in y_test]
    
    accuracy = accuracy_score(y_test_class, Y_pred_class)
    
    return accuracy, Y_pred_class, y_test_class

def plot_training_curve(train_accuracies, val_accuracies, reservoir_sizes, sample_images=None):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 10))
    
    # 子图1：不同reservoir大小的性能
    plt.subplot(2, 3, 1)
    plt.plot(reservoir_sizes, train_accuracies, 'b-o', label='Training Accuracy', linewidth=2, markersize=8)
    plt.plot(reservoir_sizes, val_accuracies, 'r-o', label='Validation Accuracy', linewidth=2, markersize=8)
    plt.xlabel('Reservoir Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Reservoir Size')
    plt.legend()
    plt.grid(True)
    
    # 子图2：准确率条形图
    plt.subplot(2, 3, 2)
    x = np.arange(len(reservoir_sizes))
    width = 0.35
    plt.bar(x - width/2, train_accuracies, width, label='Training', alpha=0.8)
    plt.bar(x + width/2, val_accuracies, width, label='Validation', alpha=0.8)
    plt.xlabel('Reservoir Size')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xticks(x, reservoir_sizes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3：准确率差异
    plt.subplot(2, 3, 3)
    accuracy_diff = np.array(train_accuracies) - np.array(val_accuracies)
    plt.plot(reservoir_sizes, accuracy_diff, 'g-o', label='Train-Val Difference', linewidth=2, markersize=8)
    plt.xlabel('Reservoir Size')
    plt.ylabel('Accuracy Difference')
    plt.title('Overfitting Analysis')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 子图4-6：显示一些样本图像
    if sample_images is not None:
        class_names = ['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 
                       'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']
        
        for i in range(3):
            plt.subplot(2, 3, i + 4)
            # 显示第i个样本图像
            if i < len(sample_images):
                img = sample_images[i]
                # 将图像从[-1,1]范围映射到[0,1]显示
                img_display = (img + 1) / 2
                img_display = np.clip(img_display, 0, 1)
                plt.imshow(img_display)
                plt.title(f'Sample Image {i+1}')
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_reservoir_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=" * 60)
    print("CIFAR-10 Classification with Reservoir Computing")
    print("=" * 60)
    
    # 加载数据
    X_train, y_train, X_test, y_test = load_and_preprocess_cifar10()
    
    # 使用全部训练数据和完整测试数据
    print("Using full CIFAR-10 dataset...")
    n_train_samples = len(X_train)  # 50000
    n_test_samples = len(X_test)    # 10000
    
    X_train_subset = X_train
    y_train_subset = y_train
    X_test_subset = X_test
    y_test_subset = y_test
    
    print(f"Training with {n_train_samples} samples")
    print(f"Testing with {n_test_samples} samples")
    
    # 转换图像为序列 - 回到原始配置：96个输入特征，重复4次增加时序长度
    input_features = 96
    repeat_times = 4
    print(f"Converting images to sequences with {input_features} input features, repeated {repeat_times} times...")
    X_train_seq = images_to_sequences(X_train_subset, input_features=input_features, repeat_times=repeat_times)
    X_test_seq = images_to_sequences(X_test_subset, input_features=input_features, repeat_times=repeat_times)
    
    print(f"Sequence shape example: {X_train_seq[0].shape}")
    total_pixels = 32 * 32 * 3  # 3072
    base_timesteps = (total_pixels + input_features - 1) // input_features  # 向上取整
    final_timesteps = base_timesteps * repeat_times
    print(f"Network architecture with repetition strategy:")
    print(f"  Input features per timestep: {input_features}")
    print(f"  Base time steps: {base_timesteps}")
    print(f"  Repeat times: {repeat_times}")
    print(f"  Final time steps: {final_timesteps}")
    print(f"  Total sequence: {total_pixels} pixels -> {base_timesteps}x{input_features} -> {final_timesteps}x{input_features}")
    
    # 测试原始网络配置：输入层96神经元，重复策略，Reservoir Size=1000
    reservoir_size = 1000
    print(f"\n--- Testing Original Network Configuration with Full Dataset ---")
    print(f"Input Layer: {input_features} neurons")
    print(f"Reservoir Layer: {reservoir_size} neurons") 
    print(f"Output Layer: 10 neurons")
    print(f"Win weights: {reservoir_size}×{input_features}")
    print(f"W weights: {reservoir_size}×{reservoir_size}")
    print(f"Wout weights: 10×{reservoir_size}")
    print(f"Strategy: Each image repeated {repeat_times} times for longer temporal dynamics")
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 创建模型
    source, reservoir, readout, model = create_reservoir_model(
        reservoir_size=reservoir_size,
        spectral_radius=0.9,
        leak_rate=0.1,
        ridge_param=1e-6
    )
    
    # 训练模型
    states_train = train_reservoir_classifier(reservoir, readout, X_train_seq, y_train_subset)
    
    # 计算训练精度
    Y_train_pred = []
    for i in range(len(states_train)):
        y_pred = readout.run(states_train[i])
        Y_train_pred.append(y_pred)
    
    train_accuracy, _, _ = evaluate_model(Y_train_pred, y_train_subset)
    
    # 测试模型
    Y_pred = predict_reservoir_classifier(reservoir, readout, X_test_seq)
    test_accuracy, Y_pred_class, y_test_class = evaluate_model(Y_pred, y_test_subset)
    
    print(f"\n--- Enhanced Network Results ---")
    print(f"Training Accuracy: {train_accuracy * 100:.3f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.3f}%")
    
    # 显示一些预测样例
    print("\nSample predictions:")
    for i in range(min(10, len(Y_pred_class))):
        true_label = class_names[y_test_class[i]]
        pred_label = class_names[Y_pred_class[i]]
        correct = "✓" if true_label == pred_label else "✗"
        print(f"Sample {i}: True={true_label}, Predicted={pred_label} {correct}")
    
    # 计算类别准确率
    print(f"\n--- Per-Class Analysis ---")
    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}
    
    for true_idx, pred_idx in zip(y_test_class, Y_pred_class):
        true_name = class_names[true_idx]
        class_total[true_name] += 1
        if true_idx == pred_idx:
            class_correct[true_name] += 1
    
    for class_name in class_names:
        if class_total[class_name] > 0:
            acc = class_correct[class_name] / class_total[class_name] * 100
            print(f"{class_name}: {acc:.1f}% ({class_correct[class_name]}/{class_total[class_name]})")
    
    # 简化的可视化
    plt.figure(figsize=(12, 8))
    
    # 子图1：总体结果对比
    plt.subplot(2, 2, 1)
    accuracies = [train_accuracy * 100, test_accuracy * 100]
    labels = ['Training', 'Testing']
    colors = ['skyblue', 'lightcoral']
    bars = plt.bar(labels, accuracies, color=colors, alpha=0.8)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Enhanced Network Performance\n(Input: {input_features}, Reservoir: {reservoir_size})')
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 子图2：类别准确率
    plt.subplot(2, 2, 2)
    class_accs = [class_correct[name] / max(class_total[name], 1) * 100 for name in class_names]
    plt.bar(range(len(class_names)), class_accs, alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(len(class_names)), [name[:4] for name in class_names], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 子图3：网络结构图
    plt.subplot(2, 2, 3)
    plt.text(0.5, 0.8, 'Enhanced Network Architecture', ha='center', fontsize=14, fontweight='bold')
    plt.text(0.5, 0.65, f'Input Layer: {input_features} neurons', ha='center', fontsize=12)
    plt.text(0.5, 0.55, f'Reservoir Layer: {reservoir_size} neurons', ha='center', fontsize=12, color='blue')
    plt.text(0.5, 0.45, f'Output Layer: 10 neurons', ha='center', fontsize=12)
    plt.text(0.5, 0.3, f'Time steps: {final_timesteps}', ha='center', fontsize=10, style='italic')
    plt.text(0.5, 0.2, f'Repeat strategy: {repeat_times}x', ha='center', fontsize=10, style='italic', color='green')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # 子图4：样本图像
    plt.subplot(2, 2, 4)
    if len(X_train_subset) > 0:
        img = X_train_subset[0]
        img_display = (img + 1) / 2
        img_display = np.clip(img_display, 0, 1)
        plt.imshow(img_display)
        true_class = class_names[np.argmax(y_train_subset[0])]
        plt.title(f'Sample Image: {true_class}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('cifar10_enhanced_reservoir_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n--- Network Summary ---")
    print(f"Configuration: {input_features} input → {reservoir_size} reservoir → 10 output")
    print(f"Final Test Accuracy: {test_accuracy * 100:.3f}%")
    print(f"Overfitting: {(train_accuracy - test_accuracy) * 100:.1f}%")

if __name__ == "__main__":
    main()
