#!/usr/bin/env python
# coding: utf-8

"""
ESN Dynamics Trajectory Visualization
观察ESN（Echo State Network）的动力学轨迹
通过随机初始状态运行ESN并进行PCA降维可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# 导入ReservoirPy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reservoirpy.nodes import Reservoir, ESN, Ridge
from reservoirpy import set_seed, verbosity

# 设置随机种子和静默模式
set_seed(42)
verbosity(0)

def create_reservoir_only(reservoir_size=1000, spectral_radius=0.9, leak_rate=0.1):
    """创建一个独立的Reservoir（不连接readout）"""
    print(f"Creating standalone reservoir with {reservoir_size} neurons...")
    print(f"  Spectral radius: {spectral_radius}")
    print(f"  Leak rate: {leak_rate}")

    np.random.seed()
    
    reservoir = Reservoir(
        units=reservoir_size,
        sr=spectral_radius,
        lr=leak_rate,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        activation='tanh',
        seed=np.random.randint(0, 10000)
    )
    
    return reservoir

def create_esn_model(reservoir_size=1000, spectral_radius=0.9, leak_rate=0.1):
    """创建完整的ESN模型（包含readout）"""
    print(f"Creating ESN model with {reservoir_size} neurons...")
    
    esn = ESN(
        units=reservoir_size,
        sr=spectral_radius,
        lr=leak_rate,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        ridge=1e-6,
        input_bias=True,
        activation='tanh'
    )
    
    return esn

def run_reservoir_dynamics(reservoir, n_steps=10000, input_dim=96, input_type='random'):
    """
    运行Reservoir的动力学演化
    
    Parameters:
    -----------
    reservoir : Reservoir节点
    n_steps : 运行步数
    input_dim : 输入维度（匹配CIFAR-10实验的96维）
    input_type : 输入类型 ('random', 'zeros', 'constant')
    """
    print(f"\nRunning reservoir dynamics for {n_steps} steps...")
    print(f"  Input dimension: {input_dim}")
    print(f"  Input type: {input_type}")
    
    # 生成输入序列
    if input_type == 'random':
        # 随机输入（范围[-1, 1]）
        inputs = np.random.uniform(-1, 1, size=(n_steps, input_dim))
    elif input_type == 'zeros':
        # 零输入（自主动力学）
        inputs = np.zeros((n_steps, input_dim))
    elif input_type == 'constant':
        # 常数输入
        inputs = np.ones((n_steps, input_dim)) * 0.1
    else:
        raise ValueError(f"Unknown input type: {input_type}")
    
    # 初始化reservoir（第一个输入用于初始化权重矩阵）
    _ = reservoir.run(inputs[:1])
    
    # 设置随机初始状态
    initial_state = np.random.randn(reservoir.output_dim)
    initial_state = initial_state / np.linalg.norm(initial_state)  # 归一化
    reservoir._state = initial_state
    
    print(f"  Initial state norm: {np.linalg.norm(initial_state):.4f}")
    
    # 运行动力学并记录轨迹
    trajectory = []
    for i in range(n_steps):
        state = reservoir.run(inputs[i:i+1], reset=False)
        trajectory.append(state.flatten())
    
    trajectory = np.array(trajectory)
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Final state norm: {np.linalg.norm(trajectory[-1]):.4f}")
    
    return trajectory, inputs

def analyze_trajectory(trajectory):
    """分析轨迹的统计特性"""
    print("\n=== Trajectory Analysis ===")
    
    # 基本统计
    print(f"Mean activation: {np.mean(trajectory):.4f}")
    print(f"Std activation: {np.std(trajectory):.4f}")
    print(f"Min activation: {np.min(trajectory):.4f}")
    print(f"Max activation: {np.max(trajectory):.4f}")
    
    # 状态范数演化
    norms = np.linalg.norm(trajectory, axis=1)
    print(f"\nState norm evolution:")
    print(f"  Initial norm: {norms[0]:.4f}")
    print(f"  Final norm: {norms[-1]:.4f}")
    print(f"  Mean norm: {np.mean(norms):.4f}")
    print(f"  Std norm: {np.std(norms):.4f}")
    
    # 活跃神经元统计
    active_neurons = np.sum(np.abs(trajectory) > 0.1, axis=1)
    print(f"\nActive neurons (|x| > 0.1):")
    print(f"  Mean: {np.mean(active_neurons):.1f}")
    print(f"  Std: {np.std(active_neurons):.1f}")
    
    return norms

def perform_pca(trajectory, n_components=3):
    """对轨迹进行PCA降维"""
    print(f"\n=== PCA Analysis ===")
    print(f"Performing PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components)
    trajectory_pca = pca.fit_transform(trajectory)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    return trajectory_pca, pca

def visualize_3d_trajectory(trajectory_pca, title="ESN Dynamics in PCA Space"):
    """3D可视化PCA降维后的轨迹"""
    fig = plt.figure(figsize=(15, 12))
    
    # 3D轨迹图
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 使用颜色表示时间演化
    n_points = len(trajectory_pca)
    colors = plt.cm.viridis(np.linspace(0, 1, n_points))
    
    # 绘制轨迹
    for i in range(n_points - 1):
        ax1.plot(trajectory_pca[i:i+2, 0], 
                trajectory_pca[i:i+2, 1], 
                trajectory_pca[i:i+2, 2], 
                color=colors[i], alpha=0.8, linewidth=1)
    
    # 标记起点和终点
    ax1.scatter(trajectory_pca[0, 0], trajectory_pca[0, 1], trajectory_pca[0, 2], 
               color='red', s=100, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax1.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 1], trajectory_pca[-1, 2], 
               color='green', s=100, marker='s', label='End', edgecolors='black', linewidth=2)
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title(f'{title}\n(Time: Blue → Yellow)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2D投影 - PC1 vs PC2
    ax2 = fig.add_subplot(222)
    scatter = ax2.scatter(trajectory_pca[:, 0], trajectory_pca[:, 1], 
                         c=np.arange(n_points), cmap='viridis', 
                         s=10, alpha=0.6)
    ax2.plot(trajectory_pca[:, 0], trajectory_pca[:, 1], 
            color='gray', alpha=0.3, linewidth=0.5)
    ax2.scatter(trajectory_pca[0, 0], trajectory_pca[0, 1], 
               color='red', s=100, marker='o', edgecolors='black', linewidth=2, zorder=5)
    ax2.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 1], 
               color='green', s=100, marker='s', edgecolors='black', linewidth=2, zorder=5)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('PC1 vs PC2 Projection')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Time Step')
    
    # 2D投影 - PC1 vs PC3
    ax3 = fig.add_subplot(223)
    scatter = ax3.scatter(trajectory_pca[:, 0], trajectory_pca[:, 2], 
                         c=np.arange(n_points), cmap='viridis', 
                         s=10, alpha=0.6)
    ax3.plot(trajectory_pca[:, 0], trajectory_pca[:, 2], 
            color='gray', alpha=0.3, linewidth=0.5)
    ax3.scatter(trajectory_pca[0, 0], trajectory_pca[0, 2], 
               color='red', s=100, marker='o', edgecolors='black', linewidth=2, zorder=5)
    ax3.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 2], 
               color='green', s=100, marker='s', edgecolors='black', linewidth=2, zorder=5)
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC3')
    ax3.set_title('PC1 vs PC3 Projection')
    ax3.grid(True, alpha=0.3)
    
    # 2D投影 - PC2 vs PC3
    ax4 = fig.add_subplot(224)
    scatter = ax4.scatter(trajectory_pca[:, 1], trajectory_pca[:, 2], 
                         c=np.arange(n_points), cmap='viridis', 
                         s=10, alpha=0.6)
    ax4.plot(trajectory_pca[:, 1], trajectory_pca[:, 2], 
            color='gray', alpha=0.3, linewidth=0.5)
    ax4.scatter(trajectory_pca[0, 1], trajectory_pca[0, 2], 
               color='red', s=100, marker='o', edgecolors='black', linewidth=2, zorder=5)
    ax4.scatter(trajectory_pca[-1, 1], trajectory_pca[-1, 2], 
               color='green', s=100, marker='s', edgecolors='black', linewidth=2, zorder=5)
    ax4.set_xlabel('PC2')
    ax4.set_ylabel('PC3')
    ax4.set_title('PC2 vs PC3 Projection')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_dynamics_analysis(trajectory, norms, trajectory_pca):
    """绘制动力学分析图表"""
    fig = plt.figure(figsize=(15, 10))
    
    # 状态范数演化
    ax1 = fig.add_subplot(231)
    ax1.plot(norms, linewidth=2, color='blue', alpha=0.8)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('State Norm')
    ax1.set_title('State Norm Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(norms), color='red', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(norms):.2f}')
    ax1.legend()
    
    # 神经元激活分布
    ax2 = fig.add_subplot(232)
    activations_flat = trajectory.flatten()
    ax2.hist(activations_flat, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Activation Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Activation Distribution')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # PCA成分的时间演化
    ax3 = fig.add_subplot(233)
    for i in range(3):
        ax3.plot(trajectory_pca[:, i], label=f'PC{i+1}', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('PCA Component Value')
    ax3.set_title('PCA Components Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 相空间图 (PC1 vs 其导数)
    ax4 = fig.add_subplot(234)
    pc1_derivative = np.diff(trajectory_pca[:, 0])
    ax4.plot(trajectory_pca[:-1, 0], pc1_derivative, linewidth=1, alpha=0.6)
    ax4.scatter(trajectory_pca[0, 0], pc1_derivative[0], color='red', s=100, marker='o', zorder=5)
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('dPC1/dt')
    ax4.set_title('Phase Space (PC1)')
    ax4.grid(True, alpha=0.3)
    
    # 活跃神经元数量
    ax5 = fig.add_subplot(235)
    active_neurons = np.sum(np.abs(trajectory) > 0.1, axis=1)
    ax5.plot(active_neurons, linewidth=2, color='purple', alpha=0.8)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Number of Active Neurons')
    ax5.set_title('Active Neurons (|activation| > 0.1)')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=np.mean(active_neurons), color='red', linestyle='--', alpha=0.5, 
                label=f'Mean: {np.mean(active_neurons):.0f}')
    ax5.legend()
    
    # 自相关函数
    ax6 = fig.add_subplot(236)
    from scipy import signal
    autocorr = signal.correlate(trajectory_pca[:, 0], trajectory_pca[:, 0], mode='same')
    autocorr = autocorr / np.max(autocorr)
    lags = np.arange(-len(autocorr)//2, len(autocorr)//2)
    ax6.plot(lags, autocorr, linewidth=2, color='orange', alpha=0.8)
    ax6.set_xlabel('Lag')
    ax6.set_ylabel('Autocorrelation')
    ax6.set_title('Autocorrelation of PC1')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim([-100, 100])
    
    plt.tight_layout()
    return fig

def compare_different_inputs(reservoir):
    """比较不同输入类型下的动力学行为"""
    print("\n" + "="*60)
    print("Comparing dynamics with different input types")
    print("="*60)
    
    input_types = ['random', 'zeros', 'constant']
    trajectories = {}
    trajectories_pca = {}
    
    fig = plt.figure(figsize=(18, 6))
    
    for idx, input_type in enumerate(input_types):
        print(f"\n--- Input type: {input_type} ---")
        
        # 重置reservoir状态
        reservoir.reset()
        
        # 运行动力学
        trajectory, _ = run_reservoir_dynamics(
            reservoir, 
            n_steps=10000, 
            input_dim=96, 
            input_type=input_type
        )
        
        # PCA分析
        trajectory_pca, pca = perform_pca(trajectory, n_components=3)
        
        trajectories[input_type] = trajectory
        trajectories_pca[input_type] = trajectory_pca
        
        # 绘制3D轨迹
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        
        n_points = len(trajectory_pca)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        
        for i in range(n_points - 1):
            ax.plot(trajectory_pca[i:i+2, 0], 
                   trajectory_pca[i:i+2, 1], 
                   trajectory_pca[i:i+2, 2], 
                   color=colors[i], alpha=0.8, linewidth=1)
        
        ax.scatter(trajectory_pca[0, 0], trajectory_pca[0, 1], trajectory_pca[0, 2], 
                  color='red', s=100, marker='o', edgecolors='black', linewidth=2)
        ax.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 1], trajectory_pca[-1, 2], 
                  color='green', s=100, marker='s', edgecolors='black', linewidth=2)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'Input: {input_type}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('ESN Dynamics with Different Input Types', fontsize=16)
    plt.tight_layout()
    
    return fig, trajectories, trajectories_pca

def main():
    """主函数"""
    print("="*60)
    print("ESN Dynamics Trajectory Visualization")
    print("="*60)
    
    # 创建与CIFAR-10实验相同配置的Reservoir
    reservoir = create_reservoir_only(
        reservoir_size=1000,
        spectral_radius=1.9,
        leak_rate=0.1
    )
    
    # 运行动力学（使用随机输入）
    print("\n" + "="*60)
    print("Running main experiment with random input")
    print("="*60)
    
    trajectory, inputs = run_reservoir_dynamics(
        reservoir,
        n_steps=10000,
        input_dim=96,
        input_type='random'
    )
    
    # 分析轨迹
    norms = analyze_trajectory(trajectory)
    
    # PCA降维
    trajectory_pca, pca = perform_pca(trajectory, n_components=10)
    
    # 打印前10个主成分的方差贡献
    print(f"\nVariance explained by first 10 components:")
    for i in range(10):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.4f}")
    print(f"  Cumulative: {np.sum(pca.explained_variance_ratio_[:10]):.4f}")
    
    # 只使用前3个主成分进行可视化
    trajectory_pca_3d = trajectory_pca[:, :3]
    
    # 3D可视化
    fig1 = visualize_3d_trajectory(trajectory_pca_3d, 
                                   title="ESN Dynamics (Random Input)")
    plt.savefig('esn_3d_trajectory.png', dpi=300, bbox_inches='tight')
    
    # 动力学分析图表
    fig2 = plot_dynamics_analysis(trajectory, norms, trajectory_pca_3d)
    plt.savefig('esn_dynamics_analysis.png', dpi=300, bbox_inches='tight')
    
    # 比较不同输入类型
    fig3, trajectories, trajectories_pca = compare_different_inputs(reservoir)
    plt.savefig('esn_input_comparison.png', dpi=300, bbox_inches='tight')
    
    # 显示所有图表
    plt.show()
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print("Results saved to ESN_experiments/")
    print("="*60)

if __name__ == "__main__":
    main()
