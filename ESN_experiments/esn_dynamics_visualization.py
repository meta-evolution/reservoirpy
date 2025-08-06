#!/usr/bin/env python
# coding: utf-8

"""
ESN Autonomous Dynamics Visualization
观察ESN的自主动力学轨迹
比较零初始状态和随机初始状态下的演化
"""

from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12

# 导入ReservoirPy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reservoirpy.nodes import Reservoir
from reservoirpy import set_seed, verbosity

# 设置随机种子和静默模式
set_seed(42)
verbosity(0)

def create_reservoir(reservoir_size=1000, spectral_radius=0.9, leak_rate=0.1):
    """创建Reservoir"""
    print(f"Creating reservoir with {reservoir_size} neurons...")
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
        seed=np.random.randint(10000)
    )
    
    return reservoir

def run_autonomous_dynamics(reservoir, n_steps=10000, input_dim=96, initial_state_type='zero'):
    """
    运行Reservoir的自主动力学（零输入）
    
    Parameters:
    -----------
    reservoir : Reservoir节点
    n_steps : 运行步数
    input_dim : 输入维度
    initial_state_type : 初始状态类型 ('zero' 或 'random')
    """
    print(f"\nRunning autonomous dynamics for {n_steps} steps...")
    print(f"  Initial state type: {initial_state_type}")
    print(f"  Input: zeros (autonomous dynamics)")
    
    # 生成零输入序列
    inputs = np.zeros((n_steps, input_dim))
    
    # 初始化reservoir（第一个输入用于初始化权重矩阵）
    _ = reservoir.run(inputs[:1])
    
    # 设置初始状态
    if initial_state_type == 'zero':
        # 零初始状态
        initial_state = np.zeros(reservoir.output_dim)
        print(f"  Initial state: all zeros")
    elif initial_state_type == 'random':
        # 随机初始状态（归一化）
        initial_state = np.random.randn(reservoir.output_dim)
        initial_state = initial_state / np.linalg.norm(initial_state)
        print(f"  Initial state: random normalized vector")
    else:
        raise ValueError(f"Unknown initial state type: {initial_state_type}")
    
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
    
    # 分析轨迹
    norms = np.linalg.norm(trajectory, axis=1)
    print(f"\nTrajectory statistics:")
    print(f"  Mean norm: {np.mean(norms):.4f}")
    print(f"  Std norm: {np.std(norms):.4f}")
    print(f"  Min norm: {np.min(norms):.4f}")
    print(f"  Max norm: {np.max(norms):.4f}")
    
    return trajectory

def calculate_effective_dimension(trajectory):
    """
    计算轨迹的有效维度（Participation Ratio）
    
    有效维度定义为：
    D_eff = (Σλ_i)^2 / Σλ_i^2
    
    其中λ_i是协方差矩阵的特征值（或PCA的方差）
    """
    # 计算协方差矩阵的特征值
    cov_matrix = np.cov(trajectory.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # 过滤掉数值噪声
    
    # 计算participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_squared = np.sum(eigenvalues**2)
    
    if sum_lambda_squared > 0:
        effective_dim = (sum_lambda**2) / sum_lambda_squared
    else:
        effective_dim = 1.0
    
    return effective_dim

def perform_pca(trajectory, n_components=10):
    """对轨迹进行PCA降维并计算有效维度"""
    print(f"\nPerforming PCA with {n_components} components...")
    
    pca = PCA(n_components=n_components)
    trajectory_pca = pca.fit_transform(trajectory)
    
    print(f"Explained variance ratio (first 3): {pca.explained_variance_ratio_[:3]}")
    print(f"Total variance explained by first 3 components: {np.sum(pca.explained_variance_ratio_[:3]):.4f}")
    print(f"Total variance explained by all {n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 计算有效维度
    effective_dim = calculate_effective_dimension(trajectory)
    print(f"Effective dimensionality (Participation Ratio): {effective_dim:.2f}")
    
    return trajectory_pca, pca, effective_dim

def visualize_3d_trajectories(trajectories_pca, titles, effective_dims):
    """3D可视化两种初始条件下的轨迹，并显示有效维度"""
    fig = plt.figure(figsize=(16, 8))
    
    for idx, (trajectory_pca, title, eff_dim) in enumerate(zip(trajectories_pca, titles, effective_dims)):
        ax = fig.add_subplot(1, 2, idx+1, projection='3d')
        
        # 使用颜色表示时间演化
        n_points = len(trajectory_pca)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))
        
        # 绘制轨迹
        for i in range(n_points - 1):
            ax.plot(trajectory_pca[i:i+2, 0], 
                   trajectory_pca[i:i+2, 1], 
                   trajectory_pca[i:i+2, 2], 
                   color=colors[i], alpha=0.8, linewidth=1.5)
        
        # 标记起点和终点
        ax.scatter(trajectory_pca[0, 0], trajectory_pca[0, 1], trajectory_pca[0, 2], 
                  color='red', s=150, marker='o', label='Start', 
                  edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(trajectory_pca[-1, 0], trajectory_pca[-1, 1], trajectory_pca[-1, 2], 
                  color='green', s=150, marker='s', label='End', 
                  edgecolors='black', linewidth=2, zorder=5)
        
        # 添加原点参考
        ax.scatter(0, 0, 0, color='black', s=100, marker='x', 
                  label='Origin', linewidth=2, zorder=4)
        
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.set_zlabel('PC3', fontsize=10)
        
        # 在标题中添加有效维度信息
        ax.set_title(f'{title}\nEffective Dimension: {eff_dim:.2f}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle('ESN Autonomous Dynamics: Zero vs Random Initial States', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("="*60)
    print("ESN Autonomous Dynamics Visualization")
    print("Comparing Zero vs Random Initial States")
    print("="*60)
    
    # 创建与CIFAR-10实验相同配置的Reservoir
    reservoir_size = 1000
    spectral_radius = 1.9
    leak_rate = 0.1
    
    # 存储两种情况的轨迹
    trajectories = []
    trajectories_pca = []
    titles = []
    
    # 情况1：零初始状态
    print("\n" + "="*60)
    print("Case 1: Zero Initial State")
    print("="*60)
    
    reservoir1 = create_reservoir(reservoir_size, spectral_radius, leak_rate)
    trajectory1 = run_autonomous_dynamics(
        reservoir1,
        n_steps=10000,
        input_dim=96,
        initial_state_type='zero'
    )
    trajectory1_pca, pca1, eff_dim1 = perform_pca(trajectory1, n_components=10)
    
    trajectories.append(trajectory1)
    trajectories_pca.append(trajectory1_pca[:, :3])  # 只取前3个主成分
    titles.append('Zero Initial State\n(Trivial Dynamics)')
    
    # 情况2：随机初始状态
    print("\n" + "="*60)
    print("Case 2: Random Initial State")
    print("="*60)
    
    reservoir2 = create_reservoir(reservoir_size, spectral_radius, leak_rate)
    trajectory2 = run_autonomous_dynamics(
        reservoir2,
        n_steps=10000,
        input_dim=96,
        initial_state_type='random'
    )
    trajectory2_pca, pca2, eff_dim2 = perform_pca(trajectory2, n_components=10)
    
    trajectories.append(trajectory2)
    trajectories_pca.append(trajectory2_pca[:, :3])  # 只取前3个主成分
    titles.append('Random Initial State\n(Autonomous Evolution)')
    
    # 比较两种情况
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    
    print("\nZero Initial State:")
    print(f"  Final norm: {np.linalg.norm(trajectories[0][-1]):.6f}")
    print(f"  Variance explained by PC1-3: {np.sum(pca1.explained_variance_ratio_[:3]):.4f}")
    print(f"  Effective dimensionality: {eff_dim1:.2f}")
    print(f"  Trajectory complexity: Trivial (stays at origin)")
    
    print("\nRandom Initial State:")
    print(f"  Final norm: {np.linalg.norm(trajectories[1][-1]):.6f}")
    print(f"  Variance explained by PC1-3: {np.sum(pca2.explained_variance_ratio_[:3]):.4f}")
    print(f"  Effective dimensionality: {eff_dim2:.2f}")
    print(f"  Trajectory complexity: Complex autonomous dynamics")
    
    # 3D可视化，包含有效维度信息
    effective_dims = [eff_dim1, eff_dim2]
    fig = visualize_3d_trajectories(trajectories_pca, titles, effective_dims)
    plt.savefig('esn_autonomous_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print("Results saved to: esn_autonomous_dynamics.png")
    print("="*60)

if __name__ == "__main__":
    main()
