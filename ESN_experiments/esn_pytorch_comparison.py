#!/usr/bin/env python
# coding: utf-8

"""
ESN PyTorch vs ReservoirPy Comparison
比较PyTorch实现的ESN与ReservoirPy的数值一致性
确保两个版本产生完全相同的轨迹
"""

import numpy as np
import torch
import torch.nn as nn
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

from reservoirpy.nodes import Reservoir
from reservoirpy import set_seed, verbosity

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)
verbosity(0)

# 设置PyTorch设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ESNPyTorch(nn.Module):
    """PyTorch实现的ESN，与ReservoirPy参数完全一致"""
    
    def __init__(self, input_dim, reservoir_size, spectral_radius=0.9, 
                 leak_rate=0.1, input_connectivity=0.1, rc_connectivity=0.1,
                 activation='tanh', dtype=torch.float64):
        super(ESNPyTorch, self).__init__()
        
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.dtype = dtype
        
        # 激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 初始化权重矩阵（暂时为None，将从ReservoirPy复制）
        self.W = None      # 递归权重
        self.Win = None    # 输入权重
        self.bias = None   # 偏置
        
        # 状态
        self.state = None
        
    def set_weights_from_reservoirpy(self, reservoir):
        """从ReservoirPy的Reservoir复制权重"""
        # 复制权重矩阵并转换为PyTorch张量
        # 处理稀疏矩阵的情况
        import scipy.sparse as sp
        
        if sp.issparse(reservoir.W):
            W_dense = reservoir.W.toarray()
        else:
            W_dense = reservoir.W
            
        if sp.issparse(reservoir.Win):
            Win_dense = reservoir.Win.toarray()
        else:
            Win_dense = reservoir.Win
            
        if sp.issparse(reservoir.bias):
            bias_dense = reservoir.bias.toarray()
        else:
            bias_dense = reservoir.bias
            
        self.W = torch.tensor(W_dense, dtype=self.dtype, device=device)
        self.Win = torch.tensor(Win_dense, dtype=self.dtype, device=device)
        self.bias = torch.tensor(bias_dense, dtype=self.dtype, device=device)
        
        # 确保bias是列向量
        if self.bias.dim() == 1:
            self.bias = self.bias.unsqueeze(1)
        
        print(f"Weights copied from ReservoirPy:")
        print(f"  W shape: {self.W.shape}, norm: {torch.norm(self.W).item():.6f}")
        print(f"  Win shape: {self.Win.shape}, norm: {torch.norm(self.Win).item():.6f}")
        print(f"  bias shape: {self.bias.shape}, norm: {torch.norm(self.bias).item():.6f}")
        
    def forward(self, input_data):
        """
        前向传播，实现与ReservoirPy相同的更新方程
        x[t+1] = (1-lr)·x[t] + lr·tanh(W·x[t] + Win·u[t+1] + bias)
        """
        if self.state is None:
            self.state = torch.zeros(self.reservoir_size, 1, dtype=self.dtype, device=device)
        
        # 确保输入是列向量
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(1)
        
        # 计算预激活值
        pre_activation = (
            self.W @ self.state +           # W·x[t]
            self.Win @ input_data +          # Win·u[t+1]
            self.bias                        # bias
        )
        
        # 应用泄漏积分更新
        self.state = (
            (1 - self.leak_rate) * self.state +                    # (1-lr)·x[t]
            self.leak_rate * self.activation(pre_activation)       # lr·tanh(...)
        )
        
        return self.state.squeeze()
    
    def reset_state(self, initial_state=None):
        """重置状态"""
        if initial_state is None:
            self.state = torch.zeros(self.reservoir_size, 1, dtype=self.dtype, device=device)
        else:
            if isinstance(initial_state, np.ndarray):
                initial_state = torch.tensor(initial_state, dtype=self.dtype, device=device)
            if initial_state.dim() == 1:
                initial_state = initial_state.unsqueeze(1)
            self.state = initial_state

def create_reservoirpy_reservoir(reservoir_size=1000, spectral_radius=1.9, leak_rate=0.1, seed=None):
    """创建ReservoirPy的Reservoir"""
    if seed is not None:
        np.random.seed(seed)
    
    reservoir = Reservoir(
        units=reservoir_size,
        sr=spectral_radius,
        lr=leak_rate,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        activation='tanh',
        seed=seed if seed is not None else np.random.randint(10000)
    )
    
    return reservoir

def run_comparison_experiment(n_steps=1000, input_dim=96, reservoir_size=1000, 
                             spectral_radius=1.9, leak_rate=0.1, 
                             initial_state_type='random'):
    """运行对比实验，确保两个版本产生相同的轨迹"""
    
    print("="*60)
    print(f"Running comparison: {initial_state_type} initial state")
    print("="*60)
    
    # 固定种子以确保可重复性
    comparison_seed = 12345
    np.random.seed(comparison_seed)
    
    # 1. 创建ReservoirPy的Reservoir
    print("\n1. Creating ReservoirPy Reservoir...")
    reservoir_rpy = create_reservoirpy_reservoir(
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        seed=comparison_seed
    )
    
    # 初始化ReservoirPy（需要一个输入来初始化权重）
    dummy_input = np.zeros((1, input_dim))
    _ = reservoir_rpy.run(dummy_input)
    
    # 2. 创建PyTorch ESN并复制权重
    print("\n2. Creating PyTorch ESN and copying weights...")
    esn_pytorch = ESNPyTorch(
        input_dim=input_dim,
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        activation='tanh',
        dtype=torch.float64  # 使用双精度以匹配NumPy
    )
    
    # 从ReservoirPy复制权重
    esn_pytorch.set_weights_from_reservoirpy(reservoir_rpy)
    
    # 3. 设置相同的初始状态
    print("\n3. Setting initial states...")
    if initial_state_type == 'zero':
        initial_state = np.zeros(reservoir_size)
    elif initial_state_type == 'random':
        np.random.seed(comparison_seed + 1)  # 使用不同的种子生成初始状态
        initial_state = np.random.randn(reservoir_size)
        initial_state = initial_state / np.linalg.norm(initial_state)
    else:
        raise ValueError(f"Unknown initial state type: {initial_state_type}")
    
    print(f"  Initial state type: {initial_state_type}")
    print(f"  Initial state norm: {np.linalg.norm(initial_state):.6f}")
    
    # 设置ReservoirPy的初始状态
    reservoir_rpy._state = initial_state.copy()
    
    # 设置PyTorch的初始状态
    esn_pytorch.reset_state(initial_state.copy())
    
    # 4. 运行两个版本并记录轨迹
    print(f"\n4. Running dynamics for {n_steps} steps...")
    
    # 生成零输入序列
    inputs = np.zeros((n_steps, input_dim))
    
    trajectory_rpy = []
    trajectory_pytorch = []
    
    for i in range(n_steps):
        # ReservoirPy步进
        state_rpy = reservoir_rpy.run(inputs[i:i+1], reset=False)
        trajectory_rpy.append(state_rpy.flatten())
        
        # PyTorch步进
        input_torch = torch.tensor(inputs[i], dtype=torch.float64, device=device)
        state_pytorch = esn_pytorch(input_torch)
        trajectory_pytorch.append(state_pytorch.cpu().numpy())
        
        # 每100步检查一次差异
        if (i + 1) % 100 == 0:
            diff = np.linalg.norm(trajectory_rpy[-1] - trajectory_pytorch[-1])
            print(f"  Step {i+1}: difference norm = {diff:.2e}")
    
    trajectory_rpy = np.array(trajectory_rpy)
    trajectory_pytorch = np.array(trajectory_pytorch)
    
    # 5. 计算差异统计
    print("\n5. Computing difference statistics...")
    
    differences = trajectory_rpy - trajectory_pytorch
    diff_norms = np.linalg.norm(differences, axis=1)
    
    print(f"\nDifference statistics:")
    print(f"  Mean difference norm: {np.mean(diff_norms):.2e}")
    print(f"  Max difference norm: {np.max(diff_norms):.2e}")
    print(f"  Min difference norm: {np.min(diff_norms):.2e}")
    print(f"  Std difference norm: {np.std(diff_norms):.2e}")
    
    # 相对误差
    rpy_norms = np.linalg.norm(trajectory_rpy, axis=1)
    relative_errors = diff_norms / (rpy_norms + 1e-10)
    print(f"\nRelative error:")
    print(f"  Mean: {np.mean(relative_errors):.2e}")
    print(f"  Max: {np.max(relative_errors):.2e}")
    
    return trajectory_rpy, trajectory_pytorch, differences

def calculate_effective_dimension(trajectory):
    """计算轨迹的有效维度（Participation Ratio）"""
    cov_matrix = np.cov(trajectory.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_squared = np.sum(eigenvalues**2)
    
    if sum_lambda_squared > 0:
        effective_dim = (sum_lambda**2) / sum_lambda_squared
    else:
        effective_dim = 1.0
    
    return effective_dim

def visualize_comparison(trajectory_rpy, trajectory_pytorch, initial_state_type):
    """可视化两个版本的轨迹对比"""
    
    # 计算PCA
    pca_rpy = PCA(n_components=3)
    trajectory_rpy_pca = pca_rpy.fit_transform(trajectory_rpy)
    
    pca_pytorch = PCA(n_components=3)
    trajectory_pytorch_pca = pca_pytorch.fit_transform(trajectory_pytorch)
    
    # 计算有效维度
    eff_dim_rpy = calculate_effective_dimension(trajectory_rpy)
    eff_dim_pytorch = calculate_effective_dimension(trajectory_pytorch)
    
    print(f"\nEffective dimensions:")
    print(f"  ReservoirPy: {eff_dim_rpy:.2f}")
    print(f"  PyTorch: {eff_dim_pytorch:.2f}")
    
    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    
    # 3D轨迹对比
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(trajectory_rpy_pca[:, 0], trajectory_rpy_pca[:, 1], trajectory_rpy_pca[:, 2], 
             'b-', alpha=0.6, linewidth=1, label='ReservoirPy')
    ax1.set_title(f'ReservoirPy Trajectory\nEff. Dim: {eff_dim_rpy:.2f}')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.legend()
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.plot(trajectory_pytorch_pca[:, 0], trajectory_pytorch_pca[:, 1], trajectory_pytorch_pca[:, 2], 
             'r-', alpha=0.6, linewidth=1, label='PyTorch')
    ax2.set_title(f'PyTorch Trajectory\nEff. Dim: {eff_dim_pytorch:.2f}')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.legend()
    
    # 叠加对比
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    ax3.plot(trajectory_rpy_pca[:, 0], trajectory_rpy_pca[:, 1], trajectory_rpy_pca[:, 2], 
             'b-', alpha=0.5, linewidth=1, label='ReservoirPy')
    ax3.plot(trajectory_pytorch_pca[:, 0], trajectory_pytorch_pca[:, 1], trajectory_pytorch_pca[:, 2], 
             'r--', alpha=0.5, linewidth=1, label='PyTorch')
    ax3.set_title('Overlaid Trajectories')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.set_zlabel('PC3')
    ax3.legend()
    
    # 状态范数演化
    ax4 = fig.add_subplot(2, 3, 4)
    norms_rpy = np.linalg.norm(trajectory_rpy, axis=1)
    norms_pytorch = np.linalg.norm(trajectory_pytorch, axis=1)
    ax4.plot(norms_rpy, 'b-', alpha=0.7, label='ReservoirPy')
    ax4.plot(norms_pytorch, 'r--', alpha=0.7, label='PyTorch')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('State Norm')
    ax4.set_title('State Norm Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 差异演化
    ax5 = fig.add_subplot(2, 3, 5)
    differences = trajectory_rpy - trajectory_pytorch
    diff_norms = np.linalg.norm(differences, axis=1)
    ax5.semilogy(diff_norms, 'g-', linewidth=2)
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Difference Norm (log scale)')
    ax5.set_title('Numerical Difference Between Implementations')
    ax5.grid(True, alpha=0.3)
    
    # 相对误差
    ax6 = fig.add_subplot(2, 3, 6)
    relative_errors = diff_norms / (norms_rpy + 1e-10)
    ax6.semilogy(relative_errors, 'orange', linewidth=2)
    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Relative Error (log scale)')
    ax6.set_title('Relative Error')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'ReservoirPy vs PyTorch ESN Comparison\n{initial_state_type.capitalize()} Initial State', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def main():
    """主函数"""
    print("="*60)
    print("ESN PyTorch vs ReservoirPy Numerical Comparison")
    print("="*60)
    
    # 实验参数（与esn_dynamics_visualization.py相同）
    n_steps = 1000  # 减少步数以便快速测试
    input_dim = 96
    reservoir_size = 1000
    spectral_radius = 1.9
    leak_rate = 0.1
    
    # 测试两种初始状态
    for initial_state_type in ['zero', 'random']:
        print(f"\n{'='*60}")
        print(f"Testing with {initial_state_type} initial state")
        print('='*60)
        
        # 运行对比实验
        trajectory_rpy, trajectory_pytorch, differences = run_comparison_experiment(
            n_steps=n_steps,
            input_dim=input_dim,
            reservoir_size=reservoir_size,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            initial_state_type=initial_state_type
        )
        
        # 可视化对比
        fig = visualize_comparison(trajectory_rpy, trajectory_pytorch, initial_state_type)
        plt.savefig(f'esn_pytorch_comparison_{initial_state_type}.png', dpi=150, bbox_inches='tight')
        
        # 验证数值一致性
        max_diff = np.max(np.abs(differences))
        mean_diff = np.mean(np.abs(differences))
        
        print(f"\n{'='*60}")
        print(f"Numerical Consistency Check ({initial_state_type} initial state):")
        print(f"  Maximum absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")
        
        if max_diff < 1e-10:
            print("  ✓ EXCELLENT: Implementations are numerically identical (diff < 1e-10)")
        elif max_diff < 1e-6:
            print("  ✓ GOOD: Implementations are very close (diff < 1e-6)")
        elif max_diff < 1e-3:
            print("  ⚠ WARNING: Small differences detected (diff < 1e-3)")
        else:
            print("  ✗ ERROR: Significant differences detected!")
    
    plt.show()
    
    print("\n" + "="*60)
    print("Comparison completed!")
    print("Results saved to: esn_pytorch_comparison_*.png")
    print("="*60)

if __name__ == "__main__":
    main()
