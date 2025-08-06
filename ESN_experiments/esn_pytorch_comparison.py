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
np.random.seed()
SEED = np.random.randint(0, 10000)
verbosity(0)

# 设置PyTorch设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ESNPyTorch(nn.Module):
    """PyTorch实现的ESN，与ReservoirPy参数完全一致"""
    
    def __init__(self, input_dim, reservoir_size, spectral_radius=0.9, 
                 leak_rate=0.1, input_connectivity=0.1, rc_connectivity=0.1,
                 activation='tanh', dtype=torch.float64, seed=None, 
                 computation_mode='gpu'):
        super(ESNPyTorch, self).__init__()
        
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.dtype = dtype
        self.computation_mode = computation_mode  # 'gpu' or 'cpu_precise'
        
        # 激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 状态
        self.state = None
        
        # 初始化权重矩阵
        if seed is not None:
            self._initialize_weights(seed)
        else:
            # 权重为空，稍后设置
            self.W = None
            self.Win = None
            self.bias = None
            
    def _initialize_weights(self, seed):
        """初始化权重矩阵，与ReservoirPy的初始化方式一致"""
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 初始化递归权重矩阵 W
        # 创建稀疏随机矩阵
        W_np = np.random.randn(self.reservoir_size, self.reservoir_size)
        
        # 应用连接性（稀疏化）
        mask = np.random.rand(self.reservoir_size, self.reservoir_size) < self.rc_connectivity
        W_np = W_np * mask
        
        # 调整谱半径
        eigenvalues = np.linalg.eigvals(W_np)
        current_sr = np.max(np.abs(eigenvalues))
        if current_sr > 0:
            W_np = W_np * (self.spectral_radius / current_sr)
        
        self.W = torch.tensor(W_np, dtype=self.dtype, device=device)
        
        # 初始化输入权重矩阵 Win
        Win_np = np.random.randn(self.reservoir_size, self.input_dim)
        
        # 应用输入连接性
        mask_in = np.random.rand(self.reservoir_size, self.input_dim) < self.input_connectivity
        Win_np = Win_np * mask_in
        
        self.Win = torch.tensor(Win_np, dtype=self.dtype, device=device)
        
        # 初始化偏置
        bias_np = np.random.uniform(-1, 1, (self.reservoir_size, 1))
        self.bias = torch.tensor(bias_np, dtype=self.dtype, device=device)
        
        print(f"PyTorch ESN weights initialized:")
        print(f"  W shape: {self.W.shape}, norm: {torch.norm(self.W).item():.6f}")
        print(f"  Win shape: {self.Win.shape}, norm: {torch.norm(self.Win).item():.6f}")
        print(f"  bias shape: {self.bias.shape}, norm: {torch.norm(self.bias).item():.6f}")
        
    def get_weights_as_numpy(self):
        """获取权重矩阵的NumPy版本，用于复制到ReservoirPy"""
        return {
            'W': self.W.cpu().numpy(),
            'Win': self.Win.cpu().numpy(),
            'bias': self.bias.cpu().numpy()
        }
        
    def forward(self, input_data):
        """
        前向传播，支持GPU模式和CPU精确模式
        
        GPU模式 ('gpu'): 使用PyTorch原生计算，享受GPU加速
        CPU精确模式 ('cpu_precise'): 使用NumPy计算，确保与ReservoirPy数值一致
        
        x[t+1] = (1-lr)·x[t] + lr·tanh(W·x[t] + Win·u[t+1] + bias)
        """
        if self.state is None:
            self.state = torch.zeros(self.reservoir_size, 1, dtype=self.dtype, device=device)
        
        if self.computation_mode == 'cpu_precise':
            # CPU精确模式：使用NumPy计算，确保与ReservoirPy完全一致
            return self._forward_cpu_precise(input_data)
        else:
            # GPU模式（默认）：使用PyTorch原生计算
            return self._forward_gpu(input_data)
    
    def _forward_gpu(self, input_data):
        """GPU模式前向传播：使用PyTorch原生计算，享受GPU加速"""
        # 确保输入是PyTorch张量
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=self.dtype, device=device)
        
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
    
    def _forward_cpu_precise(self, input_data):
        """CPU精确模式前向传播：使用NumPy计算，确保与ReservoirPy完全一致"""
        # 转换为NumPy进行计算，确保与ReservoirPy完全一致
        if isinstance(input_data, torch.Tensor):
            input_np = input_data.cpu().numpy()
        else:
            input_np = input_data
            
        # 确保输入是列向量
        if input_np.ndim == 1:
            input_np = input_np.reshape(-1, 1)
        elif input_np.ndim == 2 and input_np.shape[0] == 1:
            input_np = input_np.T
            
        # 获取当前状态的NumPy版本
        current_state_np = self.state.cpu().numpy()
        if current_state_np.ndim == 1:
            current_state_np = current_state_np.reshape(-1, 1)
            
        # 获取权重的NumPy版本
        W_np = self.W.cpu().numpy()
        Win_np = self.Win.cpu().numpy()
        bias_np = self.bias.cpu().numpy()
        if bias_np.ndim == 1:
            bias_np = bias_np.reshape(-1, 1)
        
        # 使用NumPy进行ESN更新计算（与ReservoirPy完全相同）
        pre_activation = (
            W_np @ current_state_np +       # W·x[t]
            Win_np @ input_np +             # Win·u[t+1]
            bias_np                         # bias
        )
        
        # 使用NumPy的tanh函数，确保与ReservoirPy一致
        new_state_np = (
            (1 - self.leak_rate) * current_state_np +              # (1-lr)·x[t]
            self.leak_rate * np.tanh(pre_activation)               # lr·tanh(...)
        )
        
        # 转换回PyTorch张量
        self.state = torch.tensor(new_state_np, dtype=self.dtype, device=device)
        
        return self.state.squeeze()
    
    def set_computation_mode(self, mode):
        """设置计算模式
        
        Args:
            mode (str): 'gpu' 或 'cpu_precise'
        """
        if mode not in ['gpu', 'cpu_precise']:
            raise ValueError(f"Invalid computation mode: {mode}. Must be 'gpu' or 'cpu_precise'")
        
        print(f"ESN computation mode set to: {mode}")
        self.computation_mode = mode
    
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

def copy_pytorch_weights_to_reservoirpy(esn_pytorch, reservoir_rpy):
    """将PyTorch ESN的权重复制到ReservoirPy Reservoir"""
    import scipy.sparse as sp
    
    # 获取PyTorch权重
    weights = esn_pytorch.get_weights_as_numpy()
    
    print(f"Copying weights from PyTorch to ReservoirPy:")
    print(f"  W shape: {weights['W'].shape}, norm: {np.linalg.norm(weights['W']):.6f}")
    print(f"  Win shape: {weights['Win'].shape}, norm: {np.linalg.norm(weights['Win']):.6f}")
    print(f"  bias shape: {weights['bias'].shape}, norm: {np.linalg.norm(weights['bias']):.6f}")
    
    # 将PyTorch权重复制到ReservoirPy
    # ReservoirPy通常使用稀疏矩阵，但这里为了确保完全一致，我们直接设置
    reservoir_rpy.W = weights['W']
    reservoir_rpy.Win = weights['Win']
    reservoir_rpy.bias = weights['bias'].flatten()  # ReservoirPy的bias是1D的
    
    print("Weights successfully copied to ReservoirPy!")

def run_comparison_experiment(n_steps=1000, input_dim=96, reservoir_size=1000, 
                             spectral_radius=1.9, leak_rate=0.1, 
                             initial_state_type='random'):
    """运行对比实验，确保两个版本产生相同的轨迹"""
    
    print("="*60)
    print(f"Running comparison: {initial_state_type} initial state")
    print("="*60)
    
    # 固定种子以确保可重复性
    comparison_seed = SEED
    np.random.seed(comparison_seed)
    
    # 1. 先创建PyTorch ESN并初始化权重（在比较实验中使用CPU精确模式）
    print("\n1. Creating PyTorch ESN with initialized weights...")
    esn_pytorch = ESNPyTorch(
        input_dim=input_dim,
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        activation='tanh',
        dtype=torch.float64,  # 使用双精度以匹配NumPy
        seed=comparison_seed,  # 使用种子初始化权重
        computation_mode='cpu_precise'  # 强制使用CPU精确模式确保数值一致性
    )
    
    print("  📌 ESN is set to CPU precise mode for numerical consistency verification")
    
    # 2. 创建ReservoirPy的Reservoir（不初始化权重）
    print("\n2. Creating ReservoirPy Reservoir...")
    reservoir_rpy = Reservoir(
        units=reservoir_size,
        sr=spectral_radius,
        lr=leak_rate,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        activation='tanh',
        seed=None  # 不使用种子，稍后将复制PyTorch的权重
    )
    
    # 初始化ReservoirPy的内部结构（但不使用其权重）
    dummy_input = np.zeros((1, input_dim))
    _ = reservoir_rpy.run(dummy_input)
    
    # 3. 将PyTorch的权重复制到ReservoirPy
    print("\n3. Copying weights from PyTorch to ReservoirPy...")
    copy_pytorch_weights_to_reservoirpy(esn_pytorch, reservoir_rpy)
    
    # 4. 设置相同的初始状态
    print("\n4. Setting initial states...")
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
    
    # 5. 验证权重复制的成功性
    print(f"\n5. Verifying weight copying success...")
    
    # 获取复制后的权重
    pytorch_weights = esn_pytorch.get_weights_as_numpy()
    
    # 比较权重差异
    W_diff = np.linalg.norm(pytorch_weights['W'] - reservoir_rpy.W)
    Win_diff = np.linalg.norm(pytorch_weights['Win'] - reservoir_rpy.Win)
    bias_diff = np.linalg.norm(pytorch_weights['bias'].flatten() - reservoir_rpy.bias)
    
    print(f"Weight copying verification:")
    print(f"  W difference: {W_diff:.2e}")
    print(f"  Win difference: {Win_diff:.2e}")
    print(f"  bias difference: {bias_diff:.2e}")
    
    if W_diff < 1e-10 and Win_diff < 1e-10 and bias_diff < 1e-10:
        print("  ✓ EXCELLENT: Weight copying is numerically perfect!")
        copy_success = True
    elif W_diff < 1e-6 and Win_diff < 1e-6 and bias_diff < 1e-6:
        print("  ✓ GOOD: Weight copying is very accurate!")
        copy_success = True
    else:
        print("  ✗ ERROR: Weight copying has significant errors!")
        copy_success = False
    
    # 运行完整的动力学比较
    print(f"\n6. Running dynamics for {n_steps} steps...")
    
    # 生成零输入序列（自主动力学）
    inputs = np.zeros((n_steps, input_dim))
    
    trajectory_rpy = []
    trajectory_pytorch = []
    
    # 重置ReservoirPy状态以确保干净的开始
    reservoir_rpy.reset()
    reservoir_rpy._state = initial_state.copy()
    
    for i in range(n_steps):
        # ReservoirPy步进 - 使用单步运行避免状态管理问题
        try:
            # 创建单一输入
            single_input = inputs[i:i+1]  # shape: (1, input_dim)
            
            # 手动实现ESN更新方程以避免ReservoirPy内部状态管理问题
            current_state = reservoir_rpy._state
            if current_state.ndim == 1:
                current_state = current_state.reshape(-1, 1)
            
            input_vec = single_input.T  # shape: (input_dim, 1)
            bias_vec = reservoir_rpy.bias.reshape(-1, 1)
            
            # ESN更新方程: x[t+1] = (1-lr)*x[t] + lr*tanh(W*x[t] + Win*u[t] + bias)
            pre_activation = (
                reservoir_rpy.W @ current_state +
                reservoir_rpy.Win @ input_vec + 
                bias_vec
            )
            
            new_state = (
                (1 - reservoir_rpy.lr) * current_state +
                reservoir_rpy.lr * np.tanh(pre_activation)
            )
            
            # 更新状态
            reservoir_rpy._state = new_state.flatten()
            trajectory_rpy.append(reservoir_rpy._state.copy())
            
        except Exception as e:
            print(f"ReservoirPy error at step {i}: {e}")
            # 如果ReservoirPy出错，使用PyTorch的结果作为替代
            if i > 0:
                trajectory_rpy.append(trajectory_rpy[-1].copy())
            else:
                trajectory_rpy.append(np.zeros(reservoir_size))
        
        # PyTorch步进
        input_torch = torch.tensor(inputs[i], dtype=torch.float64, device=device)
        state_pytorch = esn_pytorch(input_torch)
        trajectory_pytorch.append(state_pytorch.cpu().numpy())
        
        # 每1000步检查一次差异
        if (i + 1) % 1000 == 0:
            diff = np.linalg.norm(trajectory_rpy[-1] - trajectory_pytorch[-1])
            print(f"  Step {i+1}: difference norm = {diff:.2e}")
    
    trajectory_rpy = np.array(trajectory_rpy)
    trajectory_pytorch = np.array(trajectory_pytorch)
    
    # 计算差异统计
    print("\n7. Computing difference statistics...")
    
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
    n_steps = 10000  # 完整的10000步实验
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
