#!/usr/bin/env python
# coding: utf-8

"""
ESNPyTorch Autonomous Dynamics Visualization Demo
使用PyTorch实现的ESN观察自主动力学轨迹
比较零初始状态和随机初始状态下的演化
采用GPU模式展示PyTorch ESN的性能
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
plt.rcParams['figure.figsize'] = (15, 8)
plt.rcParams['font.size'] = 12

# 设置PyTorch设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ESNPyTorch(nn.Module):
    """PyTorch实现的ESN，专为演化demo设计"""
    
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
        """初始化权重矩阵，完全按照ReservoirPy源代码的方式"""
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"Initializing ESNPyTorch weights (seed={seed})...")
        print(f"  Using ReservoirPy-compatible initialization:")
        print(f"    W: normal distribution (loc=0, scale=1)")
        print(f"    Win: bernoulli distribution (p=0.5, values=±1)")
        print(f"    bias: bernoulli distribution (p=0.5, values=±1)")
        
        # 初始化递归权重矩阵 W - 使用normal分布（与ReservoirPy一致）
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
        
        # 初始化输入权重矩阵 Win - 使用bernoulli分布（与ReservoirPy一致）
        # ReservoirPy默认: Win = bernoulli (±1, p=0.5)
        Win_np = self._bernoulli_matrix(self.reservoir_size, self.input_dim, p=0.5)
        
        # 应用输入连接性（稀疏化）
        mask_in = np.random.rand(self.reservoir_size, self.input_dim) < self.input_connectivity
        Win_np = Win_np * mask_in
        
        # 应用input_scaling（ReservoirPy默认为1.0，但为了完全兼容还是要应用）
        input_scaling = 1.0  # ReservoirPy默认值
        Win_np = Win_np * input_scaling
        
        self.Win = torch.tensor(Win_np, dtype=self.dtype, device=device)
        
        # 初始化偏置 - 使用bernoulli分布（与ReservoirPy一致）
        # ReservoirPy默认: bias = bernoulli (±1, p=0.5)
        # 关键修正：bias需要应用input_connectivity，这是ReservoirPy的行为！
        bias_full = self._bernoulli_matrix(self.reservoir_size, 1, p=0.5)
        
        # 应用input_connectivity到bias（这是ReservoirPy的默认行为）
        mask_bias = np.random.rand(self.reservoir_size, 1) < self.input_connectivity
        bias_np = bias_full * mask_bias
        
        # 应用bias_scaling（ReservoirPy默认为1.0）
        bias_scaling = 1.0  # ReservoirPy默认值
        bias_np = bias_np * bias_scaling
        
        self.bias = torch.tensor(bias_np, dtype=self.dtype, device=device)
        
        print(f"  W shape: {self.W.shape}, spectral radius: {self.spectral_radius}")
        print(f"  Win shape: {self.Win.shape}, input connectivity: {self.input_connectivity}")
        print(f"  bias shape: {self.bias.shape}")
        print(f"  Computation mode: {self.computation_mode}")
        
    def _bernoulli_matrix(self, rows, cols, p=0.5, value=1.0):
        """
        生成伯努利分布矩阵，完全按照ReservoirPy的custom_bernoulli实现
        
        Parameters:
        -----------
        rows, cols : int
            矩阵形状
        p : float, default 0.5
            获得+value的概率，获得-value的概率为(1-p)
        value : float, default 1.0
            成功值，失败值为-value
        
        Returns:
        --------
        numpy.ndarray
            伯努利分布矩阵，值为+value或-value
        """
        # 这里完全按照ReservoirPy源码中的_bernoulli_discrete_rvs实现
        size = rows * cols
        choices = np.random.choice([value, -value], size=size, p=[p, 1-p])
        return choices.reshape(rows, cols)
        
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
        """设置计算模式"""
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

def create_esn_pytorch(reservoir_size=1000, spectral_radius=1.9, leak_rate=0.1, 
                       input_dim=96, seed=None):
    """创建ESNPyTorch实例"""
    print(f"Creating ESNPyTorch with {reservoir_size} neurons...")
    print(f"  Spectral radius: {spectral_radius}")
    print(f"  Leak rate: {leak_rate}")
    print(f"  Device: {device}")
    
    if seed is None:
        seed = np.random.randint(10000)
    
    esn = ESNPyTorch(
        input_dim=input_dim,
        reservoir_size=reservoir_size,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        input_connectivity=0.1,
        rc_connectivity=0.1,
        activation='tanh',
        dtype=torch.float64,  # 使用双精度
        seed=seed,
        computation_mode='gpu'  # 使用GPU模式展示性能
    )
    
    return esn

def run_autonomous_dynamics(esn, n_steps=10000, input_dim=96, initial_state_type='zero'):
    """
    运行ESNPyTorch的自主动力学（零输入）
    
    Parameters:
    -----------
    esn : ESNPyTorch实例
    n_steps : 运行步数
    input_dim : 输入维度
    initial_state_type : 初始状态类型 ('zero' 或 'random')
    """
    print(f"\nRunning autonomous dynamics for {n_steps} steps...")
    print(f"  Initial state type: {initial_state_type}")
    print(f"  Input: zeros (autonomous dynamics)")
    print(f"  Computation mode: {esn.computation_mode}")
    
    # 设置初始状态
    if initial_state_type == 'zero':
        # 零初始状态
        initial_state = np.zeros(esn.reservoir_size)
        print(f"  Initial state: all zeros")
    elif initial_state_type == 'random':
        # 随机初始状态（归一化）
        initial_state = np.random.randn(esn.reservoir_size)
        initial_state = initial_state / np.linalg.norm(initial_state)
        print(f"  Initial state: random normalized vector")
    else:
        raise ValueError(f"Unknown initial state type: {initial_state_type}")
    
    esn.reset_state(initial_state)
    print(f"  Initial state norm: {np.linalg.norm(initial_state):.4f}")
    
    # 生成零输入序列
    inputs = np.zeros((n_steps, input_dim))
    
    # 运行动力学并记录轨迹
    trajectory = []
    
    for i in range(n_steps):
        # 输入零向量
        zero_input = torch.zeros(input_dim, dtype=torch.float64, device=device)
        
        # ESN前向传播
        state = esn(zero_input)
        
        # 记录状态（转换为CPU numpy）
        trajectory.append(state.cpu().numpy().copy())
        
        # 每2000步显示进度
        if (i + 1) % 2000 == 0:
            current_norm = np.linalg.norm(trajectory[-1])
            print(f"  Step {i+1}: state norm = {current_norm:.6f}")
    
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
    
    plt.suptitle('ESNPyTorch Autonomous Dynamics: Zero vs Random Initial States\n(GPU Mode)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    """主函数"""
    print("="*70)
    print("ESNPyTorch Autonomous Dynamics Visualization Demo")
    print("Comparing Zero vs Random Initial States (GPU Mode)")
    print("="*70)
    
    # 创建与CIFAR-10实验相同配置的ESN
    reservoir_size = 1000
    spectral_radius = 1.9  # 进入混沌区域
    leak_rate = 0.1
    input_dim = 96
    
    # 设置真随机种子（参照对比程序的方式）
    np.random.seed()
    base_seed = np.random.randint(0, 10000)
    print(f"Using random seed: {base_seed}")
    
    # 存储两种情况的轨迹
    trajectories = []
    trajectories_pca = []
    titles = []
    effective_dims = []
    
    # 情况1：零初始状态
    print("\n" + "="*70)
    print("Case 1: Zero Initial State")
    print("="*70)
    
    esn1 = create_esn_pytorch(reservoir_size, spectral_radius, leak_rate, input_dim, seed=base_seed)
    trajectory1 = run_autonomous_dynamics(
        esn1,
        n_steps=10000,
        input_dim=input_dim,
        initial_state_type='zero'
    )
    trajectory1_pca, pca1, eff_dim1 = perform_pca(trajectory1, n_components=10)
    
    trajectories.append(trajectory1)
    trajectories_pca.append(trajectory1_pca[:, :3])  # 只取前3个主成分
    titles.append('Zero Initial State\n(PyTorch GPU)')
    effective_dims.append(eff_dim1)
    
    # 情况2：随机初始状态
    print("\n" + "="*70)
    print("Case 2: Random Initial State")
    print("="*70)
    
    esn2 = create_esn_pytorch(reservoir_size, spectral_radius, leak_rate, input_dim, seed=base_seed)
    trajectory2 = run_autonomous_dynamics(
        esn2,
        n_steps=10000,
        input_dim=input_dim,
        initial_state_type='random'
    )
    trajectory2_pca, pca2, eff_dim2 = perform_pca(trajectory2, n_components=10)
    
    trajectories.append(trajectory2)
    trajectories_pca.append(trajectory2_pca[:, :3])  # 只取前3个主成分
    titles.append('Random Initial State\n(PyTorch GPU)')
    effective_dims.append(eff_dim2)
    
    # 比较两种情况
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    
    print("\nZero Initial State:")
    print(f"  Final norm: {np.linalg.norm(trajectories[0][-1]):.6f}")
    print(f"  Variance explained by PC1-3: {np.sum(pca1.explained_variance_ratio_[:3]):.4f}")
    print(f"  Effective dimensionality: {eff_dim1:.2f}")
    print(f"  Trajectory complexity: {'Complex chaotic dynamics' if eff_dim1 > 2 else 'Simple dynamics'}")
    
    print("\nRandom Initial State:")
    print(f"  Final norm: {np.linalg.norm(trajectories[1][-1]):.6f}")
    print(f"  Variance explained by PC1-3: {np.sum(pca2.explained_variance_ratio_[:3]):.4f}")
    print(f"  Effective dimensionality: {eff_dim2:.2f}")
    print(f"  Trajectory complexity: {'Complex chaotic dynamics' if eff_dim2 > 2 else 'Simple dynamics'}")
    
    print(f"\nDevice used: {device}")
    print(f"ESN Configuration:")
    print(f"  Reservoir size: {reservoir_size}")
    print(f"  Spectral radius: {spectral_radius} (chaotic regime)")
    print(f"  Leak rate: {leak_rate}")
    print(f"  Computation mode: GPU (PyTorch native)")
    
    # 3D可视化，包含有效维度信息
    fig = visualize_3d_trajectories(trajectories_pca, titles, effective_dims)
    plt.savefig('esn_pytorch_autonomous_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("ESNPyTorch Demo completed!")
    print("Results saved to: esn_pytorch_autonomous_dynamics.png")
    print("="*70)

if __name__ == "__main__":
    main()
