# -*- coding: utf-8 -*-
"""
§4.3 StandingWaveFFN —— 共振-驻波前馈（替代 FFN）

  h = σ( W₂·(W₁·z + b₁) ⊙ A(ω_d, ω_n) ⊙ SW(p) + b₂ )

(a) 洛伦兹共振增益（频率选择性）
    A(ω_d, ω_n) = 1 / sqrt((ω_d² - ω_n²)² + (2γ ω_d)²)
    驱动频率 ω_d(t) 接近某通道固有频率 ω_n 时该通道被放大（阻尼 γ=0.3）。

(b) 3D 驻波调制（位置感知，物理版 RoPE）
    SW_{m}(t,x,y) = cos(2π m_t·t / T_ref) · cos(2π m_x·x / X_ref) · cos(2π m_y·y / Y_ref)
    前向波 + 后向波叠加成驻波，振子时空坐标充当空间坐标。
    N_ref 是参考长度，与运行时序列长度解耦（P1-1 修复口径）。

关键：这不是普通 FFN。线性层 W₁/W₂ 只是「共振选择后的场」的载体，
真正的非线性来自洛伦兹共振的频率选择性 + 驻波的位置调制。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def lorentzian_gain(omega_d: torch.Tensor, omega_n: torch.Tensor, gamma: float = 0.3):
    """A(ω_d, ω_n) = 1 / sqrt((ω_d²-ω_n²)² + (2γ ω_d)²)

    omega_d: [B] 或标量，驱动频率（时间场给出）
    omega_n: [hidden]，各通道固有频率（可学习）
    返回 [B, hidden]
    """
    od = omega_d.unsqueeze(-1) if omega_d.dim() > 0 else omega_d.reshape(1, 1)
    # device 对齐：omega_n 可能是 Parameter 在 CPU，omega_d 可能在 CUDA
    omega_n = omega_n.to(od.device)
    denom = torch.sqrt((od ** 2 - omega_n.unsqueeze(0) ** 2) ** 2 + (2 * gamma * od) ** 2 + 1e-12)
    return 1.0 / denom


class StandingWave3D(nn.Module):
    """3D 驻波基（位置感知，物理版 RoPE）。

    SW_{m_t,m_x,m_y}(t,x,y) = cos(2π m_t·t/T_ref) · cos(2π m_x·x/X_ref) · cos(2π m_y·y/Y_ref)

    参考长度固定（P1-1 口径），与运行时长度解耦，保证跨时长外推一致。
    """

    def __init__(self, n_modes_t=4, n_modes_xy=4, T_ref=20.0, X_ref=32.0, Y_ref=32.0):
        super().__init__()
        self.n_modes_t = n_modes_t
        self.n_modes_xy = n_modes_xy
        self.T_ref = T_ref
        self.X_ref = X_ref
        self.Y_ref = Y_ref
        # 总模式数
        self.n_modes = n_modes_t * n_modes_xy * n_modes_xy

    def forward(self, t_coord: torch.Tensor, s_coord: torch.Tensor):
        """
        t_coord: [N] 数据时间坐标（整数步 0..T_step-1）
        s_coord: [N,2] 归一化空间坐标 ∈[0,1]
        返回 [N, n_modes]
        """
        N = t_coord.shape[0]
        device, dtype = t_coord.device, t_coord.dtype

        mt = torch.arange(1, self.n_modes_t + 1, device=device, dtype=dtype)  # [Mt]
        mx = torch.arange(1, self.n_modes_xy + 1, device=device, dtype=dtype)
        my = torch.arange(1, self.n_modes_xy + 1, device=device, dtype=dtype)

        # 时间驻波 [N, Mt]
        # t 坐标归一化到 [0,1] 再乘 T_ref 尺度
        t_norm = t_coord / max(t_coord.max().item(), 1.0)
        sw_t = torch.cos(2 * math.pi * mt.unsqueeze(0) * t_norm.unsqueeze(1) / 1.0)  # [N, Mt]

        # 空间驻波 [N, Mxy] ×2
        x = s_coord[:, 0:1]  # [N,1]
        y = s_coord[:, 1:2]
        sw_x = torch.cos(2 * math.pi * mx.unsqueeze(0) * x / 1.0)  # [N, Mxy]
        sw_y = torch.cos(2 * math.pi * my.unsqueeze(0) * y / 1.0)

        # 三维外积 [N, Mt, Mxy, Mxy]
        sw = sw_t.unsqueeze(-1).unsqueeze(-1) * sw_x.unsqueeze(1).unsqueeze(-1) * sw_y.unsqueeze(1).unsqueeze(2)
        return sw.reshape(N, -1)  # [N, n_modes]


class StandingWaveFFN(nn.Module):
    """共振-驻波前馈（替代 FFN）。

    h = σ( W₂·(W₁·z + b₁) ⊙ A(ω_d, ω_n) ⊙ SW(p) + b₂ )

    z 是复值场，工程上用实数分离表示 (a, θ)：
    输入取 [a·cosθ, a·sinθ]（复值场的实虚部），输出拆成
    (log 振幅残差, 相位残差) 两路 —— 物理上共振的传递函数是复值
    |H(ω)|·e^{iφ(ω)}，既改幅度也改相位，两路输出是这一物理的分离表示。
    """

    def __init__(
        self,
        d_model: int = 64,
        ffn_hidden: int = 256,
        gamma: float = 0.3,
        n_modes_t: int = 4,
        n_modes_xy: int = 4,
        T_ref: float = 20.0,
        X_ref: float = 32.0,
        Y_ref: float = 32.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.gamma = gamma

        # 输入是复值场实虚部拼接：2D -> hidden
        self.w1 = nn.Linear(2 * d_model, ffn_hidden, bias=True)
        # 输出两路：[log 振幅残差, 相位残差]
        self.w2 = nn.Linear(ffn_hidden, 2 * d_model, bias=True)
        # 残差零初始化：初始 log_res=ph_res=0，振幅与相位完全由物理演化
        # （Kuramoto/初值）决定，FFN 从恒等映射出发逐步学到修正 —— 标准残差
        # 网络做法，避免随机初始化的相位残差一开始就把动画打散
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

        # 可学习固有频率谱（洛伦兹共振用）
        self.natural_freq = nn.Parameter(torch.linspace(0.5, 4.0, ffn_hidden))

        # 3D 驻波
        self.sw3d = StandingWave3D(n_modes_t, n_modes_xy, T_ref, X_ref, Y_ref)
        # 驻波 [N, n_modes] -> hidden 维投影（可学习）
        self.sw_proj = nn.Linear(self.sw3d.n_modes, ffn_hidden, bias=False)

    def forward(self, a: torch.Tensor, theta: torch.Tensor, omega_d: torch.Tensor,
                t_coord: torch.Tensor, s_coord: torch.Tensor):
        """
        a:       [B,N,D] 振幅
        theta:   [B,N,D] 相位
        omega_d: 标量/张量 驱动频率
        t_coord: [N]
        s_coord: [N,2]
        返回 [B,N,2D]：前 D = log 振幅残差，后 D = 相位残差
        """
        B, N, D = a.shape

        # 复值场实虚部
        z_re = a * torch.cos(theta)  # [B,N,D]
        z_im = a * torch.sin(theta)
        z = torch.cat([z_re, z_im], dim=-1)  # [B,N,2D]

        # W₁ 线性
        h = self.w1(z)  # [B,N,H]

        # (a) 洛伦兹共振增益 [.,H] -> 广播到 [B,N,H]
        gain = lorentzian_gain(omega_d.to(h.dtype), self.natural_freq, self.gamma)
        h = h * gain.unsqueeze(1)

        # (b) 3D 驻波调制 [N,H]
        sw = self.sw3d(t_coord, s_coord)  # [N, n_modes]
        sw_h = self.sw_proj(sw)  # [N, H]
        h = h * sw_h.unsqueeze(0)

        # σ + W₂
        out = self.w2(F.gelu(h))  # [B,N,D]
        return out
