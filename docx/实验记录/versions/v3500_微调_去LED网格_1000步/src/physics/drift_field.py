# -*- coding: utf-8 -*-
"""
§4.4 DriftField —— 漂移场

  V = V⁺ − λ·V⁻
  V⁺ = Σ_k softmax_τ(−‖z_i − z_k^tar‖²)·(z_k^tar − z_i)    （吸引：向目标靠拢）
  V⁻ = Σ_j softmax_τ(−‖z_i − z_j‖²)·(z_j − z_i)            （排斥：防坍缩）

- 多温度 τ 求和取多尺度
- λ = 0.5（排斥权重）
- 训练时目标是标签/条件嵌入；推理时用可学习原型作吸引中心
- 距离在实+虚拼接的 2D 空间上定义

显存工程（不改动任何物理公式）：
  全 pairwise 距离矩阵是 O(N·M) 的；训练时吸引中心 targets 也是 N 个振子
  （M = N 可达 10240+），单个 [B,N,N] fp32 就 400MB，连续物化 + 反向必爆
  4GB 显存。采用「分块两遍 + 梯度检查点」：
    Pass 1 分块求每行 i 的全局 logsumexp（softmax 归一化常数 logZ_i）
    Pass 2 分块累加 Σ_j w_ij·y_j 与 Σ_j w_ij，w_ij = exp(l_ij − logZ_i)
  数学与全物化严格等价（标准 logsumexp 分块合并，见 test_drift_chunk.py），
  峰值显存从 O(N·M) 降到 O(N·chunk)。反向由 checkpoint 整体重算，
  不保存任何 [N,M] 中间矩阵。

数值要求（必须 fp32）：z=ae^{iθ} 是 2D 维、元素 ±a（max≈8），‖z‖² 求和可达 ~10⁵，
超 fp16 上限 65504 直接溢出；且三数相减有灾难性抵消。距离全程升 fp32。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _pairwise_logits(x_sq, y_sq, x, y_chunk, tau):
    """logits_ij = −(‖x_i‖² + ‖y_j‖² − 2 x_i·y_j)/τ，fp32。"""
    cross = torch.einsum("bnc,bmc->bnm", x, y_chunk)
    return -(x_sq.unsqueeze(2) + y_sq.unsqueeze(1) - 2.0 * cross) / tau


def _chunk_lse(l):
    """块内 log Σ_j exp(l_ij) = m + log(Σ exp(l−m))。"""
    m = l.amax(-1)
    return m + torch.log((l - m.unsqueeze(-1)).exp().sum(-1) + 1e-30)


def _merge_lse(z1, z2):
    """合并两个 logsumexp：log(e^z1 + e^z2)。"""
    m = torch.maximum(z1, z2)
    return m + torch.log((z1 - m).exp() + (z2 - m).exp() + 1e-30)


def softmax_kernel_field(x, y, tau, self_mask=False, chunk=2048):
    """V_i = Σ_j softmax_j(−‖x_i−y_j‖²/τ)·(y_j − x_i)

    分块两遍，与全物化严格等价：
      V_i = Σ_j w_ij·y_j − x_i·Σ_j w_ij，  w_ij = softmax_j(−d²_ij/τ)

    x: [B,N,C]，y: [B,M,C]
    self_mask: x 与 y 同源时屏蔽 i==j（排斥项用，对角 logits 填 −1e4）
    返回 [B,N,C]，fp32
    """
    B, N, C = x.shape
    M = y.shape[1]
    xf, yf = x.float(), y.float()
    x_sq = (xf ** 2).sum(-1)   # [B,N]
    y_sq = (yf ** 2).sum(-1)   # [B,M]

    gi = torch.arange(N, device=x.device) if self_mask else None

    # ---- Pass 1：每行 i 的全局 logZ_i = log Σ_j exp(l_ij) ----
    logZ = None
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xf, yf[:, cs:ce], tau)
        if self_mask:
            cj = torch.arange(cs, ce, device=x.device)
            l = l.masked_fill((gi.unsqueeze(1) == cj.unsqueeze(0)).unsqueeze(0), -1e4)
        z_blk = _chunk_lse(l)
        logZ = z_blk if logZ is None else _merge_lse(logZ, z_blk)
        del l, z_blk

    # ---- Pass 2：归一化权重加权和 ----
    acc = torch.zeros(B, N, C, device=x.device, dtype=torch.float32)
    wsum = torch.zeros(B, N, device=x.device, dtype=torch.float32)
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xf, yf[:, cs:ce], tau)
        if self_mask:
            cj = torch.arange(cs, ce, device=x.device)
            l = l.masked_fill((gi.unsqueeze(1) == cj.unsqueeze(0)).unsqueeze(0), -1e4)
        w = (l - logZ.unsqueeze(-1)).exp()  # [B,N,chunk]
        acc = acc + torch.einsum("bnm,bmc->bnc", w, yf[:, cs:ce])
        wsum = wsum + w.sum(-1)
        del l, w

    return acc - xf * wsum.unsqueeze(-1)


def softmax_kernel_phase(x, y, theta, theta_tar, tau, chunk=2048):
    """相位空间漂移吸引：

      g_i = Σ_j softmax_j(−‖ẑ_i − ȳ_j‖²/τ) · sin(θ_j^tar − θ_i)

    x: [B,N,C] 当前复值场，y: [B,M,C] 目标复值场
    theta: [B,N,D] 当前相位，theta_tar: [B,M,D] 目标相位，C = 2D

    两点关键设计（修掉旧实现的死穴）：
    (1) 距离在 L2 归一化后的复值场上算。‖z‖ = a·√D，模型 a 与 target a
        相差可达 90 倍，未归一化时 logits 被 ‖z_i‖² 主导、y_j 之间差异被
        淹没，softmax 退化为均匀分布，吸引场退化成 −z_i（拉向原点），
        完全丢失 target 相位信息。归一化后 ẑ·ȳ = mean_d cos(θ_i−θ_j)，
        距离自动成为相位余弦距离，与振幅量级无关。
    (2) 力用 sin(θ_j^tar − θ_i)：周期、有界 ∈[−1,1]、天然是相位流形上的
        梯度方向（θ_i 追上 θ_j^tar 时归零，稳定）。用复值场差 (y_j − x_i)
        会被振幅量级失配污染成"压振幅"而非"对齐相位"。

    sin(θ_j − θ_i) = sinθ_j cosθ_i − cosθ_j sinθ_i，分块累加复用 logsumexp 框架。
    返回 [B,N,D]，fp32。
    """
    B, N, C = x.shape
    M = y.shape[1]
    D = theta.shape[-1]
    # L2 归一化：消除振幅量级失配
    xn = F.normalize(x.float(), dim=-1)   # [B,N,C]
    yn = F.normalize(y.float(), dim=-1)   # [B,M,C]
    x_sq = (xn ** 2).sum(-1)              # ≈1
    y_sq = (yn ** 2).sum(-1)              # ≈1

    sin_tar = torch.sin(theta_tar).float()  # [B,M,D]
    cos_tar = torch.cos(theta_tar).float()
    sin_i = torch.sin(theta).float()        # [B,N,D]
    cos_i = torch.cos(theta).float()

    # ---- Pass 1：每行 i 的全局 logZ_i ----
    logZ = None
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xn, yn[:, cs:ce], tau)
        z_blk = _chunk_lse(l)
        logZ = z_blk if logZ is None else _merge_lse(logZ, z_blk)
        del l, z_blk

    # ---- Pass 2：g_i = cosθ_i·(Σ_j w_ij sinθ_j) − sinθ_i·(Σ_j w_ij cosθ_j) ----
    acc_sin = torch.zeros(B, N, D, device=x.device, dtype=torch.float32)
    acc_cos = torch.zeros(B, N, D, device=x.device, dtype=torch.float32)
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xn, yn[:, cs:ce], tau)
        w = (l - logZ.unsqueeze(-1)).exp()  # [B,N,chunk]
        acc_sin = acc_sin + torch.einsum("bnm,bmd->bnd", w, sin_tar[:, cs:ce])
        acc_cos = acc_cos + torch.einsum("bnm,bmd->bnd", w, cos_tar[:, cs:ce])
        del l, w

    return cos_i * acc_sin - sin_i * acc_cos


def pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor):
    """dist²_nm = ‖x_n‖² + ‖y_m‖² − 2 x_n y_mᵀ

    x: [B, N, C]
    y: [B, M, C] 或 [M, C]
    返回 [B, N, M]
    全程 fp32（数值护栏）。
    """
    x = x.float()
    if y.dim() == 2:
        y = y.unsqueeze(0).expand(x.shape[0], -1, -1).float()
    else:
        y = y.float()
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [B,N,1]
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # [B,M,1]
    cross = torch.einsum("bnc,bmc->bnm", x, y)
    return x_sq + y_sq.transpose(1, 2) - 2.0 * cross


class DriftField(nn.Module):
    """漂移场：吸引 − 排斥，双分量。"""

    def __init__(self, d_model: int = 64, n_prototypes: int = 16, lam: float = 0.5,
                 n_scales: int = 2, tau: float = 1.0,
                 chunk: int = 2048, use_checkpoint: bool = True):
        super().__init__()
        self.d_model = d_model
        self.lam = lam
        self.n_scales = n_scales
        self.tau = tau
        self.chunk = chunk
        self.use_checkpoint = use_checkpoint
        # 推理时用可学习原型作吸引中心
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, 2 * d_model) * 0.02)
        # 直接目标吸引分量的权重（可学习）
        self.direct_weight = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _to_complex_field(a: torch.Tensor, theta: torch.Tensor):
        """复值场 -> 实虚拼接 [B, N, 2D]。"""
        return torch.cat([a * torch.cos(theta), a * torch.sin(theta)], dim=-1)

    def _attract(self, z, tar, tau_s):
        return softmax_kernel_field(z, tar, tau_s, self_mask=False, chunk=self.chunk)

    def _attract_phase(self, z, tar, theta, theta_tar, tau_s):
        """相位空间吸引：g_i = Σ_j softmax_j(−相位距离/τ)·sin(θ_j^tar−θ_i)。"""
        return softmax_kernel_phase(z, tar, theta, theta_tar, tau_s, chunk=self.chunk)

    def _repel(self, z, tau_s):
        return softmax_kernel_field(z, z, tau_s, self_mask=True, chunk=self.chunk)

    def forward(self, a: torch.Tensor, theta: torch.Tensor,
                targets: torch.Tensor = None):
        """
        a:     [B,N,D] 振幅
        theta: [B,N,D] 相位
        targets: [B,M,2D] 或 None。
                训练时=目标复值场（M=N）；推理时=None，用可学习 prototypes。

        返回 g: [B,N,D] 相位漂移引导，进入 Kuramoto 的 κ·g 通路。

        漂移场是模型获取 target 信息的唯一通路（初始 a0/θ0 是噪声、ω 确定性）：
        相位引导必须在相位空间上定义，否则振幅量级失配会把吸引场污染成
        "压振幅"，target 相位信息进不了相位演化，loss 贴随机地板。
        """
        z = self._to_complex_field(a, theta)  # [B,N,2D]

        # 吸引中心 + 目标相位
        if targets is None:
            tar = self.prototypes.unsqueeze(0).expand(z.shape[0], -1, -1).contiguous()
        else:
            tar = targets
            if tar.dim() == 2:
                tar = tar.unsqueeze(0).expand(z.shape[0], -1, -1).contiguous()
        re_t, im_t = tar[..., :self.d_model], tar[..., self.d_model:]
        theta_tar = torch.atan2(im_t, re_t)  # [B,M,D]

        # 训练时 M=N，[B,N,N] 爆显存 → 分块 + 梯度检查点（推理/no_grad 时无需检查点）
        use_ckpt = self.use_checkpoint and self.training and z.requires_grad

        g = torch.zeros_like(theta)
        for s in range(self.n_scales):
            tau_s = self.tau * (2 ** s)
            if use_ckpt:
                g = g + checkpoint(self._attract_phase, z, tar, theta, theta_tar,
                                   tau_s, use_reentrant=False)
            else:
                g = g + self._attract_phase(z, tar, theta, theta_tar, tau_s)

        # 直接目标吸引分量：sin(θ_i^tar − θ_i)，逐点精确注入 target 相位。
        # 空间近邻平均的对齐上限只有 0.42（实测 κ 响应饱和曲线）——FFT 系数相位
        # 在数据空间上不平滑，近邻平均把相邻块的不同相位平均掉，引入不可消除的
        # 结构误差。直接分量把零训练 cos 提到 0.998。物理上外场既可直接作用在
        # 每个振子上，也可通过振子间耦合传播，两分量叠加。
        # 仅训练时加（targets 逐点已知且 M=N）；生成时 targets=None 走原型通路。
        if targets is not None and tar.shape[1] == theta.shape[1]:
            g = g + self.direct_weight * torch.sin(theta_tar - theta)

        # 排斥项仍在复值场空间（防振幅坍缩），不进相位通路
        return g.to(a.dtype)
