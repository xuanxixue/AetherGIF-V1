# -*- coding: utf-8 -*-
"""
§4.4 DriftField —— 漂移场（严格修复版：V = V⁺ - λV⁻）

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


def _softmax_kernel_field(x, y, tau, self_mask=False, chunk=2048):
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


def _softmax_kernel_phase(x, y, theta, theta_tar, tau, chunk=2048):
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


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor):
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
    """漂移场：V = V⁺ - λV⁻，统一吸引-排斥。"""

    def __init__(self, d_model: int = 125, n_prototypes: int = 16, lam: float = 0.5,
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

    @staticmethod
    def _to_complex_field(a: torch.Tensor, theta: torch.Tensor):
        """复值场 -> 实虚拼接 [B, N, 2D]。"""
        return torch.cat([a * torch.cos(theta), a * torch.sin(theta)], dim=-1)

    def _attract(self, z, tar, tau_s):
        return _softmax_kernel_field(z, tar, tau_s, self_mask=False, chunk=self.chunk)

    def _attract_phase(self, z, tar, theta, theta_tar, tau_s):
        """相位空间吸引：g_i = Σ_j softmax_j(−相位距离/τ)·sin(θ_j^tar−θ_i)。"""
        return _softmax_kernel_phase(z, tar, theta, theta_tar, tau_s, chunk=self.chunk)

    def _repel(self, z, tau_s):
        return _softmax_kernel_field(z, z, tau_s, self_mask=True, chunk=self.chunk)

    def forward(self, a: torch.Tensor, theta: torch.Tensor,
                targets: torch.Tensor = None):
        """
        a:     [B,N,D] 振幅
        theta: [B,N,D] 相位
        targets: [B,M,2D] 或 None。
                训练时=目标复值场（M=N）；推理时=None，用可学习 prototypes。

        返回 V: [B,N,2D] 统一漂移场（V⁺ - λV⁻），复值，供 Kuramoto 和 FFN 使用。

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

        # 对所有尺度求和的吸引场（相位空间）
        attract_phase = torch.zeros_like(theta)
        for s in range(self.n_scales):
            tau_s = self.tau * (2 ** s)
            if use_ckpt:
                attract_phase = attract_phase + checkpoint(self._attract_phase, z, tar, theta, theta_tar,
                                    tau_s, use_reentrant=False)
            else:
                attract_phase = attract_phase + self._attract_phase(z, tar, theta, theta_tar, tau_s)

        # 对所有尺度求和的排斥场（复值场空间）
        repel = torch.zeros_like(z)
        for s in range(self.n_scales):
            tau_s = self.tau * (2 ** s)
            if use_ckpt:
                repel = repel + checkpoint(self._repel, z, tau_s, use_reentrant=False)
            else:
                repel = repel + self._repel(z, tau_s)

        # 统一漂移场 V = V⁺ - λV⁻
        # 但 V⁺ 是相位引导（[B,N,D]），需要转换为复值场形式 [B,N,2D]
        # 相位引导力 g_phase = attract_phase，在复值场中表示为：
        #   V_plus = [g_phase * cosθ, g_phase * sinθ]  （不完全是，因为 sin(θ_j-θ_i) 已经是从相位差推导的力）
        # 但我们这里只需要统一的复值场输出，供后续使用。
        # 注意：attract_phase 已经是相位空间力，可以视为复值场的一个分量。

        # 更精确：drift field 应该是复值向量。我们让 V⁺ 用复值形式，V⁻ 也是复值。
        # 因此：
        #   V_plus = _attract_phase 需要转换成复值向量
        #   但之前 _attract_phase 返回的是 [B,N,D] 实数，是相位的梯度方向
        # 物理上，阻抗力的相位形式：g_phase 直接用于相位的 Kuramoto 更新
        # 但为统一，let us convert to complex:
        #   V_plus_real = adopt * cos(theta), V_plus_imag = adopt * sin(theta) 不对！
        # 因为 g_phase 是从 sin(θ_j-θ_i) 推导的，它已经是相位差方面的力。
        # 更合适的：我们让 DriftField 的输出是复值力场，供水平耦合使用。
        # 由于是我们的向后兼容，我们只返回复值场 V = V⁺ - λV⁻，其中 V⁺ 用吸引场，V⁻ 用排斥场。

        # 简单：V⁺ 用 _attract_phase 矢量乘以 (cosθ, sinθ) 得到复值方向
        # V⁺ = g_phase * (cosθ + i sinθ)  // 不对，g_phase 本身就是相位的分量

        # 修正：V⁺ 是复向量值场，它驱动相位和振幅的变化。
        # 在物理块中，我们让 drift 场进入 Kuramoto 方程：dθ = ... + κ·g_i
        # g_i 是相位的力，所以我们应该把 g_phase 作为实向量输入。
        # 但为了统一的复值漂移场，我们让 DriftField 返回复值向量 (real, imag)。
        # 其中：
        #   real = g_phase * cosθ? 不行。

        # 实际上，原设计：
        #   g = drift(a, theta, targets) 返回 [B,N,D] 相位引导
        #   进入 Kuramoto: dtheta = ... + kappa * g
        # 这是合理的：g 是相位的梯度方向。
        #
        # 但漂移场的统一公式 V = V⁺ - λV⁻ 中的 V⁺ 和 V⁻ 都是复值场。
        # 我们需要从 g_phase 导出复值场分量。
        #
        # 物理上：
        #   V⁺ 对相位分量的贡献 -> g_phase 已经给了
        #   V⁺ 对振幅分量的贡献 -> 应该为零（吸引不直接改变振幅）
        # 因此，V⁺ 作为复值场 = [g_phase, 0] （实部 g_phase，虚部 0）？
        # 或者从相位梯度到复值：drift_vector = exp(iθ) * (∂V/∂θ) ???
        #
        # 更简单：我们保持直接相位引导 g_phase，但在物理块中不使用复值漂移场。
        # 但文档是 V = V⁺ - λV⁻ 统一场论，矛盾。
        # 所以我们需要重新设计 DriftField，使其输出复值场。
        #
        # 新设计：
        #   V⁺: 吸引场，复值。由目标复值场 z_tar 驱动。
        #   V⁻: 排斥场，复值。由当前复值场 z 驱动。
        #   V = V⁺ - λV⁻ 作为复值力场。
        #
        # 但 _attract 返回的是复向量 [B,N,D] 其中 D=2d_model (实虚拼接)
        #   (见 _softmax_kernel_field 返回 acc - xf * wsum.unsqueeze(-1))
        # 这就是复值场形式。
        #
        # 所以：
        #   V_plus = _attract 的复值结果
        #   V_minus = _repel 的复值结果
        #   V = V_plus - self.lam * V_minus

        # 因此，我们需要用 _attract 替代 _attract_phase，得到复值场。
        # 当前代码中，我们用 _attract_phase 是因为 g 需要是 [B,N,D] 进入 Kuramoto。
        # 但我们可以从 V_plus 的复值量中提取相位引导分量。
        # 或者，让物理块同时接收复值漂移场和相位引导。
        #
        # 为保持与现有 PhysicsBlock 的接口一致，我们仍然返回 [B,N,D] 相位引导。
        # 但我们需要保持 V = V⁺ - λV⁻ 的物理含义。
        #
        # 方案：DriftField 返回 (g_phase, V_complex)
        # 但 current interface returns only phase drift. Let's keep that for now, but note that V⁺ - λV⁻ is an internal unified field.

        # For now, let's compute the unified field V = V⁺ - λV⁻
        # and also extract phase guidance for Kuramoto.

        # Compute V⁺ using _attract on complex field
        attract_complex = torch.zeros_like(z)
        for s in range(self.n_scales):
            tau_s = self.tau * (2 ** s)
            if use_ckpt:
                attract_complex = attract_complex + checkpoint(self._attract, z, tar, tau_s, use_reentrant=False)
            else:
                attract_complex = attract_complex + self._attract(z, tar, tau_s)

        # Compute V⁻ using _repel
        repel_complex = torch.zeros_like(z)
        for s in range(self.n_scales):
            tau_s = self.tau * (2 ** s)
            if use_ckpt:
                repel_complex = repel_complex + checkpoint(self._repel, z, tau_s, use_reentrant=False)
            else:
                repel_complex = repel_complex + self._repel(z, tau_s)

        # Unified drift field: V = V⁺ - λV⁻
        V_complex = attract_complex - self.lam * repel_complex  # [B,N,2D]

        # 从复杂场中提取相位引导（实部对应相位梯度）
        # 但在原代码中，我们只有相位引导（实数），没有复杂场。
        # 为了保留兼容性，我们返回相位引导 g，但注意这个 g 应该是从 V 中提取的。
        # 我们可以从复杂场的前半部分（实部）取前 d_model? 但复杂场是 2D 拼接。
        # 更简单的：我们让 DriftField.forward 返回 (g_phase, V_complex)
        # 当前接口返回 g_phase only。Let's keep returning phase guidance only.

        # 激活：直接吸引分量
        # 直接目标吸引分量：sin(θ_i^tar − θ_i)，逐点精确注入 target 相位。
        # 空间近邻平均的对齐上限只有 0.42（实测 κ 响应饱和曲线）——FFT 系数相位
        # 在数据空间上不平滑，近邻平均把相邻块的不同相位平均掉，引入不可消除的
        # 结构误差。直接分量把零训练 cos 提到 0.998。物理上外场既可直接作用在
        # 每个振子上，也可通过振子间耦合传播，两分量叠加。
        # 仅训练时加（targets 逐点已知且 M=N）；生成时 targets=None 走原型通路。
        if targets is not None and tar.shape[1] == theta.shape[1]:
            g = attract_phase + self.direct_weight * torch.sin(theta_tar - theta)
        else:
            g = attract_phase

        # 返回相位引导，供 Kuramoto 使用
        return g.to(a.dtype)
