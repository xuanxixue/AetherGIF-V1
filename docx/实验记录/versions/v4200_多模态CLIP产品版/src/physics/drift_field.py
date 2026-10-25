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


def softmax_kernel_phase(x, y, theta, theta_tar, tau, chunk=2048,
                         mod_x=None, mod_y=None, mod_gamma=0.15):
    """相位空间漂移吸引：

      g_i = Σ_j softmax_j(−‖ẑ_i − ȳ_j‖²/τ · [模块门]) · sin(θ_j^tar − θ_i)

    x: [B,N,C] 当前复值场，y: [B,M,C] 目标复值场
    theta: [B,N,D] 当前相位，theta_tar: [B,M,D] 目标相位，C = 2D
    mod_x / mod_y: [B,N] / [B,M] 整数模块号（B 方案稀疏耦合用，None = 稠密）
    mod_gamma: 跨模块的 logits 惩罚（门控强度）。0 = 完全阻断跨模块耦合

    ★★ B 方案（2026-10-16，借鉴 Un-0 稀疏耦合论文）：
      全连接近邻吸引会导致「全局同步坍缩」——所有振子锁到同一状态、
      梯度归零（Un-0 论文原文：dense models fall into catastrophic
      synchronization into a globally locked, zero-gradient state）。
      Un-0 用稀疏/模块化耦合打破全连接，让网络维持 chimera 态
      （部分同步 + 部分异步并存），从而保住高维动力学。

      实现是纯 logits 门控：logits_ij ← logits_ij − gamma·[mod_i ≠ mod_j]。
      - 模块内：gamma=0 → 耦合强度不变，与原式严格一致
      - 模块间：softmax 被压到 e^{−gamma} 倍 → 跨模块影响趋近 0
      数学上仍是 softmax 归一化的加权和，**不是硬 mask**，梯度平滑。
      gamma 取 0.15：跨模块权重衰减到 0.861 倍（温和），
      gamma=3 才是硬阻断（e^{-3}=0.05）。UN-0 的 module 是硬不连通，
      这里留一点弱耦合让信息仍能在模块间缓慢传播。

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

    # ★ 模块门：跨模块 logits 惩罚（B 方案稀疏耦合核心）
    mod_pen = None
    if mod_x is not None and mod_y is not None:
        cross = mod_x.unsqueeze(2) != mod_y.unsqueeze(1)   # [B,N,M] True=跨模块
        mod_pen = cross.float() * mod_gamma               # 0 或 gamma

    # ---- Pass 1：每行 i 的全局 logZ_i（必须含门控，否则分块 logsumexp 合并不再等价）----
    logZ = None
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xn, yn[:, cs:ce], tau)
        if mod_pen is not None:
            l = l - mod_pen[:, :, cs:ce]
        z_blk = _chunk_lse(l)
        logZ = z_blk if logZ is None else _merge_lse(logZ, z_blk)
        del l, z_blk

    # ---- Pass 2：g_i = cosθ_i·(Σ_j w_ij sinθ_j) − sinθ_i·(Σ_j w_ij cosθ_j) ----
    acc_sin = torch.zeros(B, N, D, device=x.device, dtype=torch.float32)
    acc_cos = torch.zeros(B, N, D, device=x.device, dtype=torch.float32)
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xn, yn[:, cs:ce], tau)
        if mod_pen is not None:
            l = l - mod_pen[:, :, cs:ce]
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
                 chunk: int = 256, use_checkpoint: bool = True,
                 n_modules: int = 0, mod_gamma: float = 0.15):
        """
        ★ chunk 从 2048 降到 512（2026-10-14，纯内存优化，数学完全等价）
          分块 logsumexp 合并是精确的（见文件头说明与 test_drift_chunk.py），
          只影响峰值显存/内存，不影响任何数值结果。
          原因：单个 logits 张量是 [B, N, chunk] fp32。N=6144 时
            chunk=2048, B=4 → 201MB/个，12 个 block × 2 pass × 2 scale
            的分配 churn 把主进程提交量顶到 11.4GB（实测），4GB 机器直接换页到死。
            chunk=512 → 48MB/个。
          ★ 实测降到 256 对 GPU 峰值**无改善**（2.54GB 不变）—— 说明峰值不在
            drift chunk 上，而在驻留的 [B,N,2D] 中间张量与 12 个 block 的激活。
            所以留在 512：单步更快（4.4s vs 5.1s），显存一样。
        """
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
        # ★★ B 方案稀疏耦合（2026-10-16）：n_modules=0 退回稠密（旧行为），
        #   n_modules>0 时按时空坐标分 Z 个 module，跨模块 logits 罚 mod_gamma。
        self.n_modules = n_modules
        self.mod_gamma = mod_gamma

    @staticmethod
    def _build_modules(mod_id: torch.Tensor, n_modules: int):
        """把任意整数 id 压到 [0, n_modules) 的稠密编号（id 可能不连续）。"""
        if n_modules <= 0:
            return None
        uniq, inv = torch.unique(mod_id, return_inverse=True)
        return inv % n_modules

    @staticmethod
    def _to_complex_field(a: torch.Tensor, theta: torch.Tensor):
        """复值场 -> 实虚拼接 [B, N, 2D]。"""
        return torch.cat([a * torch.cos(theta), a * torch.sin(theta)], dim=-1)

    def _attract(self, z, tar, tau_s):
        return softmax_kernel_field(z, tar, tau_s, self_mask=False, chunk=self.chunk)

    def _attract_phase(self, z, tar, theta, theta_tar, tau_s,
                       mod_z=None, mod_tar=None):
        """相位空间吸引：g_i = Σ_j softmax_j(−相位距离/τ)·sin(θ_j^tar−θ_i)。"""
        return softmax_kernel_phase(z, tar, theta, theta_tar, tau_s,
                                    chunk=self.chunk,
                                    mod_x=mod_z, mod_y=mod_tar,
                                    mod_gamma=self.mod_gamma)

    def _repel(self, z, tau_s):
        return softmax_kernel_field(z, z, tau_s, self_mask=True, chunk=self.chunk)

    def forward(self, a: torch.Tensor, theta: torch.Tensor,
                targets: torch.Tensor = None, protos: torch.Tensor = None,
                mod_id: torch.Tensor = None, cond_anchor: torch.Tensor = None,
                anchor_weight: float = 0.3):
        """
        a:     [B,N,D] 振幅
        theta: [B,N,D] 相位
        targets: [B,M,2D] 或 None。
                训练时=目标复值场（M=N）；推理时=None。
        protos: [B,P,2D] 或 None。
                ★ 条件驱动的吸引中心（CondPrototypes 产出）。
                优先级高于 targets：**只要 protos 不为 None 就用它**，
                即使训练时也走这条通路 —— 这是让 prototypes/条件残差
                真正收到梯度的唯一办法（见文件头「内容生成通路」说明）。
        mod_id: [N] 可选。各振子的整数模块号（B 方案稀疏耦合）。
                None 或 self.n_modules=0 → 稠密耦合（旧行为，不变）。
        cond_anchor: [B,N] 可选。★ A 方案条件相位锚（2026-10-16）：
                条件给出的逐振子目标相位（通常是 θ₀ + dtheta）。
                与 protos/targets **正交叠加**，不替代它们 ——
                这样训练与生成的漂移场结构完全一致，只有 target 那一项
                在训练时存在（重建），生成时没有（内容生成）。
        anchor_weight: 条件锚的强度（默认 0.3，弱引导）。

        返回 g: [B,N,D] 相位漂移引导，进入 Kuramoto 的 κ·g 通路。

        漂移场是模型获取 target 信息的唯一通路（初始 a0/θ0 是噪声、ω 确定性）：
        相位引导必须在相位空间上定义，否则振幅量级失配会把吸引场污染成
        "压振幅"，target 相位信息进不了相位演化，loss 贴随机地板。
        """
        z = self._to_complex_field(a, theta)  # [B,N,2D]

        # 吸引中心 + 目标相位
        # ★ protos 优先：条件通路（生成模式 + 训练时的条件通路）用它。
        # ★★ A 方案条件锚优先于随机 prototypes（2026-10-16，150 步实测定位）：
        #   原逻辑 protos=None 且 targets=None 时退化到 self.prototypes
        #   （16 个随机初始化参数，**与条件无关**）。它会主导整个演化 →
        #   6 次不同条件生成的输出完全雷同（实测 vs_A 全 13.45~13.55）。
        #   条件锚给出逐振子目标相位，用它就地构造相位型吸引中心：
        #       tar = a · exp(i·anchor)   （幅度取当前 a，只定相位不定结构）
        #   这样近邻吸引有了一个**条件决定**的落点，结构由稀疏耦合 + 物理
        #   自己长出来（UN-0 范式），而不是被随机原型锁死。
        if protos is not None:
            tar = protos
        elif targets is None and cond_anchor is not None:
            # 相位型吸引中心：re = a·cos(anchor), im = a·sin(anchor)
            ca = cond_anchor.unsqueeze(-1) if cond_anchor.dim() == 2 else cond_anchor
            tar = torch.cat([a * torch.cos(ca), a * torch.sin(ca)], dim=-1)
        elif targets is None:
            tar = self.prototypes.unsqueeze(0).expand(z.shape[0], -1, -1).contiguous()
        else:
            tar = targets
            if tar.dim() == 2:
                tar = tar.unsqueeze(0).expand(z.shape[0], -1, -1).contiguous()
        re_t, im_t = tar[..., :self.d_model], tar[..., self.d_model:]
        theta_tar = torch.atan2(im_t, re_t)  # [B,M,D]

        # ★ B 方案：模块分组（当前场与吸引中心都用同一 mod_id，保证对齐）
        mod_z = mod_tar = None
        if self.n_modules > 0 and mod_id is not None:
            _m = self._build_modules(mod_id, self.n_modules)      # [N]
            mod_z = _m.unsqueeze(0).expand(z.shape[0], -1)         # [B,N]
            # ★ mod_tar 必须按 tar 的真实长度 M 索引，不是 N：
            #   原型模式 M=16、targets 模式 M=N，A 方案相位锚 M=N。
            #   用 N 展开会在 M≠N 时形状错位。
            #   ★★ 2026-10-18 修：原来 expand 到 M 时报
            #      "expanded size (16) must match existing size (6144)"——
            #   expand 只能沿 size-1 维广播，不能把6144 截成 16。
            #   正确做法是**切片取前 M 个**：原型/吸引中心与当前场按位置对齐。
            _M = int(tar.shape[1])
            if _M <= _m.shape[0]:
                mod_tar = _m[:_M].unsqueeze(0).expand(z.shape[0], -1)  # [B,M]
            else:
                # M > N（罕见）：循环补齐，保证形状合法
                _rep = _m.repeat((_M // _m.shape[0]) + 1, )[:_M]
                mod_tar = _rep.unsqueeze(0).expand(z.shape[0], -1)     # [B,M]

        # 训练时 M=N，[B,N,N] 爆显存 → 分块 + 梯度检查点（推理/no_grad 时无需检查点）
        use_ckpt = self.use_checkpoint and self.training and z.requires_grad

        g = torch.zeros_like(theta)
        for s in range(self.n_scales):
            tau_s = self.tau * (2 ** s)
            if use_ckpt:
                g = g + checkpoint(self._attract_phase, z, tar, theta, theta_tar,
                                   tau_s, mod_z, mod_tar, use_reentrant=False)
            else:
                g = g + self._attract_phase(z, tar, theta, theta_tar, tau_s,
                                            mod_z, mod_tar)

        # 直接目标吸引分量：sin(θ_i^tar − θ_i)，逐点精确注入 target 相位。
        # 空间近邻平均的对齐上限只有 0.42（实测 κ 响应饱和曲线）——FFT 系数相位
        # 在数据空间上不平滑，近邻平均把相邻块的不同相位平均掉，引入不可消除的
        # 结构误差。直接分量把零训练 cos 提到 0.998。物理上外场既可直接作用在
        # 每个振子上，也可通过振子间耦合传播，两分量叠加。
        # ★ 条件改为「tar 逐点已知」（2026-10-15，diag_cond_field2 定位）：
        #   原条件 `targets is not None` 把 protos 通路（targets=None）排除了，
        #   导致 cond_field 模式只有多尺度近邻弱吸引（上限 0.42），相位对齐慢。
        #   protos 的 shape[1]==N（cond_field 逐位置输出），完全满足逐点条件，
        #   且训练/生成都走 protos、都加直接分量 → 一致。重建模式（targets 逐点）
        #   行为不变；原型模式（P=16≠N）仍跳过。
        if tar.shape[1] == theta.shape[1]:
            g = g + self.direct_weight * torch.sin(theta_tar - theta)

        # 排斥项仍在复值场空间（防振幅坍缩），不进相位通路
        g = g.to(a.dtype)

        # ★★ A 方案：条件相位锚（2026-10-16）
        #   cond_anchor 是条件给出的逐振子目标相位 [B,N]（也可 [B,N,1]），
        #   广播到 D（D 个频率通道共享同一空间形状，与 cond_init 的设计一致）。
        #   ★ 不要 unsqueeze：cond_anchor 可能是 2 维 [B,N]，
        #     unsqueeze 后变 [B,N,1] 再与 [B,N,D] 相减会在 dim=2 撞车。
        if cond_anchor is not None:
            ca = cond_anchor
            if ca.dim() == 2:
                ca = ca.unsqueeze(-1)          # [B,N,1] → 与 [B,N,D] 广播 OK
            g = g + anchor_weight * torch.sin(ca - theta)

        return g
