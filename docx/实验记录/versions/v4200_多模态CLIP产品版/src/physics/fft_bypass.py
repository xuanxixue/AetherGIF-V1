# -*- coding: utf-8 -*-
"""
FFTBypass —— 第 4 路条件：3D FFT 场 → 初态（逐位置）

★ 为什么必须有这一路（2026-10-19，用户提出并经探针验证）
--------------------------------------------------------
CLIP 三路条件（text/first/gif）压成 cond_vec [6147] 维，
却要还原 6144×250 = 153 万个数的场。这是**信息瓶颈**：
向量再强也必然丢结构。探针实测（diag_fft_cond_probe.py）：
仅用 FFT 场逐位置解码，幅度 corr 达 **+0.8541**，
而 CLIP 条件通路经12 层物理栈后只剩 **+0.197**（diag_stack_authority.py）。

★ 但它不是拷贝通道 —— 这是本模块能用于生成的**根本原因**
----------------------------------------------------------------
本模块把每位置的 2D=250 个 FFT 系数压成 **G=5 个频段组**，
即 125 频率 → 5，**25 倍压缩**。
若去掉这个瓶颈（直接 250→250），模型会学成恒等映射，
输出 = 输入 = 自动编码器，完全背离「输出没参考过的图片」的目标。
**瓶颈就是泛化能力的来源**，不是妥协。

★ 逐位置而非全连接：4GB 显存的硬约束
----------------------------------------------------------------
全连接 hidden(512) → N×D(768K) 单层就要 1.46GB，直接 OOM。
逐位置解码器参数量与 N 解耦（本模块 ~0.15M），显存 O(D)。

★ 双模式（对应方案的 1 与 3，一次实现两用）
----------------------------------------------------------------
  mode="teacher"：FFT 场作为**初态**，直接进物理栈。
      用途：测「物理栈的上限能否被拉开」——
           混淆掉「条件通路学不学得会」，只测栈保不保结构。
  mode="cond"   ：FFT 场的预测结果作为**条件通路的一部分**，
                与 cond_vec 通路相加融合，再进物理栈。
      用途：真正的生成通路。
"""
import os
import math
import torch
import torch.nn as nn


class FFTBypass(nn.Module):
    """逐位置 FFT 场解码器：[B,N,2D] + 全局上下文 → per-band (θ̂, log â)。

    输出是 **per-band** 的（G=5），不是标量：
    FFT 每通道振幅均值跨 727 倍，频率间空间分布毫不相似，
    压成标量等于亲手抹掉全部频段结构。

    数值护栏：
      - θ̂ 用 atan 压到 (−π/2, π/2)，与 cond_init 的 θ̂ 模式同口径
      - log â 用 clamp(−8, 4) 防 exp 溢出（target p99=4.647，占得住）
    """

    def __init__(self, d_model: int = 125, n_groups: int = 5,
                 ctx_dim: int = 0, hidden: int = 256,
                 mu_init: float = None):
        super().__init__()
        self.d_model = d_model
        self.n_groups = n_groups
        self.ctx_dim = ctx_dim
        self.net = nn.Sequential(
            nn.Linear(2 * d_model + ctx_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * n_groups),
        )
        # 零初始化输出层：起步时旁路完全不起作用，
        # 与 cond_init 的 zero_init 同一原则（避免初期强调制把梯度打飞）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        #★ 口径纠错（2026-10-19，第二次自我纠正）★★
        #   我曾判「a_mu 初值 −1.609 比 log(a_tar) 真值 −6.516 差 4.91，
        #   ⇒ 12 层放大 126 倍」。**这个判断是错的**，用错了口径：
        #     · encode 用 NORM=10标定，decode 乘回 10.0，双侧必须同口径
        #       （记忆铁律：评估双侧必须同口径，否则差 0.6 dB）
        #     · 我比的是**未乘 NORM** 的 log 均值，而 OU 递推的 a 是 NORM 口径
        #     · NORM 口径下真实中心量级≈ −1.47，与 μ 初值 −1.609 恰好接近
        #
        #   真元因是**重尾本身**：log(a) 分布
        #     q0.1 = −9.210（= clamp 下界）、q0.5 = −6.822、q0.99 = +0.040
        #     ⇒ 跨 9 个数量级，均值不代表任何典型值
        #   OU 把所有系数往**一个标量** μ 收敛，在这种分布下
        #   必然摧毁 80% 的系数（这才是 corr 掉到 0.197 的机制）。
        #
        #   ⇒ 本模块的 dla 是**逐位置**的，能表达重尾；标量 μ 表达不了。
        #   这正是 FFT 旁路相对 cond_vec 通路的结构性优势。
        #
        #   保留 mu_head 但默认**不接入** PMNNCore（避免又一次基于错误
        #   前提的架构改动）。要用时显式开 FFT_BYPASS_MU=1。
        _mu0 = float(os.environ.get("MU_INIT", "") or (mu_init if mu_init is not None else -1.6094))
        self.mu_head = nn.Linear(hidden, d_model)
        nn.init.zeros_(self.mu_head.weight)
        self.mu_bias = nn.Parameter(torch.full((d_model,), _mu0))
        self._gid_cache = None

    def group_index(self, device):
        """[D] → 组号。按 w = c % 25 线性分箱（DC 与最低频同组）。

        与 cond_init._group_index 严格同口径 —— 两路旁路的频段划分
        必须一致，否则融合时同一个 g 在两路指的不是同一批频率。
        """
        if self._gid_cache is not None and self._gid_cache.device == device:
            return self._gid_cache
        c = torch.arange(self.d_model, device=device)
        w = c % 25                       # rfft2 真实频率列 0..24
        G = self.n_groups
        edges = torch.linspace(0, 25, G + 1, device=w.device)
        gid = torch.bucketize(w.float(), edges[1:-1].float()).clamp(0, G - 1)
        self._gid_cache = gid
        return gid

    def expand_to_d(self, x):
        """[B,N,G] → [B,N,D]，按组号广播到组内所有通道。"""
        if x.shape[-1] == self.d_model:
            return x
        if x.shape[-1] == 1:
            return x.expand(-1, -1, self.d_model)
        gid = self.group_index(x.device)
        out = x.new_zeros(x.shape[0], x.shape[1], self.d_model)
        for g in range(self.n_groups):
            msk = (gid == g)
            if bool(msk.any()):
                out[:, :, msk] = x[:, :, g:g + 1].expand(
                    -1, -1, int(msk.sum()))
        return out

    def forward(self, coeff: torch.Tensor, ctx: torch.Tensor = None,
                return_parts: bool = False):
        """
        coeff: [B, N, 2D] FFT 场（实部+虚部拼接）
        ctx:   [B, ctx_dim] 可选全局上下文（如 cond_vec 投影）
        返回 dth [B,N,G]、dla [B,N,G]
        """
        B, N, _ = coeff.shape
        if ctx is not None and self.ctx_dim > 0:
            x = torch.cat([coeff, ctx.unsqueeze(1).expand(-1, N, -1)], dim=-1)
        else:
            x = coeff
        o = self.net(x)
        dth = torch.atan(o[..., :self.n_groups])          # (−π/2, π/2)
        dla = o[..., self.n_groups:].clamp(-8.0, 4.0)    # log 域
        if return_parts:
            # μ 头：从同一份 FFT 特征预测 OU 平衡点，让物理栈的收敛目标
            # 落在 log(a) 的真实量级上，而不是硬编码的 −1.609。
            h = self.net[:-1](x)
            mu = self.mu_bias + self.mu_head(h.mean(dim=1))   # [B,D]
            return dth, dla, {"h": h, "mu": mu}
        return dth, dla

    def forward_mu(self, coeff: torch.Tensor, ctx: torch.Tensor = None):
        """只要 μ（[B,D]），给 PMNNCore 的 a_mu 用。"""
        B, N, _ = coeff.shape
        if ctx is not None and self.ctx_dim > 0:
            x = torch.cat([coeff, ctx.unsqueeze(1).expand(-1, N, -1)], dim=-1)
        else:
            x = coeff
        h = self.net[:-1](x)
        return self.mu_bias + self.mu_head(h.mean(dim=1))

    @staticmethod
    def supervise(dth, dla, target_field, d_model: int = 125,
                  n_groups: int = 5, w_shape: float = 1.0,
                  w_scale: float = 0.3):
        """双项监督：**形状项 + 尺度项分开加权**（二者缺一不可）。

        为什么必须分开（2026-10-19 实测）：
          只用 MSE            → corr 0.85 但像素 std 只有 5（目标 80），
                                 形状对、绝对亮度差 16 倍。
          只用逐通道能量比 log 差 → corr 0.85 掉到 0.36，
                                 它只管通道间比例、不管逐位置形状。
        单用任一个都不行，必须**形状项 + 尺度项同时在**。

        形状项：去 DC 的 sin/cos 对齐（DC 相位无意义，见记忆铁律）
        尺度项：per-channel 能量比 log 差
        """
        dev = dth.device
        re_t, im_t = target_field[..., :d_model], target_field[..., d_model:]
        a_t = torch.sqrt(re_t ** 2 + im_t ** 2 + 1e-8)
        th_t = torch.atan2(im_t, re_t)
        if a_t.shape[0] != 1:
            raise ValueError(f"supervise 只支持 B=1，收到 {a_t.shape[0]}")

        dth_D = _expand_groups(dth, d_model, n_groups, dev)
        dla_D = _expand_groups(dla, d_model, n_groups, dev)

        a_t_s, th_t_s = a_t[0], th_t[0]                # [N,D]
        # ---- 形状项：逐通道归一的 sin/cos 对齐（去 DC）----
        w_ch = (a_t_s ** 2).clamp(min=1e-8)[:, 1:]
        th_ch = th_t_s[:, 1:]
        w_norm = w_ch / w_ch.mean(dim=0, keepdim=True).clamp(min=1e-8)
        sin_c = w_norm * torch.sin(th_ch)
        cos_c = w_norm * torch.cos(th_ch)
        th_p = dth_D[0][:, 1:]
        loss_shape = (1.0 - (torch.sin(th_p) * sin_c
                             + torch.cos(th_p) * cos_c).mean())

        # ---- 尺度项：per-channel 能量比 log 差 ----
        a_p = torch.exp(dla_D[0][:, 1:].clamp(-8, 4))
        a_t2 = a_t_s[:, 1:].clamp(min=1e-8)
        E_p = (a_p ** 2).mean(dim=0, keepdim=True).clamp(min=1e-8)
        E_t = (a_t2 ** 2).mean(dim=0, keepdim=True).clamp(min=1e-8)
        loss_scale = (torch.log(E_p) - torch.log(E_t)).pow(2).mean()

        return w_shape * loss_shape + w_scale * loss_scale


def _expand_groups(x, d_model, n_groups, dev):
    """[B,N,G] → [B,N,D]。与 CondInitModulator.expand_to_d 同口径。"""
    if x.shape[-1] == d_model:
        return x
    if x.shape[-1] == 1:
        return x.expand(-1, -1, d_model)
    c = torch.arange(d_model, device=dev)
    w = c % 25
    edges = torch.linspace(0, 25, n_groups + 1, device=dev)
    gid = torch.bucketize(w.float(), edges[1:-1].float()).clamp(0, n_groups - 1)
    out = x.new_zeros(x.shape[0], x.shape[1], d_model)
    for g in range(n_groups):
        msk = (gid == g)
        if bool(msk.any()):
            out[:, :, msk] = x[:, :, g:g + 1].expand(-1, -1, int(msk.sum()))
    return out