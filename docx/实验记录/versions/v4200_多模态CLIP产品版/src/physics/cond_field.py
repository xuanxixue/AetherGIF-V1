# -*- coding: utf-8 -*-
"""
CondFieldPredictor —— 条件 → 预测场的直连通路（补齐"内容生成器"能力）

问题（2026-10-14 实测定位）
--------------------------
漂移场是唯一把 target 信息送进相位演化的通路（Kuramoto 只吃 κ·g）。
训练时永远传 `targets=真实场`，于是 drift_field.forward 里

    if targets is None:  tar = self.prototypes ...

这条**生成分支一次都没执行过**，prototypes 梯度恒为 0。
实测 best.pt：prototypes std=0.01963，而初始值 randn*0.02 的 std 就是 0.02
—— 几乎没动过。于是 generate() 拿随机原型当吸引中心，输出纯噪点。

这不是"训练不够久"，是**通路从未被优化**。跑 7 天也不会变。

本模块的解法
------------
给物理核配一个**条件直连的预测场**：把 CLIP 双模态条件映射成一个
[B, N, 2D] 的复值场，作为漂移场的吸引中心。

关键设计：**不做自由回归，而是条件驱动的驻波基混合**（物理惯用法，
与 StandingWaveFFN 里的 3D 驻波调制同源）：

    field[b, n, c] = Σ_k  w_k(cond_b) · A_k(c) · SW_k(n)
                             └── 位置基底（驻波，无参数）
                                     └── 通道基底（可学）

  - w_k(cond)  : 条件 → K 个混合权重（低秩 MLP，约 0.53M 参数）
  - A_k(c)     : 每个基的逐通道复振幅，可学 [K, 2D]
  - SW_k(n)    : 驻波位置基底，无参数，跟 N 解耦 → 任意时长可外推

为什么不直接 MLP 出 1.5M 个数：cond 只有 2049 维，自由回归必然过拟合、
且换时长就失效。基混合把可学参数压到 K×2D + 条件投影，参数量小一个量级，
且位置结构由物理基底承载（符合项目"结构生成器"定位）。

★ 物理核零改动：DriftField / Kuramoto / StandingWaveFFN / OU 递推
  的公式与结构一行未动。本模块只在**调用侧**把吸引中心的来源从
  "随机参数/真实答案"换成"条件预测"。生成动力学仍然只有一个 = PMNN 物理核。
"""
import torch
import torch.nn as nn

from .standing_wave_ffn import StandingWave3D


class CondFieldPredictor(nn.Module):
    """条件向量 → 预测复值场 [B, N, 2D]（生成模式的吸引中心）。

    与 CondPrototypes（16 原型）的区别：
        - 原型版：吸引中心只有 16 个点，只能给"粗糙吸引子"，无法表达空间结构
        - 本模块：驻波基混合，能给出**逐位置有结构的场**，才有内容生成能力

    数值护栏：
      - 输出做幅度 clamp，复值场量级与 target 同阶（FFT 系数 a 中位数~0.0015、
        p99~5.4），过大会让 Kuramoto 的 Ω_max=8 软饱和失效
      - 无可学标量门：幅度由 supervise() 的逐通道 log1p 谱包络负责
    """

    def __init__(self, cond_dim: int = 2049, d_model: int = 125,
                 n_modes_t: int = 3, n_modes_xy: int = 3,
                 hidden: int = 256,
                 max_amp: float = 3.0,
                 mode_offset: int = 1):
        super().__init__()
        self.d_model = d_model
        self.max_amp = max_amp
        # 驻波位置基底（无参数，与序列长度解耦）
        # ★ 用3×3×3=27 个基而不是 4×4×4=64：生成通路的 GPU 峰值实测
        #   3.41GB/4GB（贴边），瓶颈是驻波基与场的中间张量，不是 drift chunk。
        #   降到 27 基后峰值回落到 2.7GB 区间，留足长跑余量，
        #   而 27 个基对"结构生成器"仍是足够的表达维度。
        # ★ mode_offset（2026-10-15 fit2 定位）：默认 1 → m∈{1..n} 全振荡无 DC，
        #   实测 27/64 基纯监督 400 步 sup 只降 9%（连相位项 0.5 都没跌破）：
        #   谱包络沿时空以缓变为主，被纯振荡基底张住的难度天然高。
        #   mode_offset=0 → m∈{0..n-1}，含 DC 基（该维度常数），基数量不变、
        #   等于把最高频换成最低频。StandingWaveFFN 物理核保持默认 1 不动。
        self.sw3d = StandingWave3D(n_modes_t, n_modes_xy, 20.0, 32.0, 32.0,
                                   mode_offset=mode_offset)
        self.n_basis = self.sw3d.n_modes          # 3*3*3 = 27
        # 通道基底：每个基的逐通道实虚振幅
        self.chan_amp = nn.Parameter(torch.randn(self.n_basis, 2 * d_model) * 0.02)
        # 条件 → 混合权重（低秩）
        self.w_proj = nn.Linear(cond_dim, hidden)
        self.w_act = nn.SiLU()
        self.to_weight = nn.Linear(hidden, self.n_basis)
        # ★★ 条件 → 通道调制（关键补充，实测必需）：
        #   原来只有 27 个"基权重 w"是条件相关的，chan_amp 全局共享 ——
        #   204 个 group 挤在 27 维权重里，实测跨基 std 只有 0.00121
        #   （均匀基线 0.03634），根本分不开，条件形同摆设。
        #   现在让条件同时调制每个基的**逐通道复振幅**：
        #       amp[b,k,c] = chan_amp[k,c] · (1 + delta[b,k,c])
        #   条件自由度从 27 维升到 27×250 维，才有足够容量让不同条件
        #   长出不同的谱形状。仍是低秩线性投影 + 物理驻波基底，无 attention。
        self.to_chan = nn.Linear(hidden, self.n_basis * 2 * d_model)
        nn.init.zeros_(self.to_chan.weight)
        nn.init.zeros_(self.to_chan.bias)
        # to_weight 保持正常随机初始化：混合权重 w 必须有区分度，
        # softmax 才有意义（若全零 → w 均匀 → 预测场退化成"所有基的平均"）。
        # ★ 早期版本有一个可学标量 gate，**已移除**：
        #   supervise() 的逐位置归一化会把"整体缩放"梯度完全抵消，
        #   实测 gate 梯度仅 1.3e-08（其他参数 0.2~5.5），门形同虚设，
        #   还掩盖了 chan_amp 的真实学习率。幅度改由 supervise 的
        #   逐通道 log1p 谱包络负责，职责分离。
        # ★★ 但**不能**因此让预测场恒为零：全零复值场会让漂移场
        #   softmax_kernel_phase 里所有 y 相同 → logits 全相同 →
        #   softmax 退化、logZ 梯度 0/0 → NaN。
        #   当前靠 chan_amp 的随机初值(0.02)保证非零，supervise 会把它推向真值。

    def forward(self, cond: torch.Tensor, t_coord: torch.Tensor,
s_coord: torch.Tensor,
                return_parts: bool = False):
        """
        cond:     [B, cond_dim]  CLIP 条件（2049 = text1024 + image1024 + has_image1）
        t_coord: [N]  时间坐标
        s_coord: [N,2]  空间坐标
        return_parts: True 时额外返回中间量（诊断/监督用）
        返回:     [B, N, 2D] 实虚拼接的预测场
                  return_parts 时返回 (field, dict)
        """
        B = cond.shape[0]
        # 混合权重 [B,K]，用 softmax 保证是"混合"而非任意带符号求和
        h = self.w_act(self.w_proj(cond))                    # [B,hidden]
        w = torch.softmax(self.to_weight(h), dim=-1)         # [B,K]
        # 位置基底 [N,K]（无参数）
        sw = self.sw3d(t_coord, s_coord)                       # [N,K]
        # 通道基底：**逐样本**调制（见 __init__ 注释）
        #   [B,K,2D] = 全局 chan_amp · (1 + 条件残差)
        delta = self.to_chan(h).view(B, self.n_basis, 2 * self.d_model)
        delta = torch.tanh(delta) * 0.5                       # ±0.5，限制幅度冲击
        amp = (self.chan_amp.unsqueeze(0) * (1.0 + delta)).clamp(
            -self.max_amp, self.max_amp)                     # [B,K,2D]
        # 场：Σ_k w_bk · A_bkc · SW_k(n)
        # ★ 等价分解（2026-10-15 显存优化）：直接 einsum("bk,nk,bkc->bnc")
        #   的反向会物化 [B,K,N,C] 大张量（64 基时 786MB，生成通路显存贴边
        #   的元凶之一）。先合并条件无关的收缩系数，再与位置基底做二操作数
        #   einsum（内部走 bmm，反向只产生 [B,N,C] 与 [B,K,C] 级别的梯度）：
        #     field[b,n,c] = Σ_k coeff[b,k,c] · sw[n,k]
        #     coeff[b,k,c] = w[b,k] · amp[b,k,c]
        #   数学严格等价（autograd 自动分解 ∂/∂amp 与 ∂/∂w）。
        coeff = w.unsqueeze(-1) * amp                                # [B,K,C]
        field = torch.einsum("bkc,nk->bnc", coeff, sw)               # [B,N,2D]

        # 标量门：控制整体强度。
        # ★ gate 的梯度是"整体缩放"作用，会被后面 supervise() 的**逐位置归一化**
        #   完全抵消 —— 实测 gate 梯度仅 1.3e-08（其他参数 0.2~5.5），
        #   等于这个门白设了，还会掩盖 chan_amp 的真实学习率。
        #   职责分离后移除：幅度由 supervise 的**逐通道 log1p 谱包络**决定，
        #   不再需要一个可学标量门。
        if return_parts:
            return field, {"w": w, "sw": sw, "amp": amp}
        return field

    @staticmethod
    def supervise(pred_field: torch.Tensor, target_field: torch.Tensor,
                  d_model: int = 125) -> torch.Tensor:
        """
        ★ 直接监督：预测场 ↔ 真实场 的谱包络 + 相位余弦对齐

        为什么必须有这一项（不加就只是"能跑"，不会"学会"）：
        cond_field 的梯度原本只能从最终 loss 反传 12 个物理 block 传回来，
        路径长且弱（实测 40 步几乎不动：204 个 group 的输出场两两只差 1.7e-06，
        softmax 混合权重最大值 0.0389 而均匀基线是 1/27=0.0370 ≈ 没分化）。
        直接监督给"条件→场"一条一步到位的梯度。

        口径刻意与 physics/block.py 的 OU 递推一致（都是 **log1p 域**，
        见记忆铁律"架构递推与 loss 必须用同一变换"）：
          幅度：|field| → log1p → 与 target 的 log1p 谱包络做平方误差，
                逐通道归一（FFT 每通道振幅均值跨 727 倍，不归一会学成平的）
          相位：cos 对齐（atan2→ cos 差），相位 loss 用 a_tar² 加权
        """
        re_p, im_p = pred_field[..., :d_model], pred_field[..., d_model:]
        re_t, im_t = target_field[..., :d_model], target_field[..., d_model:]

        # ---- 幅度：逐通道 log1p 谱包络（只去均值中心，**不做范数归一**）----
        # ★ 为什么不能做范数归一（实测踩坑）：
        #   第一版做了 `lp / ||lp||`，结果 150 步训练后 cond_field 的权重
        #   **几乎没动**（chan_amp std 0.02015 vs 初始 0.02），204 个 group
        #   的输出场两两只差 5.7e-06，混合权重跨基 std 仅 0.00121
        #   （均匀基线是 0.03634）—— 条件仍然是摆设。
        #   原因：范数归一把"整体幅度"这个自由度完全消掉，而幅度恰恰是
        #   chan_amp 唯一的表达通道（形状由位置基底 sw 决定，不是它学的）。
        #   归一化 = 把要学的东西从 loss 里删掉。
        #   正确做法：只去**逐位置**中心（消掉 DC 偏移），保留幅度尺度。
        a_p = torch.sqrt(re_p ** 2 + im_p ** 2 + 1e-8)
        a_t = torch.sqrt(re_t ** 2 + im_t ** 2 + 1e-8)
        la_p = torch.log1p(a_p.clamp(min=1e-8))
        la_t = torch.log1p(a_t.clamp(min=1e-8))
        # 只做逐位置中心化（消 DC 偏移），**保留幅度尺度**
        lp = la_p - la_p.mean(dim=1, keepdim=True)
        lt = la_t - la_t.mean(dim=1, keepdim=True)
        # 按 target 的尺度归一（用 target 自身的 std，不是预测的）：
        # 这样 loss 的量级稳定在 1 附近，但**不消除**预测侧的幅度自由度
        scale = lt.std().clamp(min=1e-3).detach()
        loss_amp = ((lp - lt) / scale).pow(2).mean()

        # ---- 相位：cos 对齐，a_tar² 加权（与主 loss 同口径）----
        th_p = torch.atan2(im_p, re_p)
        th_t = torch.atan2(im_t, re_t)
        w_ph = (a_t.detach() ** 2).clamp(min=1e-8)
        cos = torch.cos(th_p - th_t)
        loss_ph = (w_ph * (1.0 - cos)).sum() / w_ph.sum().clamp(min=1e-8)

        return loss_amp + 0.5 * loss_ph

    def extra_repr(self) -> str:
        return f"n_basis={self.n_basis}, d_model={self.d_model}"