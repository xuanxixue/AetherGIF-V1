# -*- coding: utf-8 -*-
"""
AetherGIFModel —— 物理矩阵神经网络 GIF 生成模型

完整架构：
  GIF → 3D FFT → 64 复值振子 → PMNNCore → 逆 FFT → GIF

密钥原则：
  - 物理核所有算子严格按 AetherMind 公式实装
  - 不得引入任何 Transformer/注意力/FFN 结构
  - Qwen 仅作为冻结文本编码器，提供物理条件注入
  - 支持 3–10 秒可变时长（6–20 时间步）
"""
import sys

import os
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 本地导入
from physics.block import PMNNCore
from physics.time_field import TimeField
from physics.kuramoto_sync import KuramotoSync, SpacetimeToeplitzCoupling, MeanFieldCoupling, build_spacetime_index
from physics.standing_wave_ffn import StandingWaveFFN, StandingWave3D, lorentzian_gain
from physics.drift_field import DriftField
from physics.thermodynamics import Thermodynamics
from physics.cond_field import CondFieldPredictor
from physics.cond_init import CondInitModulator

# ★ A 方案：条件相位锚的强度（2026-10-16）。
#   弱引导：条件只说"往这个相位走"，具体结构由近邻吸引 + 稀疏耦合长出来。
#   太大（>1.0）会退化成"条件直接指定答案"，模型又变成抄答案。
# ★ 2026-10-24：改为环境变量可配。22 素材实验证据 ——
#   rank 8→64（23M→109M）后 sat 极差从 5.32 **降到 0.68**，越扩容越坍。
#   说明瓶颈不是 cond_init 容量，而是**条件信号在 12 层 Kuramoto 同步里被平均掉了**：
#   每层相位同步都往"共识相位"拉，初始态注入的条件差异经过 12 层后被洗平，
#   物理核索性学成一个与条件无关的平均解（相对判据 55 自己 / 55 错了 = 随机水平）。
#   所以要把每层的条件锚 ANCHOR_W 调大，让条件在每一层都被重新钉住。
ANCHOR_W = float(os.environ.get("ANCHOR_W", "0.3"))

from codec.gif_codec import load_gif_frames, resize_gif_to_10fps, split_frames_to_blocks, reconstruct_frames_from_field, gif_to_target_field, target_field_to_gif, decode_field_to_frames
from data.clip_condition import CLIP_EMB_DIM, N_COND3


# 耦合强度 κ 上限（第三十死穴修复，2026-10-19）
#   TimeField.kappa(t) 是零参数的结构性时间曲线，实测随块号从 1.9 涨到 15.85，
#   把相位全拉平（cos +1.0000 → +0.3577，且独立于其他破坏源）。
#   在调用侧截断，物理核零改动。传 0 关闭（对照实验用）。
_KAPPA_MAX = float(os.environ.get("KAPPA_MAX", "") or 6.0)


class AetherGIFModel(nn.Module):
    """
    AetherGIF 完整模型：
      1. 文本条件编码器 (Qwen 冻结)
      2. 3D FFT 振子编码器 (冻结)
      3. PMNN 物理核心 (可训练)
      4. 逆 FFT 解码器 (冻结)
      5. 文本条件调制 (注入物理参数)
    """

    def __init__(
        self,
        d_model: int = 64,
        ffn_hidden: int = 256,
        n_blocks: int = 12,
        lambda_t: float = 0.5,
        lambda_s: float = 3.0,
        temporal_window: int = 3,
        dt: float = 0.1,
        n_prototypes: int = 16,
        n_modes_t: int = 4,
        n_modes_xy: int = 4,
        T_ref: float = 20.0,
        X_ref: float = 32.0,
        Y_ref: float = 32.0,
        qwen_model_path: str = "Qwen2.5-0.5B-Instruct/snapshots/master",
        device: str = "cuda",
        patch_size: Tuple[int, int] = (4, 4),
        time_step: int = 5,
        use_qwen: bool = True,
        init_mode: str = "init",
        n_modules: int = 8,
        mod_gamma: float = 0.15,
    ):
        super().__init__()
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.n_blocks = n_blocks
        self.patch_size = patch_size
        self.time_step = time_step
        self.device = device
        self.use_qwen = use_qwen
        # ★★ A/B 方案开关（2026-10-16）
        #   init_mode='init'  A 方案：条件只调初始相位/幅度（Un-0 范式）
        #   init_mode='field' 旧版：条件预测完整吸引中心场（原 CondFieldPredictor）
        #   n_modules=0        B 方案关闭（退回稠密耦合）
        self.init_mode = init_mode
        self.n_modules = n_modules
        self.mod_gamma = mod_gamma

        # 1. PMNN 物理核心（可训练）
        self.pmnncore = PMNNCore(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            lambda_t=lambda_t,
            lambda_s=lambda_s,
            temporal_window=temporal_window,
            dt=dt,
            n_prototypes=n_prototypes,
            T_ref=T_ref,
            X_ref=X_ref,
            Y_ref=Y_ref,
            n_modules=n_modules,
            mod_gamma=mod_gamma,
        ).to(device)

        # 2. 条件编码器（v4200：Qwen3-VL 三模态同空间向量）
        # ================================================================
        # 【演进脉络】
        # v3700 时代用 jina-clip 双塔：文本塔和图像塔是**异构空间**，
        # 必须各配一套投影矩阵（CondProjector / CondProjector3）才能
        # 喂进同一组调制头 —— 投影是那个架构的刚需，不是设计偏好。
        #
        # v4200 换成 Qwen3-VL-Embedding 后，文本 / 首帧 / GIF 三路都从
        # **同一个**模型的同一个对齐空间出来（都是 2048 维 L2 归一向量），
        # 投影变成纯冗余：3×2048×896 ≈ 5.5M 参数只做"同空间到同空间"
        # 的线性变换，且其输出只喂四个旧调制头，A 方案主通路（cond_init）
        # 本就绕过它直吃 6147。
        #
        # ★ 用户定调（2026-10-16）：删掉投影，三路直接**标量加权相加**
        #   —— 同空间相加语义明确，三个可学习标量经 softmax 归一。
        #   缺省模态在 cond_vec 里本来就是零向量，加权后仍是零，
        #   无需额外门控。四个旧调制头改吃 2048（=三路向量维度）。
        #   交叉训练（overfit_cond3.py）负责让"单独输入一路还能生成"
        #   的分布被模型见过。
        # ================================================================
        self.text_dim = CLIP_EMB_DIM          # 2048：三路融合后的维度，四个调制头改吃它
        # ★ 三路标量加权权重（softmax 归一，零初始化 = 起步三路等权）
        self.cond_w = nn.Parameter(torch.zeros(3, device=device))
        self.qwen = None
        if use_qwen:
            # ★ v4200 三模态口径下该分支已停用：四个调制头固定吃 2048，
            #   Qwen2.5-0.5B 的 hidden=896 喂不进去（forward 会 loudly
            #   报错）。不再加载编码器，省 1GB 显存。全部脚本已统一
            #   use_qwen=False；保留参数仅为签名向后兼容。
            print("[WARN] use_qwen=True 在 v4200 三模态口径下已停用（896 维喂不进 2048 头），"
                  "将忽略；请传 cond_vec [B, 6147]")
        else:
            # 无条件时的可学习兜底向量（batch=1 的冒烟测试用）
            self.text_param = nn.Parameter(torch.randn(1, self.text_dim, device=device) * 0.02)

        # 3. 文本到物理条件的调制器（可训练）
        # 文本嵌入 → 模型参数（rand, scale, drift_targets, etc.）
        # 为保持物理纯粹性，我们不使用 Qwen 输出直接改变物理算子，而是提供边界条件
        # 例如：固有频率谱偏移、漂移场原型、κ课程参数
        self.omega_bias_head = nn.Linear(self.text_dim, d_model)  # 文本 → 频率扰动
        self.drift_target_head = nn.Linear(self.text_dim, d_model)
        self.kappa_scale_head = nn.Linear(self.text_dim, 1)
        self.T_scale_head = nn.Linear(self.text_dim, 1)

        # 4. 物理上下文：用于分配原型和偏移
        self.each_prototype = nn.Parameter(torch.randn(n_prototypes, 2 * d_model) * 0.02)

        # ================================================================
        # 【内容生成通路 · 2026-10-14】
        # 原架构只会"重建"不会"生成"，根因已实测定位：
        #   漂移场是唯一把 target 送进相位演化的通路，但训练时永远传
        #   targets=真实场 → `if targets is None: 用 prototypes` 这条生成分支
        #   一次都没执行，prototypes 梯度恒为 0。实测 best.pt 里
        #   prototypes std=0.01963 = 初始 randn*0.02 的 std，一动没动。
        #   → generate() 拿随机原型当吸引中心，输出纯噪点。跑 7 天也不会变。
        #
        # 修法：加这个"条件 → 预测场"预测器，训练与生成共用同一条通路。
        # 物理核一行未改，吸引中心的来源从"随机参数/真实答案"换成"条件预测"。
        #
        # ★ mode_offset=0（2026-10-15 fit2~fit4 定位）：驻波基底含 DC 基。
        #   原来 m∈{1..3} 全振荡无 DC，27/64 基纯监督 sup 只降 9%（卡死）；
        #   补 DC 后降 21%（翻倍），lr=2e-3 也不发散。基数量 27 不变
        #   （64 基仅再优 2.6%，但生成通路显存要重测——一次一个变量）。
        #   sw3d 无参数，mode_offset 不改任何参数 shape → 旧 ckpt 兼容。
        # ================================================================
        self.cond_field = CondFieldPredictor(
            cond_dim=N_COND3, d_model=d_model,
            mode_offset=0,
        ).to(device)

        # ================================================================
        # 【A 方案 · 2026-10-16】条件 → 振子初始状态（Un-0 范式）
        #
        # 病根诊断（1000 步实测）：CondFieldPredictor 要求条件一次性预测
        # 完整吸引中心场 [N=6144, 2D=250] = 153 万个数。这等于让 8.5M
        # 参数的模块去干 SD 那套 2.1B U-Net 的活。实测条件映射容量不足：
        #   图生 src1 wcos=+0.39（勉强） / 文生 src1 wcos=+0.05（≈随机，判错）
        #
        # Un-0 论文 §7 的关键分工：
        #   "oscillator dynamics organize and preserve the distribution of
        #    latent states, while the decoder turns those states into pixels"
        #   —— 条件不预测像素，只设定振子初始状态（附录：encoder 就是
        #      "a linear layer from the data to the initial conditions"）。
        #
        # 本模块：条件 → (θ₀ 位移 [N,1], ln a₀ 增益 [N,1])，
        #   输出维度从 153 万降到 6144×2，且不含 250 通道的组合爆炸。
        #   零初始化输出头 → 起步时调制量恒 0，从纯随机起点平滑过渡。
        #
        # n_osc 需要在 forward 时才知道 N，这里用 TS×GRID² 的典型值
        # 初始化（6144），forward 里若 N 不同会按需重建（见 _cond_init）。
        # ================================================================
        self._cond_init_n = 6144
        # ★ E3 旋钮（2026-10-17）：amp_scale 环境变量透传
        #   dlna = tanh(to_amp(h)) * amp_scale，默认 1.5
        #   E3a 探针 = 3.0：幅度增益空间翻倍，攻判别（区分主要靠 dlna 承担）
        import os as _os
        _amp_scale_env = _os.environ.get("AMP_SCALE")
        _amp_scale = float(_amp_scale_env) if _amp_scale_env else 1.5
        # ★★ per-channel 扩容开关（2026-10-21，第 36 个发现的直接修复）★★
        # teacher 天花板探针实测：fix_r4 ckpt 在 teacher 口径（答案直接喂给
        # G=5 的 fft_bypass + G=5 的 cond_init）下重建 PSNR 13.76/9.64 dB ≈
        # 噪声基线 13.03/7.44 —— G=5 per-group 结构（125 通道共享 5 相位，
        # content 层 w=0 的 5 个时间频率通道共享每位置 1 个相位）就是天花板。
        # 对照 v3600：drift 拿逐通道 target → wcos 0.9991（瓶颈不在物理栈）。
        # COND_INIT_PC=1 → 换 CondInitPC（θ̂/â 逐通道 [B,N,D]，
        # R=COND_PC_RANK 低秩头，默认 8；参数量 ~21.5M << 500M 上限）。
        if _os.environ.get("COND_INIT_PC", "").lower() in ("1", "true", "yes", "on"):
            from physics.cond_init_pc import CondInitPC as _CondInitCls
            _pc_rank_env = _os.environ.get("COND_PC_RANK")
            _pc_kwargs = {"rank": int(_pc_rank_env)} if _pc_rank_env else {}
            print(f"[INFO] 条件初始态：CondInitPC（逐通道，rank={_pc_kwargs.get('rank', 8)}）")
        else:
            _CondInitCls = CondInitModulator
            _pc_kwargs = {}
        self.cond_init = _CondInitCls(
            cond_dim=N_COND3,
            n_osc=self._cond_init_n,
            amp_scale=_amp_scale,
            **_pc_kwargs,
        ).to(device)

        # ★★ 2026-10-25：条件 → 振幅谱包络偏置（解决 sat 锁死成常数）★★
        #   问题链（全部实测）：
        #     1. 8 层递推 log_a ← log_a + res − γ(log_a − μ_c)，μ_c 全样本共享；
        #        γ=0.3 → (1−γ)^12≈0.014，γ=0.5 → 2.4e-4，**初值 m_0 被洗掉**；
        #     2. 于是终态层级 ≡ μ_c，与条件无关 → 所有素材 sat 锁在 6.0~6.9；
        #     3. 实测 2 素材：src0(素材5.35) 生成6.85、src1(素材10.58) 生成5.93，
        #        4 个条件之间差 <0.15 —— 颜色完全不跟随条件。
        #   修法：平衡点本身随条件变。mu_bias = LayerScale·tanh(head(cond_vec))，
        #   加到 μ_c 上。收敛到平衡点是物理性质，但**每个条件有自己的平衡点**。
        #   零初始化起步（不破坏已训好的 μ_c），训练中逐渐学会按条件调谱。
        # ★ 2026-10-25 v2：输入接 cond_init.proj 输出的 h（512 维），**不再接原始 cond_vec**。
        #   证据：MU_SCALE 0.5→2.0→6.0 时，两素材 sat 差只从 −0.30→1.27→1.34（三倍放大只换来 5%），
        #   且 src0/src1 一起被抬高 —— 说明 head 学到的是「整体偏移」而非「条件差异」。
        #   原因：原始 cond 向量本身余弦 0.19~0.33（差异小），而 cond_init.proj 的 h 已被
        #   DISC_W=8 训练出区分度（余弦 0.307）。所以改为复用 h。
        self.mu_head = nn.Sequential(
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, self.d_model),
        )
        self.mu_scale = float(os.environ.get("MU_SCALE", "0.5"))
        with torch.no_grad():
            nn.init.zeros_(self.mu_head[-1].weight)
            nn.init.zeros_(self.mu_head[-1].bias)
        self._mu_head_on = os.environ.get("MU_HEAD", "1") == "1"

        # ================================================================
        # ★★ 第 4 路条件：3D FFT 场旁路（2026-10-19，用户提出）★★
        #用户判断：「向量再怎么样，包含的东西也不够全面。既然 3D FFT
        #   本身就能还原整个 GIF，那把它结合进去。既有泛化能力，
        #   又能理解大量文本，还能输出没参考过的图片。」
        #
        # 探针验证（diag_fft_cond_probe.py）：仅用 FFT 场逐位置解码，
        #   幅度 corr 达 +0.8541；而 CLIP 条件通路经 12 层物理栈后
        #   只剩 +0.1969。数量级优势明确。
        #
        # 逐位置而非全连接：4GB 显存下 hidden(512)→N×D(768K) 单层就要
        #   1.46GB 直接 OOM。旁路参数量与 N 解耦（~0.15M）。
        #
        # 天然瓶颈：每位置 2D=250 个系数 → G=5 个频段组（25× 压缩）。
        #   这个瓶颈是**泛化能力的来源**，不是妥协——
        #   去掉它模型会学成恒等映射（自动编码器），
        #   反而背离「输出没参考过的图片」的目标。
        #
        # FFT_FFTW=0 关闭（对照实验用）。
        # ================================================================
        self.fft_bypass = None
        if float(_os.environ.get("FFT_FFTW", "") or 1.0) > 0:
            from physics.fft_bypass import FFTBypass
            self.fft_bypass = FFTBypass(
                d_model=d_model, n_groups=5, ctx_dim=0, hidden=256,
            ).to(device)
            print(f"[INFO] 第4 路 FFT 场旁路已启用（逐位置，25× 频段瓶颈）")

        print(f"[INFO] AetherGIFModel 初始化完成：d_model={d_model}, n_blocks={n_blocks}, 参数量 = {sum(p.numel() for p in self.parameters()):,}")
        print(f"[INFO] A/B 方案：init_mode={init_mode}, n_modules={n_modules}, mod_gamma={mod_gamma}")

    def _fuse_three_way(self, cond_vec: torch.Tensor) -> torch.Tensor:
        """三路同空间向量 → 标量加权融合 [B, 2048]。

        ★ 用户定调（2026-10-16）：文本 / 首帧 / GIF 三路都出自 Qwen3-VL
          的**同一个**对齐向量空间（同为 2048 维 L2 归一），不再需要
          投影矩阵（CondProjector3 是 jina-clip 双塔异构空间时代的历史
          包袱），直接标量加权相加，语义明确。
        ★ softmax 归一化：三个权重和恒为 1，防止三路整体同步放大；
          单路权重仍可自由学到趋近 0（丢弃该路）或趋近 1（独占）。
        ★ 缺省模态在 cond_vec 里本来就是零向量（交叉训练与评测缓存都
          这么拼），零向量加权后仍是零，**不需要**额外的 has 门控；
          has 标志位只随 6147 整体喂给 cond_init，不参与本融合。

        cond_vec: [B, 6147] = [text(2048)|first(2048)|gif(2048)|has×3]
        返回: [B, 2048]
        """
        t_e = cond_vec[:, :CLIP_EMB_DIM]
        f_e = cond_vec[:, CLIP_EMB_DIM:CLIP_EMB_DIM * 2]
        g_e = cond_vec[:, CLIP_EMB_DIM * 2:CLIP_EMB_DIM * 3]
        w = F.softmax(self.cond_w, dim=0)              # [3]，零初始化=等权起步
        return w[0] * t_e + w[1] * f_e + w[2] * g_e    # [B, 2048]

    def _init_omega(self, t_coord: torch.Tensor, s_coord: torch.Tensor) -> torch.Tensor:
        """确定性固有频率场 [1, N, D]。

        ω 是系统的固有属性（参数），不是初始状态。若每次前向都 randn 重采样，
        同一个样本两次前向就是两个不同的动力系统，相位演化无法学习——这条
        通路与漂移场同样致命：ω 随机 + 漂移场失效时，模型完全没有可学习的
        确定性动力学。用时空坐标 × 通道索引的确定性准随机场给出空间异质频率，
        保证同一位置每次前向频率一致。
        """
        d_idx = torch.arange(self.d_model, device=t_coord.device).view(1, -1)  # [1,D]
        base = 0.4 * torch.sin(
            s_coord[..., 0:1] * 12.9898
            + s_coord[..., 1:2] * 78.233
            + t_coord.unsqueeze(-1) * 37.719
            + d_idx * 4.1
        )  # [N, D]
        return base.unsqueeze(0)  # [1, N, D]

    def forward(self, texts: List[str], target_field: torch.Tensor,
                t_coord: torch.Tensor, s_coord: torch.Tensor,
                batch_size: int = 1, duration_target: Optional[int] = None,
                text_embs: Optional[torch.Tensor] = None,
                cond_vec: Optional[torch.Tensor] = None,
                use_cond_field: bool = False,
                force_init: Optional[tuple] = None):
        """
        条件生成前向：

        1. 条件编码（v4200：CLIP 双模态，cond_vec 直传离线特征）
        2. 物理参数调制：基于条件向量产生 ω bias, drift targets, κ/T 缩放
        3. 初始化噪声场
        4. PMNN 物理演化：从噪声场积分到目标场
        5. 返回最终振子场和诊断

        ★ v4200 三模态条件入口（三个生成能力共用一个 forward）：
            cond_vec: [B, 6147] = [text(2048)|first(2048)|gif(2048)|has_text(1)|has_first(1)|has_gif(1)]
              - 文生 GIF：has_text=1，first/gif 段填零
              - 图生 GIF：has_first=1，text/gif 段填零
              - 图文生：has_text=has_first=1，gif 段填零
              - GIF 本身：has_gif=1，其余填零
              - 三路全给：三个 has 全 1
            三路同空间（Qwen3-VL 对齐空间），_fuse_three_way 标量加权融合成
            [B, 2048] 喂四个调制头；cond_vec 整体 6147 另喂 cond_init（A 方案）。
            旧接口 texts / text_embs（Qwen）已停用（维度不匹配，见路径 B 报错）。

        返回值：
          - (a, theta): [B, N, D] 最终振子场
          - diag: 诊断标签字典
        """
        # 1. 条件编码
        # ---- 路径 A（v4200 主路径）：离线三模态特征直传 ----
        if cond_vec is not None:
            cond_vec = cond_vec.to(self.device).float()
            if cond_vec.dim() == 2 and cond_vec.shape[1] == N_COND3:
                text_embs = self._fuse_three_way(cond_vec)  # [B, 2048]
            else:
                # 已经是融合后的条件向量，直接用
                text_embs = cond_vec
        # ---- 路径 B：兼容旧的 Qwen 文本 ----
        elif text_embs is not None:
            text_embs = text_embs.to(self.device)
            if text_embs.shape[-1] != self.text_dim:
                raise ValueError(
                    f"text_embs 维度 {text_embs.shape[-1]} != 调制头维度 "
                    f"{self.text_dim}：v4200 三模态口径下四个头固定吃 2048，"
                    f"Qwen（hidden=896）路径已停用，请改传 cond_vec [B, {N_COND3}]")
        else:
            text_embs = self.text_param.expand(len(texts) if texts else batch_size, -1).to(self.device)

        # 2. 物理参数调制
        omega_bias = self.omega_bias_head(text_embs)  # [B, D]
        drift_targets = self.drift_target_head(text_embs)  # [B, D]
        kappa_scale = torch.sigmoid(self.kappa_scale_head(text_embs)).squeeze(-1)  # [B]
        T_scale = torch.sigmoid(self.T_scale_head(text_embs)).squeeze(-1)  # [B]

        # 3. 初始化/加载噪声场 (若 target_field 是 None，则使用原型)
        if target_field is None:
            # 生成模式：无 target_field
            # 使用可学习原型作为吸引中心（作为漂移场的吸引子）
            # 在 PMNNCore 内部处理原型
            pass
        else:
            # 训练模式：有 target_field
            pass

        # 4. 物理演化
        # 初始状态 a0/θ0 随机（生成模型从噪声出发，正确）；ω 必须确定性
        # a0 在 log 域采样以匹配 FFT 幅度的重尾量级（几何均值 ≈0.08，与 target 同阶）；
        # 振幅走乘性演化后，起点量级几层就能调过来，但同阶起点能显著加速收敛
        a0 = torch.exp(torch.randn(batch_size, len(t_coord), self.d_model, device=self.device) * 0.8 - 2.5)
        theta0 = torch.rand(batch_size, len(t_coord), self.d_model, device=self.device) * 2 * math.pi - math.pi
        # ★ 仅供诊断：强制指定初态，绕开条件通路，直接测物理栈的**上限**。
        #   正常生成/训练路径永远为 None，行为逐位不变。
        if force_init is not None:
            a0 = force_init[0].to(self.device).float()
            theta0 = force_init[1].to(self.device).float()
        omega_base = self._init_omega(t_coord, s_coord)  # [1,N,D] 确定性

        # ================================================================
        # 【A 方案】条件 → 初始状态调制（Un-0 范式）
        # ================================================================
        protos = None
        drift_target = target_field
        init_parts = None
        dtheta_cached = dlna_cached = None
        anchor_t = None
        theta0_m = None
        fft_parts = None
        if use_cond_field and cond_vec is not None:
            cv = cond_vec.to(self.device).float()
            # ★★ 第 4 路：FFT 场旁路（与 CLIP 三路**相加融合**）★★
            #   FFT 场逐位置自带答案，CLIP 提供全局语义。
            #   两者都走「θ̂/â 模式」，量纲一致（rad / log 幅度），可直接相加。
            #
            #   ★ 变量初始化教训：_fs/_fdth_D 曾在分支内赋值，却在下游
            #     融合点引用 → UnboundLocalError（旁路关闭或 N 不匹配时）。
            #     诊断/融合代码不能假设上游分支一定执行过。
            _fs = None
            _fdth_D = _fdla_D = None
            _fft_src = None
            if self.fft_bypass is not None:
                # 训练期用 target_field 当 FFT 场（方案 3 的 teacher旁路）；
                # 生成期由调用方通过 _fft_cond_cache 注入。
                _fft_src = target_field
                if _fft_src is None:
                    _fft_src = getattr(self, "_fft_cond_cache", None)
            if self.fft_bypass is not None and _fft_src is not None:
                _fs = _fft_src.to(self.device).float()
                if _fs.dim() == 2:
                    _fs = _fs.unsqueeze(0)
                if _fs.shape[1] != len(t_coord):
                    # N 不匹配（N=TS×GRID² 变了）→ 跳过，保持与旧 ckpt 兼容
                    _fs = None
                if _fs is not None:
                    _fdth, _fdla, _fparts = self.fft_bypass(_fs, return_parts=True)
                    _fdth_D = self.fft_bypass.expand_to_d(_fdth)   # [B,N,G]→[B,N,D]
                    _fdla_D = self.fft_bypass.expand_to_d(_fdla)
                    _fparts["dth"] = _fdth
                    _fparts["dla"] = _fdla
                    fft_parts = _fparts
            _cond_h = None
            if self.init_mode == "init":
                # A 方案：只调初始相位与幅度，吸引中心留给物理近邻自洽
                dtheta_cached, dlna_cached, init_parts = self._cond_init(
                    cv, len(t_coord), theta0, a0)
                _cond_h = init_parts.get("h")   # [B,512]，已含 DISC 训练出的区分度
                dtheta, dlna = dtheta_cached, dlna_cached
                # ★ 第二十三死穴修复（2026-10-18）：按频段分组展开到 D 维。
                #   旧代码直接广播 [B,N,1] → D，125 个通道共享同一套空间形状
                #   （实测跨通道 std 精确=0）。现在 expand_to_d 按 w 列分箱，
                #   让每个频段有独立的条件空间结构。
                # ★ 第二十八死穴修复（2026-10-19）：θ̂ 模式下 dtheta 已是 [B,N,D]
                #   （由 θ0 反算而来），expand_to_d 原样返回，不做任何广播。
                dtheta = self.cond_init.expand_to_d(dtheta)   # [B,N,D]
                dlna = self.cond_init.expand_to_d(dlna)       # [B,N,D]
                # dlna/dtheta 形状 [B,N,D]，直接作用于 a0/θ0。
                # ★ 不要再 unsqueeze，否则变 4 维爆显存。
                a0 = a0 * torch.exp(dlna)
                theta0 = theta0 + dtheta
                # ★★ 第 4 路融合点：FFT 旁路的 θ̂/â 与 cond_init 结果**相加**。
                #   两者都走「θ̂/â 模式」（第二十八死穴修复）——
                #   dtheta = wrap(θ̂ − θ0) 形式，随机初相位被完全抵消，
                #   所以这里可以直接把旁路的修正量加到 dtheta/dlna 上：
                #     最终 θ0 = θ0 + dθ_clip + dθ_fft ≡ θ̂_clip + θ̂_fft
                #   零初始化输出层保证起步时旁路贡献为 0，平滑接入。
                if self.fft_bypass is not None and _fs is not None:
                    # dlna/dtheta 已在上方施加到 a0/theta0，这里**只改量**，
                    # 不能重复乘 exp（dlna 已含旁路项）。
                    dtheta = dtheta + _fdth_D
                    dlna = dlna + _fdla_D
                    a0 = a0 * torch.exp(_fdla_D)
                    theta0 = theta0 + _fdth_D
                # ★★★ 初态级教师强制（2026-10-22，第三十七死穴修复）★★★
                #   实测（diag_pure_dynamics.py）：把初态**强制成目标场本身**后，
                #   12 层动力学仍把它搅到 11.79/8.95 dB ≈ 噪声 —— 动力学
                #   从未学过「保持好结构」，ph_res（FFN 相位残差）单独贡献
                #   −4.8/−3.4 dB。给训练加教师强制步骤（force_init=(a_tar,θ_tar)，
                #   概率 FORCE_INIT_P），让动力学在 loss 监督下学会对好结构
                #   只做小修正。与 GUID_P 同模式；target 只作训练输入与
                #   对齐目标（「不注入数据先验」铁律：不进权重、不进 μ）。
                #   force_init=None 时这三行逐位不执行，历史行为完全不变。
                if force_init is not None:
                    a0 = force_init[0].to(self.device).float()
                    theta0 = force_init[1].to(self.device).float()
                # ★ 条件相位锚：**只用条件增量 dtheta，不含随机 θ₀ 噪声**。
                #   之前用 theta0.detach()（= 随机 θ₀ + dtheta），随机部分
                #   是每次前向重采的噪声，会把条件信号稀释掉 —— 吸引中心
                #   变成"随机 + 一点点条件"，6 次生成自然雷同。
                #   现在锚是纯条件量：不同条件 → 不同锚 → 不同落点。
                theta0_m = theta0
                # ★ 审查修正（2026-10-20）：锚必须是「目标相位 θ̂ = θ0+dtheta (mod 2π)」，
                #   不是修正量 dtheta。θ̂ 模式下 dtheta = wrap(θ̂−θ0) 含随机 θ0 分量；
                #   若锚=dtheta，漂移场力 = sin(dtheta−θ̂) = sin(−θ0)，依赖随机初相位，
                #   每块都在把相位往随机方向拽，主动抵消 θ̂ 模式的修复。
                #   supervise（diag["theta0"]）用的是 θ̂ 口径，锚却用 dtheta —— 不一致。
                anchor_t = theta0_m.detach()
                # ★★ 关键修正（2026-10-16，150 步实测定位）：
                #   训练时 target **绝不能**再进漂移场当吸引中心。
                #   之前误写 `drift_target = target_field`，导致：
                #     训练：target 当吸引中心 → wcos=+0.97（抄答案，指标漂亮）
                #     生成：target_field=None → 漂移场退化到随机 prototypes
                #     结果：条件的作用被 target 完全挤掉，6 次生成一模一样
                #   这就是 2026-10-14 定位的「训练/生成两条路脱节」老病根，
                #   换个条件通路（cond_init）它照样复发 —— 因为病根不在
                #   「条件从哪来」，而在「target 有没有进物理核」。
                #
                #   UN-0 的做法：条件是唯一信息源。target 只用来算 loss。
                #   这就是 A 方案「不抄答案」的真正含义。
                drift_target = None
            else:
                # 旧版：条件预测完整吸引中心场（三模态 cond_dim=6147）
                protos = self.cond_field(cv, t_coord, s_coord)   # [B,N,2D]
                drift_target = None

        # ================================================================
        # 【B 方案】稀疏耦合的振子模块号
        # 按时空坐标分 Z 个 module（Un-0 modular 稀疏耦合）：
        # 阻断全局同步坍缩、保留 chimera 态的高维动力学。
        # 分法用现成的 build_spacetime_index 的时空结构，纯整数运算，
        # 无参数、不进梯度。
        # ================================================================
        mod_id = None
        if self.n_modules > 0:
            gi = torch.arange(len(t_coord), device=self.device) // self.n_modules
            mod_id = gi % self.n_modules

        # 文本偏移
        omega = omega_base + omega_bias.unsqueeze(1)  # [B, N, D]

        # 时间场参数平均化
        # ★ kappa_scale/T_scale 是 [B]，必须 unsqueeze 两次得到 [B,1,1]。
        #   只 unsqueeze 一次 → [B,1]，与 g 的 [B,N,D] 相乘时会在 dim=1 上撞车：
        #   B=1 时 [1,1] 恰好能广播（所以 batch=1 一直没暴露问题），
        #   B=2 时报 "size of tensor a (2) must match tensor b (6144)"。
        #   修在这里而不是 Kuramoto 内部：物理核保持零参数、对 batch 维无感知。
        kappa_base, T_base = 2.0, 0.5
        kappa = (kappa_base * kappa_scale + 0.0).view(-1, 1, 1)  # [B,1,1]
        T = (T_base * T_scale + 0.0).view(-1, 1, 1)              # [B,1,1]
        # ★★★ 第三十死穴修复（2026-10-19）：耦合强度 κ 上限 ★★★
        #   实测（diag_kuramoto_parts.py，关掉 FFN/OU 只跑 Kuramoto）：
        #     κ 随块号从 1.907 **单调涨到 15.851**（8.3×），
        #     cos(θ) 从 +1.0000 逐层掉到 **+0.3577**，
        #     且该崩塌**完全独立于** w2 范数与幅度 res
        #     （消融表里 cos 恒钉 +0.36 一个数都不动）。
        #   ⇒ Kuramoto 的 `kappa * g` 同步项在深层把相位全拉平，
        #     这是第三个独立破坏源（幅度 res、OU、κ）。
        #
        #   修法：在**调用侧**给 κ 加上限，不改 TimeField/Kuramoto 公式
        #   （物理核零改动铁律）。实测 κ=15.85 时 cos 已掉到 0.36，
        #   故上限取 KAPPA_MAX（环境变量，默认 6.0）。
        kappa = kappa.clamp(max=_KAPPA_MAX)

        # 5. PMNN 物理演化（时间步迭代）
        a, theta = a0, theta0
        for i in range(self.n_blocks):
            tf = self.pmnncore.time_field.forward(i, self.device)
            # 漂移引导由 PhysicsBlock 内部的 DriftField(a, theta, targets) 给出，
            # 是唯一携带 target 信息的通路；这里不再重复计算
            # ★ 审查修正（2026-10-20）：KAPPA_MAX 必须 clamp「有效 κ」= tf["kappa"]*kappa，
            #   不是 clamp kappa_base*kappa_scale（sigmoid 后 ≤2，永远不触发）。
            #   深块 tf["kappa"] 涨到 ~15.85，乘积可达 ~31 → 相位被 κ·g 暴力同步拉平
            #   （第三十死穴第③破坏源，此前 KAPPA_MAX 实为空操作）。
            _k_eff = (tf["kappa"] * kappa).clamp(max=_KAPPA_MAX)
            _mu_b = None
            if getattr(self, "_mu_head_on", False) and _cond_h is not None:
                _mu_b = self.mu_scale * torch.tanh(self.mu_head(_cond_h))
            a, theta, d = self.pmnncore.blocks[i](
                a, theta, omega, t_coord, s_coord,
                kappa=_k_eff,
                T=tf["T"] * T,
                omega_d=tf["omega_d"],
                targets=drift_target,
                protos=protos,
                mod_id=mod_id,
                cond_anchor=anchor_t,
                anchor_weight=ANCHOR_W,
                mu_bias=_mu_b,
            )
            # 可选：记录 diag

        # 返回最终状态和诊断
        diag = self._collect_diag(a, theta, t_coord, s_coord)
        # 预测场一并返回：训练侧要用它做**直接监督**（否则 cond_field 的梯度
        # 只能穿过 12 个物理 block 反传，实测 40 步几乎学不动）。
        if protos is not None:
            diag["pred_field"] = protos
        # ★ A 方案：初始调制量返回给训练侧做直接监督
        #   （梯度要穿 12 个物理 block 才到，路径长且弱）
        if init_parts is not None:
            diag["dtheta"] = dtheta_cached
            diag["dlna"] = dlna_cached
            # ★ 第二十七死穴（2026-10-19）：把实际使用的 θ0 一并返回。
            #   绝对相位对齐需要「θ0 + dθ ≈ θ_tar」的监督，
            #   而 θ0 是每次前向重采的随机量 —— 训练侧必须拿到
            #   **同一个** θ0，不能自己重采一个（否则监督的是错的初态）。
            if theta0_m is not None:
                diag["theta0"] = theta0_m
            if init_parts is not None and "theta_hat" in init_parts:
                diag["theta_hat"] = init_parts["theta_hat"]
            if init_parts is not None and "a_hat" in init_parts:
                diag["a_hat"] = init_parts["a_hat"]
            if fft_parts is not None:
                diag["fft_dth"] = fft_parts.get("dth")
                diag["fft_dla"] = fft_parts.get("dla")
                diag["fft_h"] = fft_parts.get("h")
        return a, theta, diag

    def _cond_init(self, cv: torch.Tensor, n_osc: int,
                   theta0: torch.Tensor = None,
                   a0: torch.Tensor = None):
        """按需重建 cond_init（N 变化时），并做 A 方案的前向。

        N 在本项目是固定的（TS×GRID²），但保留重建能力以便换时长/分辨率。
        theta0 不为 None 时启用 θ̂ 模式（第二十八死穴修复）。
        """
        if self.cond_init.n_osc != n_osc:
            # ★ 按实例的实际类重建（CondInitModulator / CondInitPC），
            #   PC 无 phase_scale，用 hasattr 探测兼容字段。zero_init 重建
            #   会重新零初始化，但 N 变化在本项目不发生（TS×GRID² 固定）。
            _Cls = type(self.cond_init)
            _kw = {}
            if hasattr(self.cond_init, "phase_scale"):
                _kw["phase_scale"] = self.cond_init.phase_scale
            if hasattr(self.cond_init, "rank"):
                _kw["rank"] = self.cond_init.rank
            self.cond_init = _Cls(
                cond_dim=N_COND3, n_osc=n_osc,
                amp_scale=self.cond_init.amp_scale, **_kw,
            ).to(self.device)
        return self.cond_init(cv, return_parts=True, theta0=theta0, a0=a0)

    def _get_static_drift(self, a, theta, drift_target):
        # 生成模式下，使用可学习原型 + drift_target 注入
        # 为了简化演示，这里返回一个固定漂移
        B, N, D = a.shape
        return torch.zeros(B, N, D, device=a.device)

    def _collect_diag(self, a, theta, t_coord, s_coord):
        # 收集一些诊断量
        diag = {}
        # 计算 r (平均场序参量)
        w = torch.nn.functional.softplus(a)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        rcos = (w * torch.cos(theta)).sum(dim=1) / w_sum
        rsin = (w * torch.sin(theta)).sum(dim=1) / w_sum
        r = torch.sqrt(rcos ** 2 + rsin ** 2 + 1e-12).clamp(max=1.0)
        diag["r"] = r.mean().item()
        diag["theta_diff"] = (torch.diff(theta, dim=1).abs().mean().item())
        return diag

    @torch.inference_mode()
    def generate(self, text: str = None, duration: int = 5, save_path: str = None,
                 image_cond=None, cond_vec=None, clip_encoder=None) -> Union[torch.Tensor, str]:
        """
        推理：从随机噪声生成 GIF。**v4200 支持三种生成模式**：

          1. 文生 GIF：    generate(text="一个弹跳的小球")
          2. 图生 GIF：    generate(image_cond=pil_img 或 np.ndarray)
          3. 图文生 GIF：  generate(text="...", image_cond=img)

        参数:
          text: 文本条件（可 None）
          image_cond: PIL Image / np.ndarray[H,W,3]
          cond_vec: 直接给 [1, 6147] 离线三模态条件向量（最高效，训练产物自带）
          clip_encoder: 在线编码用的 ClipConditionEncoder（传离线特征时不需要）
        返回:
          (a, theta) 或 save_path

        ★ 遗留说明（2026-10-16）：本方法的物理演化段（cond_field 原型场）
          是 v4200 之前 B 方案时代的写法，与 init_mode='init' 的 A 方案
          forward（cond_init 调初始状态 + cond_anchor 相位锚）不一致 ——
          A 方案训练下 cond_field 无梯度，protos 是随机的。
          v4200 的训练/评测一律走 forward 的生成模式（target_field=None
          + use_cond_field=True），**不走本方法**。这里保留投影清理
          （_fuse_three_way）保持口径一致，A 方案适配待后续 generate
          改造任务处理。
        """
        # 1. 条件编码 → cond_vec [1, 6147]（三模态）
        if cond_vec is None:
            if clip_encoder is None:
                from data.clip_condition import ClipConditionEncoder
                clip_encoder = ClipConditionEncoder(device=self.device)
            t_e = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
            f_e = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
            g_e = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
            h_t = h_f = h_g = 0.0
            if text:
                t_e = clip_encoder.encode_text([text])
                h_t = 1.0
            if image_cond is not None:
                from PIL import Image
                if isinstance(image_cond, np.ndarray):
                    arr = image_cond
                    if arr.dtype != np.uint8:
                        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                    pil_im = Image.fromarray(arr)
                else:
                    pil_im = image_cond
                f_e = clip_encoder.encode_images([pil_im])
                h_f = 1.0
            cond_vec = np.concatenate([
                t_e, f_e, g_e,
                np.array([[h_t, h_f, h_g]], np.float32)], 1)
        if not torch.is_tensor(cond_vec):
            cond_vec = torch.from_numpy(np.asarray(cond_vec, dtype=np.float32))
        cond_vec = cond_vec.to(self.device).float()
        if cond_vec.dim() == 1:
            cond_vec = cond_vec.unsqueeze(0)

        if cond_vec.shape[1] == N_COND3:
            text_embs = self._fuse_three_way(cond_vec)  # [1, 2048]
        else:
            raise ValueError(f"cond_vec 维度 {cond_vec.shape[1]} 不是三模态 {N_COND3}")

        # 2. 物理参数调制
        omega_bias = self.omega_bias_head(text_embs)  # [1, D]
        kappa_scale = torch.sigmoid(self.kappa_scale_head(text_embs)).squeeze(-1)  # [1]
        T_scale = torch.sigmoid(self.T_scale_head(text_embs)).squeeze(-1)  # [1]

        # 3. 构造时空坐标
        fps = 10
        T_step = duration * fps // self.time_step
        H, W = 128, 128
        patch_h, patch_w = self.patch_size
        n_h = H // patch_h
        n_w = W // patch_w
        hw = n_h * n_w
        t_coord, s_coord = build_spacetime_index(T_step, n_h, n_w, self.device)
        N = len(t_coord)

        # 4. 初始化噪声场
        a0 = torch.nn.functional.softplus(torch.randn(1, N, self.d_model, device=self.device))
        theta0 = torch.rand(1, N, self.d_model, device=self.device) * 2 * math.pi - math.pi
        omega_base = self._init_omega(t_coord, s_coord)  # [1,N,D] 确定性

        # ★ 条件预测场作为吸引中心（内容生成通路的唯一入口）
        protos = self.cond_field(cond_vec, t_coord, s_coord)   # [1,N,2D]

        # 5. 文本偏移
        omega = omega_base + omega_bias.unsqueeze(1)  # [1, N, D]

        # 6. 迭代演化
        a, theta = a0, theta0
        kappa_base, T_base = 2.0, 0.5
        # 与 forward 同一口径：unsqueeze 两次 → [B,1,1]
        kappa = (kappa_base * kappa_scale + 0.0).view(-1, 1, 1)
        T = (T_base * T_scale + 0.0).view(-1, 1, 1)
        for i in range(self.n_blocks):
            tf = self.pmnncore.time_field.forward(i, self.device)
            # ★ 生成模式：吸引中心 = 条件预测场（不再是随机 prototypes）
            a, theta, d = self.pmnncore.blocks[i](
                a, theta, omega, t_coord, s_coord,
                kappa=tf["kappa"] * kappa,
                T=tf["T"] * T,
                omega_d=tf["omega_d"],
                targets=None,
                protos=protos,
            )

        # 7. 逆 FFT 重构
        # 将振子场还原为 GIF 帧
        frames = self._field_to_frames(a, theta, t_coord, s_coord)

        # 8. 保存
        if save_path:
            self._save_gif(frames, save_path)
            print(f"[GEN] 输出到 {save_path}")
            return save_path
        return a, theta

    def _field_to_frames(self, a, theta, t_coord, s_coord):
        """振子场 → GIF 帧列表。

        a/theta: [1, N, D]。先合成复值场 z = a·exp(iθ)，
        再走确定性逆时空 FFT（decode_field_to_frames）还原像素。
        """
        T_step = int(t_coord.max().item()) + 1
        N = a.shape[1]
        grid = int(round((N / T_step) ** 0.5))
        patch_h, patch_w = self.patch_size
        win = self.time_step

        z = (a[0] * torch.exp(1j * theta[0])).to(torch.complex64)   # [N, D]
        field = torch.cat([z.real, z.imag], dim=-1).float().cpu()   # [N, 2D]，decode 在 CPU 做
        frames = decode_field_to_frames(
            field, time_step=T_step, grid=grid,
            patch=patch_h, win=win, img=128, d_model=self.d_model,
            normalize=True,
        )
        return frames

    def _save_gif(self, frames: List[np.ndarray], path: str) -> None:
        """保存 GIF（frames 为 numpy 数组，转 PIL）。"""
        from PIL import Image
        pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
        pil[0].save(path, save_all=True, append_images=pil[1:], loop=0, duration=100, quality=95)

    def save_checkpoint(self, path: str, epoch: int = 0):
        """保存权重。"""
        state = {
            "model": self.state_dict(),
            "epoch": epoch,
            "d_model": self.d_model,
            "n_blocks": self.n_blocks,
        }
        torch.save(state, path)
        print(f"[SAVE] 检查点到 {path}")

    def load_checkpoint(self, path: str, map_location="cpu"):
        """加载权重。"""
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state["model"])
        print(f"[LOAD] 从 {path} 加载，epoch={state.get('epoch', '?')}")


# 测试函数
if __name__ == "__main__":
    # 快速测试
    model = AetherGIFModel()
    model.eval()
    # 测试条件前向
    texts = ["简约线条动画", "简单的gif"]
    # 构造 dummy 时长和坐标
    duration = 3
    fps = 10
    T_step = duration * fps // 5
    H, W = 128, 128
    patch_h, patch_w = 4, 4
    n_h, n_w = H // patch_h, W // patch_w
    t_coord, s_coord = build_spacetime_index(T_step, n_h, n_w, "cuda")
    # 生成流程测试
    with torch.no_grad():
        a, theta, diag = model.generate("简约线条动画", duration=duration)
        print("生成完成:", diag)
        # 保存
        gpath = f"test_{duration}s.gif"
        model._save_gif([] if not isinstance(a, torch.Tensor) else [], gpath)
        # 尝试用 AetherGIFModel 生成的 GIF 
        print("路径:", gpath)
