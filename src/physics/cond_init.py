# -*- coding: utf-8 -*-
"""
CondInitModulator —— A 方案：条件 → 振子初始状态（Un-0 范式）

问题（2026-10-16 用户指出 + 诊断确认）
----------------------------------------
原 CondFieldPredictor 要求条件**一次性预测完整吸引中心场** [N, 2D]：
N=6144 × 250 维 = 153 万个数。这等于让一个 8.5M 参数的模块去干
Stable Diffusion 那套 2.1B U-Net 的活（SD 的条件映射容量全靠参数堆）。

实测后果（1000 步训练后）：
  - 图生 src1  wcos=+0.39   （勉强判对）
  - 文生 src1  wcos=+0.05   ≈ 随机相位，判错
即条件映射容量不足，纯文生更是信息瓶颈。

★ Un-0 的做法（论文 §7）
  "the oscillator dynamics organize and preserve the distribution of
   latent states, while the conventional decoder turns those states
   into recognizable pixels."
  —— 条件**不预测像素**，只设定振子的初始状态（§附录：encoder 是
  "a linear layer from the data to the initial conditions"）。
  结构由物理动力学自己长出来。

本模块的解法（A 方案）
----------------------
条件 → 两类**低维**初始调制量，输出形状与 a0/θ0 相同但数值极轻：

    1. θ0 位移 [B,N,1] : 逐振子的初始相位偏移 ∈ (−π, π)
       —— 一个振子一个数，N 个数。这是"起跑姿势"，不是"答案"。
    2. ln a0 增益 [B,N,1] : 逐振子的初始幅度对数增益
       —— 让不同条件的能量分布起点不同。

  参数量：cond_dim→hidden→N 的线性层，约 4097×256 + 256×6144 ≈ 2.6M，
  但**输出维度从 153 万降到 6144+6144**，且不含 250 个频率通道的
  组合爆炸——每个频率通道 c 的初始相位偏移 = 该通道的 base + δ，
  δ 只有 N 个自由参数，250 个通道共享同一套空间形状（物理合理：
  图像的空间结构在不同频率上是同一个轮廓）。

★ 与物理核的关系（铁律：物理核零改动）
  DriftField / Kuramoto / StandingWaveFFN / OU 递推的公式与结构一行未动。
  本模块只在**调用侧**改初始条件的来源：a0/θ0 从"纯随机"变成
  "随机 + 条件调制"。吸引中心退回弱引导（target 仍可作引导，
  但不再是唯一信息来源）。

★ 与原 CondFieldPredictor 的关系
  两者是**互补**而非替代：
    - CondInitModulator：条件 → 初始状态（提供"方向"）
    - DriftField 近邻吸引：物理自洽（提供"结构"）
  原 CondFieldPredictor（预测完整场）保留在代码里，
  需要时可用 init_mode='field' 切回对照。
"""
import os
import math
import torch
import torch.nn as nn


# ★ 激活开关（2026-10-17 机制实验矩阵 E2）：
#   tanh  有界 (−1,1)，饱和区梯度→0，一旦贴边回不来（铁律 8 的死锁机制）
#   arctan 同值域 (−1,1)（2/π 归一），梯度 = 1/(1+x²) 永不消失，
#         饱和后仍可被守卫/方向监督拉回线性区 —— 治本路线
# 环境变量 PHASE_ACT=arctan 切换；不影响 state_dict（参数 shape 不变），
# 老 ckpt 可直接换激活做诊断对照。
_PHASE_ACT = os.environ.get("PHASE_ACT", "tanh").strip().lower()
# 绝对相位项的相位差上限（rad）。π/2 ≈ 1.5708
#   —— 超过这个量级就饱和，避免该项压垮其余 loss（phabs2k 退步教训）。
_IDEAL_CLAMP = float(os.environ.get("IDEAL_CLAMP", "") or (math.pi / 2.0))
if _PHASE_ACT not in ("tanh", "arctan"):
    _PHASE_ACT = "tanh"

# ★ 绝对相位项的权重口径（2026-10-21，方向2）
#   病灶：w_norm 逐通道归一后高频权重被压没 → |pe| 中位 1.51rad 卡在 clamp，
#   且跨素材×条件恒定（与条件无关）= 高频相位学不动。
#   PH_ABS_W_MODE 可选 norm(默认) / sqrt / uniform / log / band。
_PH_ABS_W_MODE = os.environ.get("PH_ABS_W_MODE", "norm").strip().lower()


def _make_ph_abs_weight(w_ch: torch.Tensor,
                        a_t_s: torch.Tensor) -> torch.Tensor:
    """构造绝对相位项的逐通道权重 [N, D-1]（已去DC）。

    w_ch: (a_t_s[:,1:]**2)，即去 DC 后的 a_tar²。
    a_t_s: [N, D] 完整幅度（含 DC），供 log/uniform 口径取用。
    ★ 无论哪种口径，调用方都必须保留 IDEAL_CLAMP（量级对账铁律）。
    """
    mode = _PH_ABS_W_MODE
    if mode == "uniform":
        return torch.ones_like(w_ch)
    if mode == "sqrt":
        return a_t_s[:, 1:].clamp(min=1e-8)
    if mode == "log":
        return 1.0 / (1.0 + torch.log1p(a_t_s[:, 1:].clamp(min=1e-8)))
    if mode == "band":
        # per-band 内均匀：以 D-1 个非 DC 通道均分5 个 band 的列号
        Dc = w_ch.shape[1]
        cidx = torch.arange(1, Dc + 1, device=w_ch.device)
        gid = cidx % 25
        edges = torch.linspace(0, 25, 6, device=w_ch.device)
        gb = torch.bucketize(gid.float(), edges[1:-1].float()).clamp(0, 4)
        w = torch.zeros_like(w_ch)
        for g in range(5):
            msk = (gb == g)
            if bool(msk.any()):
                w[:, msk] = 1.0
        return w
    # norm（旧，默认）
    return w_ch / w_ch.mean(dim=0, keepdim=True).clamp(min=1e-8)


class CondInitModulator(nn.Module):
    """条件向量 → 振子初始相位偏移 + 幅度增益。

    输出（默认 mode='init'）：
        dtheta: [B, N, 1]  初始相位偏移，单位 rad，建议乘 phase_scale
        dlna:   [B, N, 1]  初始幅度对数增益

    数值护栏：
      - dtheta 用 tanh/arctan 压到 (−1,1) 再乘 phase_scale，
        防止初期把振子推到完全无序（那等于纯噪声起步）
        ★★★ phase_scale 天花板已解除（2026-10-19，第二十七死穴）：
          旧默认 π/4=0.785 的理由是「1000 步后 |dtheta| 顶满 π、
          两素材余弦 0.9053，区分度被饱和区压掉」。
          **该理由已失效**：区分度问题（第二十六死穴）的真因是
          supervise 口径 + 低秩瓶颈，已由 h 层判别项根治（cos_h 0.985→−0.055）。
          而 π/4 这个天花板本身正是**根因**：
            - 随机 θ0 与 target 的 cos = 0.00004（完全无关）
            - 实测 |dtheta| = 0.77，而理想相位差尺度 ≈ π ≈ 3.14
            - 若 dtheta = 0.5×理想差 → cos = +0.4877；1.0× → +0.3733
          ⇒ **dtheta 幅度差 4 倍，物理上搬不动随机初态到 target**，
            所以 cos(θ,target) 12 层只从 0.0007 走到 −0.0266。
          换言之：π/4 治的是「区分度」，代价是「搬不动」—— 治标反伤本。
          现在 phase_scale 由环境变量 PHASE_SCALE 控制（默认 π/2=1.5708，
          恰为 0.5×理想差，是实测 cos 增益最大处），
          且 supervise 的相位项改为**直接对齐 target 相位**，
          幅度会自己长到需要的量级，不需要手工天花板。
      - dlna 用 tanh 压到 (−1,1) 再乘 amp_scale（默认 1.5），
        幅度演化是 log 域乘性（a_new = a·exp(clamp(ffn,−6,3))），
        起点差异在几层内会被 FFN 调回来，这里只是"倾向"不是"锁定"
      - **supervise 幅度守卫**（2026-10-16 任务 #69）：cos 方向对齐
        对 |dtheta| 大小无代价 → tanh 被推进饱和区 → 两条件都顶到
        ±phase_scale 边界、区分度被压掉（1000 步实测两素材 dtheta
        余弦 +0.9988）。supervise 里对 |dtheta| > 0.4 的部分二次惩罚，
        把幅度拉回 tanh 线性区。详见 supervise docstring。
      - **无初始化为零的分支**：两路都用 `tanh(zero_init)` 保证起步
        时调制量为 0，模型从"纯随机起点"平滑过渡到"条件引导起点"。
        这是刻意的：初期若调制过强，梯度全在把振子推回随机，
        反而学不动（与 to_chan 全零初值导致条件形同摆设是同类教训）。
    """

    def __init__(self, cond_dim: int = 4097, n_osc: int = 6144,
                 hidden: int = 512,
                 phase_scale: float = 0.0,
                 amp_scale: float = 1.5,
                 zero_init: bool = True,
                 n_groups: int = 0,
                 d_model: int = 125,
                 w_fft: int = 25):
        super().__init__()
        # ★ 第二十七死穴（2026-10-19）：phase_scale 改由环境变量控制，
        #   默认 π/2（=1.5708，恰为 0.5× 理想相位差，实测 cos 增益最大处）。
        #   旧的硬编码 π/4 天花板太低，dtheta 搬不动随机初态（差 4 倍）。
        #   传 0 表示「用默认」，避免调用方需要知道这件事。
        if phase_scale <= 0.0:
            phase_scale = float(os.environ.get("PHASE_SCALE", "") or
                                (math.pi / 2.0))
        self.n_osc = n_osc
        self.phase_scale = phase_scale
        self.amp_scale = amp_scale
        self.d_model = d_model

        # ★★★ 第二十三死穴修复：per-band 条件调制（2026-10-18）
        #   旧设计假设「图像空间结构在各频率上是同一个轮廓，250 个通道
        #   共享同一套空间形状」→ 输出 [B,N,1] 广播到全部 D 通道。
        #   **这个假设是错的**，实测两处硬证据：
        #     ① 广播后跨通道 std = 0.000e+00（精确为零，见diag_info_bottleneck）
        #     ② FFT 每通道振幅均值跨 727 倍，频率间空间分布毫不相似
        #   后果：条件通路只能给「全图统一偏移」，无法表达任何空间结构；
        #   结构必须从与 target 无关的随机种子长出来 → 学到的必然是
        #   条件无关的通用纹理 → **这就是「生成全是噪声」的字面成因**。
        #
        #   修法：按频率分 w_fft 列分成 n_groups 组，每组一套空间形状。
        #   组内仍共享（省参数），组间独立（能表达频率结构）。
        #   选 n_groups=5：6144×5=30720 输出/路，参数 512×30720×2 ≈ 31M，
        #   加上原有 ≈ 40M，4GB 卡放得下（AdamW 状态 ×3 ≈ 480MB）。
        #   n_groups=0 → 退回旧行为（[B,N,1]），保留 A/B 对照能力。
        _ng = int(os.environ.get("COND_GROUPS", str(n_groups)) or 0)
        self.n_groups = max(1, _ng)
        #低秩因子秩。COND_LOWRANK=0 退回全秩（调试对照用）。
        _lowrank = int(os.environ.get("COND_LOWRANK", "16") or 0)
        self.lowrank = _lowrank
        self.w_fft = w_fft
        out_dim = n_osc * self.n_groups if self.n_groups > 1 else n_osc
        self.out_dim = out_dim

        # ★ hidden=512（用户定调：能跑就算不奢侈，拉顶配）。
        #   条件通路独占 8.65M（总参数 8.49M→17.16M），但这一层做的事只是
        #   "给振子起跑姿势"——宽一点让条件映射有余量，纯文本那种模糊条件
        #   才有机会被解出来（128 宽时文生 src1 wcos 仅 +0.05 ≈ 随机）。
        #   代价 = 参数 8.65M + AdamW 状态 ≈ 105MB，4GB 卡需实测能否跑。
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            # ★★ 归一化修复（2026-10-21，diag_cond_stall.py 决定性验证）：
            #   h 爆炸 → _pre 爆炸 → arctan 100% 饱和 → 梯度死亡 →
            #   cond_init 150/300 步都贴着基线不动（学不动）。
            #   实测（fix_r2 ckpt，探针 diag_hnorm.py）：
            #     h 原始 std=23.48，_pre mean_abs=241，饱和率 100%，
            #        arctan*2 梯度均值 = 5.99e-05（死亡）。
            #     h 归一化(std→1) 后，_pre mean_abs=10.3，梯度均值 = 2.86e-02
            #        （复活 476 倍），cos(θ̂,θ_tar) 维持 +0.50。
            #   LayerNorm 是标准 MLP 归一化（非数据先验），把 h 固定到 ~N(0,1)，
            #   根治 h 爆炸，arctan 回到线性工作区。
            #   ★ 兼容性：新增参数 load 时 missing=1，strict=False 忽略，
            #     LN 初始化为恒等（γ=1,β=0），老 ckpt 行为与无权归一一致。
            nn.LayerNorm(hidden),
        )
        # 两路输出头：θ0 位移 + ln a0 增益
        # ★★ 低秩分解（2026-10-18，4GB 卡必需）：直接 nn.Linear(hidden, 6144*5)
        #   要34.9M 参数，AdamW 状态 ×3 → 显存 3904/4096MB 已顶满，
        #   ★★ 实测时延（bench_step_real.py，2026-10-21，R=16 低秩现状）：
        #      单步 = 6.22 s（forward 2.16 / backward 4.04 / opt 0.02），
        #      300 步 ≈ 31 分钟。
        #   ★ 上方注释曾写「1.7 分钟/步，150 步 = 4 小时」——那是**全秩 34.9M
        #     参数时代**的数字，低秩后早了一个量级，别再照搬（我曾据此误判时长）。
        #   改成 rank-R 的因子化：hidden → hidden → out_dim，
        #   参数 = 2·hidden·(hidden + out_dim)。R=16 时 ≈2.1M，省 16 倍。
        #   表达力：per-band 调制的瓶颈是「空间形状 × 频段」这一项，
        #   而条件向量到 hidden 的映射是低秩的（hidden=512 已远大于条件
        #   自由度），实测 group 间差异不受影响（见 verify_perband_cond.py）。
        if _lowrank > 0:
            self.to_phase = nn.Sequential(
                nn.Linear(hidden, _lowrank, bias=False),
                nn.Linear(_lowrank, out_dim))
            self.to_amp = nn.Sequential(
                nn.Linear(hidden, _lowrank, bias=False),
                nn.Linear(_lowrank, out_dim))
            self._zero_head = (self.to_phase[0], self.to_phase[1],
                               self.to_amp[0], self.to_amp[1])
        else:
            self.to_phase = nn.Linear(hidden, out_dim)
            self.to_amp = nn.Linear(hidden, out_dim)
            self._zero_head = (self.to_phase, self.to_amp)

        if zero_init:
            # 零初始化**最后一层**输出头 → 起步时调制量恒 0（纯随机起点）
            # 梯度仍能流回（因为前面的因子非零），不会像全零chan_amp
            # 那样让 softmax 退化。低秩模式下只零最后一层，
            # 保证梯度能经第一层流回 proj。
            if _lowrank > 0:
                for _lin in (self.to_phase[1], self.to_amp[1]):
                    nn.init.zeros_(_lin.weight)
                    if _lin.bias is not None:
                        nn.init.zeros_(_lin.bias)
            else:
                nn.init.zeros_(self.to_phase.weight)
                nn.init.zeros_(self.to_phase.bias)
                nn.init.zeros_(self.to_amp.weight)
                nn.init.zeros_(self.to_amp.bias)

    def _group_index(self, device):
        """[N, D] → [N, G] 组号。按 w 列切（DC 列=0 单独成组）。"""
        c = torch.arange(self.d_model, device=device)
        w = c % self.w_fft                       # w = 0..24 是真实频率列
        G = self.n_groups
        if G <= 1:
            return torch.zeros(self.d_model, dtype=torch.long, device=device)
        # 线性分箱：w∈[0,25) 切成 G 段，DC(0) 与最低频同组
        edges = torch.linspace(0, self.w_fft, G + 1, device=w.device)
        gid = torch.bucketize(w.float(), edges[1:-1].float())
        return gid.clamp(0, G - 1)

    def forward(self, cond: torch.Tensor, return_parts: bool = False,
                n_pos: int = None, theta0: torch.Tensor = None,
                a0: torch.Tensor = None):
        """
        cond: [B, cond_dim]
        返回: dtheta [B,N,G]（rad）、dlna [B,N,G]（log 域增益）
              G = n_groups；调用侧需按组展开到 D 通道。
              n_groups=1 时形状退化为 [B,N,1]，与旧行为一致。

        ★★ 第二十八死穴修复（2026-10-19）：theta0 不为 None 时，
          to_phase 的输出**不再当增量**，而当成「目标相位的预测 θ̂」，
          再由 θ0 反算修正量：
              dtheta = wrap(θ̂ − θ0)
          于是 θ0 + dtheta ≡ θ̂ (mod 2π)，**随机初相位被完全抵消**，
          cos(θ_final, θ̂) 恒等于 1，条件信号不再被随机噪声稀释。
          探针 diag_theta0_ceiling.py 实测：
            dtheta 与 θ0 独立时，幅值 π/4→2π 全部 cos≈±0.003（死路）
            dtheta 依赖 θ0 时，bound=π/4→0.474、π/2→0.818、π→1.000
        """
        h = self.proj(cond)                              # [B,hidden]
        _pre_ph = self.to_phase(h)                # Sequential 时输出同形状
        _pre_am = self.to_amp(h)
        _pre = _pre_ph
        N = int(n_pos) if n_pos is not None else self.n_osc
        B = h.shape[0]
        if theta0 is not None:
            # ---- 第二十八死穴修复路径：θ̂ 模式 ----
            # ★ 归一化缩放修正（2026-10-21，diag_arctan_fold.py 决定性验证）：
            #   arctan 值域是**半圆** (−π/2, +π/2)，宽 π。
            #   要让 θ̂ 覆盖**全圆** (−π, π] 必须乘 **2**，而非乘 π！
            #   旧代码 `* math.pi` 得 (−π²/2, π²/2)≈(−4.93,+4.93)，**超出全圆**。
            #   而 to_phase 低秩头把 _pre 学到 ±900（全饱和在 arctan 端点 ±π/2），
            #   乘 π 后 θ̂=±4.93，**100% 超出全圆**，wrap 后翻转成 ∓1.35
            #   —— 模型方向学对了却被系统性翻转成负相关（探针实测）：
            #        arctan*π : cos(θ̂,θ_tar) = −0.107（学反了，折叠率 100%）
            #        arctan*2 : cos(θ̂,θ_tar) = +0.520（强正相关，折叠率 0%）
            #   ⇒「乘 π」是根因，只改这一个常数相位对齐度从 −0.107 → +0.52。
            theta_hat = (torch.arctan(_pre) * 2.0).reshape(
                B, N, self.n_groups if self.n_groups > 1 else 1)
            if self.n_groups <= 1:
                th_D = theta_hat.expand(-1, -1, self.d_model)
            else:
                th_D = self.expand_to_d(theta_hat)          # [B,N,D]
            dtheta = torch.remainder(th_D - theta0 + math.pi,
                                     2 * math.pi) - math.pi
            # ★ 幅度侧同构修复（第二十八死穴的第二例）。
            #   a0 = lognormal(−2.5, 0.8) 的逐位置抖动 dlna 看不到 a0，
            #   乘性注入能整体平移分布但消不掉 σ=0.8 的抖动 —— 探针
            #   diag_a0_ceiling.py 实测：最优常数增益下 log 误差仍 1.718，
            #   抖动 100% 残留，落到像素域就是**灰雾**（std 只有 1.7）。
            #   修法与相位侧同构：dlna 改成从 a0 反算，抖动被完全抵消。
            if a0 is not None:
                # â = exp(pre) 直接预测目标幅度（正数，无界→ 用 softplus 保正）
                a_hat = torch.nn.functional.softplus(_pre_am).reshape(
                    B, N, self.n_groups if self.n_groups > 1 else 1) + 1e-6
                if self.n_groups <= 1:
                    aD = a_hat.expand(-1, -1, self.d_model)
                else:
                    aD = self.expand_to_d(a_hat)                # [B,N,D]
                dlna = torch.log(aD.clamp(min=1e-6)) - torch.log(
                    a0.clamp(min=1e-6))
                if return_parts:
                    return dtheta, dlna, {"h": h, "theta_hat": theta_hat,
                                          "a_hat": a_hat}
                return dtheta, dlna
            dlna = (torch.tanh(self.to_amp(h))
                    * self.amp_scale).reshape(B, N, self.n_groups)
            if return_parts:
                return dtheta, dlna, {"h": h, "theta_hat": theta_hat}
            return dtheta, dlna
        if _PHASE_ACT == "arctan":
            _p = torch.arctan(_pre) * (2.0 / math.pi)   # 同值域(−1,1)，梯度不消失
        else:
            _p = torch.tanh(_pre)                       # 默认（老行为）
        if self.n_groups <= 1:
            # 旧行为：输出 [B,N]，保持与原实现逐位一致
            dtheta = (_p * self.phase_scale).reshape(B, N, 1)
            dlna = (torch.tanh(_pre_am)
                    * self.amp_scale).reshape(B, N, 1)
        else:
            # out_dim = n_osc × n_groups → reshape 成 [B, n_osc, n_groups]
            dtheta = (_p * self.phase_scale).reshape(B, N, self.n_groups)
            dlna = (torch.tanh(self.to_amp(h))
                    * self.amp_scale).reshape(B, N, self.n_groups)
        if return_parts:
            return dtheta, dlna, {"h": h}
        return dtheta, dlna

    def expand_to_d(self, x):
        """[B,N,G] → [B,N,D]，按组号把每组空间形状广播到组内所有通道。"""
        if x.shape[-1] == self.d_model:
            return x
        if x.shape[-1] == 1:
            return x.expand(-1, -1, self.d_model)
        gid = self._group_index(x.device)             # [D]
        return x.index_select(-1, gid)

    @staticmethod
    def supervise(dtheta, dlna, target_field, d_model: int = 125,
                  w_ph: torch.Tensor = None,
                  guard_bot: float = 0.4, guard_w: float = 1.0,
                  sep_w: float = 1.0,
                  theta0: torch.Tensor = None,
                  abs_w: float = 0.0,
                  theta_hat: torch.Tensor = None,
                  a_hat: torch.Tensor = None):
        """
        abs_w: 绝对相位对齐权重（2026-10-19，第二十七死穴修复）。
          需同时传 theta0（当前随机初相位 [B,N,D]）。
          监督「θ0 + dtheta ≈ θ_tar」的相位差MSE——
          这是唯一对|dtheta|**幅度**产生真实压力的项。
          abs_w=0 时该分支完全不激活（向后兼容）。
        theta0: [B,N,D] 当前初相位（含随机 θ0）。abs_w>0 时必需。
        sep_w: 频段分离正则权重（2026-10-18 新增）。
          实测不加此项时 5 个频段学出来几乎一模一样（组间差异 2.5e-8
          vs 组内 0.53，ratio≈0），per-band 自由度完全闲置 ——
          因为主监督按组均值对齐，梯度对「各组怎么分」无偏好，
          5 组会自然收敛到同一个平均解。加此项显式鼓励组间分化。

        ★ 直接监督：初始调制量是否指向正确方向。

        为什么需要（用户铁律：不注入数据先验，让模型自己学）：
        初始调制量的梯度必须穿过 12 个物理 block 才能到达，
        路径长且弱。所以给一条**一步到位**的监督。

        监督对象不是"预测完整场"（那又是原方案的老毛病），
        而是**低维的方向量**：
          - 相位：模型给的 dtheta 应该与 target 相位同向。
            衡量 = cos(target_phase, argmax-free 平滑后) 的加权相关。
            实际用可微代理：把 dtheta 当作对相位的"拉力方向"，
            监督它与 target 相位的加权余弦相似度。
          - 幅度：dlna 应该与 log1p(target_a) 的空间分布相关
            （条件的能量分布倾向）。

        ★★ 幅度守卫（2026-10-16，任务 #69）：根治 cos 对齐的
            「只管方向不管幅度」病根 ——
            cos_align = cos(pd − φ_target)，pd 幅度再大只要方向对就
            不罚；tanh 有界 (±1×phase_scale)，梯度于是把 tanh 一路
            推向 ±1 饱和区。实测 1000 步后 |dtheta| 顶满 phase_scale、
            两素材 dtheta 余弦 +0.9988（两边都顶到边界、方向退化为
            符号，条件区分度被压掉）。守卫 = |dtheta| 超过 tanh 线性区
            上界的部分二次惩罚（阈值默认 0.4 rad ≈ 0.5×π/4）。
            纯输出端正则、不碰 target，不违反「不注入数据先验」。

        w_ph: [B,N,D] 或 None。默认用 a_tar² 加权（与主 loss 同口径）。
        guard_bot: 守卫阈值（rad），超过此幅度的 dtheta 被二次惩罚。
        guard_w:   守卫权重。起效量级 ≈ (phase_scale−0.4)² ≈ 0.15，
                   与 loss_ph 同阶；dtheta≈0 时守卫恒 0，不干扰方向学习。
        """
        # 模型方向
        # ★ 2026-10-18 per-band 修复（第二十三死穴的后半段）：
        #   旧写法 pd = dtheta.mean(dim=-1) 把 G 个频段取平均，
        #   **等于亲手抹掉 per-band 差异** —— 实测组间差异只有 2.5e-8，
        #   5 个频段学出来几乎一模一样，per-band 白加。
        #   改成：相位/幅度主监督仍按组均值（与 target 的单一方向向量对齐），
        #   **额外加一项「频段分离」正则** —— 鼓励各组输出彼此不同，
        #   否则它们会收敛到同一个平均解（这正是实测到的现象）。
        G = dtheta.shape[-1] if dtheta.dim() == 3 else 1
        dl_raw = dlna
        # ★★ 第二十八死穴修复（2026-10-19）：θ̂ 模式下监督对象换成 θ̂。
        #   dtheta 在该模式下是 wrap(θ̂−θ0) 的派生物 —— 它把随机 θ0 夹带
        #   进来，监督它等于让模型去拟合一个每次前向都在变的随机量。
        #   真正该学的是条件预测出的相位场 θ̂ 本身。
        if theta_hat is not None:
            pd_g = theta_hat                                   # [B,N,G]
            dl_g = dl_raw
            # 频段分离在 θ̂ 上算（pd_g 已是 θ̂，逻辑与旧分支一致）
            mu = pd_g.mean(dim=1, keepdim=True)
            spread_in = (pd_g - mu).pow(2).mean()
            spread_out = (mu - mu.mean(dim=-1, keepdim=True)
                          ).pow(2).mean()
            # ★ 符号修复（2026-10-21，第三十二死穴）：原 clamp(max=0) 把
            #   注释「ratio<1 时才罚」写成了「ratio<1 时发负奖励」——
            #   组间越无差异、组内越像噪声，loss 越低（diag_sep_sign 实测
            #   -5.674），模型狂刷它（gen=-9.77 以 GEN_W=2 主导总 loss，
            #   与日志 loss=-18.4589 量级对账吻合），导致 |dθ| 顶满 π/2、
            #   dθcos≈0、生成图 ac1=-0.199（负相关=高频棋盘振荡）。
            #   改为 relu(log(spread_in/spread_out))：ratio<1 → 正罚（推
            #   组间分化）；ratio>=1 → 0（不罚）。恢复注释的原始设计意图。
            loss_sep = (torch.log(spread_in.clamp(min=1e-8))
                        - torch.log(spread_out.clamp(min=1e-8))).clamp(min=0.0)
            loss_sep = loss_sep.mean()
        elif G > 1:
            pd_g = dtheta                                       # [B,N,G]
            dl_g = dl_raw                                       # [B,N,G]
            # ---- 频段分离：组间差异 / 组内方差，ratio 越大越好 ----
            # 目标：组间散布至少不小于组内散布（ratio ≥ 1）
            mu = pd_g.mean(dim=1, keepdim=True)                 # [B,1,G] 跨位置均值
            spread_in = (pd_g - mu).pow(2).mean()               # 组内（跨位置）散布
            spread_out = (mu - mu.mean(dim=-1, keepdim=True)
                          ).pow(2).mean()                # 组间散布
            # 用 log 比值，尺度无关；ratio<1 时才罚
            # ★ 符号修复（2026-10-21，第三十二死穴）：原 clamp(max=0) 把
            #   注释「ratio<1 时才罚」写成了「ratio<1 时发负奖励」——
            #   组间越无差异、组内越像噪声，loss 越低（diag_sep_sign 实测
            #   -5.674），模型狂刷它（gen=-9.77 以 GEN_W=2 主导总 loss，
            #   与日志 loss=-18.4589 量级对账吻合），导致 |dθ| 顶满 π/2、
            #   dθcos≈0、生成图 ac1=-0.199（负相关=高频棋盘振荡）。
            #   改为 relu(log(spread_in/spread_out))：ratio<1 → 正罚（推
            #   组间分化）；ratio>=1 → 0（不罚）。恢复注释的原始设计意图。
            loss_sep = (torch.log(spread_in.clamp(min=1e-8))
                        - torch.log(spread_out.clamp(min=1e-8))).clamp(min=0.0)
            loss_sep = loss_sep.mean()
        else:
            loss_sep = torch.zeros((), device=dtheta.device)
            pd_g = theta_hat if theta_hat is not None else dtheta
            dl_g = dl_raw
        # ★ 幅度侧：a_hat 模式下监督 log(â) 而非 dlna。
        #   dlna = log(â) − log(a0) 含随机 a0，监督它 = 让模型拟合随机量
        #   （与 θ̂ 模式下不监督 dtheta 同理）。log 口径与下方 la_c 一致。
        if a_hat is not None:
            dl_g = torch.log(a_hat.clamp(min=1e-6))
        pd = pd_g.mean(dim=-1)                                  # [B,N]
        lp = dl_g.mean(dim=-1)                                 # [B,N]
        sin_p, cos_p = torch.sin(pd), torch.cos(pd)             # [B,N]
        re_t, im_t = target_field[..., :d_model], target_field[..., d_model:]
        a_t = torch.sqrt(re_t ** 2 + im_t ** 2 + 1e-8)          # [B,N,D]
        th_t = torch.atan2(im_t, re_t)                          # [B,N,D]
        # supervise 只在 batch=1 的 overfit 场景调用，squeeze(0) 让 [B,N,D]
        # 变 [N,D] 以便逐通道归一（真实批量化时应改成 [B,N,D] 版本）。
        if a_t.shape[0] == 1:
            a_t_s, th_t_s = a_t[0], th_t[0]                     # [N,D]
        else:
            raise ValueError(
                f"supervise 目前只支持 batch=1，收到 B={a_t.shape[0]}。"
                f"批量化需把下面的逐通道归一改成 [B,N,D] 形状。")

        if w_ph is None:
            w_ph = (a_t.detach() ** 2).clamp(min=1e-8)           # [B,N,D]

        # ---- 相位方向：dtheta 与 target 相位的加权圆均值一致性 ----
        # 用 sin/cos 双通道：把 dtheta 当作相位力方向，
        # 监督 (cos dtheta, sin dtheta) 与 target 相位单位向量对齐。
        # 这比"预测相位值"低维得多，且物理上就是「初始相位偏移」。
        # ★★ DC 通道必须排除（2026-10-16 实测定位，diag_init_snr.py）：
        #   实测 w_ph = a_tar² 的通道构成：
        #     8-ball:      ch0(DC) 占 82.6%，ch0/ch50 = 55.4x
        #     二次元表情包:  ch0(DC) 占 67.9%，ch0/ch50 = 51.9x
        #   更糟的是**空间一致性**（该位置相位方向的一致程度）：
        #     全通道加权平均:  src0=+0.0009  src1=-0.0356  ← ≈0 = 目标被平均成噪声
        #     排除 ch0 后:    src0=+0.2467  src1=-0.1022  ← 有信号了
        #   机制：DC 相位与其余 124 通道相位不一致，按 a² 加权平均时
        #   互相抵消 → 监督目标退化成空间白噪声，模型学不到任何东西。
        #   （与记忆铁律「DC 通道承载 97% 能量」「per-channel 必须」同源）
        #
        #   修法：相位监督只走 ch1~（去掉 DC），并**逐通道归一**让每个
        #   频率通道贡献可比，而不是被少数大能量通道支配。
        w_ch = (a_t_s.detach() ** 2).clamp(min=1e-8)[:, 1:]      # [N, D-1] 去 DC
        th_ch = th_t_s[:, 1:]
        # 逐通道按自身均值归一（保留空间变化，消掉通道间量级差）
        w_norm = w_ch / w_ch.mean(dim=0, keepdim=True).clamp(min=1e-8)
        sin_ch = w_norm * torch.sin(th_ch)                      # [N, D-1]
        cos_ch = w_norm * torch.cos(th_ch)

        # ★★ 第二十三死穴后一半（2026-10-18）：per-band 相位监督。
        #   旧写法把**所有通道压成一个方向向量** sin_t/cos_t [N]，
        #   却只给模型**一个标量** pd [B,N] 去匹配 ——
        #   标量在数学上匹配不了向量场的空间分布，这是维度错配。
        #   实测后果：cos 对齐恒为 −0.003，150 步纹丝不动。
        #
        #   修法：按频段分组，每组算自己的目标方向向量 [N]，
        #   与该组的 dtheta [B,N] 对齐 —— 维度匹配，梯度直达。
        G = int(pd_g.shape[-1])
        if G > 1:
            # 组号：c → w=c%25，按 w 线性分G 箱（与 _group_index 一致）
            c_idx = torch.arange(1, d_model, device=dtheta.device)
            w_idx = c_idx % 25
            edges = torch.linspace(0, 25, G + 1, device=dtheta.device)
            gid = torch.bucketize(w_idx.float(),
                                  edges[1:-1].float()).clamp(0, G - 1)
            losses_ph = []
            for g in range(G):
                m = (gid == g)
                if not bool(m.any()):
                    continue
                st = sin_ch[:, m].sum(dim=1)                      # [N]
                ct = cos_ch[:, m].sum(dim=1)
                nt = torch.sqrt(st ** 2 + ct ** 2).clamp(min=1e-6)
                st, ct = st / nt, ct / nt                          # 该组目标方向
                pg = pd_g[..., g]                                 # [B,N]
                sp, cp = torch.sin(pg), torch.cos(pg)
                losses_ph.append(
                    1.0 - (sp * ct + cp * st).mean(dim=1))# [B]
            loss_ph = torch.stack(losses_ph, 0).mean()
        else:
            sin_t = sin_ch.sum(dim=1)                             # [N]
            cos_t = cos_ch.sum(dim=1)
            norm_t = torch.sqrt(sin_t ** 2 + cos_t ** 2).clamp(min=1e-6)
            sin_t, cos_t = sin_t / norm_t, cos_t / norm_t
            sin_p, cos_p = torch.sin(pd), torch.cos(pd)
            cos_align = (cos_p * cos_t + sin_p * sin_t).mean(dim=1)
            loss_ph = (1.0 - cos_align).mean()

        # ================================================================
        # ★★★ 第二十七死穴修复（2026-10-19）：绝对相位对齐
        #   实测铁律：
        #     随机 θ0 与 target 的 cos = 0.00004（完全无关）
        #     若 dtheta = 0.5× 理想相位差 → cos = +0.4877
        #     实测 |dtheta| = 0.77，理想相位差尺度 ≈ π ⇒ **差 4 倍**
        #     ⇒ dtheta 搬不动随机初态，cos(θ,target) 12 层只走到 −0.0266
        #
        #   为什么上面那个 loss_ph 修不了：
        #     它只监督**方向场**（该组的 sin/cos 叠加后归一），
        #     对 |dtheta| 的**大小没有任何要求**——
        #     dtheta=0.1·理想差 和 dtheta=1.0·理想差 的 loss_ph 几乎相同。
        #     ⇒ 幅度不会自己长起来。
        #
        #   修法：直接监督 **dtheta ≈ (θ_tar − θ0)**，用相位差MSE。
        #     这会对幅度产生**真实压力**：只有幅度够大才能把差压下去。
        #     权重按 w_norm 逐通道归一，避免被大能量通道支配。
        # ================================================================
        loss_ph_abs = torch.zeros((), device=dtheta.device)
        if abs_w > 0 and theta0 is not None:
            # 把 per-band 的 dtheta 展开到 D 维（与 aether_gif_model 同口径：
            # 通道 c ↔ w = c % 25，按 w 线性分 G 箱）
            _G = int(dtheta.shape[-1])
            if _G > 1:
                _c = torch.arange(1, d_model, device=dtheta.device)
                _w = _c % 25
                _ed = torch.linspace(0, 25, _G + 1, device=dtheta.device)
                _gi = torch.bucketize(
                    _w.float(), _ed[1:-1].float()).clamp(0, _G - 1)
                _dD = torch.empty(dtheta.shape[0], dtheta.shape[1],
                                  d_model, device=dtheta.device)
                _dD[:, :, 0] = dtheta[..., 0]          # DC 沿用组 0
                for _g in range(_G):
                    _mm = _gi == _g
                    if bool(_mm.any()):
                        _dD[:, :, 1:][:, :, _mm] = \
                            dtheta[..., _g:_g + 1].expand(
                                -1, -1, int(_mm.sum()))
                dtheta_D = _dD
            else:
                dtheta_D = dtheta.expand(-1, -1, d_model)
            # 理想相位差：θ_tar − (θ0 + dθ)，逐通道按 w_norm 归一后 MSE
            # ★ 必须用 th_ch（= th_t_s[:, 1:]，**已去 DC**，[N,D-1]），
            #   不能用 th_t_s（[N,D]）—— 这是本条崩过一次的地方。
            #   w_norm 同为 [N, D-1]，三者必须同口径。
            _th0_d = theta0[0][:, 1:]
            _dth_d = dtheta_D[0][:, 1:]
            ideal = torch.remainder(
                th_ch - (_th0_d + _dth_d) + math.pi,
                2 * math.pi) - math.pi
            # ★★ clamp 上限（2026-10-19，phabs2k 退步的修复）：
            #   phabs2k 把 PH_ABS_W 从 3.0 提到 8.0，wcos 0.25→0.85、
            #   phase 1.99→0.40（指标暴涨），但**视觉退步成均匀灰白噪点**：
            #   ac1 虚高（靠全图偏灰）、sobel 爆到 37.5、PSNR 反降。
            #   根因：本项是 `w_norm·MSE`，**无上界**；权重 8 ⇒ 量级 ~80，
            #   直接压垮 amp（只剩 0.0064）等其余所有项。
            #   铁律：加权 loss 必须逐项量级对账，
            #   「指标暴涨 + 视觉退步」= 某项已压垮其余项。
            #
            #   修法：clamp 到 IDEAL_CLAMP。数学上仍允许把相位差
            #   从 π（随机）压到 0，但**梯度在达到目标前就饱和**，
            #   它只能贡献有限的压力，不会挤掉其他项。
            #
            # ★ 权重口径可选（2026-10-21，方向2：PH_ABS_W_MODE）
            #   实测病灶（fix_r3 四路交叉+ 口径矩阵）：
            #     相位误差 |pe| 中位 1.51 rad ≈ IDEAL_CLAMP(π/2)，
            #     且 **6 个素材×条件组合上恒为 1.51**（与条件无关）
            #     → 高频相位学不动，w_norm 逐通道归一把高频权重压没了。
            #   这里让**绝对相位项**用独立权重口径（loss_ph 的方向项不变）：
            #     norm（默认，旧）: w_norm = a_tar²/通道均值（DC 去后仍重尾）
            #     sqrt          : w = a_tar（缓和重尾）
            #     uniform       : w = 1（高频与低频同权，最大化高频梯度）
            #     log           : w = 1/(1+log1p a_tar)（反重尾，更压 DC）
            #     band          : per-band 内均匀（band 间按能量）
            #   量级对账铁律：本项仍必须配clamp，不能只加权重。
            _w_abs = _make_ph_abs_weight(w_ch, a_t_s)
            loss_ph_abs = (_w_abs * ideal.clamp(max=_IDEAL_CLAMP)
                           .pow(2)).mean()

        loss_ph = loss_ph + abs_w * loss_ph_abs

        # ---- 幅度方向：dlna 与 log1p(a_tar) 空间分布的相关 ----
        # 同样排除 DC（DC 占 82.6% 能量，会把空间形状压成一个常数）。
        # per-band 版：每组用自己的通道算形状目标，与该组 dlna 对齐。
        la_ch = torch.log1p(a_t_s.clamp(min=1e-8))[:, 1:]        # [N, D-1]
        la_c = la_ch - la_ch.mean(dim=1, keepdim=True)
        if G > 1:
            losses_amp = []
            for g in range(G):
                m = (gid == g)
                if not bool(m.any()):
                    continue
                stg = la_c[:, m].mean(dim=1)                       # [N]
                lpg = dl_g[..., g]                                 # [B,N]
                lpc = lpg - lpg.mean(dim=-1, keepdim=True)
                stg_c = stg - stg.mean()
                ca = ((lpc * stg_c).sum(dim=-1)
                      / (lpc.norm(dim=-1) * stg_c.norm() + 1e-6))
                losses_amp.append(1.0 - ca)
            loss_amp = torch.stack(losses_amp, 0).mean()
        else:
            shape_t = la_c.mean(dim=1)
            lp_c = lp - lp.mean()
            st_c = shape_t - shape_t.mean()
            cos_align = (lp_c * st_c).sum() / (lp_c.norm() * st_c.norm() + 1e-6)
            loss_amp = 1.0 - cos_align

        # ---- dtheta 幅度守卫：不让 tanh 冲进饱和区 ----
        # over = |dtheta| − guard_bot 的正部，二次惩罚（梯度线性过渡，
        # relu 一次会让梯度阶跃）。pd 幅度小于阈值时守卫恒为 0，
        # 完全不影响方向监督。
        over = (pd.abs() - guard_bot).clamp(min=0.0)               # [B,N]
        loss_guard = (over ** 2).mean()                            # 标量

        return loss_amp + loss_ph + guard_w * loss_guard + sep_w * loss_sep

    def extra_repr(self) -> str:
        return f"n_osc={self.n_osc}, phase_scale={self.phase_scale:.3f}, amp_scale={self.amp_scale}"
