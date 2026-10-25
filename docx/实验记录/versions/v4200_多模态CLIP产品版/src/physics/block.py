# -*- coding: utf-8 -*-
"""
物理块（Physical Block）：与 Transformer Block 对位的通用层。

单轮数据流（AetherMind §5，逐字照搬）：

  (a, θ) ──→ DriftField ──→ 漂移引导 g ──┐
                                          ↓
         ──→ KuramotoSync(κ, g, T) ──→ θ_new     （相位演化，Euler 积分）
                                          ↓
         ──→ StandingWaveFFN(ω_d) ──→ ffn_out    （振幅更新）
                                          ↓
         a_new = softplus(LayerNorm(a + ffn_out))（振幅非负 + 尺度稳定）
                                          ↓
         Thermodynamics(κ, T) ──→ {U, S, F, r}    （判据/诊断）

- 相位不做 LayerNorm（保持周期流形），只归一振幅
- 12 轮物理块串联，每轮的 t = i/(n_blocks-1) 驱动时间场
"""
import math
import os
import torch
import torch.nn as nn

import os as _os

# 相位残差激活（第二十一死穴修复开关，见 PhysicsBlock.forward 内注释）
#   clamp（默认，历史口径）| soft（tanh 软饱和，梯度不消失）
_PHASE_RES_ACT = _os.environ.get("PHASE_RES_ACT", "clamp").strip().lower()
if _PHASE_RES_ACT not in ("clamp", "soft", "soft2"):
    _PHASE_RES_ACT = "clamp"

# 幅度残差硬上限（第三十死穴修复，2026-10-19）
#   旧值硬编码 2.0（res = 2.0·tanh(log_res/2.0)）。实测证明它是破坏主源：
#   ffn.w2 范数 1.72→14.50 无约束长大，res_max 顶满 2.0，
#   关掉幅度 res 后 corr(log a) 0.478 → 0.895。
#   默认 0.5：每层 ≤ e^0.5=1.65 倍，12 层 e^6=403 倍，够表达真实动态。
#   传0 恢复旧的 2.0（用于对照实验）。
_RES_LIMIT = float(_os.environ.get("RES_LIMIT", "") or 0.5)
if _RES_LIMIT <= 0:
    _RES_LIMIT = 2.0

# FFN 输出层权重范数上限（同死穴配套约束）
#   w2 范数从 1.72 单调涨到 14.50，无结构性上限 ⇒ 灌巨量扰动。
#   默认 3.0（实测 corr 0.478→0.500 区间内最优）。传 0 关闭。
_W2_NORM_MAX = float(_os.environ.get("W2_NORM_MAX", "") or 3.0)

from .time_field import TimeField
from .kuramoto_sync import KuramotoSync, build_spacetime_index
from .standing_wave_ffn import StandingWaveFFN
from .drift_field import DriftField
from .thermodynamics import Thermodynamics


class PhysicsBlock(nn.Module):
    """单轮物理块：五算子各就位，零简化。"""

    def __init__(
        self,
        d_model: int = 64,
        ffn_hidden: int = 256,
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
        a_mu: torch.Tensor = None,
        n_modules: int = 0,
        mod_gamma: float = 0.15,
    ):
        super().__init__()
        self.drift = DriftField(d_model, n_prototypes,
                                n_modules=n_modules, mod_gamma=mod_gamma)
        self.kuramoto = KuramotoSync(lambda_t, lambda_s, temporal_window, dt)
        self.ffn = StandingWaveFFN(d_model, ffn_hidden, n_modes_t=n_modes_t,
                                   n_modes_xy=n_modes_xy, T_ref=T_ref, X_ref=X_ref, Y_ref=Y_ref)
        # ★★★ 第三十死穴配套约束（2026-10-19）：FFN 输出层权重范数上限 ★★★
        #   实测（diag_res_by_block.py）：w2 范数从层0 的 1.72 单调涨到
        #   层11 的 14.50（8.4×），res_std 涨 14×，末层 res 顶满 tanh 上限。
        #   权重只有 weight decay 压制，**没有结构性上限** ⇒
        #   后 4 层（梯度健康的那几层）往幅度里灌巨量扰动，把结构冲掉。
        #
        #   干预实验（diag_clip_ffn_w2.py，无需训练直接缩放）证实有效：
        #     w2≤6.0 → corr 0.4122 / w2≤3.0 → 0.4781 / w2≤1.72 → 0.5003
        #
        #   用**梯度投影**做硬约束：backward 结束时把 w₂ 的梯度按
        #   ‖w₂‖/上限 的比例缩放，等价于「每步后投影回上限」，
        #   **不改权重数值口径** ⇒ 加载历史 ckpt 完全兼容。
        #   （不用 spectral_norm：它对矩阵做幂迭代归一化，会改变
        #     数值口径，让全部历史 ckpt 失效。）
        #   W2_NORM_MAX=0 关闭该约束（对照实验用）。
        # ★★★ 约束实现改为「step 后权重投影」而非梯度 hook（2026-10-20）★★★
        #   上一版用 register_full_backward_hook 缩放 grad_output，有**两处根本错误**
        #   （fft4k 长训第 1 步就崩：'tuple' object has no attribute 'mul_'）：
        #     ① full_backward_hook 的 gout 是 tuple 且不可索引赋值，
        #        `gout.mul_()` 与 `gout[i] = _g*k` 都会抛错；
        #     ② 即使不崩，缩放 grad_output 也只是缩小当步更新量，
        #        **不能把已经超标（实测 14.50）的权重范数拉回上限**。
        #   有效做法来自 diag_clip_ffn_w2.py（直接缩放权重实测）：
        #     每次 optimizer.step() 之后，把 ffn.w2.weight 按
        #     ‖w2‖/上限 的比例投影回上限，corr(log a) 0.478 → 0.500，
        #     数值口径不变、历史 ckpt 完全兼容。
        #   由训练循环在 opt.step() 后显式调用 project_w2()。
        self._w2_norm_max = _W2_NORM_MAX
        self.thermo = Thermodynamics()
        # 振幅 OU 回复力参数（Ornstein-Uhlenbeck，层级/形状分离版）：
        #   m_new = m + γ_lvl·(μ − m)          层级几何收敛到 OU 锚
        #   δ_new = (1−γ_shape)·δ + res        形状阻尼累积残差
        # FFT 幅度在对数空间近似对数正态、演化应有界；纯累加残差在 12 层
        # 堆叠下失控（实测 a_out 爆到 100×target）。γ_lvl 回复率把通道层级 m
        # 拉向可学习先验 μ，既允许偏离又防漂移 —— 统计物理经典稳定化机制
        # （布朗运动 + 回复力，Fokker-Planck 稳态有界）。γ_shape 独立控制形状
        # 记忆：0 = 完全保留上一块的位置结构，1 = 每块从零重画。
        # γ_lvl / γ_shape 是架构超参数（收敛速率与形状记忆），不是可学习参数。
        # 之前让它们可训练（diagnosis 前），Adam 会走"把 γ 调小以保留层级残留"
        # 的捷径——γ→0.04 时 Π(1−γ)=0.61，初值层级洗不掉，μ 怎么学都压不住；
        # 这正是层级/形状单 γ 绑死时代留下的病根（2500 步 γ 自学到 0.211~0.04）。
        # 拆成两个常量后：γ_lvl=0.5 ⇒ 12 块后层级收敛 0.5^12≈2e-4（初值彻底洗掉，
        # 终态层级 = μ_c，与数据无关）；γ_shape=0.1 ⇒ 形状保留 0.9^12≈0.28 的
        # 上一块位置结构 + 累加 12 层 res，FFN 有足够深度去塑形。固定后梯度压力
        # 全部压到 μ_c（学谱包络）和 FFN（学位置级细节），职责正交、无捷径可抄。
        # 【第十三死穴修复回滚】v4200 曾退化回单一可训练 gamma（方案 D 形态），
        # 实测 block4~10 的 a_gamma 学成负数（-0.006~-0.032），OU 回复力变发散力；
        # 且单 γ 同时控制层级收敛与形状衰减（第九死穴）。恢复项目根已验证的
        # 层级/形状分离 + 固定 γ 形态：γ_lvl/γ_shape 是架构超参数，不可训练。
        # [BASELINE 2500] 单一可训练 gamma（方案 D 形态）
        self.a_gamma = nn.Parameter(torch.tensor(0.3))
        # OU 平衡点必须 per-channel（第七个死穴），且必须 block 间共享（第十个死穴）。
        # 【per-channel】FFT 振幅谱是重尾的——实测同一 GIF 每通道振幅均值跨 727 倍
        # （低频通道 0.573、中高频通道 0.0247，DC 通道达 7.84）。此前 μ 是标量
        # log(0.2) 把所有通道拉向同一 a=0.2：中高频被抬高 ~8x、低频被压低，整张谱
        # 被压平（用户看到的"有轨迹没细节"，帧内对比度只有 target 的 1/3）。
        # 【block 间共享】150 步实验证明：11 个块各一个 μ 时，只有最后 1~2 块的 μ
        # 能收到有效梯度（第 i 块对终态的贡献 ∝ γ_lvl·(1−γ_lvl)^(11−i)，前面块指数
        # 衰减），前面块的 μ 学不动、还把早期层级来回拉回初值 −3.0，谱形状 span 只
        # 到 0.1（target 6.589）。一份共享 μ_c 让 12 次递推都在靠近同一个平衡点，
        # 梯度沿线累加（Σγ_lvl(1−γ_lvl)^k≈1），谱包络成为一条直通路径。
        # 【初值】target 逐通道 log 均值 −3.12（几何均值 ≈0.044），初值 log(0.05)
        # =−3.0 贴近靶心。注意 OU 作用在 log 域，相关量是几何均值而非算术均值 0.195。
        # 【初值，2026-10-11 第十三死穴修正】μ 现在是 **log1p 域**的谱包络，
        # 初值必须用 log1p(0.05)=0.0488 而不是 log(0.05)=-3.0。
        # 用错量纲会让 μ 从第一步就偏离目标（实测差 3.05 个单位）。
        # target 逐通道 log1p 均值：ch0=2.09ch55=0.029，跨度仅 2.06。
        if a_mu is None:
            a_mu = torch.full((d_model,), math.log(0.2))
        self.a_mu = a_mu if isinstance(a_mu, nn.Parameter) else nn.Parameter(a_mu)

    def project_w2(self):
        """step 后把 FFN 输出层权重投影回 ‖w2‖ ≤ W2_NORM_MAX（第三十死穴约束）。

        直接缩放权重（非梯度 hook）：数值口径不变、历史 ckpt 兼容。
        每步后由训练循环显式调用；W2_NORM_MAX<=0 时为空操作。
        """
        _mx = getattr(self, "_w2_norm_max", 0.0)
        if _mx is None or _mx <= 0:
            return
        w = self.ffn.w2.weight
        with torch.no_grad():
            n = float(w.norm())
            if n > _mx and n > 0.0:
                w.mul_(_mx / n)

    def forward(self, a, theta, omega, t_coord, s_coord, kappa, T, omega_d,
                targets=None, protos=None, mod_id=None, cond_anchor=None,
                anchor_weight: float = 0.3, mu_bias=None):
        """
        a, theta: [B,N,D]
        omega:   [B,N,D] 固有频率（文本条件注入）
        kappa, T, omega_d: 标量（时间场）
        targets: 漂移场吸引中心（训练时给真实场）
        protos:  条件驱动的吸引中心 [B,P,2D]（生成通路，见 cond_prototypes.py）。
                 与 targets 同时给时 protos 优先。
        mod_id:  [N] 可选，稀疏耦合的振子模块号（B 方案，透传给 DriftField）。
        cond_anchor: [B,N] 可选，条件相位锚（A 方案，透传给 DriftField）。
        """
        # 漂移引导
        g = self.drift(a, theta, targets, protos, mod_id=mod_id,
                       cond_anchor=cond_anchor, anchor_weight=anchor_weight)

        # 相位演化（Kuramoto，Euler dt=0.1，折叠 [-π,π]）
        theta_new, sync_diag = self.kuramoto(
            theta, a, omega, g, kappa, T, t_coord, s_coord
        )

        # 共振-驻波 FFN：同时更新振幅与相位
        # 物理上共振传递函数是复值 |H(ω)|·e^{iφ(ω)}，既改幅度也改相位；此前 FFN
        # 只输出振幅残差，相位演化完全由零参数的 Kuramoto/DriftField 决定 ——
        # 梯度诊断（diag_grad.py）证实 drift_field/kuramoto/time_field 梯度恒为 0
        # （零参数结构先验），相位只剩 ω 偏移与 κ 缩放两个弱自由度，200 步
        # wcos 振荡不收敛。给 FFN 加相位输出头后，相位有了 per-position、
        # per-channel 的可学习驱动力。
        ffn_out = self.ffn(a, theta_new, omega_d, t_coord, s_coord)  # [B,N,2D]
        log_res, ph_res = ffn_out.chunk(2, dim=-1)

        # 振幅：log 域乘性残差 + OU 回复力（层级/形状正交分离版）
        # FFT 幅度跨数量级重尾（mean≈0.2、std/mean≈5.75），log 域乘性残差让
        # 梯度在对数尺度均匀（0.75→0.001 只是 ln 尺度 −6.6 的位移）。但纯累加
        # 在 12 层堆叠下失控（a_out 爆到 100×target，decode 归一化被爆炸值
        # 主导、frame-diff 塌到 0.02）。OU 回复力把 log a 拉回可学习先验 μ：
        #   log a_new = log a + res − γ·(log a − μ)
        # tanh 软饱和（每层 ≤e²）+ OU 双重护栏，res 原点附近线性不压缩弱信号。
        # ==================================================================
        # 【第十三死穴：架构在 log 域递推 μ，loss 在 log1p 域——量纲错位】
        # 症状（500 步实测）：μ 报 -1.21，模型实际层级 +0.34，μ 震荡 4~16。
        #   block.py 递推  m_new = m + γ_lvl·(μ − m)，其中 m = mean_N(log a)
        #   => a_mu 的稳态目标是 mean_N(log a)，而 loss_amp 惩罚的是
        #      mean_N(log1p(a))，两套量纲从第一步就不同。
        #   实测 log 域真值 ch0=+1.94 / ch25=-5.52，log1p 域真值 ch0=+2.09 /
        #      ch25=+0.34 —— μ 拿着 log 域的结构去追 log1p 域的目标，
        #      永远差一个变换，只能靠震荡硬凑。
        # 根因：log 在 a→0 处无界（第十二死穴）。但真正解法不是改 loss，
        #   而是让**架构和 loss 用同一个有界变换**。
        #
        # 【修法】层级递推整体搬到 log1p 域：
        #   L = log1p(a)   （a→0 时 L→0，梯度=1，有界）
        #   m = mean_N(L)  （层级）
        #   m_new = m + γ_lvl·(μ − m)，μ 即 log1p 域的谱包络
        #   a_new = expm1(L_new)（exp溢出保护见下）
        # 这样 μ 的稳态 = mean_N(log1p(a))，与 loss_amp 严格同量纲。
        # 初值 μ = log1p(0.05) = 0.0488，target ch0 log1p=2.09/ch55=0.029，
        # 跨度仅 2.06（有界可学），不再需要 ×80 这种激进放大。
        # ==================================================================
        # ★★★ 第三十死穴修复（2026-10-19）：残差幅度硬上限 ★★★
        #   实测（diag_res_by_block.py，真权重逐块）：
        #     ffn.w2 范数从层0 的 1.72 **单调涨到**层11 的 14.50（8.4×），
        #     res_std 从 0.014涨到 0.196（14×），末层 res_max 顶满 tanh 上限 2.0。
        #   后果（diag_ablate_parts.py 消融）：关掉幅度 res 后
        #     corr(log a) 0.4781 → **0.8949**，证明 res 是最大破坏源。
        #
        #   旧上限 2.0意味着每层可加 e^2=7.4倍幅度，12 层累积 e^24 ≈ 2.6e10。
        #   现改为 RES_LIMIT（环境变量，默认 0.5）：
        #     每层 ≤ e^0.5 = 1.65 倍，12 层 e^6 = 403 倍——仍够表达真实动态，
        #     但灌不进「把结构冲掉」的量级。
        #   ★ 不改递推公式，只改res 的值域上界（纯输出端约束，
        #     与 ph_res 的 clamp(±1.5) 同一性质）。
        res = _RES_LIMIT * torch.tanh(log_res / _RES_LIMIT)
        # 【第九个死穴：层级与形状必须分离（单 γ 的结构性矛盾）】
        # 单 γ 的 OU 递推 log a_new = (1−γ)log a + γμ + res 里，同一个 γ 同时
        # 控制 ① 层级 m 向 μ 的收敛速率 ② 形状 δ 的衰减速率——两者是同一个数，
        # 无法兼得。诊断（diag_amp.py，两个版本）：
        #   - γ 自学到 0.04：初值层级残留 61%（Π(1−γ)=0.61），μ 压不住初值，
        #     终态层级仍偏 1.85×；μ 的 span 只学到 0.05~0.10（target 6.589）
        #   - γ=0.3：层级 12 块收敛 98.6%（0.7^12=0.014）但形状也被压到 1.4%，
        #     res 白学（第七死穴修复前的实验同样印证）
        # 拆成两个独立 γ，让层级与形状的动力学正交：
        #   m_new   = m + γ_lvl·(μ − m)            层级几何收敛到 OU 锚
        #   δ_new   = (1−γ_shape)·δ + res          形状阻尼累积 FFN 残差
        # 验证（diag_amp.py）：mean_pos(δ_new) = (1−γ_shape)·0 + 0 = 0 精确成立，
        # 所以 mean_pos(log a_new) = m_new 精确成立——层级 loss 对 res 的梯度
        # 严格为 0，只训练 μ_c 与 γ_lvl；形状 loss 只经 res 训练 FFN。监督正交。
        # [BASELINE 2500] 单γ OU 回复力（方案 D 形态）：
        #   log a_new = log a + res − γ·(log a − μ)
        # 同一个 γ 同时控制层级收敛与形状衰减，这是方案 D 的已知缺陷
        # （第九死穴），但它是当前已知最优视觉效果的版本，作为基线保留。
        log_a = torch.log(a.clamp(min=1e-8))
        # ★ 2026-10-25：μ 条件调制（解决「所有素材 sat 锁死成常数」）
        #   原式 log_a_new = log_a + res − γ·(log_a − μ_c)，μ_c 是**全样本共享**的单一
        #   d_model 向量。γ 初始 0.3，12 层后 (1−γ)^12≈0.014；γ 学到 0.5 时仅 2.4e-4。
        #   即：无论条件注入什么初值 m_0，终态层级都收敛到同一个 μ_c
        #   → 所有素材输出相同的振幅谱 → sat 固定 ≈6.5（实测 2 素材与 22 素材均如此）。
        #   修法：让平衡点本身随条件变，mu_bias = head(cond_vec) [B, D]。
        #   这样即使收敛到平衡点，不同条件也有不同 μ，颜色才能跟随条件。
        _mu = self.a_mu.view(1, 1, -1)
        if mu_bias is not None:
            _mu = _mu + mu_bias.view(mu_bias.shape[0], 1, -1)
        log_a_new = log_a + res - self.a_gamma * (log_a - _mu)
        log_a_new = log_a_new.clamp(-20.0, 5.0)
        a_new = torch.exp(log_a_new)

        # 相位：共振相位响应，残差叠加。不折叠——演化链保持连续，
        # remainder 只在 loss 处做（见 kuramoto_sync 同名注释）。
        # clamp ±1.5：实测 ±π 时帧间相位断裂（frame-diff 14.7）、±0.5 时
        # 修正力被锁死（cos 钉在 0.416 不动，Kuramoto 吸引子拉不回来）。
        # ±1.5 与 Kuramoto 每步 0.8 rad 同量级，12 块预算 18 rad。
        #
        # ================== 第二十一死穴（2026-10-18 定位）==================
        # 【症状】相位永远学不动。隔离实验（exp_phase_isolated.py）：
        #   P0 旧 loss_ph 50 步 old_ph 2.7653→2.7515，band_cos ≈ +0.04
        #   P2 新分组监督（高频梯度给到 14~21 倍）25~50 步 new_ph 0.9535→0.9505，
        #      band_cos 仍 ≈ +0.045，**给多少梯度都不动**。
        # 【根因】diag_phase_grad.py 实测 |ph_res|max 逐 block：
        #   b0~b7 = 0.03~0.79（正常），b8=1.28，b9=3.84，b10=13.85，b11=**37.01**
        #   clamp 阈值 1.5 → 末端三层饱和率 0.25% / 0.45%。
        #   **深层 block 的相位修正量全部被 clamp 削掉，clamp 的次梯度为 0
        #   → 相位梯度在最后三层彻底断流。**
        #   而深层 block 恰恰是最后也是最关键的相位修正机会。
        # 【为什么之前没人发现】loss 侧梯度已经很小，看 loss 数值不动
        #   会误判成「学不动就调权重」，而真因在架构的硬 clamp 上。
        #   与第十二死穴（log 无界）、铁律 8（有界激活必饱和）同源。
        # 【修法】PHASE_RES_ACT=soft 时用 tanh 软饱和：
        #   同值域 (−1.5, 1.5)，但梯度 1.5(1−tanh²) 永不消失
        #   → 饱和后仍可被拉回线性区（与 arctan 攻破铁律 8 同一思路）。
        #   clamp 保留为默认以维持历史ckpt 口径一致。
        # 【但 tanh 单独不够（实测）】|ph_res|max 达 39.5 → tanh(x/1.5)=1.0000，
        #   梯度 1.5(1−tanh²(26.3)) ≈ 1e-24，仍等于断流。根因是 ph_res
        #   **逐 block 指数放大**（b7=0.80 → b11=39.5，每层 ~1.8×），
        #   12 层放大 1.8^12 ≈ 200×。所以必须**每层重归一化**：
        #   soft2 模式先按张量自身的 RMS 归一化再 tanh：
        #     r = ph_res / (rms(ph_res) + eps) * 1.5
        #     theta += 1.5·tanh(r/1.5) = 1.5·tanh(ph_res/rms)
        #   这样输入恒在 O(1)，梯度 ~0.75，**与量级无关**。
        #   代价：丢失「本层想改多少」的绝对幅度信息 ——但相位残差
        #   本来就是「方向修正」，量级由 Kuramoto 的 κ 决定，更合理。
        # =====================================================================
        if _PHASE_RES_ACT == "soft":
            _k = 1.5
            theta_new = theta_new + _k * torch.tanh(ph_res / _k)
        elif _PHASE_RES_ACT == "soft2":
            # 每层 RMS 归一化 + 软饱和：梯度与 ph_res 量级解耦
            _k = 1.5
            _rms = ph_res.pow(2).mean(dim=-1, keepdim=True).add(1e-8).sqrt()
            theta_new = theta_new + _k * torch.tanh(ph_res / _rms)
        else:
            theta_new = theta_new + ph_res.clamp(-1.5, 1.5)

        return a_new, theta_new, sync_diag


class PMNNCore(nn.Module):
    """PMNN 物理核心：R 轮物理块堆叠 = Transformer 的层堆叠。

    每块共享时空坐标，时间场沿块编号给出 κ(t)/T(t)/ω_d(t)。
    """

    def __init__(
        self,
        d_model: int = 64,
        n_blocks: int = 12,
        ffn_hidden: int = 256,
        lambda_t: float = 0.5,
        lambda_s: float = 3.0,
        temporal_window: int = 3,
        dt: float = 0.1,
        n_prototypes: int = 16,
        T_ref: float = 20.0,
        X_ref: float = 32.0,
        Y_ref: float = 32.0,
        n_modules: int = 0,
        mod_gamma: float = 0.15,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.time_field = TimeField(n_blocks=n_blocks)
        # ★★★ 回归根因（2026-10-21）：v4200 曾退化回每 block 独立 a_mu。
        #   恢复项目根：顶层共享一份 a_mu 供 12 块引用（第十个死穴，见 PhysicsBlock 注释）。
        self.blocks = nn.ModuleList(
            [
                PhysicsBlock(
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
                    a_mu=None,
                    n_modules=n_modules,
                    mod_gamma=mod_gamma,
                )
                for _ in range(n_blocks)
            ]
        )

    def project_w2(self):
        """对全部 block 执行 FFN 输出层权重投影（第三十死穴约束）。"""
        for blk in self.blocks:
            blk.project_w2()

    def forward(self, a, theta, omega, t_coord, s_coord, targets=None, return_diag=False,
                protos=None, mod_id=None, cond_anchor=None, anchor_weight: float = 0.3,
                mu_bias=None):
        """
        a, theta: [B,N,D]
        omega:   [B,N,D] 固有频率（由 Qwen 文本条件经物理调制给出）
        t_coord: [N] 数据时间坐标
        s_coord: [N,2] 空间坐标
        targets: 漂移场吸引中心（训练时）
        protos:  条件驱动的吸引中心（A/B 方案：与 targets 同时给时 protos 优先）
        mod_id:  [N] 可选，稀疏耦合模块号（B 方案）
        cond_anchor: [B,N] 可选，条件相位锚（A 方案）
        """
        diags = []
        dev = a.device
        # ★ 梯度检查点（activation checkpointing）：用重算换显存。
        #   本架构 12 个物理块每块的中间激活都很大，而 Transformer 那种
        #   fp16 混合精度在这里**不能用**（相位是周期量、振幅跨 5~6 个数量级，
        #   fp16 会把高频系数直接下溢成 0）。所以省显存只能靠重算，不能靠降精度。
        #   USE_CKPT=1 开启；训练态且需要梯度时生效，推理/Eval 自动跳过。
        _use_ckpt = os.environ.get("USE_CKPT", "0") == "1"
        for i, blk in enumerate(self.blocks):
            tf = self.time_field.forward(i, dev)
            _kw = dict(kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
                       targets=targets, protos=protos, mod_id=mod_id,
                       cond_anchor=cond_anchor, anchor_weight=anchor_weight,
                       mu_bias=mu_bias)
            if _use_ckpt and self.training and torch.is_grad_enabled() \
                    and (a.requires_grad or theta.requires_grad):
                from torch.utils.checkpoint import checkpoint as _ck
                a, theta, d = _ck(blk, a, theta, omega, t_coord, s_coord,
                                  use_reentrant=False, **_kw)
            else:
                a, theta, d = blk(a, theta, omega, t_coord, s_coord, **_kw)
            if return_diag:
                d["layer"] = i
                d["kappa"] = tf["kappa"].detach()
                d["T"] = tf["T"].detach()
                diags.append(d)

        # 末层热力学诊断
        t_steps = int(t_coord.max().item()) + 1
        hw = a.shape[1] // t_steps
        thermo_diag = self.blocks[-1].thermo(a, theta,
                                             self.time_field.forward(self.n_blocks - 1, dev)["T"],
                                             t_steps, hw)
        if return_diag:
            return a, theta, {"per_block": diags, "thermo": thermo_diag}
        return a, theta
