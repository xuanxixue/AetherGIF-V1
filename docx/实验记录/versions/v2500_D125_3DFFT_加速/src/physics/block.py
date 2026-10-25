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
import torch
import torch.nn as nn

from .time_field import TimeField
from .kuramoto_sync import KuramotoSync, build_spacetime_index
from .standing_wave_ffn import StandingWaveFFN
from .drift_field import DriftField
from .thermodynamics import Thermodynamics


class PhysicsBlock(nn.Module):
    """单轮物理块：五算子各就位，零简化。"""

    def __init__(
        self,
        d_model: int = 125,
        ffn_hidden: int = 512,
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
    ):
        super().__init__()
        self.drift = DriftField(d_model, n_prototypes, use_checkpoint=True, chunk=2048)
        self.kuramoto = KuramotoSync(lambda_t, lambda_s, temporal_window, dt)
        self.ffn = StandingWaveFFN(d_model, ffn_hidden, n_modes_t=n_modes_t,
                                   n_modes_xy=n_modes_xy, T_ref=T_ref, X_ref=X_ref, Y_ref=Y_ref)
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

    def forward(self, a, theta, omega, t_coord, s_coord, kappa, T, omega_d,
                targets=None):
        """
        a, theta: [B,N,D]
        omega:   [B,N,D] 固有频率（文本条件注入）
        kappa, T, omega_d: 标量（时间场）
        targets: 漂移场吸引中心（训练时给）
        """
        # 漂移引导
        g = self.drift(a, theta, targets)

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
        log_a = torch.log(a.clamp(min=1e-8))
        res = 2.0 * torch.tanh(log_res / 2.0)
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
        log_a_new = log_a + res - self.a_gamma * (log_a - self.a_mu.view(1, 1, -1))
        log_a_new = log_a_new.clamp(-20.0, 5.0)
        a_new = torch.exp(log_a_new)

        # 相位：共振相位响应，残差叠加。不折叠——演化链保持连续，
        # remainder 只在 loss 处做（见 kuramoto_sync 同名注释）。
        # clamp ±1.5：实测 ±π 时帧间相位断裂（frame-diff 14.7）、±0.5 时
        # 修正力被锁死（cos 钉在 0.416 不动，Kuramoto 吸引子拉不回来）。
        # ±1.5 与 Kuramoto 每步 0.8 rad 同量级，12 块预算 18 rad。
        theta_new = theta_new + ph_res.clamp(-1.5, 1.5)

        return a_new, theta_new, sync_diag


class PMNNCore(nn.Module):
    """PMNN 物理核心：R 轮物理块堆叠 = Transformer 的层堆叠。

    每块共享时空坐标，时间场沿块编号给出 κ(t)/T(t)/ω_d(t)。
    """

    def __init__(
        self,
        d_model: int = 125,
        n_blocks: int = 12,
        ffn_hidden: int = 512,
        lambda_t: float = 0.5,
        lambda_s: float = 3.0,
        temporal_window: int = 3,
        dt: float = 0.1,
        n_prototypes: int = 16,
        T_ref: float = 20.0,
        X_ref: float = 32.0,
        Y_ref: float = 32.0,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.time_field = TimeField(n_blocks=n_blocks)
        # [BASELINE 2500] 每 block 各自持有独立的 a_mu（方案 D 形态）。
        # 基线不使用共享 μ，也不注册多余的顶层 a_mu 参数 ——
        # 参数量 1,135,658 是基线的身份指纹，改动此行必须重新核对该数字。
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
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, a, theta, omega, t_coord, s_coord, targets=None, return_diag=False):
        """
        a, theta: [B,N,D]
        omega:   [B,N,D] 固有频率（由 Qwen 文本条件经物理调制给出）
        t_coord: [N] 数据时间坐标
        s_coord: [N,2] 空间坐标
        targets: 漂移场吸引中心（训练时）
        """
        diags = []
        dev = a.device
        for i, blk in enumerate(self.blocks):
            tf = self.time_field.forward(i, dev)
            a, theta, d = blk(
                a, theta, omega, t_coord, s_coord,
                kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
                targets=targets,
            )
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
