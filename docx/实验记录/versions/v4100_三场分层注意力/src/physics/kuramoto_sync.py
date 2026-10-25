# -*- coding: utf-8 -*-
"""
§4.2 KuramotoSync —— 相位同步（替代注意力）

状态方程（AetherMind §4.2，逐字照搬）：

  dθ_i/dt = ω_i + Σ_j W_ij · sin(θ_j - θ_i) + κ(t)·g_i + η(t)

四条耦合通路：

  (1) 平均场序参量耦合（全局，秩-1，无参数）
      Σ_j sin(θ_j-θ_i) → r·sin(ψ-θ_i) = (r sinψ)cosθ_i - (r cosψ)sinθ_i
      振幅加权 + 归一化口径修正：分母必须用 Σw_i，不能是 N（否则 r 虚高 >1）

  (2) 低秩/结构化耦合矩阵 W
      本项目用时空 Toeplitz 结构（零参数物理先验）：
      W_ij = exp(-λ_t·|t_i-t_j| - λ_s·||p_i-p_j||²)
      时间衰减给光锥因果性，空间高斯衰减给局部性。

  (3) 漂移引导项 κ(t)·g_i，g 由 DriftField 给出。

  (4) 朗之万噪声 η ~ N(0, 2T·Δt)（默认关），涨落-耗散定理。

数值积分：Euler，步长 dt=0.1；积分后折叠回 [-π,π]。
"""
import math
import torch
import torch.nn as nn


def build_spacetime_index(t_steps: int, h: int, w: int, device, dtype=torch.float32):
    """构造时空振子位置索引。

    返回:
      t_coord: [N] 数据时间坐标（振子网格时间步，0..t_steps-1）
      s_coord: [N,2] 空间坐标（x,y），范围归一化到 [0,1]
      N = t_steps * h * w
    """
    tt = torch.arange(t_steps, device=device, dtype=dtype)
    yy = torch.arange(h, device=device, dtype=dtype)
    xx = torch.arange(w, device=device, dtype=dtype)
    # meshgrid 顺序: t, y, x
    T, Y, X = torch.meshgrid(tt, yy, xx, indexing="ij")
    t_coord = T.reshape(-1)
    s_coord = torch.stack([X.reshape(-1) / max(w - 1, 1), Y.reshape(-1) / max(h - 1, 1)], dim=-1)
    return t_coord, s_coord


class SpacetimeToeplitzCoupling(nn.Module):
    """时空 Toeplitz 耦合矩阵 W（零参数，纯物理结构先验）。

    W_ij = exp(-λ_t·|t_i - t_j| - λ_s·||p_i - p_j||²)

    - λ_t: 时间衰减率。跨帧距离衰减 → 光锥因果性
    - λ_s: 空间衰减率。同帧内空间距离高斯衰减 → 局部性

    复杂度：不物化 N×N 矩阵（N 可达 20480，N²=4e8 爆显存），
    而是把 W 的作用拆成「时间维 cumsum 因果前缀和 × 空间维局部窗」两步，
    与 AetherMind §4.2(2) 的 cumsum 并行加速同源。

    但注意：sin(θ_j - θ_i) 的耦合需要 W 逐对作用。为兼顾物理正确性与显存，
    采用「分块物化 + 流式累加」：把振子按时间步分组，只在时间局部窗内物化。
    """

    def __init__(self, lambda_t: float = 0.5, lambda_s: float = 3.0, temporal_window: int = 3):
        super().__init__()
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
        self.temporal_window = temporal_window  # 时间局部窗（因果：只看前面 + 窗内后面）

    def apply_coupling(self, theta: torch.Tensor, t_coord: torch.Tensor, s_coord: torch.Tensor,
                       couple_g: torch.Tensor = None):
        """计算 Σ_j W_ij · sin(θ_j - θ_i)。

        输入:
          theta: [B, N, D] 相位
          t_coord: [N] 数据时间坐标（整数步）
          s_coord: [N,2] 归一化空间坐标
        输出:
          coupling: [B, N, D]

        实现：按时间步分块，在 temporal_window 内逐对计算。
        N 个振子被分成 T_step 组（每组 h*w=1024 个），
        每次只物化 (T_win × 1024) × 1024 的子矩阵，显存可控。
        """
        B, N, D = theta.shape
        t_steps = int(t_coord.max().item()) + 1
        hw = N // t_steps

        # 时间分组索引
        t_int = t_coord.long()  # [N]
        # 排序后的组内顺序：t, y, x 已经是有序的（build_spacetime_index 保证）
        # theta_view: [B, T, hw, D]
        theta_view = theta.reshape(B, t_steps, hw, D)
        s_view = s_coord.reshape(t_steps, hw, 2)

        # 三场门控：couple_g ∈ [D]，1=完全参与耦合，0=外层场噪声被隔离。
        # 门控只作用于「源端 j」的 sinθ_j / cosθ_j（被耦合进来的那一方），
        # 目标端 i 的原幅不变 —— 否则会连 i 的响应一起缩，物理语义错位。
        if couple_g is not None:
            gate = couple_g.to(theta.dtype)
        else:
            gate = torch.ones(D, device=theta.device, dtype=theta.dtype)

        out = torch.zeros_like(theta)
        W = self.temporal_window
        # 空间尺度对齐：s_coord 归一化到 [0,1]，而 λ_s 是按格子单位标定的
        # 衰减率。若直接用归一化距离，32×32 网格相邻间距仅 1/31，
        # exp(−λ_s·(1/31)²)≈1，耦合矩阵退化为全 1、行和≈1000，正磁场
        # 炸到 O(10³)，相位每块转数十圈，漂移场方向信息被完全抹掉。
        # 乘 (grid−1)² 换算回格子单位，λ_s 语义恢复"格子单位的衰减率"。
        grid = int(round(hw ** 0.5))
        s_scale = float(max(grid - 1, 1)) ** 2

        for dt in range(-0, W + 1):
            # dt=0: 帧内耦合（含自身，sin(0)=0 自然无自耦）
            # dt>0: 当前帧 t 与未来帧 t+dt 的耦合（因果：本帧受未来影响，因为生成时整场联合演化）
            # 注意：GIF 生成是整场联合演化，不存在严格因果遮罩，时间衰减本身就是因果性来源
            if dt == 0:
                # 帧内：全物化 hw×hw（1024×1024 = 1M 元素，可接受）
                ref = theta_view
                # 空间距离矩阵 [hw, hw]
                diff = s_view[0].unsqueeze(0) - s_view[0].unsqueeze(1)  # [hw,hw,2]
                spatial_W = torch.exp(-self.lambda_s * s_scale * (diff ** 2).sum(-1))  # [hw,hw]
                # sin(θ_j - θ_i) = sinθ_j cosθ_i - cosθ_j sinθ_i，分头计算
                # 源端 j（门控）、目标端 i（原幅）
                sin_j = torch.sin(ref) * gate  # [B,T,hw,D]
                cos_j = torch.cos(ref) * gate
                sin_i = torch.sin(ref)
                cos_i = torch.cos(ref)
                Ws = spatial_W.to(theta.dtype)  # [hw,hw]
                ws_sum = torch.einsum("ij,btjd->btid", Ws, sin_j)  # [B,T,hw,D]
                wc_sum = torch.einsum("ij,btjd->btid", Ws, cos_j)
                contrib = cos_i * ws_sum - sin_i * wc_sum
                out += contrib.reshape(B, N, D)
            else:
                # 跨帧：t 与 t+dt
                if t_steps <= dt:
                    continue
                # 时间衰减权重（标量，按 dt）
                w_t = math.exp(-self.lambda_t * dt)
                # 空间仍用局部衰减
                diff = s_view[0].unsqueeze(0) - s_view[0].unsqueeze(1)
                spatial_W = torch.exp(-self.lambda_s * s_scale * (diff ** 2).sum(-1))
                Ws = (w_t * spatial_W).to(theta.dtype)
                # 未来帧 / 当前帧分离正余弦，避免「同名张量两用」误门控目标端 i
                sin_fut = torch.sin(theta_view[:, dt:])              # 未来帧
                cos_fut = torch.cos(theta_view[:, dt:])
                sin_cur = torch.sin(theta_view[:, : t_steps - dt])   # 当前帧
                cos_cur = torch.cos(theta_view[:, : t_steps - dt])
                # 正向：j=未来帧（门控），i=当前帧（原幅）
                ws_sum = torch.einsum("ij,btjd->btid", Ws, sin_fut * gate)
                wc_sum = torch.einsum("ij,btjd->btid", Ws, cos_fut * gate)
                contrib = cos_cur * ws_sum - sin_cur * wc_sum
                out[:, : (t_steps - dt) * hw] += contrib.reshape(B, -1, D)
                # 反向（对称）：j=当前帧（门控），i=未来帧（原幅）
                ws_sum_r = torch.einsum("ij,btjd->btid", Ws, sin_cur * gate)
                wc_sum_r = torch.einsum("ij,btjd->btid", Ws, cos_cur * gate)
                contrib_r = cos_fut * ws_sum_r - sin_fut * wc_sum_r
                out[:, dt * hw:] += contrib_r.reshape(B, -1, D)

        return out


class MeanFieldCoupling(nn.Module):
    """(1) 平均场序参量耦合（全局，秩-1，无参数）。

    Σ_j sin(θ_j - θ_i) → r·sin(ψ - θ_i)
                      = (r sinψ)·cosθ_i - (r cosψ)·sinθ_i

    归一化口径修正（重要）：分母必须用 w 求和，不能是 N。
    旧实现分子 w 加权、分母用 N，当 w=softplus(a)>1 时 r 虚高甚至 >1
    （理论 r∈[0,1]）。修正后 r 严格 ∈[0,1]。
    """

    @staticmethod
    def forward(a: torch.Tensor, theta: torch.Tensor):
        """
        a:     [B, N, D] 振幅
        theta: [B, N, D] 相位
        返回 coupling: [B, N, D]，以及诊断 r, psi
        """
        w = torch.nn.functional.softplus(a)  # [B,N,D]
        # 分母必须用 w 求和（口径修正）
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B,1,D]
        rcos = (w * torch.cos(theta)).sum(dim=1, keepdim=True) / w_sum  # [B,1,D]
        rsin = (w * torch.sin(theta)).sum(dim=1, keepdim=True) / w_sum  # [B,1,D]
        r = torch.sqrt(rcos ** 2 + rsin ** 2 + 1e-12).clamp(max=1.0)
        psi = torch.atan2(rsin, rcos)
        # r·sin(ψ-θ_i) = r sinψ cosθ_i - r cosψ sinθ_i
        coupling = rsin * torch.cos(theta) - rcos * torch.sin(theta)
        return coupling, {"r": r, "psi": psi}


class KuramotoSync(nn.Module):
    """完整的相位同步层（替代注意力）。

    dθ_i/dt = ω_i + Σ_j W_ij sin(θ_j-θ_i) + κ(t)·g_i + η(t)
    """

    def __init__(
        self,
        lambda_t: float = 0.5,
        lambda_s: float = 3.0,
        temporal_window: int = 3,
        dt: float = 0.1,
        use_langevin: bool = True,
        omega_max: float = 8.0,
    ):
        super().__init__()
        self.dt = dt
        self.use_langevin = use_langevin
        self.omega_max = omega_max
        self.toeplitz = SpacetimeToeplitzCoupling(lambda_t, lambda_s, temporal_window)
        self.mean_field = MeanFieldCoupling()

    def forward(
        self,
        theta: torch.Tensor,
        a: torch.Tensor,
        omega: torch.Tensor,
        g: torch.Tensor,
        kappa: torch.Tensor,
        T: torch.Tensor,
        t_coord: torch.Tensor,
        s_coord: torch.Tensor,
        couple_g: torch.Tensor = None,
    ):
        """
        theta:   [B,N,D] 相位
        a:       [B,N,D] 振幅（用于平均场权重）
        omega:   [B,N,D] 固有频率（由文本条件/输入驱动）
        g:       [B,N,D] 漂移场引导
        kappa:   标量 κ(t)
        T:       标量温度
        t_coord: [N] 数据时间坐标
        s_coord: [N,2] 空间坐标

        返回 theta_new: [B,N,D]，以及诊断字典
        """
        # 通路 (2): 时空结构化耦合
        coupling_struct = self.toeplitz.apply_coupling(theta, t_coord, s_coord, couple_g)

        # 通路 (1): 平均场秩-1 耦合（无参数）
        coupling_mf, diag = self.mean_field.forward(a, theta)

        # 合并
        dtheta = omega + coupling_struct + coupling_mf + kappa * g

        # 软饱和（运算放大器饱和物理）：dθ = Ω_max·tanh(dθ/Ω_max)
        # 相位演化原本每元素 RMS≈4.0、dt=0.1 时每步移动 0.4 rad，量级大的位置
        # >π，相位在 ±π 间疯狂反弹（frame-diff 14.7 抖动型 / 1.39 静态型的根源）。
        # tanh 原点附近近似线性（弱信号不受影响）、全域有界可导。
        # Ω_max=8：正常信号（RMS 4）几乎不压缩（tanh(0.5)×8=3.7），只压极端；
        # 每步 ≤0.8 rad，12 块总演化预算 9.6 rad，同时满足相位修正（初始偏差
        # ~1.2 rad）与时间驻波动画（每时间步需 ~1 rad 相位演化）。
        # 演化链已去掉中途折叠（见下），不存在 remainder 断梯度，可放心放宽。
        dtheta = self.omega_max * torch.tanh(dtheta / self.omega_max)

        # 通路 (4): 朗之万噪声 η ~ N(0, 2T·Δt)
        # 扩散模型热力学：温度 T 由时间场退火 T(t)=T_max(1-t)^p+T_min（高→低），
        # 前期高温跨越势垒（探索）、后期低温锁定（利用）= 模拟退火。
        # Kuramoto 本身是保守系统（相位空间体积保持、无耗散），不会自发收敛到
        # 稳态；朗之万项把相位演化变成耗散随机动力学，稳态 ∝ exp(-U/T)。
        # 标准差必须是 sqrt(2T·Δt)，漏掉 Δt 会让噪声量级随步长错误缩放
        if self.use_langevin and self.training:
            noise_std = torch.sqrt(2.0 * T * self.dt).clamp(min=0.0)
            dtheta = dtheta + noise_std * torch.randn_like(dtheta)

        # Euler 积分
        theta_new = theta + self.dt * dtheta

        # 演化链不折叠：θ 沿物理演化连续累加，remainder 只在 loss/诊断处做。
        # 全链路下游（Kuramoto 的 sin(θ_j−θ_i)、漂移场的 z=a·e^{iθ}、驻波 FFN
        # 的输入、decode 的 z=a·e^{iθ}）都通过 sin/cos 使用相位，天然周期，
        # 不需要中途折叠；中途折叠的 remainder 在边界处梯度=0，是 12 层梯度链
        # 反复断裂的元凶。θ 绝对值由软饱和限制在小范围，无精度损失。

        diag.update({"dtheta_norm": dtheta.norm(dim=-1).mean().detach()})
        return theta_new, diag
