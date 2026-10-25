# -*- coding: utf-8 -*-
"""
§4.5 Thermodynamics —— 热力学判据层

  U(Z) = −Σ_ij W_ij·cos(θ_j − θ_i)     （XY 能量：锁相 = 能量最小化）
  S(Z) = −Σ_i p_i·log p_i, p_i = softmax(a_i)   （振幅分布熵）
  F = U − T·S                          （自由能）

核心恒等式（AetherMind §4.5 / AetherSentry §2.3）：
  ∂U/∂θ_i = Σ_j W_ij·sin(θ_j − θ_i)

即 Kuramoto 的 sin 耦合就是 XY 能量梯度。同步层不是「模拟一个物理现象」，
而是在做能量最小化。这个同源性必须保留，不能简化掉。

AetherSentry §8.3 的教训（通道正交）：
  U 依赖时间维序参量，S 依赖通道维振幅分布。
  若要用 F = U − T·S 修时间通道的问题，S 必须显式计算在时间维度上。
  本实现提供两种 S：通道维熵 S_chan 与时间维熵 S_time，禁止混用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Thermodynamics(nn.Module):
    """热力学判据层：U / S / F，只做诊断，不参与梯度。"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def order_parameter(a: torch.Tensor, theta: torch.Tensor):
        """序参量 r 与平均相位 ψ（振幅加权 + 归一化口径修正）。

        r cosψ = Σw_i cosθ_i / Σw_i,  w_i = softplus(a_i)
        返回 r: [B,D], psi: [B,D]
        """
        w = F.softplus(a)
        w_sum = w.sum(dim=1).clamp(min=1e-8)  # [B,D]
        rcos = (w * torch.cos(theta)).sum(dim=1) / w_sum
        rsin = (w * torch.sin(theta)).sum(dim=1) / w_sum
        r = torch.sqrt(rcos ** 2 + rsin ** 2 + 1e-12).clamp(max=1.0)
        psi = torch.atan2(rsin, rcos)
        return r, psi

    @staticmethod
    def xy_energy(theta: torch.Tensor, W: torch.Tensor = None):
        """U = −Σ_ij W_ij cos(θ_j − θ_i)

        若不给 W（时空耦合是隐式的、分块计算的），用平均场近似：
        U ≈ −Σ_i w_i·r·cos(ψ − θ_i) / Σw_i
        （AetherMind §4.5 的平均场近似能量）
        """
        if W is not None:
            cos_diff = torch.cos(theta.unsqueeze(2) - theta.unsqueeze(1))  # [B,N,N,D]
            return -(W.unsqueeze(-1) * cos_diff).sum(dim=(1, 2)).mean()
        # 平均场近似
        w = F.softplus(theta.abs())  # 退化情形：无 a 时用 |θ|
        r, psi = Thermodynamics.order_parameter(w, theta)
        U = -(w * r.unsqueeze(1) * torch.cos(psi.unsqueeze(1) - theta)).sum(dim=1)
        U = U / w.sum(dim=1).clamp(min=1e-8)
        return U.mean()

    @staticmethod
    def amplitude_entropy_channel(a: torch.Tensor):
        """通道维振幅分布熵 S_chan = −Σ p log p, p = softmax(a_i)

        a: [B,N,D]，沿 N 维 softmax（每个通道一个分布）
        返回 [B,D] 归一化熵 ∈[0,1]
        """
        p = F.softmax(a, dim=1)  # 沿 N
        S = -(p * torch.log(p + 1e-12)).sum(dim=1)
        return S / torch.log(torch.tensor(float(a.shape[1]), device=a.device))

    @staticmethod
    def amplitude_entropy_time(a: torch.Tensor, t_steps: int, hw: int):
        """时间维振幅分布熵 S_time（AetherSentry §8.3 教训：修时间通道必须用时间维的 S）

        把 [B, N, D] 重排为 [B, t_steps, hw, D]，
        沿时间维 softmax 计算熵。用于时间通道的诊断与正则。
        """
        B, N, D = a.shape
        a_view = a.reshape(B, t_steps, hw, D)
        # 沿时间维 softmax（固定空间位置与通道）
        p = F.softmax(a_view, dim=1)  # [B,T,hw,D]
        S = -(p * torch.log(p + 1e-12)).sum(dim=1)  # [B,hw,D]
        return S.mean(dim=(1, 2)) / torch.log(torch.tensor(float(t_steps), device=a.device))

    def forward(self, a: torch.Tensor, theta: torch.Tensor, T: torch.Tensor,
                t_steps: int = None, hw: int = None):
        """返回 {U, S_chan, S_time, F, r, psi}，全部 detach 只做诊断。"""
        with torch.no_grad():
            r, psi = self.order_parameter(a, theta)
            U = self.xy_energy(theta)
            S_chan = self.amplitude_entropy_channel(a)
            S_time = (
                self.amplitude_entropy_time(a, t_steps, hw)
                if t_steps is not None and hw is not None
                else torch.zeros_like(S_chan)
            )
            F_val = U - T * S_chan
            return {
                "U": U.detach(),
                "S_chan": S_chan.detach(),
                "S_time": S_time.detach(),
                "F": F_val.detach(),
                "r": r.detach(),
                "psi": psi.detach(),
            }
