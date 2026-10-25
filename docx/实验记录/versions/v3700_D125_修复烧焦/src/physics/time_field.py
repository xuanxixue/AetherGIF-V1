# -*- coding: utf-8 -*-
"""
§4.1 TimeField —— 深度 = 物理时间

t = round / (n_blocks - 1) ∈ [0,1]，即层数编号归一化为物理时间。
三条协议曲线：

  κ(t) = κ_max · σ((t - t_c) / τ_c)          耦合强度课程：混沌 → 锁相
  T(t) = T_max·(1-t)^p + T_min                温度退火：冷却
  ω_d(t) = ω_lo · (ω_hi / ω_lo)^t             驱动频率几何扫描：低频 → 高频

扫描区间端点由可学习固有频率谱 natural_freq 的低/高分位均值给出，
这样时间场课程本身可训练（梯度可回传），否则是死参数。
"""
import math
import torch
import torch.nn as nn


class TimeField(nn.Module):
    """时间场：把层数编号翻译成三条物理协议曲线。"""

    def __init__(
        self,
        n_blocks: int = 12,
        kappa_max: float = 16.0,
        t_c: float = 0.3,
        tau_c: float = 0.15,
        T_max: float = 1.0,
        T_min: float = 0.05,
        p: float = 1.5,
        n_drive_freq: int = 16,
        freq_lo: float = 0.5,
        freq_hi: float = 3.0,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.kappa_max = kappa_max
        self.t_c = t_c
        self.tau_c = tau_c
        self.T_max = T_max
        self.T_min = T_min
        self.p = p

        # 可学习固有频率谱：时间场课程本身可训练（梯度可回传）
        self.natural_freq = nn.Parameter(torch.linspace(freq_lo, freq_hi, n_drive_freq))

    def t_of_block(self, block_idx: int) -> float:
        """第 block_idx 层的物理时间 ∈ [0,1]。"""
        if self.n_blocks <= 1:
            return 0.0
        return block_idx / (self.n_blocks - 1)

    def kappa(self, t):
        """κ(t) = κ_max · σ((t - t_c)/τ_c) —— 耦合强度课程。"""
        return self.kappa_max * torch.sigmoid((t - self.t_c) / self.tau_c)

    def temperature(self, t):
        """T(t) = T_max(1-t)^p + T_min —— 温度退火。"""
        return self.T_max * (1.0 - t) ** self.p + self.T_min

    def drive_freq(self, t):
        """ω_d(t) = ω_lo · (ω_hi/ω_lo)^t —— 驱动频率几何扫描。

        端点由 natural_freq 低/高分位均值给出（可训练）。
        """
        with torch.no_grad():
            k = max(1, self.natural_freq.numel() // 4)
            sorted_f = torch.sort(self.natural_freq).values
            w_lo = sorted_f[:k].mean()
            w_hi = sorted_f[-k:].mean()
        # 保证 lo < hi，避免几何扫描方向翻转
        w_lo = torch.clamp(w_lo, min=1e-3)
        w_hi = torch.clamp(w_hi, min=w_lo + 1e-3)
        return w_lo * (w_hi / w_lo) ** t

    def forward(self, block_idx: int, device=None):
        """返回第 block_idx 层的三条曲线值 (kappa, T, omega_d)。

        device: 若给定，把标量张量搬到该 device（避免后续运算 device 不匹配）。
        """
        t = torch.tensor(float(self.t_of_block(block_idx)), dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return {
            "t": t,
            "kappa": self.kappa(t),
            "T": self.temperature(t),
            "omega_d": self.drive_freq(t),
        }

    def all_curves(self):
        """返回全部 n_blocks 层的曲线，用于诊断与日志。"""
        rows = []
        for i in range(self.n_blocks):
            c = self.forward(i)
            rows.append(
                {
                    "layer": i,
                    "t": c["t"].item(),
                    "kappa": c["kappa"].item(),
                    "T": c["T"].item(),
                    "omega_d": c["omega_d"].item(),
                }
            )
        return rows
