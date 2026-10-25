# -*- coding: utf-8 -*-
"""
CondPrototypes —— 条件驱动的吸引中心（补上"内容生成"通路）

问题（2026-10-14 实测定位）
--------------------------
漂移场是唯一把 target 信息送进相位演化的通路（Kuramoto 只吃 κ·g）。
训练时永远传 targets=真实场，于是 `drift_field.forward` 里

    if targets is None:  tar = self.prototypes...

这条**生成分支一次都没被执行过**，prototypes 的梯度恒为 0。
实测 best.pt 里 prototypes std=0.01963，初始化值 randn*0.02 的 std 就是
0.02 —— 几乎没动过。于是 generate() 拿随机原型当吸引中心，输出纯噪点。

这不是训练不足，是**通路从未被优化**。跑 7 天也不会变。

修法（本模块）
--------------
把 prototypes 从「静态参数」升级成「静态原型 + 条件残差」：

    proto(B) = prototypes[None] + tanh(gate) · Δ(cond)

  - prototypes 仍是可学习参数（保留原行为，收敛更快）
  - Δ(cond) 是**低秩线性投影**：cond → hidden → n_proto×2D
    （cond 是 CLIP 的 2049 维双模态条件）
  - gate 初始 0 → 起步时等价于静态原型，梯度会把它推向"学会用条件"

★ 物理核零改动：DriftField/Kuramoto/StandingWaveFFN/OU 的公式与结构一行未动，
  本模块只在**调用侧**把吸引中心的来源从"随机参数"换成"条件驱动"。
  生成动力学仍然只有一个 = PMNN 物理核。
"""
import torch
import torch.nn as nn


class CondPrototypes(nn.Module):
    """条件向量 → 动态原型集 [B, n_proto, 2D]。

    刻意保持简单：两层 nn.Linear + 一个标量门。没有 attention、没有 FFN，
    与"物理核之外不引入 Transformer 架构"的约束一致（线性头是项目一贯做法，
    四个物理调制头 omega_bias/drift_target/kappa/T 也是 nn.Linear）。

    gate 用 tanh 而不是 sigmoid：tanh(0)=0，初始时条件残差**完全为零**，
    起点严格等于原静态原型（不破坏已有权重），梯度又能自然把它推离 0。
    """

    def __init__(self, cond_dim: int = 2049, d_model: int = 125,
                 n_proto: int = 16, hidden: int = 256, gate_init: float = 0.0):
        super().__init__()
        self.cond_dim = cond_dim
        self.d_model = d_model
        self.n_proto = n_proto
        self.out_dim = n_proto * 2 * d_model
        # 低秩瓶颈：2049→256→(16×250)。参数量 ~1.55M，
        # 直接 2049→4000 是 8.2M，在 4GB 卡上没必要。
        self.proj = nn.Linear(cond_dim, hidden)
        self.act = nn.SiLU()
        self.to_proto = nn.Linear(hidden, self.out_dim)
        # 零初始化最后一层：起步时 Δ≈0（等价于恒零映射），
        # 避免随机初始化直接给原型灌一大团噪声（那正是原来噪点的来源）。
        nn.init.zeros_(self.to_proto.weight)
        nn.init.zeros_(self.to_proto.bias)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # 残差尺度上限：条件最多能把原型拉出 1.0（复数场量级 ~1），
        # 防止早期训练把吸引中心推飞导致场崩。
        self.max_delta = 1.0

    def forward(self, cond: torch.Tensor, base_protos: torch.Tensor) -> torch.Tensor:
        """
        cond:        [B, cond_dim]  CLIP 条件（2049 = text1024+image1024+has_image1）
        base_protos: [n_proto, 2D]  静态可学原型（来自 DriftField.prototypes）
        返回:        [B, n_proto, 2D]
        """
        h = self.act(self.proj(cond))                    # [B, hidden]
        delta = self.to_proto(h).view(-1, self.n_proto, 2 * self.d_model)
        delta = torch.tanh(self.gate) * delta.clamp(-3.0, 3.0)
        delta = delta * self.max_delta
        return base_protos.unsqueeze(0) + delta

    def extra_repr(self) -> str:
        return (f"cond_dim={self.cond_dim}, n_proto={self.n_proto}, "
                f"d_model={self.d_model}, gate={self.gate.item():.4f}")