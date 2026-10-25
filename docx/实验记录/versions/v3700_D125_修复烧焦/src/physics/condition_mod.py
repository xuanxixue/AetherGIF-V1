# -*- coding: utf-8 -*-
"""
条件化物理调制器：条件 → prototypes

核心：CLIP cond (2049 维) → MLP → 16 个原型 [16, 2*D]
用于 DriftField 的吸引中心（推理时），训练时 targets 仍是真实场。
"""
import torch
import torch.nn as nn

class ConditionModulator(nn.Module):
    def __init__(self, cond_dim, d_model, n_prototypes, hidden_dim=256):
        super().__init__()
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.d_model = d_model
        self.n_prototypes = n_prototypes
        self.proto_proj = nn.Linear(hidden_dim, n_prototypes * 2 * d_model)

        with torch.no_grad():
            self.proto_proj.weight.normal_(0, 0.02)

    def forward(self, cond):
        x = self.cond_net(cond)
        proto = self.proto_proj(x)  # [B, n_proto*2*D]
        return proto.view(cond.shape[0], self.n_prototypes, 2 * self.d_model)
