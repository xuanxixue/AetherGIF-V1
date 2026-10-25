# -*- coding: utf-8 -*-
"""验证幅度 â 模式：a0 的随机抖动是否被完全抵消"""
import sys, math, torch
sys.path.insert(0, "src")
from physics.cond_init import CondInitModulator
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
B, N, D, G = 1, 512, 125, 5
m = CondInitModulator(cond_dim=1024, n_osc=N, n_groups=G).to(dev)
cond = torch.randn(B, 1024, device=dev)
th0 = torch.rand(B, N, D, device=dev) * 2*math.pi - math.pi

res = []
for t in range(2):
    a0 = torch.exp(torch.randn(B, N, D, device=dev) * 0.8 - 2.5)
    dt, dl, parts = m(cond, return_parts=True, n_pos=N, theta0=th0, a0=a0)
    a_out = a0 * torch.exp(dl)
    ah = parts["a_hat"]
    aD = m.expand_to_d(ah)
    err = (torch.log(a_out.clamp(min=1e-9)) - torch.log(aD.clamp(min=1e-9))).abs().max().item()
    res.append(err)
    print(f"[trial {t}] log 域 max|err| = {err:.3e}  a_hat 均值={ah.mean().item():.4f}")
print(f"\n差异 = {abs(res[0]-res[1]):.2e}")
print("✔ a0 抖动已完全抵消" if max(res) < 1e-3 else "✘ 未抵消")
