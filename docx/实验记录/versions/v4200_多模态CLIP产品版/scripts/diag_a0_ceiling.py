# -*- coding: utf-8 -*-
"""幅度侧天花板：a0 = exp(N(-2.5, 0.8)) · exp(dlna)，dlna 不看 a0

问：dlna 最好能到多少？理想情形 dlna = log(a_tar) − log(a0)（依赖 a0）。
"""
import torch, math
torch.manual_seed(0)
N = 500000
a_tar = torch.exp(torch.randn(N) * 2.0 - 2.5)   # 重尾 target
a0    = torch.exp(torch.randn(N) * 0.8 - 2.5)   # 当前初态

def relerr(x):
    return (torch.log(x.clamp(min=1e-9)) - torch.log(a_tar.clamp(min=1e-9))).abs().mean().item()

print("=== dlna 依赖 a0（理想修正）===")
ideal = torch.log(a_tar.clamp(min=1e-9)) - torch.log(a0.clamp(min=1e-9))
out = a0 * torch.exp(ideal)
print(f"  log 域平均绝对误差 = {relerr(out):.6f}   -> 完全可达")

print()
print("=== dlna 依赖 a0，但被 amp_scale 截断（当前 amp_scale 见训练脚本）===")
for s in [0.5, 1.0, 1.5, 3.0, 6.0]:
    d = ideal.clamp(-s, s)
    out = a0 * torch.exp(d)
    print(f"  amp_scale={s:.1f}  log 误差 = {relerr(out):.4f}  std(线性)={out.std():.3f}")

print()
print("=== dlna 不依赖 a0（只由 cond 决定）时的极限：取最优常数增益 ===")
lg = (torch.log(a_tar.clamp(min=1e-9)) - torch.log(a0.clamp(min=1e-9)))
best = lg.mean()
out = a0 * torch.exp(best)
print(f"  最优常数增益 {best:+.4f} -> log 误差 = {relerr(out):.4f}（σ=0.8 的抖动完全残留）")
print(f"  残留抖动 = {torch.log(a0).std().item():.4f}（log 域标准差，无法被常数增益消除）")
