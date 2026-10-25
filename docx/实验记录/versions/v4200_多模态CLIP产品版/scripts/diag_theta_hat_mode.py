# -*- coding: utf-8 -*-
"""第二十八死穴修复验证：θ̂ 模式下随机 θ0 是否真被完全抵消"""
import sys, os, math, torch
sys.path.insert(0, "src")
from physics.cond_init import CondInitModulator

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
B, N, D, G = 1, 512, 125, 5

m = CondInitModulator(cond_dim=1024, n_osc=N, n_groups=G).to(dev)
cond = torch.randn(B, 1024, device=dev)

# 模拟 target 相位
th_tar = torch.rand(B, N, D, device=dev) * 2 * math.pi - math.pi

# 两次不同随机 θ0，同一条件
res = []
for trial in range(2):
    th0 = torch.rand(B, N, D, device=dev) * 2 * math.pi - math.pi
    dt, dl, parts = m(cond, return_parts=True, n_pos=N, theta0=th0)
    th_final = th0 + dt
    cosv = torch.cos(th_final - th_tar).mean().item()
    # θ̂ 与 target 的对齐（这才是模型真正要学的量）
    cos_hat = torch.cos(parts["theta_hat"] - th_tar[:, :, :G]).mean().item()
    res.append(cosv)
    print(f"[trial {trial}] θ0 seed 不同")
    print(f"  cos(θ0+dθ, θ_tar)  = {cosv:+.6f}")
    print(f"  |dθ| 均值           = {dt.abs().mean().item():.4f}")

print()
print(f"两次结果差异 = {abs(res[0]-res[1]):.2e}")
if abs(res[0]-res[1]) < 1e-5:
    print("✔ 随机 θ0 被完全抵消（结果与 θ0 无关）—— 结构修复生效")
else:
    print("✘ 仍依赖 θ0 —— 修复未生效")

# 反证：旧模式（不传 theta0）下 dθ 与 θ0 独立
dt_old, _, _ = m(cond, return_parts=True, n_pos=N)
print(f"\n旧模式 |dθ| 均值 = {dt_old.abs().mean().item():.4f} (被 phase_scale 锁死)")
