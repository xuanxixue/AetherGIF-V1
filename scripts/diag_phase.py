# -*- coding: utf-8 -*-
"""相位通路诊断：定位 cos(θ−θ_tar) 卡在 0.39 的根因。

四个数据点：
  1. a_tar 分布与分层 cos —— 大量 a_tar≈0 的高频相位是不是纯噪声（atan2 数值噪声）
  2. 逐 block cos 演化 —— 相位是被拉向 target 还是远离 / 平台
  3. 各通路 norm —— κ·g 相对 ω / struct / mf 的实际量级
  4. 分通道 cos —— D=64 个通道各自的对齐情况
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import torch
import numpy as np
from pathlib import Path
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
device = "cuda"

gp = next(Path("data/gif_raw/noto").glob("8-ball_1f3b1.gif"))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
th_tar = torch.atan2(im, re)

# ---- 1. a_tar 分布 ----
at = a_tar[0].flatten()
q = torch.quantile(at, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99], device=device))
print("[a_tar 分位] " + " ".join(f"p{int(p*100)}={v:.4f}"
      for p, v in zip([.1, .25, .5, .75, .9, .99], q.tolist())))
print("[a_tar] mean=%.4f  <0.01 占比=%.1f%%  <0.05 占比=%.1f%%"
      % (at.mean(), (at < 0.01).float().mean() * 100, (at < 0.05).float().mean() * 100))

# ---- 模型前向，逐 block 记录 ----
model = AetherGIFModel(device=device).to(device)
model.eval()
text_embs = model.qwen.encode(["台球"])
omega_bias = model.omega_bias_head(text_embs)
kappa_scale = torch.sigmoid(model.kappa_scale_head(text_embs)).squeeze(-1)
T_scale = torch.sigmoid(model.T_scale_head(text_embs)).squeeze(-1)
omega_base = model._init_omega(t, s)
omega = omega_base + omega_bias.unsqueeze(1)

torch.manual_seed(42)
a0 = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
th0 = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi

a, th = a0, th0
cos_traj, a_traj, norms = [], [], []
with torch.inference_mode():
    for i in range(model.n_blocks):
        tf = model.pmnncore.time_field.forward(i, device)
        a_prev, th_prev = a, th
        a, th, d = model.pmnncore.blocks[i](
            a, th, omega, t, s,
            kappa=tf["kappa"] * kappa_scale,
            T=tf["T"] * T_scale,
            omega_d=tf["omega_d"],
            targets=target,
        )
        pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
        cos_traj.append(torch.cos(pe).mean().item())
        a_traj.append(a.mean().item())
        if i in (0, 5, 11):
            norms.append((i, d.get("dtheta_norm", torch.tensor(float("nan"))).mean().item(),
                          d.get("r", torch.tensor(float("nan"))).mean().item()))

print("\n[逐 block cos] " + " ".join(f"b{i}={c:+.3f}" for i, c in enumerate(cos_traj)))
print("[逐 block a]   " + " ".join(f"b{i}={v:.3f}" for i, v in enumerate(a_traj)))
for i, dn, r in norms:
    print(f"[block {i:2d}] dtheta_norm={dn:.3f}  r(平均场序参量)={r:.4f}")

# ---- 2. 分层 cos（按 a_tar 大小） ----
with torch.inference_mode():
    pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
    cos_all = torch.cos(pe)[0]                     # [N,D]
at2 = a_tar[0]                                     # [N,D]
bins = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.0), (1.0, 1e9)]
print("\n[分层 cos]")
for lo, hi in bins:
    m = (at2 >= lo) & (at2 < hi)
    if m.sum() > 0:
        print(f"  a_tar ∈ [{lo:.2f},{hi:.2f})  n={m.sum().item():6d} ({m.float().mean().item()*100:5.1f}%)  cos={cos_all[m].mean().item():+.4f}")

# ---- 4. 分通道 cos ----
cos_ch = cos_all.mean(dim=0)                        # [D]
print("\n[分通道 cos] mean=%.3f  min=%.3f(ch%d)  max=%.3f(ch%d)"
      % (cos_ch.mean(), cos_ch.min(), cos_ch.argmin(), cos_ch.max(), cos_ch.argmax()))
print("  前8通道: " + " ".join(f"ch{i}={v:+.2f}" for i, v in enumerate(cos_ch[:8].tolist())))
print("  后8通道: " + " ".join(f"ch{i}={v:+.2f}" for i, v in enumerate(cos_ch[-8:].tolist())))

# ---- 5. 最终 cos 与各通路量级（最后 block） ----
print("\n[总结] 最终 cos=%.4f  a_out mean=%.4f  a_tar mean=%.4f"
      % (cos_all.mean(), a.mean(), a_tar.mean()))
