# -*- coding: utf-8 -*-
"""诊断 AC 分量过冲的训练端根因：a_mu / a_gamma 学到哪了。

加载 v2500 checkpoint，打印：
  1. 每个 block 的 a_gamma（单可训练 γ）
  2. 每个 block 的 a_mu（per-channel 64 维，log 域）的逐通道剖面
  3. 对比 target 的逐通道 log1p 均值（loss 的量纲）与 log 均值（架构递推的量纲）
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import torch
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
CKPT = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt"
device = "cuda"

field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)  # [1,N,D]

model = AetherGIFModel(device=device, use_qwen=False).to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd)

# target 逐通道统计
at_ch = a_tar[0].mean(dim=0).cpu()          # 算术均值（log1p 域偏算术）
lt_ch = torch.log1p(at_ch)                   # log1p 域真值（loss 量纲）
logm = torch.log(a_tar[0].clamp(min=1e-8)).mean(dim=0).cpu()  # log 域真值（架构递推量纲）

print("=" * 78)
print("target 逐通道（抽样）:")
idx = [0, 25, 50, 5, 12, 30, 40, 63]
print("  %-6s %-12s %-12s %-12s" % ("c", "a_mean", "log1p_mean", "log_mean"))
for c in idx:
    print("  %-6d %-12.4f %-12.4f %-12.4f" % (c, at_ch[c], lt_ch[c], logm[c]))
print("-" * 78)

# 每个 block 的 a_gamma 和 a_mu
print("各 block 的 a_gamma（单可训练 γ）与 a_mu 剖面（log 域）：")
for i, blk in enumerate(model.pmnncore.blocks):
    g = blk.a_gamma.item()
    mu = blk.a_mu.detach().cpu()  # [64] log 域
    # 压缩成频段均值（按真实 rfft2 w 维）
    bands = {
        "DC(w0)": [c for c in range(D) if c % 25 == 0],
        "低频w1-6": [c for c in range(D) if 1 <= c % 25 <= 6],
        "中频w7-16": [c for c in range(D) if 7 <= c % 25 <= 16],
        "高频w17-24": [c for c in range(D) if 17 <= c % 25 <= 24],
    }
    bstr = " ".join("%s=%.2f" % (n, mu[idxs].mean()) for n, idxs in bands.items())
    print("  blk%02d γ=%.4f  a_mu: %s" % (i, g, bstr))

# target 在 log 域的对应频段均值
print("-" * 78)
print("target 逐频段 log 均值（对照 a_mu 应学到的稳态）:")
bands = {
    "DC(w0)": [c for c in range(D) if c % 25 == 0],
    "低频w1-6": [c for c in range(D) if 1 <= c % 25 <= 6],
    "中频w7-16": [c for c in range(D) if 7 <= c % 25 <= 16],
    "高频w17-24": [c for c in range(D) if 17 <= c % 25 <= 24],
}
for n, idxs in bands.items():
    print("  %-12s log_mean=%.2f   log1p_mean=%.2f   (算术mean=%.4f)"
          % (n, logm[idxs].mean(), lt_ch[idxs].mean(), at_ch[idxs].mean()))
print("=" * 78)