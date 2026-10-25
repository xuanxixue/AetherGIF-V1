# -*- coding: utf-8 -*-
"""精确测 codec 表征天花板：LED 网格感能否靠表征维度修复。

一次性测清：
  1. 全谱(125) 与 D=64 的往返 PSNR（全帧口径，clamp，与训练一致）
  2. patch=4 与 patch=8 的影响
  3. 逐频段能量：被 D=64 砍掉的那 61 个通道（w_fft 维度的高频）到底丢了多少
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from PIL import Image
import torch
from codec.gif_codec import load_gif_frames, resample_frames

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
IMG = 128

def roundtrip(D_keep, grid=32, patch=4, win=5, ts=6):
    frames = load_gif_frames(GIF, (IMG, IMG))
    frames = resample_frames(frames, ts * win)
    x = frames.astype(np.float32) / 127.5 - 1.0
    xt = x.reshape(ts, win, grid, patch, grid, patch, 3)
    blocks = xt.transpose(0, 2, 4, 1, 3, 5, 6).reshape(ts * grid * grid, win, patch * patch * 3)
    flat = torch.from_numpy(np.ascontiguousarray(blocks))
    spec = torch.fft.rfft2(flat, dim=(-2, -1)) / 10.0
    spec = spec.reshape(flat.shape[0], -1)
    full = spec.shape[-1]
    # 保留前 D_keep，其余补零
    z = spec.clone()
    if D_keep < full:
        z[:, D_keep:] = 0
    # decode
    n_pix = win * patch * patch * 3
    s_dim = patch * patch * 3
    h_fft = win
    w_fft = s_dim // 2 + 1
    zz = (z * 10.0).reshape(flat.shape[0], h_fft, w_fft)
    rec = torch.fft.irfft2(zz, s=(win, s_dim), dim=(-2, -1)).real.reshape(flat.shape[0], n_pix)
    rec = rec.clamp(-1.2, 1.2)
    b = rec.reshape(ts, grid, grid, win, patch, patch, 3)
    xt2 = b.permute(0, 3, 1, 4, 2, 5, 6)
    xr = xt2.reshape(ts * win, IMG, IMG, 3)
    arr = ((xr + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.float32)
    ref = ((x + 1.0) * 127.5).astype(np.float32)
    mse = ((arr - ref) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12)), full

print("=" * 70)
for D_keep in [125, 64, 96, 48]:
    ps, full = roundtrip(D_keep)
    pct = D_keep / full * 100
    print("D=%3d (保留 %.1f%% 谱)  ->  往返 PSNR = %.2f dB" % (D_keep, pct, ps))
print("-" * 70)
for pt in [4, 8]:
    ps, full = roundtrip(64, patch=pt, win=5, ts=6)
    print("D=64, patch=%d  ->  往返 PSNR = %.2f dB" % (pt, ps))
print("=" * 70)

# 逐频段能量：64 通道覆盖 vs 125 全谱，看被砍的 61 个通道的能量占比
frames = load_gif_frames(GIF, (IMG, IMG))
frames = resample_frames(frames, 30)
x = frames.astype(np.float32) / 127.5 - 1.0
xt = x.reshape(6, 5, 32, 4, 32, 4, 3)
blocks = xt.transpose(0, 2, 4, 1, 3, 5, 6).reshape(6 * 32 * 32, 5, 48)
flat = torch.from_numpy(np.ascontiguousarray(blocks))
spec = torch.fft.rfft2(flat, dim=(-2, -1)) / 10.0  # [N,5,25]
amp2 = (spec.real ** 2 + spec.imag ** 2).mean(dim=0)  # [5,25] 平均能量
total = amp2.sum()
# 展平通道 c = h*25 + w
flat_amp2 = amp2.reshape(-1)  # [125]
# 前 64 通道 vs 后 61 通道
keep = flat_amp2[:64].sum()
drop = flat_amp2[64:].sum()
print("能量分布：前64通道能量占比 = %.2f%%  被砍61通道 = %.2f%%" % (keep/total*100, drop/total*100))
# 按 w_fft 分频段（空间频率）
print("按空间频率 w_fft 分段的能量占比（h 维全累加）：")
for wgrp, name in [([0], "w=0(空间DC)"), (list(range(1,7)), "w1-6(低频)"),
                   (list(range(7,17)), "w7-16(中频)"), (list(range(17,25)), "w17-24(高频)")]:
    e = 0.0
    for h in range(5):
        for w in wgrp:
            e += amp2[h, w].item()
    print("  %-14s = %.2f%%" % (name, e / total * 100))