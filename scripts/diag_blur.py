# -*- coding: utf-8 -*-
"""诊断"糊"的瓶颈：从 checkpoint 前向，量化相位误差分布 + 逐通道振幅偏差。

回答三个问题：
1. 相位误差有多大？在携带结构的（|z| 大的）系数上 cos 是多少？
2. 振幅偏差集中在哪些通道？DC/低频学够了吗？
3. 剩余 PSNR 缺口里，相位 vs 振幅各占多少？（消融：把相位替换成 target 看能涨多少 dB）
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import torch
import numpy as np
from pathlib import Path
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
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

model = AetherGIFModel(device=device, use_qwen=False).to(device)
sd = torch.load("outputs_real/overfit_ckpt.pt", map_location=device, weights_only=True)
model.load_state_dict(sd)
model.eval()

with torch.inference_mode():
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)

pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
c = torch.cos(pe)

print("=" * 70)
print("① 相位误差分布（全 6144×64 系数）")
print("=" * 70)
ape = pe.abs()
print(f"  |pe| 中位数 = {ape.median():.4f} rad ({math.degrees(float(ape.median())):.1f}°)")
print(f"  |pe| 均值   = {ape.mean():.4f} rad")
print(f"  |pe| p90    = {ape.quantile(0.9):.4f} rad")
print(f"  cos 均值    = {c.mean():.4f}")
print(f"  |pe| > 1.0 rad 的比例 = {(ape > 1.0).float().mean():.4f}  (这些是'糊'的主因)")
print()

print("=" * 70)
print("② 按系数功率分桶的相位精度（|z_tar| 越大越携带结构）")
print("=" * 70)
mag = a_tar[0]  # [N, D]
flat_mag = mag.reshape(-1)
flat_cos = c[0].reshape(-1)
qs = torch.quantile(flat_mag, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], device=device))
print(f"  |z_tar| 分位: p50={qs[1]:.4f} p90={qs[2]:.4f} p99={qs[3]:.4f} max={qs[4]:.4f}")
bins = [(0.0, 0.01, "近零(<0.01)"), (0.01, 0.1, "小"), (0.1, 1.0, "中"),
        (1.0, 5.0, "大"), (5.0, 1e9, "巨大(>5)")]
for lo, hi, n in bins:
    m = (flat_mag >= lo) & (flat_mag < hi)
    if bool(m.any()):
        print(f"  {n:12s} n={int(m.sum()):6d}  cos={flat_cos[m].mean():+.4f}  "
              f"|pe|中位={ape[0].reshape(-1)[m].median():.3f}  功率占比={flat_mag[m].pow(2).sum()/flat_mag.pow(2).sum():.3f}")
print()

print("=" * 70)
print("③ 逐通道振幅偏差（前 12 通道 = 时空最低频）")
print("=" * 70)
am_ch = a[0].mean(dim=0)
at_ch = a_tar[0].mean(dim=0)
print(f"{'ch':>3} {'target':>9} {'model':>9} {'比值':>7}")
for i in range(0, 16):
    r = float(am_ch[i]) / max(float(at_ch[i]), 1e-8)
    print(f"{i:3d} {float(at_ch[i]):9.4f} {float(am_ch[i]):9.4f} {r:7.2f}")
print(f"  ... (ch 16-63 平均) target={at_ch[16:].mean():.4f} model={am_ch[16:].mean():.4f} "
      f"比值={float(am_ch[16:].mean()/at_ch[16:].mean().clamp(min=1e-8)):.2f}")
print()

print("=" * 70)
print("④ 消融：剩余缺口里相位 vs 振幅各占多少 dB")
print("=" * 70)


def psnr_of(a_, th_):
    z = (a_[0] * torch.exp(1j * th_[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    fr = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
    return fr


with torch.inference_mode():
    zt = (a_tar[0] * torch.exp(1j * th_tar[0])).to(torch.complex64)
    ft = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()
fr_tar = np.stack(decode_field_to_frames(ft, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


def calc_psnr(arr):
    return 10 * math.log10(255.0 ** 2 / max(((arr - fr_tar) ** 2).mean(), 1e-12))


base = calc_psnr(psnr_of(a, th))
ph_fix = calc_psnr(psnr_of(a, th_tar))       # 相位换成 target
am_fix = calc_psnr(psnr_of(a_tar, th))       # 振幅换成 target
print(f"  当前输出                  PSNR = {base:6.2f} dB")
print(f"  相位→target（只修相位）  PSNR = {ph_fix:6.2f} dB  (+{ph_fix-base:.2f})")
print(f"  振幅→target（只修振幅）  PSNR = {am_fix:6.2f} dB  (+{am_fix-base:.2f})")
print(f"  天花板（全 target）       PSNR = {calc_psnr(fr_tar):6.2f} dB")
print()
if ph_fix - base > am_fix - base:
    print(f"  => 【相位是主要瓶颈】(+{ph_fix-base:.2f} vs +{am_fix-base:.2f})")
else:
    print(f"  => 【振幅是主要瓶颈】(+{am_fix-base:.2f} vs +{ph_fix-base:.2f})")
