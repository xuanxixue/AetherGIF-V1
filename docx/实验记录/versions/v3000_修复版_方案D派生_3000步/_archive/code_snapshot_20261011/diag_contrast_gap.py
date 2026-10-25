# -*- coding: utf-8 -*-
"""定位 v3000 对比度缺口（24.9 vs target 42.6），为修 loss 定参数。

已确认事实：
  能量占比 v3000 极准（DC 96.2% vs target 97.2%，均匀度 0.094）
  μ 与真值 corr=0.9953（μ 没问题）
  但像素对比度 v3000 std=24.9 vs codec顶 36.0 vs 原始 42.6 → 只有 58~69%
  → 缺口不在能量占比，那在哪？候选：
     (a) 相位错 → irfft2 后干涉相消，能量被摊平（能量守恒但对比度降）
     (b) a_max 溢出（v3000 max=87 vs target 24）→ 少数系数爆，clamp 后高光糊成一片
     (c) decode 口径
  消融（固定 seed）：逐个替换测 PSNR/std
"""
import os, sys, math
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "src"))
import numpy as np
import torch
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
device = "cuda"
field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D)
tgt_a = torch.sqrt(field[:, :D] ** 2 + field[:, D:] ** 2 + 1e-12)
tgt_th = torch.atan2(field[:, D:], field[:, :D])
tgt_pix = np.stack([np.asarray(f, dtype=np.float32) for f in
                    decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)]).mean(-1)
t, sc = build_spacetime_index(TS, GRID, GRID, device)
fc = field.unsqueeze(0).to(device)


def run(ck, seed=0):
    m = AetherGIFModel(device=device, use_qwen=False).to(device)
    m.load_state_dict(torch.load(ck, map_location=device), strict=False)
    m.eval()
    torch.manual_seed(seed)
    with torch.inference_mode():
        a, th, _ = m(texts=["台球"], target_field=fc, t_coord=t, s_coord=sc, batch_size=1)
    return a[0].cpu(), th[0].cpu()


def z_of(a, th):
    z = torch.zeros(a.shape[0], 2 * D)
    z[:, :D] = a * torch.cos(th); z[:, D:] = a * torch.sin(th)
    return z


def ev(z, name):
    p = np.stack([np.asarray(f, dtype=np.float32) for f in
                  decode_field_to_frames(z, TS, GRID, PATCH, WIN, IMG, D, normalize=False)]).mean(-1)
    mse = ((p - tgt_pix) ** 2).mean()
    ps = 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))
    return "%-38s PSNR=%5.2f dB  std=%5.2f  p1-p99=%6.1f  过曝=%.2f%%" % (
        name, ps, p.std(), np.percentile(p, 99) - np.percentile(p, 1), (p >= 254).mean() * 100)


a, th = run(os.path.join(BASE, "outputs_real", "overfit_ckpt.pt"))

print("=" * 104)
print("消融：对比度缺口归因（v3000, seed=0）")
print("=" * 104)
print(ev(z_of(tgt_a, tgt_th), "target(基准)"))
print("-" * 104)
print(ev(z_of(a, th), "v3000 原样"))
print(ev(z_of(a, tgt_th), "v3000振幅 + target相位"))
print(ev(z_of(tgt_a, th), "target振幅 + v3000相位"))
print("-" * 104)

# (b) 溢出假设：把超过 target_max 的系数截到 target_max
thr = tgt_a.max()
a_cl = a.clamp(max=thr)
print(ev(z_of(a_cl, th), "v3000振幅(削峰到target_max) + 原相位"))
print(ev(z_of(a_cl, tgt_th), "v3000振幅(削峰) + target相位"))

# 只削 DC 列溢出（实测溢出集中在 c=0/25/50）
dc = [0, 25, 50]
a_dc = a.clone()
a_dc[:, dc] = a_dc[:, dc].clamp(max=thr)
print(ev(z_of(a_dc, th), "v3000振幅(只削c=0/25/50) + 原相位"))

print()
print("=" * 104)
print("溢出系数分布（定位病灶）")
print("=" * 104)
print("target: a_max=%.2f  p99=%.2f  p99.9=%.2f" % (tgt_a.max(), tgt_a.quantile(0.99), tgt_a.quantile(0.999)))
print("v3000 : a_max=%.2f  p99=%.2f  p99.9=%.2f" % (a.max(), a.quantile(0.99), a.quantile(0.999)))
over = a > thr
print("v3000 超 target_max 的系数: %d 个 (%.4f%%), 分布在 c=%s"
      % (int(over.sum()), over.float().mean().item() * 100,
         sorted(set(torch.nonzero(over)[:, 1].tolist()))[:12]))
# 每个溢出系数贡献多少像素能量
print()
print("单系数溢出量级（irfft2 基函数幅值 ~ 1/N）：")
print("  超出部分 %.2f / N=%d → 单像素污染量级 %.4f（像素量纲 0-255，约 %.1f 灰阶）"
      % (float(a.max() - thr), a.shape[0], float(a.max() - thr) / a.shape[0],
         float(a.max() - thr) / a.shape[0] * 255))
print()
print("=" * 104)
print("结论用：像素域 p1-p99 动态范围")
print("=" * 104)
print("  target %.1f / v3000原样 %.1f / 削峰后 见上" % (np.percentile(tgt_pix, 99) - np.percentile(tgt_pix, 1)))
print("=" * 104)