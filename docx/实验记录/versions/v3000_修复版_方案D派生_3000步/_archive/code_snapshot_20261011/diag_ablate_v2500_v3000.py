# -*- coding: utf-8 -*-
"""决定性消融：把 v2500 / v3000 的输出拆成 (振幅, 相位) 分别与 target 组合解码，
分离「糊」的真正来源：振幅通路 or 相位通路 or DC 相位漂移。

背景（diag_linear 实测）：
  target 振幅分布  p50=0.0015 p90=0.182 p99=5.44 max=24.0  （极度重尾）
  v2500           p50=0.160  p90=0.269 p99=0.447 max=4.2   （重尾被抹平，扁平）
  v3000           p50=0.0018 p90=0.174 p99=5.13  max=87.0  （重尾完美复现，但DC 列溢出3.6x）
  像素域：target std=36.2 / v2500 std=11.0 / v3000 std=20.9，v3000 mean=159（偏亮发白）
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
V2500 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt"
V3000 = os.path.join(BASE, "outputs_real", "overfit_ckpt.pt")
device = "cuda"

field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D)
tgt_a = torch.sqrt(field[:, :D] ** 2 + field[:, D:] ** 2 + 1e-12)
tgt_th = torch.atan2(field[:, D:], field[:, :D])
t, sc = build_spacetime_index(TS, GRID, GRID, device)
fc = field.unsqueeze(0).to(device)


def run(ck):
    m = AetherGIFModel(device=device, use_qwen=False).to(device)
    m.load_state_dict(torch.load(ck, map_location=device), strict=False)
    m.eval()
    with torch.inference_mode():
        a, th, _ = m(texts=["台球"], target_field=fc, t_coord=t, s_coord=sc, batch_size=1)
    return a[0].cpu(), th[0].cpu()


def pol2cplx(a, th):
    z = torch.zeros(a.shape[0], 2 * D)
    z[:, :D] = a * torch.cos(th)
    z[:, D:] = a * torch.sin(th)
    return z


def px(fr):
    p = np.stack([np.asarray(f, dtype=np.float32) for f in fr])
    return p


def rep(name, z):
    fr = decode_field_to_frames(z, TS, GRID, PATCH, WIN, IMG, D, normalize=False)
    p = px(fr)
    t = px(decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False))
    mse = ((p - t) ** 2).mean()
    return "%-30s PSNR=%5.2f dB | mean=%6.2f std=%6.2f p1=%5.1f p50=%5.1f p99=%6.1f" % (
        name, 10 * math.log10(255.0 ** 2 / max(mse, 1e-12)), p.mean(), p.std(),
        np.percentile(p, 1), np.percentile(p, 50), np.percentile(p, 99))


a25, th25 = run(V2500)
a30, th30 = run(V3000)

print("=" * 96)
print("消融 1：谁贡献了 PSNR？（振幅/相位交叉组合）")
print("=" * 96)
print(rep("target(基准)", pol2cplx(tgt_a, tgt_th)))
print(rep("v2500 原样", pol2cplx(a25, th25)))
print(rep("v3000 原样", pol2cplx(a30, th30)))
print("-" * 96)
print(rep("v2500振幅 + target相位", pol2cplx(a25, tgt_th)))
print(rep("v3000振幅 + target相位", pol2cplx(a30, tgt_th)))
print("-" * 96)
print(rep("target振幅 + v2500相位", pol2cplx(tgt_a, th25)))
print(rep("target振幅 + v3000相位", pol2cplx(tgt_a, th30)))
print("=" * 96)

print("DC 相位漂移诊断（c=0/25/50 是 DC 列，占 96.4% 能量）")
print("=" * 96)
print("target DC 列相位: mean|cos|=%.4f  (应≈1.0，DC 是纯实正数)" % tgt_th[:, ::25].cos().abs().mean())
for tag, a, th in [("v2500", a25, th25), ("v3000", a30, th30)]:
    dc = th[:, ::25]
    print("%s DC 列相位: mean|cos|=%.4f  相位std=%.3f rad  → DC 能量被搬成低频波动的程度"
          % (tag, dc.cos().abs().mean(), dc.std()))
print("=" * 96)

print("RMS 相位误差（大振幅权重）")
print("=" * 96)
for tag, a, th in [("v2500", a25, th25), ("v3000", a30, th30)]:
    d = torch.atan2(torch.sin(th - tgt_th), torch.cos(th - tgt_th)).abs()
    print("%s  w=1: %.3f rad (%.1f°) | 大振幅加权: %.3f rad (%.1f°)"
          % (tag, d.mean(), d.mean() * 57.3,
             (d * tgt_a).sum() / tgt_a.sum(), (d * tgt_a).sum() / tgt_a.sum() * 57.3))
print("=" * 96)