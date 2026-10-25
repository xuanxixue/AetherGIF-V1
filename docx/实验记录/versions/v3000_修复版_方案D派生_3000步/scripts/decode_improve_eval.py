# -*- coding: utf-8 -*-
"""P0：用 v2500 checkpoint 换出图口径，量化对比度/噪声改善（零训练成本）。

对比出图口径：
  A. normalize=True  全局 min-max（v2500 主产物 overfit.gif 的现状）
  B. normalize=False 固定 clamp ±1.2（honest 诚实重建）
  C. 稳健分位数拉伸：p0.5/p99.5 映射到 [-1,1] 后再 clamp
  D. 直方图匹配：模型 flat 的排序值对齐 target flat 的排序值（过拟合场景可参考上限）
  E. 固定 clamp + 全局均值/标准差对齐 target（仿射重标定）

同时输出：逐通道振幅比（模型 vs target），确认"能量堆错通道"的假设。
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from PIL import Image
import torch
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
CKPT = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt"
device = "cuda"

torch.manual_seed(1234)
field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)

model = AetherGIFModel(device=device, use_qwen=False).to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd)
model.eval()

with torch.inference_mode():
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()

# target 复值场（用于 D 直方图匹配 & 对比重建）
zt = (a_tar[0] * torch.exp(1j * torch.atan2(im[0], re[0]))).to(torch.complex64)
fld_t = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()

# ---- 出图口径 ----
def decode(fld, normalize=True, clamp_lo=-1.2, clamp_hi=1.2):
    return decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D, normalize=normalize)

# A / B
frA = np.stack(decode(fld, normalize=True)).astype(np.float32)
frB = np.stack(decode(fld, normalize=False)).astype(np.float32)

# C 稳健分位数拉伸：在 irfft2 后的 flat 上做（需重写 decode）
from codec.gif_codec import decode_field_to_frames as _dec
import codec.gif_codec as gic

def decode_percentile(fld, lo_p=0.5, hi_p=99.5):
    N = fld.shape[0]
    zz = fld[:, :D] + 1j * fld[:, D:]
    n_pix = WIN * PATCH * PATCH * 3
    s_dim = PATCH * PATCH * 3
    full = WIN * (s_dim // 2 + 1)
    if D < full:
        zz = torch.cat([zz, torch.zeros(N, full - D, dtype=zz.dtype)], dim=-1)
    zz = zz * 10.0
    h_fft, w_fft = WIN, s_dim // 2 + 1
    flat = torch.fft.irfft2(zz.reshape(N, h_fft, w_fft), s=(WIN, s_dim), dim=(-2, -1)).real
    flat = flat.reshape(N, n_pix)
    lo = torch.quantile(flat, lo_p / 100.0)
    hi = torch.quantile(flat, hi_p / 100.0)
    flat = (flat - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    flat = flat.clamp(-1.0, 1.0)
    blocks = flat.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)
    x = xt.reshape(TS * WIN, IMG, IMG, 3)
    arr = ((x + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]

frC = np.stack(decode_percentile(fld)).astype(np.float32)

# target 重建（clamp，作为 D/E 的分布参考）
frT = np.stack(decode(fld_t, normalize=False)).astype(np.float32)

# D 直方图匹配：对最后像素级做（先各自 honest clamp 再匹配全局排序）
def hist_match(src, ref):
    s = src.ravel()
    r = ref.ravel()
    # 排序映射
    s_idx = np.argsort(s)
    r_idx = np.argsort(r)
    mapped = np.empty_like(s_idx)
    # 用分位插值
    qs = np.linspace(0, 1, len(s))
    mapped[s_idx] = np.interp(qs, np.linspace(0, 1, len(r)), r[r_idx])
    return mapped.reshape(src.shape)

fra = ((frB + 1.0) * 127.5).clip(0, 255)  # honest 像素空间
fra_ref = ((frT + 1.0) * 127.5).clip(0, 255)
frD = hist_match(fra, fra_ref)

# E 全局仿射对齐 target 均值/方差（像素空间）
def affine_match(src, ref):
    s_f = src.astype(np.float32); r_f = ref.astype(np.float32)
    s_m, s_s = s_f.mean(), s_f.std()
    r_m, r_s = r_f.mean(), r_f.std()
    return ((s_f - s_m) / (s_s + 1e-8) * r_s + r_m).clip(0, 255)
frE = affine_match(fra, fra_ref)

def psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))

def contrast(f):
    g = f.astype(np.float32).mean(-1)
    return float(g.std()), float(np.percentile(g, 95) - np.percentile(g, 5)), float(f.min()), float(f.max())

frame_pairs = [("target", frT)]
print("=" * 78)
print("各出图口径（对比 target 重建 frT，注意模型场≠target场，PSNR 衡量像素逼近）：")
print("=" * 78)
trash = None
for name, fr in [("A min-max", frA), ("B honest clamp", frB), ("C 分位数", frC),
                 ("D 直方图匹配", frD), ("E 仿射对齐", frE)]:
    fr_u8 = ((fr + 1.0) * 127.5).clip(0, 255) if fr.max() <= 1.3 else fr
    sd, span, mn, mx = contrast(fr_u8 if fr_u8.max() <= 255 else fr)
    print("  %-14s PSNR=%.2f  std=%.1f  span=%.1f  min/max=%.0f/%.0f"
          % (name, psnr(fr_u8, frT), sd, span, mn, mx))
sd, span, mn, mx = contrast(frT)
print("  %-14s PSNR=%.2f  std=%.1f  span=%.1f  min/max=%.0f/%.0f  (参考)"
      % ("target-recon", 999.0, sd, span, mn, mx))
print("=" * 78)

# 逐通道振幅比（模型 vs target），确认能量堆错通道
am_ch = a[0].mean(dim=0).cpu()
at_ch = a_tar[0].mean(dim=0).cpu()
ratio = (am_ch / at_ch.clamp(min=1e-6)).numpy()
# 空间 DC 通道 c=0,25,50；空间高频区 w 大
print("逐通道振幅比（模型/target，跨 N 平均）：")
for name, idx in [("c0(spDC·tDC)", 0), ("c25(spDC·t1)", 25), ("c50(spDC·t2)", 50),
                  ("c5", 5), ("c12", 12), ("c30", 30), ("c40", 40), ("c63", 63)]:
    print("  %-16s ratio=%.2f   model=%.4f  target=%.4f" % (name, ratio[idx], am_ch[idx], at_ch[idx]))
# 分频段：真实 rfft2 网格 h_fft=5, w_fft=25
bands = {"空间DC(w=0)": [i for i in range(D) if i % 25 == 0],
         "低频(w1-6)": [i for i in range(D) if 1 <= i % 25 <= 6],
         "中频(w7-16)": [i for i in range(D) if 7 <= i % 25 <= 16],
         "高频(w17-24)": [i for i in range(D) if 17 <= i % 25 <= 24]}
print("分频段（按真实 rfft2 w 维切分）：")
for name, idxs in bands.items():
    r = ratio[idxs].mean()
    print("  %-14s 平均 ratio=%.2f   (%.2f~%.2f)" % (name, r, ratio[idxs].min(), ratio[idxs].max()))

# 拼对比图
cells = [("A min-max", frA), ("B honest", frB), ("C 分位", frC), ("D 直方图", frD)]
cell = IMG + 16
canvas = np.full((cell * 2 + 8, cell * 2 + 8, 3), 255, dtype=np.uint8)
for i in range(4):
    fr = cells[i][1]
    fr_u8 = (((fr + 1.0) * 127.5).clip(0, 255) if fr.max() <= 1.5 else fr).astype(np.uint8)
    y, x = 8 + (i // 2) * cell, 8 + (i % 2) * cell
    canvas[y:y + IMG, x:x + IMG] = fr_u8[0]
out = os.path.join(BASE, "outputs_real", "diag_v3000_出图口径对比.png")
Image.fromarray(canvas).save(out)
print("[SAVE]", out)