# -*- coding: utf-8 -*-
"""视觉诊断：把 v2500 现状的「LED 网格 / 噪声 / 对比度偏淡」三个症状量化钉死。

不做任何训练，只读 4 张图：
  1. target      —— 原 8-ball GIF 首帧（128x128，训练口径）
  2. ceiling     —— encode→decode 往返（64 通道表征天花板）
  3. v2500 main  —— overfit.gif（normalize=True 出图口径）
  4. v2500 honest—— overfit_honest.gif（normalize=False 固定 clamp）
产出 diag_v3000_现状.png（2x2）+ 关键统计。
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from PIL import Image
import torch
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames, load_gif_frames, resample_frames

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
V2500 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real"


def load_first_frame(path, img_size=(IMG, IMG)):
    img = Image.open(path)
    img.seek(0)
    f = np.array(img.convert("RGB"), dtype=np.uint8)
    if f.shape[0] != img_size[0]:
        f = np.array(Image.fromarray(f).resize(img_size, Image.Resampling.LANCZOS), dtype=np.uint8)
    return f


# 1) target 首帧（与训练口径一致：resize 128 -> 重采样到 30 帧 -> 取第一帧）
frames_raw = load_gif_frames(GIF, (IMG, IMG))
frames = resample_frames(frames_raw, TS * WIN)
target = frames[0]

# 2) 表征天花板：encode 取前 64 通道 -> decode 补零重建（normalize=False 互逆口径）
field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D)
ceil_frames = decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)
ceiling = ceil_frames[0]

# 3) v2500 模型产物
main = load_first_frame(os.path.join(V2500, "overfit.gif"))
honest = load_first_frame(os.path.join(V2500, "overfit_honest.gif"))


def psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = ((a - b) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def contrast(f):
    f = f.astype(np.float32)
    g = f.mean(-1)
    return float(g.std()), float(np.percentile(g, 95) - np.percentile(g, 5))


print("=" * 70)
print("ceiling vs target:  PSNR = %.2f dB" % psnr(ceiling, target))
print("v2500 main  vs target: PSNR = %.2f dB" % psnr(main, target))
print("v2500 honest vs target: PSNR = %.2f dB" % psnr(honest, target))
for name, f in [("target", target), ("ceiling", ceiling), ("main", main), ("honest", honest)]:
    sd, span = contrast(f)
    print("  %-8s 灰度std=%.2f  5-95%%跨度=%.1f  均值=%.1f  像素min/max=%d/%d"
          % (name, sd, span, f.mean(), f.min(), f.max()))
print("=" * 70)


def dc_jumpiness(frames_u8):
    """DC 通道（块的均值能量）在相邻 4x4 patch 间的跳变度 —— LED 网格的直接量化。"""
    f = np.stack([frames_u8] * (TS * WIN), axis=0)
    x = f.astype(np.float32) / 127.5 - 1.0
    xt = x.reshape(TS, WIN, GRID, PATCH, GRID, PATCH, 3)
    blocks = xt.transpose(0, 2, 4, 1, 3, 5, 6).reshape(TS * GRID * GRID, WIN, PATCH * PATCH * 3)
    flat = torch.from_numpy(np.ascontiguousarray(blocks))
    spec = torch.fft.rfft2(flat, dim=(-2, -1)) / 10.0
    spec = spec.reshape(flat.shape[0], -1)
    z = spec[:, :D]
    amp = torch.sqrt(z.real ** 2 + z.imag ** 2 + 1e-12).numpy().reshape(TS, GRID, GRID, D)
    # DC 通道 c=0,25,50 在 t=0 帧的块均值能量场 [GRID,GRID]
    dc = amp[0, :, :, 0] + amp[0, :, :, 25] + amp[0, :, :, 50]
    jump = (np.abs(np.diff(dc, axis=0)).mean() + np.abs(np.diff(dc, axis=1)).mean()) / 2
    return float(jump), dc


for name, f in [("ceiling", ceiling), ("main", main), ("honest", honest)]:
    j, dc = dc_jumpiness(f)
    print("  DC相邻patch跳变 %-8s = %.4f   (DC场std=%.3f, mean=%.3f)"
          % (name, j, dc.std(), dc.mean()))


# 拼 2x2 大图
cell = IMG + 16
rows = [
    [("target(原图)", target), ("ceiling(天花板)", ceiling)],
    [("v2500-main", main), ("v2500-honest", honest)],
]
canvas = np.full((cell * 2 + 8, cell * 2 + 8, 3), 255, dtype=np.uint8)
for ri in range(2):
    for ci in range(2):
        y, x = 8 + ri * cell, 8 + ci * cell
        canvas[y:y + IMG, x:x + IMG] = rows[ri][ci][1]
out = os.path.join(BASE, "outputs_real", "diag_v3000_现状.png")
Image.fromarray(canvas.astype(np.uint8)).save(out)
print("[SAVE]", out)