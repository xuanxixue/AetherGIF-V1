# -*- coding: utf-8 -*-
"""一致口径重算 PSNR + 像素级可看性指标。

overfit.py 的 [CMP] 用 normalize=True（模型）vs normalize=False（target），
尺度不一致导致 PSNR 虚低。这里两者同口径，并给出可看性指标。
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
device = "cuda"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gp = next(Path(ROOT, "data/gif_raw/noto").glob("8-ball_1f3b1.gif"))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)

# target 重建（与模型同口径 normalize=True）
re_t, im_t = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re_t ** 2 + im_t ** 2 + 1e-12)
th_tar = torch.atan2(im_t, re_t)
zt = (a_tar[0] * torch.exp(1j * th_tar[0])).to(torch.complex64)
ft = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()
fr_tar = decode_field_to_frames(ft, TS, GRID, PATCH, WIN, IMG, D, normalize=True)
arr_t = np.stack(fr_tar).astype(np.float32)

# 已生成的 overfit.gif
# 最新输出会被自动重命名为带编号的文件，取最大编号的 overfit-*.gif
cands = sorted(Path(ROOT, "outputs_real").glob("overfit*.gif"),
               key=lambda p: p.stat().st_mtime)
gif_path = cands[-1]
print(f"[评估对象] {gif_path.name}  ({gif_path.stat().st_size} bytes)")
gif = Image.open(str(gif_path))
fr_gen = []
for i in range(gif.n_frames):
    gif.seek(i)
    fr_gen.append(np.asarray(gif.convert("RGB"), dtype=np.float32))
arr_g = np.stack(fr_gen).astype(np.float32)

n = min(len(arr_g), len(arr_t))
ps = [10 * math.log10(255.0 ** 2 / max(((arr_g[i] - arr_t[i]) ** 2).mean(), 1e-12))
      for i in range(n)]
print(f"[一致口径 PSNR] 均值={np.mean(ps):.2f} dB  最好={np.max(ps):.2f} dB  最差={np.min(ps):.2f} dB")
print("  逐帧: " + " ".join(f"{p:.1f}" for p in ps[:10]))

# 像素级可看性
print(f"\n[亮度] gen={arr_g.mean():.1f} | tar={arr_t.mean():.1f}")
print(f"[帧内对比度] gen={arr_g.std():.1f} | tar={arr_t.std():.1f}")
gd = [np.abs(arr_g[i+1]-arr_g[i]).mean() for i in range(len(arr_g)-1)]
td = [np.abs(arr_t[i+1]-arr_t[i]).mean() for i in range(len(arr_t)-1)]
print(f"[帧间变化] gen={np.mean(gd):.2f} | tar={np.mean(td):.2f}")

def grad_e(f):
    return (np.abs(np.diff(f, axis=1)).mean() + np.abs(np.diff(f, axis=0)).mean()) / 2
print(f"[空间结构能] gen={np.mean([grad_e(f) for f in arr_g]):.2f} | tar={np.mean([grad_e(f) for f in arr_t]):.2f}")

# target_recon 自身相对原 GIF 的上限（表征瓶颈参照）
raw = Image.open(str(gp))
fr_raw = []
for i in range(raw.n_frames):
    raw.seek(i)
    fr_raw.append(np.asarray(raw.convert("RGB"), dtype=np.float32)[:128, :128])
arr_r = np.stack(fr_raw).astype(np.float32)
m = min(len(arr_r), len(arr_t))
psr = [10 * math.log10(255.0 ** 2 / max(((arr_t[i] - arr_r[i]) ** 2).mean(), 1e-12)) for i in range(m)]
print(f"\n[表征上限参照] target_recon vs 原GIF PSNR={np.mean(psr):.2f} dB（d_model=64 截断 49% 频谱的瓶颈）")
