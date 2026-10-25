# -*- coding: utf-8 -*-
"""决定性实验：网格感到底来自哪里？

三个候选：
  (A) FFT 变换本身有噪声      -> 纯往返误差应 > 1e-3
  (B) 逐 patch 独立 FFT 导致块边界不连续  -> 全图一次 FFT 应无网格
  (C) 频谱硬截断（只留 64/125）-> 全保留 125 应无网格

口径与训练完全一致：PATCH=4, WIN=5, IMG=128, TS=6, NORM=10.0
"""
import sys
sys.path.insert(0, './src')
import numpy as np
import torch
from codec.gif_codec import load_gif_frames

PATCH, WIN, TS, IMG = 4, 5, 6, 128
GRID = IMG // PATCH
NORM = 10.0
GIF = 'data/gif_raw/noto/3D87B14FBB731ED3A948AC6DD9EA6EBC.gif'


def to_blocks(x):
    """[TS*WIN, IMG, IMG, 3] -> [TS, WIN, GRID, PATCH, GRID, PATCH, 3]"""
    t = TS
    return x.reshape(t, WIN, GRID, PATCH, GRID, PATCH, 3)


def blocks_to_img(blocks):
    """[TS, GRID, GRID, WIN, PATCH, PATCH, 3] -> [TS*WIN, IMG, IMG, 3]"""
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)     # [t,w,gy,py,gx,px,c]
    return xt.reshape(TS * WIN, IMG, IMG, 3)


def grid_ratio(arr):
    """patch 边界跳变 / 内部跳变"""
    b, i = [], []
    for f in arr:
        d = np.abs(np.diff(f, axis=1))
        xs = np.arange(d.shape[1])
        m = (xs % PATCH) == (PATCH - 1)
        b.append(d[:, m].mean()); i.append(d[:, ~m].mean())
    return float(np.mean(b)), float(np.mean(i)), float(np.mean(b)) / max(np.mean(i), 1e-9)


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / max(((a - b) ** 2).mean(), 1e-12))


def stats(name, arr, ref=None):
    b, i, r = grid_ratio(arr)
    s = f"{name:<38} std={arr.std():6.2f}  边界={b:6.2f} 内部={i:6.2f} 网格比={r:5.3f}"
    if ref is not None:
        s += f"  PSNR={psnr(arr, ref):5.2f}dB"
    print(s)
    return r


# ---------- 载入素材 ----------
frames = load_gif_frames(GIF, (IMG, IMG))
T0 = len(frames)
idx = np.linspace(0, T0 - 1, TS * WIN)
lo, hi = np.floor(idx).astype(int), np.minimum(np.floor(idx).astype(int) + 1, T0 - 1)
w = (idx - lo).astype(np.float32)[..., None, None, None]
src = (frames[lo] * (1 - w) + frames[hi] * w).astype(np.float32)
x = src / 127.5 - 1.0                                        # [-1,1]
blocks = to_blocks(x)                                        # [TS,WIN,GRID,PATCH,GRID,PATCH,3]

# ---------- (A) 纯往返：全保留 ----------
flat = torch.from_numpy(np.ascontiguousarray(
    blocks.transpose(0, 2, 4, 1, 3, 5, 6).reshape(TS * GRID * GRID, WIN, PATCH * PATCH * 3)
)).float()
spec_full = torch.fft.rfft2(flat, dim=(-2, -1)) / NORM      # [N,5,25] = 125 系数
print(f"[A] rfft2 输出形状 {tuple(spec_full.shape)} → 共 {spec_full[0].numel()} 个系数/块，"
      f"D=64 只保留 {64/spec_full[0].numel()*100:.0f}%\n")

rec = torch.fft.irfft2(spec_full, s=(WIN, PATCH * PATCH * 3), dim=(-2, -1)).real * NORM
err = (rec.reshape(flat.shape) - flat).abs().max().item()
print(f"[A] 纯 rfft2/irfft2 全保留往返最大误差 = {err:.3e}  → FFT 本身{'有噪声' if err>1e-3 else '零噪声'}\n")


def decode_from_spec(spec, shape):
    """spec: [N, h_fft, w_fft] 已含 NORM 缩放 → 图像 uint8 数组"""
    rec = torch.fft.irfft2(spec, s=(WIN, PATCH * PATCH * 3), dim=(-2, -1)).real * NORM
    rec = rec.clamp(-1.2, 1.2)
    blk = rec.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
    img = blocks_to_img(blk)
    return ((img + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.float32)


# ---------- 当前实现：D=64 前 64 个 ----------
s64 = spec_full.reshape(spec_full.shape[0], -1)[:, :64]
spec64 = torch.zeros(spec_full.shape[0], 5, 25, dtype=spec_full.dtype)
spec64.reshape(spec_full.shape[0], -1)[:, :64] = s64      # 与 encode 的 spec[:, :D] 逐字对齐
a64 = decode_from_spec(spec64, flat.shape)

# ---------- (B) 全保留 125（不截断）----------
a125 = decode_from_spec(spec_full, flat.shape)

# ---------- (C) 矩形低通：保留 h<3 且 w<13（真正的低频角落 3*13=39，再补齐）----------
lowcorner = spec_full[:, :3, :13].clone()
spec_lc = torch.zeros(spec_full.shape[0], 5, 25, dtype=spec_full.dtype)
spec_lc[:, :3, :13] = lowcorner
a_lc = decode_from_spec(spec_lc, flat.shape)

# ---------- (D) Hann 窗平滑衰减（Gibbs 标准解法）----------
win_h = torch.hann_window(5 + 2)[1:-1]
win_w = torch.hann_window(25 + 2)[1:-1]
spec_h = spec_full * (win_h[:, None] * win_w[None, :])
a_hann = decode_from_spec(spec_h, flat.shape)

print("=" * 104)
print("网格感来源定位（素材 = 二次元头像 std=83.7）")
print("=" * 104)
stats("① 原始素材", src)
r125 = stats("② 全保留 125 系数（不截断）", a125, src)
r64 = stats("③ 当前实现：前 64 个（非对称截断）", a64, src)
rlc = stats("④ 矩形低通 h<3,w<13（39 系数）", a_lc, src)
rh = stats("⑤ Hann 窗平滑衰减（全 125 带窗）", a_hann, src)
print("-" * 104)
print(f"网格比：全保留 {r125:.3f} ← 基准 | 当前 {r64:.3f} | 低通 {rlc:.3f} | Hann {rh:.3f}")
print(f"素材自身网格比 = {grid_ratio(src)[2]:.3f}")
print("=" * 104)
