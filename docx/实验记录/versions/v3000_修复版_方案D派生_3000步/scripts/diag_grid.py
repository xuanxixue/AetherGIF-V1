# -*- coding: utf-8 -*-
"""量化「LED 颗粒感」的可修复边界：patch 网格结构在 target 表征里就存在多少。

核心问题：target 也是用 codec encode→decode 生成的，4×4 patch 结构是表征层固有属性。
如果 target 自身就带明显的块间跳变，那模型再准也消不掉颗粒感——这是 codec 上限，
只能靠「相邻 patch 平滑约束」在模型端缓解（让块间过渡更自然）。

测量：
  1. 对 target / ceiling / 模型输出，计算相邻 patch 边界的亮度跳变 vs patch 内部跳变。
     "颗粒感" = 边界跳变 >> 内部跳变 的比值越大越明显。
  2. 空间自相关：相邻 patch 均值差的标准差。
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from PIL import Image
from codec.gif_codec import load_gif_frames, resample_frames

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
IMG = 128


def measure_grid_artifact(frames_u8):
    """返回 (边界跳变均值, patch内部跳变均值, 比值)。"""
    f = frames_u8.astype(np.float32)
    if f.ndim == 3:
        f = f[None]  # [1,H,W,C]
    # 取灰度
    g = f.mean(-1)  # [T,H,W]
    T, H, W = g.shape
    patch = 4
    n_h, n_w = H // patch, W // patch

    # patch 内部跳变：相邻像素差
    intra = []
    # 边界跳变：跨 patch 边界的相邻像素差（每 4 行/列一条边界）
    inter = []
    for t in range(T):
        img = g[t]
        # 所有相邻像素差（水平+垂直）
        dh = np.abs(np.diff(img, axis=1))  # [H,W-1]
        dv = np.abs(np.diff(img, axis=0))  # [H-1,W]
        # 边界列：x = patch-1, 2*patch-1, ... 即 (x+1) % patch == 0
        inter_h = dh[:, [i for i in range(W - 1) if (i + 1) % patch == 0]]
        intra_h = dh[:, [i for i in range(W - 1) if (i + 1) % patch != 0]]
        inter_v = dv[[i for i in range(H - 1) if (i + 1) % patch == 0], :]
        intra_v = dv[[i for i in range(H - 1) if (i + 1) % patch != 0], :]
        inter.append(np.concatenate([inter_h.ravel(), inter_v.ravel()]).mean())
        intra.append(np.concatenate([intra_h.ravel(), intra_v.ravel()]).mean())
    return float(np.mean(inter)), float(np.mean(intra))


print("=" * 70)
print("LED 颗粒感量化：边界跳变 vs patch 内部跳变（比值越接近 1 颗粒感越弱）")
print("=" * 70)

# 1. 原图（无分块，理想上限）
raw = load_gif_frames(GIF, (IMG, IMG))
raw_res = resample_frames(raw, 30)
inter, intra = measure_grid_artifact(raw_res)
print("原图(无patch)        边界跳变=%.3f 内部跳变=%.3f 比值=%.2f" % (inter, intra, inter / max(intra, 1e-6)))

# 2. codec 往返（表征层，patch=4 结构）
import torch
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
D, GRID, PATCH, WIN, TS = 64, 32, 4, 5, 6
field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D)
ceil_frames = decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)
ceiling = np.stack(ceil_frames)
inter, intra = measure_grid_artifact(ceiling)
print("codec往返(天花板)     边界跳变=%.3f 内部跳变=%.3f 比值=%.2f" % (inter, intra, inter / max(intra, 1e-6)))

# 3. v2500 模型产物
for tag, p in [("v2500-honest", r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_honest.gif"),
               ("v2500-main", r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit.gif")]:
    img = Image.open(p)
    frames = []
    for i in range(getattr(img, "n_frames", 1)):
        img.seek(i)
        fr = np.array(img.convert("RGB"), dtype=np.uint8)
        if fr.shape[0] != IMG:
            fr = np.array(Image.fromarray(fr).resize((IMG, IMG), Image.Resampling.LANCZOS))
        frames.append(fr)
    frames = np.stack(frames)
    inter, intra = measure_grid_artifact(frames)
    print("%-22s 边界跳变=%.3f 内部跳变=%.3f 比值=%.2f" % (tag, inter, intra, inter / max(intra, 1e-6)))

print("=" * 70)
print("解读：比值≈1 表示颗粒感消失；比值>>1 表示每 4 像素一条明显边界线。")
print("若 ceiling 的比值已接近原图的比值，说明颗粒感是 codec 固有、模型无法消除。")