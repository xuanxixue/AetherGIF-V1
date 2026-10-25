# -*- coding: utf-8 -*-
"""V3000 验收评估：对比 v2500（基线）vs v3000（方案 I）在三个修复指标上的表现。

三个指标（均对应本轮修复目标）：
  1. 对比度：灰度 std + 5-95% 跨度（越接近 target 越好）
  2. 颗粒感：边界跳变 / patch 内部跳变 比值（target≈1.01，codec 天花板≈1.23，越接近 1.23 越好）
  3. 噪声/过冲：逐通道振幅比（理想 1.0，AC 过冲是噪声源）
  4. PSNR（clamp 口径，与训练一致）
"""
import os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
from PIL import Image
import torch
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames, load_gif_frames, resample_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
V2500_DIR = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real"
V3000_DIR = os.path.join(BASE, "outputs_real")
CKPT = os.path.join(V3000_DIR, "overfit_ckpt.pt")
device = "cuda"


def load_gif_frames_u8(path, target_T=30):
    img = Image.open(path)
    frames = []
    for i in range(getattr(img, "n_frames", 1)):
        img.seek(i)
        fr = np.array(img.convert("RGB"), dtype=np.uint8)
        if fr.shape[0] != IMG:
            fr = np.array(Image.fromarray(fr).resize((IMG, IMG), Image.Resampling.LANCZOS))
        frames.append(fr)
    return np.stack(frames)


def psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    return 10 * math.log10(255.0 ** 2 / max(((a - b) ** 2).mean(), 1e-12))


def contrast(f):
    g = f.astype(np.float32).mean(-1)
    return float(g.std()), float(np.percentile(g, 95) - np.percentile(g, 5))


def grid_ratio(frames):
    g = frames.astype(np.float32).mean(-1)  # [T,H,W]
    T, H, W = g.shape
    p = PATCH
    intra, inter = [], []
    for t in range(T):
        img = g[t]
        dh = np.abs(np.diff(img, axis=1))
        dv = np.abs(np.diff(img, axis=0))
        inter_h = dh[:, [i for i in range(W - 1) if (i + 1) % p == 0]]
        intra_h = dh[:, [i for i in range(W - 1) if (i + 1) % p != 0]]
        inter_v = dv[[i for i in range(H - 1) if (i + 1) % p == 0], :]
        intra_v = dv[[i for i in range(H - 1) if (i + 1) % p != 0], :]
        inter.append(np.concatenate([inter_h.ravel(), inter_v.ravel()]).mean())
        intra.append(np.concatenate([intra_h.ravel(), intra_v.ravel()]).mean())
    return np.mean(inter) / max(np.mean(intra), 1e-6)


# --- 参考对象 ---
raw = load_gif_frames(GIF, (IMG, IMG))
raw = resample_frames(raw, TS * WIN)
target = raw

field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D)
ceil = np.stack(decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False))

# target 逐通道振幅分布（loss 口径）
a_tar = torch.sqrt(field[:, :D] ** 2 + field[:, D:] ** 2 + 1e-12).mean(0)

print("=" * 78)
print("V3000 验收：v2500（基线） vs v3000（方案 I）")
print("=" * 78)
print("参考：target 灰度std=%.1f 跨度=%.1f  颗粒感=%.2f"
      % (contrast(target)[0], contrast(target)[1], grid_ratio(target)))
print("参考：codec 天花板 灰度std=%.1f 跨度=%.1f  颗粒感=%.2f  PSNR=%.2f dB"
      % (contrast(ceil)[0], contrast(ceil)[1], grid_ratio(ceil), psnr(ceil, target)))
print("-" * 78)

# v2500（诚实重建口径）
for tag, d in [("v2500-honest", V2500_DIR), ("v3000-honest", V3000_DIR)]:
    p = os.path.join(d, "overfit_honest.gif")
    if not os.path.exists(p):
        print("%s: 未找到产物，跳过" % tag)
        continue
    fr = load_gif_frames_u8(p)
    sd, span = contrast(fr)
    print("%-16s PSNR=%.2f dB  对比度std=%.1f 跨度=%.1f  颗粒感=%.2f"
          % (tag, psnr(fr, target), sd, span, grid_ratio(fr)))

print("-" * 78)
# 用 checkpoint 做逐通道振幅比（噪声/过冲）
print("逐通道振幅比（模型/target，1.0=校准）：")
for tag, ckpt in [("v2500", os.path.join(V2500_DIR, "overfit_ckpt.pt")),
                  ("v3000", CKPT)]:
    if not os.path.exists(ckpt):
        print("  %s: checkpoint 缺失" % tag)
        continue
    model = AetherGIFModel(device=device, use_qwen=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    t, s = build_spacetime_index(TS, GRID, GRID, device)
    with torch.inference_mode():
        a, th, _ = model(texts=["台球"], target_field=field.unsqueeze(0), t_coord=t, s_coord=s, batch_size=1)
    ratio = (a[0].mean(0) / a_tar.clamp(min=1e-6)).cpu()
    bands = {
        "DC(w0)": [c for c in range(D) if c % 25 == 0],
        "低频w1-6": [c for c in range(D) if 1 <= c % 25 <= 6],
        "中频w7-16": [c for c in range(D) if 7 <= c % 25 <= 16],
        "高频w17-24": [c for c in range(D) if 17 <= c % 25 <= 24],
    }
    bstr = " ".join("%s=%.2f" % (n, ratio[idxs].mean()) for n, idxs in bands.items())
    print("  %-8s %s  (a_out=%.4f)" % (tag, bstr, a.mean().item()))
print("=" * 78)