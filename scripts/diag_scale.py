# -*- coding: utf-8 -*-
"""振幅缩放扫描：定位 PSNR 峰值与 decode clamp 的影响。

用途（诊断 PSNR 反常）：方案 G 振幅偏得最少（频段比 0.8）却 PSNR 最低（17.71），
方案 D 振幅偏得最多（2.3~3.0）却 PSNR 最高（22.46）——反直觉。本脚本对模型输出
振幅乘一个标量 k，扫 PSNR / frame-diff / 像素均值，定位：
  1. PSNR 峰值在哪个 k（若 k≠1，说明模型自身振幅标度没校准）
  2. normalize=False（固定 clamp）与 normalize=True（min-max）两种解码下峰值是否一致
     ——若一致，说明不是解码 clamp 造成的假象，是模型真的错
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
CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs_real/overfit_ckpt.pt"
device = "cuda"

gp = next(Path("data/gif_raw/noto").glob("8-ball_1f3b1.gif"))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
th_tar = torch.atan2(im, re)

model = AetherGIFModel(device=device, use_qwen=False).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
with torch.inference_mode():
    a, th, _ = model(texts=["台球"], target_field=target,
                     t_coord=t, s_coord=s, batch_size=1)


def dec(a_, th_, norm):
    z = (a_[0] * torch.exp(1j * th_[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    return np.stack(
        decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D, normalize=norm)
    ).astype(np.float32)


zt = (a_tar[0] * torch.exp(1j * th_tar[0])).to(torch.complex64)
ft = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()

for norm in (False, True):
    fr_tar = np.stack(
        decode_field_to_frames(ft, TS, GRID, PATCH, WIN, IMG, D, normalize=norm)
    ).astype(np.float32)
    print(f"--- normalize={norm}  target pix_mean={fr_tar.mean():.1f} ---")
    best = (-1, None)
    for k in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 2.0):
        arr = dec(a * k, th, norm)
        p = 10 * math.log10(255.0 ** 2 / max(((arr - fr_tar) ** 2).mean(), 1e-12))
        fd = float(np.abs(np.diff(arr, axis=0)).mean())
        mark = ""
        if p > best[0]:
            best = (p, k)
            mark = "  <== PSNR peak"
        print(f"  scale={k:4.2f}  PSNR={p:6.2f} dB  frame-diff={fd:5.2f}  "
              f"pix_mean={arr.mean():6.1f}{mark}")
    print(f"  => PSNR 峰值在 scale={best[1]:.2f}（{best[0]:.2f} dB）")
    print()