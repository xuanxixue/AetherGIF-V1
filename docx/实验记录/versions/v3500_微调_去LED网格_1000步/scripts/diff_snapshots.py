# -*- coding: utf-8 -*-
"""诊断：扫描 v3500 各步快照，找出对比度/帧间差是从哪一步开始跌破 90% 的。"""
import sys, os, glob
import numpy as np, torch
from PIL import Image

sys.path.insert(0, "."); sys.path.insert(0, "src")
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
device = "cuda"
BASE = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt"

field = encode_gif_to_field("data/gif_raw/noto/8-ball_1f3b1.gif", TS, GRID, PATCH, WIN, IMG, D)
t, s = build_spacetime_index(TS, GRID, GRID, device)
fc = field.unsqueeze(0).to(device)


def decode(a0, th0):
    """与 overfit.py 的 honest 重建同口径：固定 clamp，不做 min-max 归一化"""
    N = a0.shape[0]
    z = torch.zeros(N, 2 * D, device=device)
    z[:, :D] = a0 * torch.cos(th0); z[:, D:] = a0 * torch.sin(th0)
    zc = torch.complex(z[:, :D], z[:, D:])
    s_dim = PATCH * PATCH * 3
    full = WIN * (s_dim // 2 + 1)
    if D < full:
        zc = torch.cat([zc, torch.zeros(N, full - D, device=device, dtype=zc.dtype)], -1)
    zc = zc * 10.0
    rec = torch.fft.irfft2(zc.reshape(N, WIN, s_dim // 2 + 1), s=(WIN, s_dim), dim=(-2, -1)).real
    rec = rec.clamp(-1, 1) * 127.5 + 127.5
    # 还原成 [TS*WIN, 128, 128, 3]
    rec = rec.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3).permute(0, 3, 1, 4, 2, 5, 6)
    rec = rec.reshape(TS * WIN, GRID * PATCH, GRID * PATCH, 3)
    return rec.mean(-1).cpu().numpy()          # 灰度 [T,128,128]


def metrics(g):
    dh = np.abs(np.diff(g, axis=2)); dv = np.abs(np.diff(g, axis=1))
    hb = np.array([(j + 1) % PATCH == 0 for j in range(dh.shape[2])])
    vb = np.array([(i + 1) % PATCH == 0 for i in range(dv.shape[1])])
    E = dh[:, :, hb].mean() + dv[:, vb, :].mean()
    I = dh[:, :, ~hb].mean() + dv[:, ~vb, :].mean()
    gx = np.diff(g, axis=2); gy = np.diff(g, axis=1)
    mh, mw = min(gx.shape[1], gy.shape[1]), min(gx.shape[2], gy.shape[2])
    sob = np.sqrt(gx[:, :mh, :mw] ** 2 + gy[:, :mh, :mw] ** 2).mean()
    return dict(ratio=E / I, edge=E, inner=I, std=g.std(), sobel=sob,
                fdiff=np.abs(np.diff(g, axis=0)).mean())


def run(ckpt):
    m = AetherGIFModel(device=device, use_qwen=False).to(device)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd); m.eval()
    torch.manual_seed(1234)
    with torch.inference_mode():
        a, th, _ = m(texts=["台球"], target_field=fc, t_coord=t, s_coord=s, batch_size=1)
    return metrics(decode(a[0], th[0]))


print("=" * 92)
print("各快照指标演化（找对比度跌破 90% 的临界步）")
print("=" * 92)
print(f"{'ckpt':26s}{'网格比':>8s}{'边界跳变':>9s}{'内部跳变':>9s}{'像素std':>9s}{'Sobel':>8s}{'帧间差':>8s}")
print("-" * 92)

rows = [("v2500 基线", BASE)]
for p in sorted(glob.glob("outputs_real/v3500/overfit_ckpt_step*.pt")):
    rows.append((os.path.basename(p).replace("overfit_ckpt_", "").replace(".pt", ""), p))
rows.append(("v3500 best", "outputs_real/v3500/overfit_ckpt_best.pt"))
rows.append(("v3500 终态", "outputs_real/v3500/overfit_ckpt.pt"))

res = []
for tag, p in rows:
    m = run(p)
    res.append((tag, m))
    print(f"{tag:26s}{m['ratio']:8.3f}{m['edge']:9.2f}{m['inner']:9.2f}"
          f"{m['std']:9.2f}{m['sobel']:8.2f}{m['fdiff']:8.2f}")

b = res[0][1]
print("\n--- 相对 v2500 基线保留率（<90% 即跌破）---")
for tag, m in res[1:]:
    line = f"{tag:26s}"
    for k, nm in (("std", "std"), ("fdiff", "帧间差"), ("sobel", "Sobel"), ("inner", "内部")):
        keep = m[k] / b[k] * 100
        line += f" {nm}={keep:5.1f}%{'*' if keep < 90 else ' '}"
    print(line)
print("\n* = 跌破 90% 守卫")
