# -*- coding: utf-8 -*-
"""v3500 验收：只盯两个目标（去 LED 网格 + 降噪），四个守卫（细节不许变差）。

对照口径全部用「图像空间差分」，与版本说明.md 第 4 节一致。
用法: python scripts/accept_v3500.py [v3500的honest.gif] [v2500的honest.gif]
"""
import sys, os
import numpy as np
from PIL import Image

IMG, PATCH = 128, 4
V3500 = sys.argv[1] if len(sys.argv) > 1 else "outputs_real/v3500/overfit_honest.gif"
V2500 = sys.argv[2] if len(sys.argv) > 2 else \
    r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_honest.gif"
RAW = "data/gif_raw/noto/8-ball_1f3b1.gif"


def load_gray(path, n=30):
    im = Image.open(path)
    fr = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        a = np.array(im.convert("L"), dtype=np.float32)
        if a.shape[0] != IMG:
            a = np.array(Image.fromarray(a.astype(np.uint8)).resize((IMG, IMG), Image.LANCZOS),
                         dtype=np.float32)
        fr.append(a)
    g = np.stack(fr)
    if len(g) >= n:
        g = g[np.linspace(0, len(g) - 1, n).round().astype(int)]
    return g


def edge_inner(img):
    """边界跳变 / 内部跳变（PATCH=4 网格口径）"""
    dh = np.abs(np.diff(img, axis=2)); dv = np.abs(np.diff(img, axis=1))
    hb = np.array([(j + 1) % PATCH == 0 for j in range(dh.shape[2])])
    vb = np.array([(i + 1) % PATCH == 0 for i in range(dv.shape[1])])
    E = dh[:, :, hb].mean() + dv[:, vb, :].mean()
    I = dh[:, :, ~hb].mean() + dv[:, ~vb, :].mean()
    return float(E), float(I)


def sobel(img):
    gx = np.diff(img, axis=2); gy = np.diff(img, axis=1)
    mh, mw = min(gx.shape[1], gy.shape[1]), min(gx.shape[2], gy.shape[2])
    return float(np.sqrt(gx[:, :mh, :mw] ** 2 + gy[:, :mh, :mw] ** 2).mean())


def frame_diff(img):
    return float(np.abs(np.diff(img, axis=0)).mean())


def report(tag, g):
    E, I = edge_inner(g)
    return dict(tag=tag, edge=E, inner=I, ratio=E / I, std=float(g.std()),
                sobel=sobel(g), fdiff=frame_diff(g))


rows = [report("原始素材", load_gray(RAW)),
        report("v2500(基线)", load_gray(V2500)),
        report("v3500(微调)", load_gray(V3500))]
base, new = rows[1], rows[2]

print("=" * 78)
print("v3500 验收（只盯两个目标，四个守卫）")
print("=" * 78)
hdr = f"{'':14s}{'网格比':>8s}{'边界跳变':>9s}{'内部跳变':>9s}{'像素std':>9s}{'Sobel':>8s}{'帧间差':>8s}"
print(hdr)
print("-" * 78)
for r in rows:
    print(f"{r['tag']:14s}{r['ratio']:8.3f}{r['edge']:9.3f}{r['inner']:9.3f}"
          f"{r['std']:9.2f}{r['sobel']:8.2f}{r['fdiff']:8.2f}")

print("\n--- 目标 1：去 LED 网格 ---")
gr_b, gr_n = base["ratio"], new["ratio"]
print(f"  网格比  v2500 {gr_b:.3f} → v3500 {gr_n:.3f}   目标 ≤1.30")
print(f"  降幅 {(1 - gr_n / gr_b) * 100:.1f}%   {'PASS' if gr_n <= 1.30 else 'FAIL'}")

print("\n--- 目标 2：降噪（边界跳变绝对值）---")
print(f"  边界跳变  v2500 {base['edge']:.2f} → v3500 {new['edge']:.2f}   原始素材 4.99")
print(f"  降幅 {(1 - new['edge'] / base['edge']) * 100:.1f}%   "
      f"{'PASS' if new['edge'] < base['edge'] else 'FAIL'}")

print("\n--- 守卫：细节 / 质感 不许变差（≥ 基线 90%）---")
ok = True
for key, name in (("inner", "内部跳变(细节)"), ("std", "像素std(对比度)"),
                  ("sobel", "Sobel锐度(质感)"), ("fdiff", "帧间差(运动轨迹)")):
    keep = new[key] / base[key]
    good = keep >= 0.90
    ok = ok and good
    print(f"  {name:16s} {base[key]:8.2f} → {new[key]:8.2f}  保留 {keep * 100:5.1f}%  "
          f"{'OK' if good else '跌破90% FAIL'}")

print("\n" + "=" * 78)
final = (gr_n <= 1.30) and (new["edge"] < base["edge"]) and ok
print(f"总判定：{'PASS —— 两个目标达成且细节未损失' if final else 'FAIL'}")
print("=" * 78)
sys.exit(0 if final else 1)
