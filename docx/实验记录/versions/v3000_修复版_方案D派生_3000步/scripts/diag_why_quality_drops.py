# -*- coding: utf-8 -*-
"""回答「为什么质量一直在降」—— 空间频率域取证。

假设：每加一个标量 loss（振幅比/ wcos / 全局对比度），都在把模型
      从「视觉讨巧」推向「指标正确」，而这两者在当前架构下冲突。

要测三件事：
  1. 空间频率能量分布：v2500 的「细节」到底落在哪个频率？是真结构还是高频噪声？
  2. 空间相关性：v2500 的高频纹理与 target 的高频位置是否对应？
     对应 = 真结构；不对应 = 伪造细节（人眼仍会读作"质感"）
  3. 频段 vs 视觉锐度的关系：解释为什么"能量对了反而糊"
"""
import os, sys, math
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "src"))
import numpy as np
from PIL import Image

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
V25 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_honest.gif"
VJ = os.path.join(BASE, "outputs_real", "v3000J", "overfit_honest.gif")
VK = os.path.join(BASE, "outputs_real", "v3000K", "overfit_honest.gif")


def rd(p, n=30):
    im = Image.open(p)
    fr = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        a = np.array(im.convert("L"), dtype=np.float32)
        if a.shape[0] != IMG:
            a = np.array(Image.fromarray(a.astype(np.uint8)).resize((IMG, IMG), Image.LANCZOS),
                         dtype=np.float32)
        fr.append(a)
    g = np.stack(fr)
    idx = np.linspace(0, len(g) - 1, n).round().astype(int)
    return g[idx]


src = {"原始GIF": rd(GIF), "v2500": rd(V25)}
for k, p in (("v3000J", VJ), ("v3000K", VK)):
    if os.path.exists(p):
        src[k] = rd(p)

# ---------- 1. 空间频率能量分布（径向） ----------
H = 128
yy, xx = np.mgrid[0:H, 0:H]
cy, cx = H // 2, H // 2
rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
NB = H // 2
RINGS = [(1, 4), (5, 12), (13, 24), (25, 40), (41, 63)]
NAMES = ["最低频(轮廓)", "低频(大块)", "中频(形体)", "中高频(细节)", "高频(纹理/噪点)"]

print("=" * 96)
print("1. 空间频率能量分布（径向 FFT，各频段占总能量 %）—— 灰度图，30帧平均")
print("=" * 96)
print("%-14s %s" % ("版本", "  ".join("%-14s" % n for n in NAMES)))
spec = {}
for name, g in src.items():
    gf = g - g.mean(axis=(1, 2), keepdims=True)
    F = np.abs(np.fft.fftshift(np.fft.fft2(gf), axes=(1, 2))) ** 2
    tot = F.sum(axis=(1, 2)).mean()
    row = []
    for (a, b) in RINGS:
        mask = (rr >= a) & (rr <= b)
        e = F[:, mask].sum() / (len(g) * tot)
        row.append(e * 100)
    spec[name] = F
    print("%-14s %s" % (name, "  ".join("%-14s" % ("%.2f%%" % v) for v in row)))

print()
print("=" * 96)
print("2. 关键对比：最高频段(纹理) 能量的相对倍数 —— 相对 原始GIF")
print("=" * 96)
base = spec["原始GIF"][:, (rr >= 41) & (rr <= 63)].sum()
for name in src:
    hf = spec[name][:, (rr >= 41) & (rr <= 63)].sum()
    print("  %-10s 高频能量 = 原始GIF 的 %6.2f 倍" % (name, hf / base))

print()
print("=" * 96)
print("3. 空间相关性：v2500 的高频纹理 与 原始GIF 的高频位置 对应吗？")
print("   对应 → v2500 复原了真实结构；不对应 → 是自己编的纹理（人眼仍读作'质感'）")
print("=" * 96)
F_ref = spec["原始GIF"].mean(0)
for name in src:
    if name == "原始GIF":
        continue
    F_n = spec[name].mean(0)
    a = F_n.ravel(); b = F_ref.ravel()
    cc = np.corrcoef(a, b)[0, 1]
    # 只在高频环上算相关
    m = ((rr >= 25) & (rr <= 63)).ravel()
    cc_hf = np.corrcoef(a[m], b[m])[0, 1]
    print("  %-10s 全谱 corr=%.4f   高频环corr=%.4f" % (name, cc, cc_hf))

print()
print("=" * 96)
print("4. 视觉锐度 vs 高频能量的矛盾（解释「能量对了反而糊」）")
print("=" * 96)
print("  %-10s %12s %12s %12s" % ("版本", "高频倍数", "像素std", "Sobel锐度"))
for name, g in src.items():
    hf = spec[name][:, (rr >= 41) & (rr <= 63)].sum() / base
    sd = g.std()
    gx = np.diff(g, axis=2); gy = np.diff(g, axis=1)
    mh, mw = min(gx.shape[1], gy.shape[1]), min(gx.shape[2], gy.shape[2])
    sob = np.sqrt(gx[:, :mh, :mw] ** 2 + gy[:, :mh, :mw] ** 2).mean()
    print("  %-10s %11.2fx %12.2f %12.2f" % (name, hf, sd, sob))
print("=" * 96)