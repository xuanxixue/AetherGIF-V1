# -*- coding: utf-8 -*-
"""精确定位 v2500 高频里的两类成分：
   A. 真实细节（运动轨迹、残影、球色分层）→ 用户要保住
   B. LED 网格感（4x4 patch 边界跳变）      → 用户讨厌

关键：两者在空间频率上可分离吗？
   - patch 边界跳变是**周期性**的，能量集中在特定的空间频率
   - 真实细节是**非周期**的，散布在所有高频
做法：把 v2500 的高频按「是否与 patch 边界对齐」分开，
     算各自占多少，再看原图对应成分是多少。
"""
import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "src"))
import numpy as np
from PIL import Image

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
V25 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_honest.gif"
VJ = os.path.join(BASE, "outputs_real", "v3000J", "overfit_honest.gif")


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
    return g[np.linspace(0, len(g) - 1, n).round().astype(int)]


im = Image.open(GIF)
fr = []
for i in range(getattr(im, "n_frames", 1)):
    im.seek(i)
    fr.append(np.array(im.convert("L"), dtype=np.float32))
raw = np.stack(fr)
src = {"原始GIF": raw[np.linspace(0, len(raw) - 1, 30).round().astype(int)],
       "v2500": rd(V25), "v3000J": rd(VJ)}


def split_patch_energy(g):
    """分离 patch 边界跳变 与 patch 内部细节 的能量。
    注意：差分后维度少 1，边界掩码必须按差分后的长度构造。"""
    inter, intra = [], []
    for t in range(len(g)):
        img = g[t]
        dh = np.abs(np.diff(img, axis=1))    # [H, W-1]
        dv = np.abs(np.diff(img, axis=0))    # [H-1, W]
        # 水平差分的列 j 对应像素 j 与 j+1 之间的边界 -> 边界在 (j+1)%PATCH==0
        hb = np.array([(j + 1) % PATCH == 0 for j in range(dh.shape[1])])
        inter.append(dh[:, hb].mean()); intra.append(dh[:, ~hb].mean())
        vb = np.array([(i + 1) % PATCH == 0 for i in range(dv.shape[0])])
        inter.append(dv[vb, :].mean()); intra.append(dv[~vb, :].mean())
    return float(np.mean(inter)), float(np.mean(intra))


def radial(g):
    gf = g - g.mean(axis=(1, 2), keepdims=True)
    F = np.abs(np.fft.fftshift(np.fft.fft2(gf), axes=(1, 2))) ** 2
    H = g.shape[1]
    yy, xx = np.mgrid[0:H, 0:H]
    rr = np.sqrt((yy - H // 2) ** 2 + (xx - H // 2) ** 2).astype(np.int32)
    return F, rr


print("=" * 100)
print("1. LED 网格感的直接度量：patch 边界跳变 / patch 内部跳变")
print("   =1.0 表示完全没有网格结构；>1 表示边界比内部跳变更剧烈（网格感）")
print("=" * 100)
print("%-12s %14s %14s %14s" % ("版本", "边界跳变", "内部跳变", "网格比"))
res = {}
for name, g in src.items():
    iv, ia = split_patch_energy(g)
    res[name] = iv / max(ia, 1e-6)
    print("%-12s %14.3f %14.3f %14.3f" % (name, iv, ia, res[name]))
print()
print(">>> codec 天花板的网格比由 rfft2 截断决定，是**固有代价**，躲不掉。")

print()
print("=" * 100)
print("2. 高频能量：拆成「patch 周期分量」和「非周期分量」")
print("   做法：频域做 4x4 周期掩膜 —— 与 patch 周期对齐的频率成分 = 网格感来源")
print("=" * 100)
for name, g in src.items():
    F, rr = radial(g)
    H = g.shape[1]
    fx = np.fft.fftshift(np.fft.fftfreq(H)) * H
    fy = np.fft.fftshift(np.fft.fftfreq(H)) * H
    FX, FY = np.meshgrid(fx, fy)
    # 与 4x4 网格周期对齐的频率（GRID=32 个 patch，频率 = 32*k）
    per = np.abs(((FX * GRID / H).round() - FX * GRID / H) < 0.15) & \
          np.abs(((FY * GRID / H).round() - FY * GRID / H) < 0.15) & (np.abs(FX) > 2) & (np.abs(FY) > 2)
    tot = F.sum()
    hp = rr >= 25
    e_hf = F[:, hp].sum() / tot
    e_per = F[:, hp & per].sum() / tot
    print("  %-10s 高频总量=%6.2f%%  其中 patch周期=%6.2f%%  非周期(真实细节)=%6.2f%%  周期占比=%.0f%%"
          % (name, e_hf * 100, e_per * 100, (e_hf - e_per) * 100,
             (e_per / max(e_hf, 1e-9)) * 100))

print()
print("=" * 100)
print("3. 时间维度：运动轨迹 / 残影 靠 frame-diff 体现")
print("=" * 100)
print("%-12s %16s %16s" % ("版本", "frame-diff", "逐帧std 变异系数"))
for name, g in src.items():
    fd = np.abs(np.diff(g, axis=0)).mean()
    per = g.reshape(len(g), -1).std(1)
    print("%-12s %16.3f %16.3f" % (name, fd, per.std() / max(per.mean(), 1e-9)))
print()
print("=" * 100)