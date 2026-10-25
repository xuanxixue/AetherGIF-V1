# -*- coding: utf-8 -*-
"""v3500-t 验收（修正版）：以「原始素材」为基准，不再拿 v2500 当及格线。

旧版验收的错误：拿一个本身重度噪点的 v2500 当降噪基准
（v2500 边界跳变 19.92，是素材 4.995 的 4 倍），导致「降了 3%」也判 PASS。

正确基准：原始素材 = 用户给的图2（平滑、干净、有立体感）。
架构上限 = codec 天花板（编解码引入少量振铃，略高于素材）。
"""
import sys
import numpy as np
from PIL import Image

IMG, PATCH = 128, 4
NEW = sys.argv[1] if len(sys.argv) > 1 else "outputs_real/v3500_t/overfit_honest.gif"
RAW = "data/gif_raw/noto/8-ball_1f3b1.gif"
# 用户给的图2（target 重建）与 v2500 作为参照
REF_TARGET = r"D:/AetherGIF-V1/outputs_real/target_recon_ball.gif"
V2500 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_honest.gif"


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


def metrics(g):
    dh = np.abs(np.diff(g, axis=2)); dv = np.abs(np.diff(g, axis=1))
    hb = np.array([(j + 1) % PATCH == 0 for j in range(dh.shape[2])])
    vb = np.array([(i + 1) % PATCH == 0 for i in range(dv.shape[1])])
    E = dh[:, :, hb].mean() + dv[:, vb, :].mean()
    I = dh[:, :, ~hb].mean() + dv[:, ~vb, :].mean()
    gx = np.diff(g, axis=2); gy = np.diff(g, axis=1)
    mh, mw = min(gx.shape[1], gy.shape[1]), min(gx.shape[2], gy.shape[2])
    sob = np.sqrt(gx[:, :mh, :mw] ** 2 + gy[:, :mh, :mw] ** 2).mean()
    return dict(edge=E, inner=I, ratio=E / I, std=g.std(), sobel=sob,
                fdiff=np.abs(np.diff(g, axis=0)).mean())


raw = metrics(load_gray(RAW))
tgt = metrics(load_gray(REF_TARGET))
base = metrics(load_gray(V2500))
new = metrics(load_gray(NEW))

print("=" * 88)
print("v3500-t 验收（基准已修正：以原始素材为及格线，非 v2500）")
print("=" * 88)
print(f"{'':16s}{'边界跳变':>9s}{'内部跳变':>9s}{'网格比':>8s}{'像素std':>9s}{'Sobel':>8s}{'帧间差':>8s}")
print("-" * 88)
for tag, r in (("原始素材(目标)", raw), ("codec天花板", tgt), ("v2500(起点)", base),
               ("v3500-t(本版)", new)):
    print(f"{tag:16s}{r['edge']:9.2f}{r['inner']:9.2f}{r['ratio']:8.3f}"
          f"{r['std']:9.2f}{r['sobel']:8.2f}{r['fdiff']:8.2f}")

print("\n--- 相对素材的达成度（越接近 100% 越好）---")
items = [
    ("边界跳变(噪点)", "edge", True, 1.6),   # ≤ 素材×1.6（天花板 6.81 是素材 4.995 的 1.36x）
    ("内部跳变(噪点)", "inner", True, 1.6),
    ("像素std(立体感)", "std", False, 0.65),  # ≥ 素材×0.65
    ("Sobel(锐度)", "sobel", True, 2.2),      # ≤ 素材×2.2（天花板 4.89 是素材 4.20 的 1.16x）
]
allpass = True
for name, key, lower_is_better, thr in items:
    ratio = new[key] / raw[key]
    if lower_is_better:
        ok = ratio <= thr
        print(f"  {name:16s} {new[key]:7.2f} = 素材的 {ratio:5.2f}x  要求 ≤{thr}x  "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        ok = ratio >= thr
        print(f"  {name:16s} {new[key]:7.2f} = 素材的 {ratio * 100:5.1f}%  要求 ≥{thr * 100:.0f}%  "
              f"{'PASS' if ok else 'FAIL'}")
    allpass = allpass and ok

print("\n--- 相对 v2500 的改善（参照，非判据）---")
for name, key in (("边界跳变", "edge"), ("内部跳变", "inner"), ("Sobel", "sobel")):
    print(f"  {name:12s} {base[key]:7.2f} → {new[key]:7.2f}  "
          f"{(1 - new[key] / base[key]) * 100:+.1f}%")
print(f"  像素std     {base['std']:7.2f} → {new['std']:7.2f}  "
      f"{(new['std'] / base['std'] - 1) * 100:+.1f}%")

print("\n" + "=" * 88)
print(f"总判定：{'PASS —— 噪点与立体感均达素材门槛' if allpass else 'FAIL'}")
print("=" * 88)
sys.exit(0 if allpass else 1)
