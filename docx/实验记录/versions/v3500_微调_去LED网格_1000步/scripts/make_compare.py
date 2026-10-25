# -*- coding: utf-8 -*-
"""生成 v2500 vs v3500 并排对比图，直观看到 LED 网格感的减弱。"""
import sys
import numpy as np
from PIL import Image

IMG = 128
V2500 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_honest.gif"
V3500 = "outputs_real/v3500_e/overfit_honest.gif"


def load(path):
    im = Image.open(path)
    fr = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        fr.append(np.array(im.convert("L"), dtype=np.float32))
    return np.stack(fr)


def grid_highlight(g):
    """高亮 patch 边界跳变：把边界列/行加强，其余压暗，让网格结构显形"""
    out = np.zeros_like(g)
    for t in range(len(g)):
        img = g[t]
        dh = np.abs(np.diff(img, axis=1))
        dv = np.abs(np.diff(img, axis=0))
        m = np.zeros_like(img)
        m[:, :-1] += dh
        m[:-1, :] += dv
        # 归一化增强
        m = m / (m.max() + 1e-6) * 255
        out[t] = np.clip(m, 0, 255)
    return out


a = load(V2500)
b = load(V3500)
n = min(6, len(a), len(b))
idx = np.linspace(0, min(len(a), len(b)) - 1, n).round().astype(int)

# 拼接：上排=v2500，下排=v3500，同一帧竖直对齐
# 每格 128px，间距 8px，边框 2px
CELL, GAP, BORDER = 128, 8, 2
W = n * (CELL + GAP) + GAP
H = 2 * (CELL + GAP) + GAP + 30  # 底部留标题
canvas = np.full((H, W), 245, dtype=np.uint8)

for col, fi in enumerate(idx):
    x0 = GAP + col * (CELL + GAP)
    for row, src in ((0, a), (1, b)):
        y0 = GAP + row * (CELL + GAP)
        img = np.clip(src[fi], 0, 255).astype(np.uint8)
        canvas[y0:y0 + CELL, x0:x0 + CELL] = img
        # 边框
        canvas[y0, x0:x0 + CELL] = 120
        canvas[y0 + CELL - 1, x0:x0 + CELL] = 120
        canvas[y0:y0 + CELL, x0] = 120
        canvas[y0:y0 + CELL, x0 + CELL - 1] = 120

im = Image.fromarray(canvas)
dr = ImageDraw = None
try:
    from PIL import ImageDraw
    dr = ImageDraw.Draw(im)
    dr.text((GAP, GAP + 2 * (CELL + GAP) + 2), "v2500 (上) vs v3500 (下) —— 同一帧上下对齐",
            fill=40)
except Exception:
    pass
im.save("outputs_real/v3500_e/compare_v2500_v3500.png")
print(f"[GEN] outputs_real/v3500_e/compare_v2500_v3500.png  ({n} 帧)")

# 网格高亮对比
ha, hb = grid_highlight(a), grid_highlight(b)
W2 = 2 * (CELL + GAP) + GAP
H2 = CELL + GAP + 20
c2 = np.full((H2, W2), 245, dtype=np.uint8)
for row, src in ((0, ha), ):
    y0 = GAP
    x0 = GAP
    c2[y0:y0 + CELL, x0:x0 + CELL] = np.clip(src[idx[2]], 0, 255).astype(np.uint8)
im2 = Image.fromarray(c2)
im2.save("outputs_real/v3500_e/_grid_probe_a.png")

# 网格感直接量化：把边界列的跳变均值 vs 内部列，做一张 1D 曲线
import numpy as np
PATCH = 4
def band(g):
    dh = np.abs(np.diff(g, axis=2))
    hb = np.array([(j + 1) % PATCH == 0 for j in range(dh.shape[2])])
    return dh[:, :, hb].mean(axis=(0, 1)), dh[:, :, ~hb].mean(axis=(0, 1))
ea, ia = band(a.astype(np.float64))
eb, ib = band(b.astype(np.float64))
print(f"\n帧均边界跳变  v2500 {ea.mean():.2f} → v3500 {eb.mean():.2f}")
print(f"帧均内部跳变  v2500 {ia.mean():.2f} → v3500 {ib.mean():.2f}")
print(f"网格比        v2500 {ea.mean()/ia.mean():.3f} → v3500 {eb.mean()/ib.mean():.3f}")
