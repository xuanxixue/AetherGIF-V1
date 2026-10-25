# -*- coding: utf-8 -*-
"""生成 素材 vs v2500 vs v3500_t 三方对比图。"""
import sys
import numpy as np
from PIL import Image, ImageDraw

IMG, PATCH = 128, 4
TARGET = r"D:/AetherGIF-V1/outputs_real/target_recon_ball.gif"
V2500 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_honest.gif"
V3500T = "outputs_real/v3500_t/overfit_honest.gif"


def load(path):
    im = Image.open(path)
    fr = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        fr.append(np.array(im.convert("L"), dtype=np.float32))
    return np.stack(fr)


a, b, c = load(TARGET), load(V2500), load(V3500T)
n = min(6, len(a), len(b), len(c))
idx = np.linspace(0, min(len(a), len(b), len(c)) - 1, n).round().astype(int)

CELL, GAP = 128, 10
W = n * (CELL + GAP) + GAP
H = 3 * (CELL + GAP) + GAP + 46
canvas = np.full((H, W), 250, dtype=np.uint8)

rows = [("素材 (目标)", a), ("v2500 (起点)", b), ("v3500_t (本版)", c)]
for r, (label, src) in enumerate(rows):
    for col, fi in enumerate(idx):
        x0 = GAP + col * (CELL + GAP)
        y0 = GAP + r * (CELL + GAP)
        img = np.clip(src[fi], 0, 255).astype(np.uint8)
        canvas[y0:y0 + CELL, x0:x0 + CELL] = img
        canvas[y0, x0:x0 + CELL] = 110
        canvas[y0 + CELL - 1, x0:x0 + CELL] = 110
        canvas[y0:y0 + CELL, x0] = 110
        canvas[y0:y0 + CELL, x0 + CELL - 1] = 110

im = Image.fromarray(canvas)
dr = ImageDraw.Draw(im)
for r, (label, src) in enumerate(rows):
    y0 = GAP + r * (CELL + GAP)
    dr.text((GAP, y0 + CELL + 1), label, fill=30)
dr.text((GAP, H - 24),
        "同一帧上下三行对齐。v3500_t 噪点:边界 19.9->7.5 / 内部 12.3->6.4 / Sobel 11.3->5.4",
        fill=30)
out = "outputs_real/v3500_t/compare_3way.png"
im.save(out)
print(f"[GEN] {out}")
