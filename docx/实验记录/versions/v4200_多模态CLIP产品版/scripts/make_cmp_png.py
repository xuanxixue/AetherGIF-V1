# -*- coding: utf-8 -*-
"""把生成 GIF 的关键帧与素材并排导出成PNG，直接看图判断内容对不对。

铁律：ac1 / sobel 等平均值指标已被证实会在规则纹理与均匀场上骗人两次，
     任何验收都必须看图。本脚本就是为此存在。
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else "thetahat150"
src = ROOT / "data" / "gif_raw"
out = ROOT / "outputs_real" / tag
gdir = out

def gif_frames(p):
    im = Image.open(p)
    fr = []
    for i in range(im.n_frames):
        im.seek(i)
        fr.append(np.array(im.convert("RGB")))
    return np.stack(fr)

# 素材
mats = sorted(src.glob("*.gif"))
gen_files = [gdir / f"三路全给_src{i}.gif" for i in (0, 1)]

rows = []
labels = []
for i, mp in enumerate(mats[:2]):
    m = gif_frames(mp)
    idx = np.linspace(0, len(m) - 1, 4).astype(int)
    rows.append(np.concatenate([m[j] for j in idx], axis=1))
    labels.append(f"素材{i}")
for i, gp in enumerate(gen_files):
    if not gp.exists():
        continue
    g = gif_frames(gp)
    idx = np.linspace(0, len(g) - 1, 4).astype(int)
    rows.append(np.concatenate([g[j] for j in idx], axis=1))
    labels.append(f"生成_src{i}")

hmin = min(r.shape[0] for r in rows)
wmin = min(r.shape[1] for r in rows)
rows = [r[:hmin, :wmin] for r in rows]
canvas = np.concatenate(rows, axis=0)
p = out / "_cmp.png"
Image.fromarray(canvas).save(p)
print("saved", p)
for lb, r in zip(labels, rows):
    print(f"  {lb}: std={r.std():.2f} shape={r.shape}")
