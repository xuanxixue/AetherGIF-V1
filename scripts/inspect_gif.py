# -*- coding: utf-8 -*-
"""检查生成 GIF 的帧统计：尺寸、帧数、像素分布、帧间差。"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

def inspect(p):
    with Image.open(p) as img:
        n = getattr(img, "n_frames", 1)
        size = img.size
        frames = []
        for i in range(n):
            try:
                img.seek(i)
                frames.append(np.array(img.convert("RGB"), dtype=np.float32))
            except EOFError:
                break
    arr = np.stack(frames)
    print(f"\n=== {p} ===")
    print(f"frames={len(arr)} size={size}")
    print(f"per-frame mean : {arr.reshape(len(arr), -1).mean(1).round(1).tolist()}")
    print(f"per-frame std  : {arr.reshape(len(arr), -1).std(1).round(1).tolist()}")
    print(f"global  min/max: {arr.min():.0f} / {arr.max():.0f}")
    print(f"unique vals (q): {np.unique(np.quantile(arr, [0.01,0.25,0.5,0.75,0.99]).round(0)).tolist()}")
    if len(arr) > 1:
        d = np.abs(np.diff(arr, axis=0)).mean()
        print(f"mean frame-diff: {d:.2f}")
    # 极端值占比：判断是否过曝/全黑
    frac_sat = float(((arr <= 2) | (arr >= 253)).mean())
    print(f"frac saturated : {frac_sat:.3f}")

for f in sys.argv[1:]:
    inspect(Path(f))
