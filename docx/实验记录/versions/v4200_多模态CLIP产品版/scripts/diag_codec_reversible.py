# -*- coding: utf-8 -*-
"""codec 可逆性：绕开帧口径争议，只测field → decode → encode 是否回到原 field

这是回答「是不是缺解码器」最干净的判据：
若 field 能无损往返（误差~1e-7），说明 decode 是精确逆变换，
**「没有可学习解码器」不是瓶颈** —— 问题在别处。
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
_p = ROOT
for _ in range(8):
    if (_p / "deps312").is_dir(): break
    if _p.parent == _p: break
    _p = _p.parent
sys.path.insert(0, str(_p))
if (_p / "deps312").is_dir(): sys.path.insert(0, str(_p / "deps312"))
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from PIL import Image

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
DATA = ROOT / "data" / "gif_raw"
g = [x for x in sorted(DATA.rglob("*.gif")) if "ball" in x.name.lower()][0]
print(f"素材 {g.name}")

fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).float()
print(f"field = {tuple(fld.shape)}")

# decode → 帧 → 存成 png → 再 encode 回field
frames = decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)
arr = np.stack(frames).astype(np.uint8)
tmp = ROOT / "_tmp_rt.gif"
im = Image.fromarray(arr[0])
im.save(tmp, save_all=True, append_images=[Image.fromarray(a) for a in arr[1:]],
         duration=100, loop=0)
fld2 = encode_gif_to_field(str(tmp), TS, GRID, PATCH, WIN, IMG, D).float()
tmp.unlink(missing_ok=True)

d = (fld - fld2).abs()
print(f"\n=== field 往返误差 ===")
print(f"  max|d| = {d.max():.3e}   mean|d| = {d.mean():.3e}")
rel = d.max() / fld.abs().max().clamp(min=1e-9)
print(f"  相对误差 = {rel:.3e}")
if rel < 1e-4:
    print("\n★ codec 是**精确可逆**的 ⇒ 「没有可学习解码器」不是瓶颈")
    print("  decode_field_to_frames 是参数为零的确定性逆 FFT，")
    print("  它已经把 field 完整还原成像素，不存在「还原不了」的问题。")
else:
    print(f"\n★★ codec 本身有损（相对误差 {rel:.2e}）⇒ 这才是真瓶颈")
