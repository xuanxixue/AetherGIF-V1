# -*- coding: utf-8 -*-
"""直接回答「是不是缺解码器」：codec 往返的天花板到底有多高？

流程：真实 GIF → encode_gif_to_field → decode_field_to_frames → 与原图比 PSNR
若PSNR 极高（如 100+ dB），说明「没有可学习解码器」不是瓶颈，
因为固定逆变换已经是精确逆运算。
"""
import sys, math
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
_p = ROOT
for _ in range(8):
    if (_p / "deps312").is_dir(): break
    if _p.parent == _p: break
    _p = _p.parent
sys.path.insert(0, str(_p))
if (_p / "deps312").is_dir(): sys.path.insert(0, str(_p / "deps312"))
import torch
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames, load_gif_frames, resample_frames

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
DATA = ROOT / "data" / "gif_raw"

for g in sorted(DATA.rglob("*.gif")):
    fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).float()
    fr = decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)
    # ★ 基准口径：encode 内部做了 resample_frames(frames, TS)，
    #   所以基准也必须走同一路径，否则是苹果比橘子（上一次测出 12 dB 的原因）。
    base = resample_frames(load_gif_frames(str(g), (IMG, IMG)), TS).astype(np.float32)
    #   decode 出来的是 TS*WIN 帧（每个时间片展开 WIN 帧），
    #   基准同样按 nearest 展开到同一长度再比。
    n = len(fr)
    idx = np.linspace(0, len(base) - 1, n).round().astype(int)
    raw = base[idx].astype(np.float64)
    fo = np.stack(fr).astype(np.float64)
    mse = ((fo - raw) ** 2).mean()
    psnr = 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))
    print(f"{g.name}: field{tuple(fld.shape)} decode{fo.shape}")
    print(f"    ★ codec 往返 PSNR = {psnr:.2f} dB   最大像素误差 = {np.abs(fo-raw).max():.2f}")
