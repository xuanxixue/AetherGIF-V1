# -*- coding: utf-8 -*-
"""逐步定位动画在编解码哪一步丢失。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import numpy as np, torch
from pathlib import Path
from codec.gif_codec import load_gif_frames, resample_frames, encode_gif_to_field, decode_field_to_frames

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
gp = "data/gif_raw/noto/8-ball_1f3b1.gif"

raw = load_gif_frames(gp, (IMG, IMG))
print(f"[1] load: T={raw.shape[0]} frame-diff={np.abs(np.diff(raw,axis=0)).mean():.2f}")

fr = resample_frames(raw, TS * WIN)
print(f"[2] resample->{fr.shape[0]}: frame-diff={np.abs(np.diff(fr,axis=0)).mean():.2f}")

# 手动走一遍 encode 的重排，看时间维是否真有变化
x = fr.astype(np.float32) / 127.5 - 1.0
xt = x.reshape(TS, WIN, GRID, PATCH, GRID, PATCH, 3)
# 每个时空块内 win 帧的方差：动画意味着块内 win 帧不同
blk = xt.transpose(0,2,4,1,3,5,6).reshape(TS*GRID*GRID, WIN*PATCH*PATCH*3)
per_block_std = blk.std(-1)
print(f"[3] per-block std: mean={per_block_std.mean():.4f} max={per_block_std.max():.4f} "
      f"(>0.01 的块占比={float((per_block_std>0.01).mean()):.3f})")

field = encode_gif_to_field(gp, TS, GRID, PATCH, WIN, IMG, D)
print(f"[4] encode field: {tuple(field.shape)} "
      f"re={field[:,:D].abs().mean():.5f} im={field[:,D:].abs().mean():.5f}")

frames = decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)
arr = np.stack(frames)
print(f"[5] decode: T={arr.shape[0]} frame-diff={np.abs(np.diff(arr,axis=0)).mean():.2f}")
print(f"    逐帧 mean: {arr.reshape(30,-1).mean(1)[:6].round(1).tolist()} ...")
