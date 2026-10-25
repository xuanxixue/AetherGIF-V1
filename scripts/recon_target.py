# -*- coding: utf-8 -*-
"""把训练样本 encode→decode 往返重建，作为模型生成图的参照。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
from PIL import Image
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
name = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
gp = next(Path("data/gif_raw/noto").glob(name))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D)
frames = decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)
out = sys.argv[1] if len(sys.argv) > 1 else "outputs_real/target_recon.gif"
pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
pil[0].save(out, save_all=True, append_images=pil[1:], loop=0, duration=100)
print(f"target recon -> {out}  frames={len(pil)}  src={gp.name}")
