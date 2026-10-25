# -*- coding: utf-8 -*-
"""C 项验证：素材1 的 codec 天花板 12 dB 到底是不是截断造成的？（2026-10-18）

疑点：训练 D=125，rfft2 真实网格 = WIN*(PATCH²*3//2+1) = 5*25 = 125
      → D=125 就是**全保留，零截断**。截断不可能是天花板低的真凶。
      怀疑是 abs_quality_probe.py 参考帧对齐 bug：
      raw.shape[0] >= NEED 时用了 arange(NEED)（取前 30 帧），
      而 encode_gif_to_field 用 resample_frames = linspace 重采样 → 时序错位。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames, load_gif_frames, resample_frames

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
FULL = WIN * (PATCH * PATCH * 3 // 2 + 1)      # 125
gif_dir = ROOT / "data" / "gif_raw" / "noto"
GIF_SPECS = [
    ("8-ball_1f3b1.gif", "8号球"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "表情包"),
]
NEED = TS * WIN


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / max(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(), 1e-12))


print("=" * 100)
print(f"[C-0] D={D} vs FULL={FULL} → {'零截断（全保留）' if D >= FULL else f'截断 {FULL - D} 个系数'}")
print("=" * 100)

for name, desc in GIF_SPECS:
    gp = gif_dir / name
    raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
    print(f"\n--- {desc} ({name}) 原始 {raw.shape} ---")

    # encode 内部用的就是 resample_frames(frames, TS*WIN)
    enc_ref = resample_frames(raw, NEED).astype(np.float32)

    # abs_quality_probe 里用的对齐方式（有 bug 的那个）
    bad_idx = (np.linspace(0, raw.shape[0] - 1, NEED).round().astype(int)
               if raw.shape[0] < NEED else np.arange(NEED))
    bad_ref = raw[bad_idx].astype(np.float32)

    # 往返
    fld = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D)
    rec = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D,
                                          normalize=False)).astype(np.float32)

    print(f"  往返 PSNR vs encode 同口径参考(linspace) : {psnr(rec, enc_ref):7.2f} dB   <-- 真实天花板")
    print(f"  往返 PSNR vs abs_probe 的参考(arange)   : {psnr(rec, bad_ref):7.2f} dB   <-- 误报来源")
    print(f"  两个参考自身差异 PSNR                   : {psnr(bad_ref, enc_ref):7.2f} dB")
    print(f"  rec.std={rec.std():.2f}  enc_ref.std={enc_ref.std():.2f}  bad_ref.std={bad_ref.std():.2f}")