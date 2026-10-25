# -*- coding: utf-8 -*-
"""验证 GIF <-> 振子场 编解码的往返保真度。

截断到 D=64 个低频系数会有损失，但对低细节线条表情包应足够。
用 PSNR 量化。
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np
import torch
from codec.gif_codec import load_gif_frames, resample_frames, encode_gif_to_field, decode_field_to_frames

TIME_STEP, GRID, PATCH, WIN, IMG, D = 6, 32, 4, 5, 128, 64
T_TARGET = TIME_STEP * WIN


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = ((a - b) ** 2).mean()
    if mse < 1e-12:
        return 99.0
    return 10.0 * np.log10((255.0 ** 2) / mse)


def main():
    data_dir = Path("data/gif_raw/noto")
    gifs = sorted(data_dir.glob("*.gif"))
    if not gifs:
        print("[FATAL] data/gif_raw/noto 下没有 GIF")
        sys.exit(1)

    # 只测前 6 个
    for gp in gifs[:6]:
        try:
            frames = load_gif_frames(str(gp), (IMG, IMG))
            frames = resample_frames(frames, T_TARGET)
            field = encode_gif_to_field(str(gp), TIME_STEP, GRID, PATCH, WIN, IMG, D)
            rec = decode_field_to_frames(field, TIME_STEP, GRID, PATCH, WIN, IMG, D)
            rec_arr = np.stack(rec, axis=0)
            if rec_arr.shape != frames.shape:
                print(f"[WARN] {gp.name} 形状不一致 {rec_arr.shape} vs {frames.shape}")
                continue
            p = psnr(frames, rec_arr)
            print(f"[{gp.name:36s}] N={field.shape[0]} PSNR={p:.2f} dB  {'OK' if p > 18 else 'LOW'}")
        except Exception as e:
            print(f"[ERROR] {gp.name}: {e}")

    # 形状自检
    gp = gifs[0]
    field = encode_gif_to_field(str(gp), TIME_STEP, GRID, PATCH, WIN, IMG, D)
    assert field.shape == (TIME_STEP * GRID * GRID, 2 * D), f"field shape {field.shape}"
    print(f"\n[SHAPE] field = {tuple(field.shape)}  (N={TIME_STEP*GRID*GRID}, 2D={2*D})  OK")


if __name__ == "__main__":
    main()
