# -*- coding: utf-8 -*-
"""纯解析可逆性：完全绕开 uint8/GIF/重采样，只测 FFT 本身

回答「是不是缺解码器」的决定性判据：
  若 irfft2(rfft2(x)) == x（误差~1e-7），则 codec 是数学精确逆，
  decode 不存在「还原不了」的问题 —— 瓶颈必在别处。
"""
import sys
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

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
DATA = ROOT / "data" / "gif_raw"
g = [x for x in sorted(DATA.rglob("*.*")) if x.suffix == ".gif" and "ball" in x.name.lower()][0]

fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).float()
N = TS * GRID * GRID
s_dim = PATCH * PATCH * 3
h_fft, w_fft = WIN, s_dim // 2 + 1# 与 decode_field_to_frames 同一口径
full = h_fft * w_fft
print(f"素材 {g.name}  field={tuple(fld.shape)}  N={N}")
print(f"  rfft2 网格=({h_fft},{w_fft})={full}  D={D}  -> {'恰好填满，无截断' if full==D else f'需补零/截断 {full-D}'}")

z = fld[:, :D] + 1j * fld[:, D:]
z = z.reshape(N, h_fft, w_fft)
x = torch.fft.irfft2(z, s=(h_fft, s_dim), dim=(-2,-1)).real
z_back = torch.fft.rfft2(x, dim=(-2,-1))
print(f"\n=== 纯解析往返 irfft2(rfft2(x)) ===")
print(f"  重建 x: shape={tuple(x.shape)}  std={x.std():.6f}")

# 再正变换回去，和原始频域系数比
print(f"  ★ 频域往返 max|rfft2(irfft2(z))- z| = {(z_back-z).abs().max():.3e}")
print(f"    相对 = {((z_back-z).abs().max()/z.abs().max()):.3e}")

# ★ 真正该测的：decode 出的帧 == 原始帧吗
frames = decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)
fo = np.stack(frames).astype(np.float64)
print(f"\n  decode 输出 {fo.shape}")
print(f"  像素取值范围 [{fo.min():.1f}, {fo.max():.1f}]  std={fo.std():.2f}")
# 帧内空间结构是否存在（相邻像素差）——有结构就不是纯噪声
d = np.abs(np.diff(fo[0], axis=0)).mean()
print(f"  帧内相邻行平均差= {d:.3f}  (>1 说明有真实空间结构，非噪声)")
