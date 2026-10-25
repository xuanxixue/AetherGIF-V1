# -*- coding: utf-8 -*-
"""定位 3000 步新版的块状 plateau 伪影根因。

假设：`a_new = expm1(L_new.clamp(0, 6))` 不是严格可逆。
  - L_new < 0  -> clamp 到 0-> a = 0（硬截断，梯度为0）
  - L_new > 6  -> clamp 到 6 -> a = 402（硬截断）
截断破坏「零均值残差 -> 层级对 res 梯度严格为0」的不变量：
  delta 本应保持可正可负，被 clamp 后负半轴全部压成 0，
  L_new = m_new + delta_new 里 delta 的负半轴消失 => 低幅位置全部归零 => 块状 plateau

验证方式：统计 L_new 的截断比例 + 逐块统计 a=0 的空间分布。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')

import numpy as np
import torch
from pathlib import Path

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

gp = next(Path('data/gif_raw/noto').glob('8-ball_1f3b1.gif'))
f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(dev)
t, s = build_spacetime_index(TS, GRID, GRID, dev)
target = f.unsqueeze(0)
a_tar = torch.sqrt(target[..., :D] ** 2 + target[..., D:] ** 2 + 1e-12)

m = AetherGIFModel(device=dev, use_qwen=False).to(dev)
m.load_state_dict(torch.load('outputs_real/overfit_ckpt.pt',
                             map_location=dev, weights_only=True))
m.eval()
with torch.inference_mode():
    a, th, _ = m(texts=['台球'], target_field=target,
                 t_coord=t, s_coord=s, batch_size=1)

print('=' * 74)
print('[根因 1] a_new=expm1(clamp(0,6)) 的截断比例')
print('=' * 74)
a0 = (a[0] < 1e-6).float().mean().item()
print(f'  a==0（被 clamp(0) 硬截断）的位置占比 = {a0*100:.2f}%')
print(f'  target对应的位置占比            = {(a_tar[0] < 1e-6).float().mean().item()*100:.2f}%')
L = torch.log1p(a[0].clamp(min=0))
print(f'  L=log1p(a) 的 min={L.min().item():.4f}（clamp 下界是0）')
print(f'  L 的最大值          = {L.max().item():.4f}（clamp 上界是6）')
n_hi = (L > 5.99).float().mean().item()
print(f'  L 撞上界(>5.99) 的占比 = {n_hi*100:.3f}%')

print()
print('=' * 74)
print('[根因 2] 逐通道看被截断的程度')
print('=' * 74)
W_FFT = (PATCH * PATCH * 3) // 2 + 1
print(f'{"ch":>5}{"a均值":>12}{"target均值":>12}{"a=0占比":>10}')
for c in [0, 25, 50, 1, 20, 40, 63]:
    z = (a[0, :, c] < 1e-6).float().mean().item()
    print(f'{c:>5}{a[0,:,c].mean().item():>12.5f}{a_tar[0,:,c].mean().item():>12.5f}{z*100:>9.2f}%')

print()
print('=' * 74)
print('[根因 3] 块状伪影的空间证据：相邻块的 DC 通道跳变')
print('=' * 74)
# ch0 是每个块的均值（DC），块间跳变大 => 块状可见
dc0 = a[0, :, 0].reshape(TS, GRID, GRID)
tgt0 = a_tar[0, :, 0].reshape(TS, GRID, GRID)
jump_m = (dc0[:, 1:, :] - dc0[:, :-1, :]).abs().mean().item()
jump_t = (tgt0[:, 1:, :] - tgt0[:, :-1, :]).abs().mean().item()
print(f'  相邻块 ch0 跳变：模型={jump_m:.5f}  target={jump_t:.5f}  '
      f'比值={jump_m/max(jump_t,1e-9):.2f}x')

print()
print('=' * 74)
print('[根因 4] 分位裁剪出图会不会放大 plateau')
print('=' * 74)
z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
raw = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D,
                                       normalize=False)).astype(np.float32)
zt = (a_tar[0] * torch.exp(1j * torch.atan2(target[0, :, D:], target[0, :, :D]))).to(torch.complex64)
ft = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()
rawt = np.stack(decode_field_to_frames(ft, TS, GRID, PATCH, WIN, IMG, D,
                                        normalize=False)).astype(np.float32)
# 图像域"平坦块"占比：4x4 块内标准差极小的块
def flat_ratio(frames, thr=2.0):
    out = []
    for fr in frames:
        b = fr.reshape(TS * 0 + fr.shape[0] // 4, 4, fr.shape[1] // 4, 4, 3) \
            if False else None
        h, w = fr.shape[:2]
        g = fr.mean(-1).reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(-1, 4, 4)
        sd = g.std(axis=(1, 2))
        out.append((sd < thr).float().mean().item())
    return np.mean(out)

print(f'  4x4 块内 std<2 的平坦块占比：模型={flat_ratio(raw)*100:.1f}%  '
      f'target={flat_ratio(rawt)*100:.1f}%')
print('  （模型远高于 target => 块状 plateau 确实存在）')
