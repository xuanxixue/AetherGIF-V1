# -*- coding: utf-8 -*-
"""第十三死穴修复验证：架构 log1p 域递推后，μ 应精确等于模型层级。

修复前（log 域架构 + log1p loss）：μ 报 -1.21、模型实际层级 +0.34，量纲错位。
修复后（log1p 域架构 + log1p loss）：两者必须严格相等（γ_lvl=0.5, 12 块 => 0.5^12=2.4e-4）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')

import math
import torch
from pathlib import Path

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

gp = next(Path('data/gif_raw/noto').glob('8-ball_1f3b1.gif'))
f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(dev)
t, s = build_spacetime_index(TS, GRID, GRID, dev)
target = f.unsqueeze(0)
a_tar = torch.sqrt(target[..., :D] ** 2 + target[..., D:] ** 2 + 1e-12)

m = AetherGIFModel(device=dev, use_qwen=False).to(dev)
m.eval()
with torch.inference_mode():
    a, th, _ = m(texts=['台球'], target_field=target,
                 t_coord=t, s_coord=s, batch_size=1)

mu = m.pmnncore.a_mu.detach()
lt = torch.log1p(a_tar[0].mean(0))
lm = torch.log1p(a[0].mean(0))

print('=' * 70)
print('[验证 1] μ 是否精确等于模型层级（架构 log1p 域递推的核心证据）')
print('=' * 70)
d = (lm - mu).abs()
print(f'  |模型层级(log1p域) - μ|  mean={d.mean():.6f}   max={d.max():.6f}')
print(f'  判定：{"通过 —— 架构与 loss 同量纲" if d.max() < 0.01 else "失败"}')
print(f'  （修复前此差值为 1.55，μ=−1.21 vs 实际 +0.34）')

print()
print('=' * 70)
print('[验证 2] μ 初值量纲是否正确（修复前误用 log(0.05)=−3.0）')
print('=' * 70)
print(f'  μ 初值应= log1p(0.05) = {math.log1p(0.05):.4f}   实际 = {mu[0].item():.4f}')
print(f'  判定：{"通过" if abs(mu[0].item() - math.log1p(0.05)) < 1e-5 else "失败"}')

print()
print('=' * 70)
print('[验证 3] 目标跨度与所需 lr（log1p 域是有界的）')
print('=' * 70)
print(f'  target log1p: ch0={lt[0]:.4f}  ch25={lt[25]:.4f}  ch50={lt[50]:.4f}  ch55={lt[55]:.4f}')
print(f'  跨度 = {lt[0].item() - lt.min().item():.4f}（有界可学）')
print(f'  对比 log 域跨度 = {(a_tar[0].mean(0).log().mean() - a_tar[0].mean(0).log().min()).item():.4f}（无界）')

print()
print('=' * 70)
print('[验证 4] 溢出保护')
print('=' * 70)
print(f'  L_new clamp [0,6] => a_new 上限 expm1(6) = {math.expm1(6):.1f}')
print(f'  target ch0 最大振幅 = {a_tar[0, :, 0].max().item():.1f}'
      f'  余量 = {math.expm1(6) / a_tar[0, :, 0].max().item():.1f}x')
print(f'  a_out（零训练）= {a.mean().item():.5f}   target mean = {a_tar.mean().item():.5f}')

print()
print('=' * 70)
print('[验证 5] a_new 非负性（expm1 保证）')
print('=' * 70)
print(f'  a.min() = {a.min().item():.8f}  (必须 >= 0)')
print(f'  判定：{"通过" if a.min().item() >= 0 else "失败"}')
