# -*- coding: utf-8 -*-
"""频域逐通道消融：定位「高频过冲 / 网格感」到底出在哪些 FFT 通道。

配合 diag_block.py 的图像域结论（振幅->target +6.44 dB，方块完全消失）
从频域侧独立验证，并回答：过冲是「少数通道」还是「全局放大」？

用法：
    python scripts/diag_spec.py [ckpt]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')

import torch
from pathlib import Path

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'outputs_real/ckpt_schemeG_1000.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# rfft2 展平顺序 -> (h_fft, w_fft) 坐标
H_FFT, W_FFT = WIN // 2 + 1, (PATCH * PATCH * 3) // 2 + 1


def to_2d(idx):
    return idx // W_FFT, idx % W_FFT


gp = next(Path('data/gif_raw/noto').glob('8-ball_1f3b1.gif'))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
a_tar = torch.sqrt(target[..., :D] ** 2 + target[..., D:] ** 2 + 1e-12)

model = AetherGIFModel(device=device, use_qwen=False).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
with torch.inference_mode():
    a, th, _ = model(texts=['台球'], target_field=target,
                     t_coord=t, s_coord=s, batch_size=1)

at = a_tar[0].mean(0)     # [D] 逐通道平均振幅
am = a[0].mean(0)
ratio = (am / at.clamp(min=1e-12))

print('=' * 82)
print(f'[频域] 逐通道平均振幅比（模型/target）  ckpt={CKPT}')
print(f'  rfft2 网格 = {H_FFT}(时间freq) x {W_FFT}(空间freq)，展平为 {D} 通道')
print(f'  真实全谱 = {H_FFT * W_FFT}，只取前 {D} -> 保留 {D/(H_FFT*W_FFT)*100:.0f}%')
print('=' * 82)
print(f'{"通道":<6}{"h":>4}{"w":>4}{"径向":>7}{"target":>12}{"模型":>12}{"比值":>9}')
rows = []
for c in range(D):
    h, w = to_2d(c)
    rad = (h / max(H_FFT - 1, 1) * 0.5 + w / max(W_FFT - 1, 1) * 0.5) * 2
    rows.append((c, h, w, rad, at[c].item(), am[c].item(), ratio[c].item()))

for c, h, w, rad, tv, mv, r in rows:
    flag = ''
    if r > 2.0:
        flag = '  <<< 过冲'
    elif r < 0.5:
        flag = '  <<< 不足'
    print(f'{c:<6}{h:>4}{w:>4}{rad:>7.2f}{tv:>12.5f}{mv:>12.5f}{r:>9.3f}{flag}')

print()
print('-' * 82)
print('按径向频段汇总（判断「全局放大」还是「局部高频过冲」）')
print('-' * 82)
bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
print(f'{"径向频段":<14}{"通道数":>8}{"能量占比target":>18}'
      f'{"能量占比model":>17}{"能量比":>10}')
et = (at ** 2)
em = (am ** 2)
for lo, hi in bands:
    idx = [c for c, h, w, rad, *_ in rows if lo <= rad < hi]
    if not idx:
        continue
    i = torch.tensor(idx, device=et.device)
    pt = (et[i].sum() / et.sum()).item()
    pm = (em[i].sum() / em.sum()).item()
    print(f'{lo:.1f}-{hi:.1f}{"":<6}{len(idx):>8}{pt:>18.4f}{pm:>17.4f}{pm/max(pt,1e-12):>10.3f}')

print()
print(f'全局：sum(model)/sum(target) = {(am.sum()/at.sum()).item():.3f}')
print(f'     能量比 sum(m^2)/sum(t^2) = {((em.sum()/et.sum())).item():.3f}')
over = [c for c, h, w, rad, tv, mv, r in rows if r > 2.0]
print(f'过冲通道(>2x)：{len(over)}/{D}  索引={over[:20]}')
if over:
    i = torch.tensor(over, device=et.device)
    print(f'  这 {len(over)} 个通道占 model 总能量 '
          f'{(em[i].sum()/em.sum()).item()*100:.1f}%')
