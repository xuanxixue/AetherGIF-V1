# -*- coding: utf-8 -*-
"""核查：训练日志里的 [死穴12] μ层级误差 判据本身口径是否正确。

疑点：日志显示 ch0  mdl=1.8272  target=2.1793  （差 0.35）
                ch0  mu =0.8326              （差 1.35）
若架构的 m=mean_N(log1p(a)) 精确收敛到 μ，两者必须相等。矛盾说明判据口径错。

两种可能：
  A) 架构 m= mean_N(log1p(a))      -> 判据应比 mean_N(log1p(a)) 与 μ
  B) 判据用的是 log1p(mean_N(a))   -> 与 A 不是同一个量，口径错
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

mu = m.pmnncore.a_mu.detach()
W_FFT = (PATCH * PATCH * 3) // 2 + 1
DC = [c for c in range(D) if c % W_FFT == 0]

# 架构里实际递推的量：m = L.mean(dim=1)，L = log1p(a)
m_架构 = torch.log1p(a[0]).mean(dim=0)              # mean_N(log1p(a))  ← 架构口径
m_日志 = torch.log1p(a[0].mean(dim=0))              # log1p(mean_N(a))  ← 日志口径
t_架构 = torch.log1p(a_tar[0]).mean(dim=0)          # target 的架构口径
t_日志 = torch.log1p(a_tar[0].mean(dim=0))          # target 的日志口径

print('=' * 78)
print('两种口径对比（DC 通道）')
print('=' * 78)
print(f'{"ch":>4}{"μ":>10}{"架构口径mdl":>14}{"日志口径mdl":>14}'
      f'{"架构target":>14}{"日志target":>14}')
for c in DC:
    print(f'{c:>4}{mu[c].item():>10.4f}{m_架构[c].item():>14.4f}'
          f'{m_日志[c].item():>14.4f}{t_架构[c].item():>14.4f}{t_日志[c].item():>14.4f}')

print()
print('=' * 78)
print('判据正确性检查：架构递推量是否等于 μ')
print('=' * 78)
d = (m_架构 - mu).abs()
print(f'  |mean_N(log1p(a)) − μ|  max={d.max().item():.6f}   '
      f'{"通过" if d.max().item() < 0.01 else "失败"}')

print()
print('=' * 78)
print('真正的层级误差（架构口径 vs 架构口径）')
print('=' * 78)
print(f'{"ch":>4}{"target":>12}{"模型":>12}{"误差":>12}')
err_a, err_l = [], []
for c in DC:
    e = m_架构[c].item() - t_架构[c].item()
    print(f'{c:>4}{t_架构[c].item():>12.4f}{m_架构[c].item():>12.4f}{e:>+12.4f}')
    err_a.append(abs(e))
print(f'  DC 真实层级误差均值 = {sum(err_a)/len(err_a):.4f}')

nd = [c for c in range(D) if c not in DC]
nd_e = (m_架构[nd] - t_架构[nd]).abs().mean().item()
print(f'  非DC 真实层级误差均值 = {nd_e:.4f}')

print()
print('=' * 78)
print('对比：日志口径算出的误差（错误的判据）')
print('=' * 78)
nd_e2 = (mu[nd] - t_日志[nd]).abs().mean().item()
dc_e2 = max(abs(mu[c].item() - t_日志[c].item()) for c in DC)
print(f'  日志口径 DC_max={dc_e2:.4f}  非DC均值={nd_e2:.4f}')
print(f'  架构口径 DC_max={max(err_a):.4f}  非DC均值={nd_e:.4f}')
