# -*- coding: utf-8 -*-
"""决定性验证：把模型振幅的「块内高频通道」替换为 target，
看图像域的网格感 / 高频抬升是否消失。

背景：diag_spec.py（块内频域）显示 0 个通道过冲、能量占比全面偏低；
      diag_block.py（整帧图像域）显示 21-128px 频段抬升 2.3-5.8 倍。
      两者矛盾，必须用「替换单类通道 -> 看图像域变化」直接判定。

整帧频率 <-> 块内通道的映射：
    patch=4 的块内空间频率 w，每像素 w/8 周期
    整帧 128px 上同频率 = 128 * w/8 = 16w 周期
    => 整帧 21-42px 频段(周期 3-6) <-> 块内 w=1..6 附近

用法：
    python scripts/diag_which_ch.py [ckpt]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
H_FFT, W_FFT = WIN // 2 + 1, (PATCH * PATCH * 3) // 2 + 1
CKPT = sys.argv[1] if len(sys.argv) > 1 else 'outputs_real/ckpt_schemeG_1000.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gp = next(Path('data/gif_raw/noto').glob('8-ball_1f3b1.gif'))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
a_tar = torch.sqrt(target[..., :D] ** 2 + target[..., D:] ** 2 + 1e-12)
th_tar = torch.atan2(target[..., D:], target[..., :D])

model = AetherGIFModel(device=device, use_qwen=False).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.eval()
with torch.inference_mode():
    a, th, _ = model(texts=['台球'], target_field=target,
                     t_coord=t, s_coord=s, batch_size=1)


def dec(amp, phase):
    z = (amp[0] * torch.exp(1j * phase[0])).to(torch.complex64)
    f = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    return np.stack(decode_field_to_frames(
        f, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


def radial(frames, nb=24):
    acc = np.zeros(nb)
    for fr in frames[:8]:
        g = fr.mean(-1)
        g = g - g.mean()
        F = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2).astype(int)
        rmax = min(h, w) // 2
        for i in range(nb):
            m = (r >= i * rmax / nb) & (r < (i + 1) * rmax / nb)
            if m.any():
                acc[i] += F[m].mean()
    return acc / acc.sum()


def psnr(x, y):
    return 10 * np.log10(255.0 ** 2 / max(((x - y) ** 2).mean(), 1e-12))


# 通道分组：按块内空间频率 w 切
# 整帧 128px：patch=4 -> 块内 w 频率对应整帧 16w 周期
#   w=0    -> 整帧 DC
#   w=1-2  -> 整帧 32-64px 频段（低频，球的轮廓）
#   w=3-6  -> 整帧 21-32px（中频，边缘/网格感所在）
#   w>=7   -> 整帧 <18px（高频细节）
groups = {
    'w=0(DC)': [c for c in range(D) if (c % W_FFT) == 0],
    'w=1-2(低频)': [c for c in range(D) if 1 <= (c % W_FFT) <= 2],
    'w=3-6(中频)': [c for c in range(D) if 3 <= (c % W_FFT) <= 6],
    'w>=7(高频)': [c for c in range(D) if (c % W_FFT) >= 7],
}

tgt = dec(a_tar, th_tar)
rt = radial(tgt)

print('=' * 84)
print(f'[决定性验证] 逐组把「模型振幅」换成「target 振幅」，看整帧图像域频谱变化')
print(f'  ckpt={CKPT}')
print(f'  通道总数 {D}，各组占比见下')
print('=' * 84)
print(f'{"替换组":<16}{"通道数":>7}{"振幅能量占比":>16}'
      f'{"PSNR":>8}{"中频比(21-42px)":>18}')

base = dec(a, th)
rb = radial(base)
pb = psnr(base, tgt)
mb = rb[1:2].sum() / max(rt[1:2].sum(), 1e-12)
print(f'{"(不替换)":<16}{"-":>7}{"-":>16}{pb:>8.2f}{mb:>18.2f}')

rows = {}
for name, idx in groups.items():
    ii = torch.tensor(idx, device=a.device)
    a2 = a.clone()
    a2[0, :, ii] = a_tar[0][:, ii]     # 该组振幅 -> target，相位保持模型
    fr = dec(a2, th)
    r = radial(fr)
    # 该组在模型振幅里的能量占比
    share = ((a[0][:, ii] ** 2).sum() / (a[0] ** 2).sum()).item()
    p = psnr(fr, tgt)
    m = r[1:2].sum() / max(rt[1:2].sum(), 1e-12)
    rows[name] = fr
    print(f'{name:<16}{len(idx):>7}{share:>15.1%}{p:>8.2f}{m:>18.2f}')

# 全替换（振幅 -> target）作为上界
full = dec(a_tar, th)
print(f'{"全部->target":<16}{D:>7}{100.0:>15.1%}'
      f'{psnr(full, tgt):>8.2f}'
      f'{radial(full)[1:2].sum()/max(rt[1:2].sum(),1e-12):>18.2f}')

print()
print('-' * 84)
print('判读：')
print('  · 某组替换后「中频比」明显下降 => 网格感/中频抬升来自该组振幅')
print('  · 某组振幅能量占比极小但中频比大幅下降 => 少数离群通道在放大中频')
print('  · 所有组替换都无效=> 中频抬升不是振幅问题（去查相位或 codec）')
print('-' * 84)

out = {
    'target': tgt,
    '模型原样': base,
}
for name, idx in groups.items():
    out[f'替换{name}'] = rows[name]
cols = [0, 10, 20]
Hh, Ww = tgt.shape[1:3]
cv = np.zeros((len(out) * Hh, len(cols) * Ww, 3), np.float32)
for r, (n, fs) in enumerate(out.items()):
    for c, i in enumerate(cols):
        cv[r * Hh:(r + 1) * Hh, c * Ww:(c + 1) * Ww] = fs[i]
Image.fromarray(cv.astype(np.uint8)).save('outputs_real/cmp_which_ch.png')
print('[SAVED] outputs_real/cmp_which_ch.png')
print('  行序: ' + ' / '.join(out.keys()))
