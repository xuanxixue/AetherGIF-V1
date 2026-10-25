# -*- coding: utf-8 -*-
"""离线消融：把"方块（网格感）"和"糊"分别归因到振幅/ 相位 / 编解码层。

三个问题一次性分清：
  Q1「糊」归谁？   -> 图像域径向功率谱：模型 vs target 哪个频段抬升/衰减
  Q2「方块」归谁？ -> 误差能量在 patch 边界（每 4 像素）vs 块内部的分布
  Q3「振幅/相位各贡献多少 dB」-> 像素级消融 PSNR

方法论与前十死穴一致：先消融分清，再动手改架构。

用法：
    python scripts/diag_block.py [ckpt]
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


# ---------- 解码工具：两个口径，绝不用 min-max ----------
def to_z(amp, phase):
    z = (amp[0] * torch.exp(1j * phase[0])).to(torch.complex64)
    return torch.cat([z.real, z.imag], dim=-1).float().cpu()


def dec_clamp(f):
    """decode_field_to_frames 已返回 uint8 [0,255]，不要再做量纲变换。"""
    return np.stack(decode_field_to_frames(
        f, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


def dec_robust(f, lo_pct=0.5, hi_pct=99.5):
    """分位裁剪归一化：抗离群系数的正确口径（替代 min-max）。输入 uint8。"""
    raw = dec_clamp(f)
    out = []
    for fr in raw:
        a_, b_ = np.percentile(fr, [lo_pct, hi_pct])
        out.append(np.clip((fr - a_) *255.0 / max(b_ - a_, 1e-6), 0, 255))
    return np.stack(out)


def load_gif(p):
    img = Image.open(p)
    fr = []
    for i in range(img.n_frames):
        img.seek(i)
        fr.append(np.asarray(img.convert('RGB'), dtype=np.float32))
    return np.stack(fr)


def to_img(x):
    """占位：decode 已返回 uint8，此处恒等。保留以免漏改。"""
    return x


# ================= Q3：像素级消融 =================
zt = to_z(a_tar, th_tar)
cases = {
    '模型原样':        to_z(a, th),
    '振幅->target':    to_z(a_tar, th),
    '相位->target':    to_z(a, th_tar),
    '两者->target':    zt,
}
tgt_frames = to_img(dec_clamp(zt))

print('=' * 78)
print(f'[Q3] 像素级消融（固定 clamp 口径）  ckpt={CKPT}')
print('=' * 78)
print(f'{"方案":<16}{"PSNR(dB)":>10}{"Δ vs 原样":>12}'
      f'{"亮度":>9}{"对比度":>9}{"帧间差":>9}')
fdiff_t = np.mean([np.abs(tgt_frames[i + 1] - tgt_frames[i]).mean()
                   for i in range(len(tgt_frames) - 1)])
base = None
for k, v in cases.items():
    fr = to_img(dec_clamp(v))
    mse = ((fr - tgt_frames) ** 2).mean()
    p = 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))
    if base is None:
        base = p
    fd = np.mean([np.abs(fr[i + 1] - fr[i]).mean() for i in range(len(fr) - 1)])
    print(f'{k:<16}{p:>10.2f}{p - base:>+12.2f}'
          f'{fr.mean():>9.2f}{fr.std():>9.2f}{fd:>9.2f}')
print(f'{"target(参照)":<16}{"-":>10}{"-":>12}'
      f'{tgt_frames.mean():>9.2f}{tgt_frames.std():>9.2f}{fdiff_t:>9.2f}')

print()
print('  读法：振幅->target 涨得多 => 「糊/对比度」归振幅；')
print('        相位->target 涨得多 => 「方块/结构」归相位；两者都涨得少 => 归编解码层。')


# ================= Q1：径向功率谱，「糊」归谁 =================
def radial(frames, nb=24):
    """帧内 2D 径向功率谱，归一化到总能量 1。"""
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
    acc /= acc.sum()
    return acc


tgt_px = tgt_frames
mod_px = to_img(dec_clamp(cases['模型原样']))
rt, rm = radial(tgt_px), radial(mod_px)
print()
print('=' * 78)
print('[Q1] 图像域径向功率谱：「糊」= 中低频抬升，「锐」= 高频过冲')
print('=' * 78)
print(f'{"频段":<10}{"target":>12}{"模型":>12}{"比值":>10}   判读')
for i in range(0, 24, 4):
    lab = f'{i*128//24}-{(i+4)*128//24}px'
    r = rm[i:i + 4].sum() / max(rt[i:i + 4].sum(), 1e-12)
    if r > 1.15:
        j = '抬升 -> 该段被灌水 = 糊'
    elif r < 0.85:
        j = '衰减 -> 该段细节丢失'
    else:
        j = '基本对齐'
    print(f'{lab:<10}{rt[i:i+4].sum():>12.5f}{rm[i:i+4].sum():>12.5f}{r:>10.2f}   {j}')


# ================= Q2：分块边界能量，「方块」归谁 =================
def boundary_profile(err):
    """误差能量在 patch 边界行/列 vs 块内部的分布。

    patch=4 -> 边界像素在 0,4,8,...,124 及其 +1 偏移（取 0::4 与 3::4）。
    必须用平方能量：带符号误差会在边界处正负相消，比值失真成 1e10 量级。
    返回 (边界能量, 内部能量, 比值)
    """
    b = PATCH
    m = np.zeros(err.shape[-1], dtype=bool)
    m[0::b] = True
    m[b - 1::b] = True
    inner = ~m
    # 行列两个方向都算。err 已是 2D [H,W]
    eb, ei = [], []
    for g in err[:8]:
        eb.append((g[:, m] ** 2).mean()); ei.append((g[:, inner] ** 2).mean())
        eb.append((g[m, :] ** 2).mean()); ei.append((g[inner, :] ** 2).mean())
    return np.mean(eb), np.mean(ei), np.mean(eb) / max(np.mean(ei), 1e-9)


print()
print('=' * 78)
print(f'[Q2] patch边界能量（patch={PATCH}px，每{PATCH}像素一条边界）')
print('    比值 >>1 => 误差集中在分块边界 = 真·编解码/分块问题')
print('=' * 78)
print(f'{"方案":<16}{"边界误差":>12}{"内部误差":>12}{"比值":>10}')
for k in ['模型原样', '振幅->target', '相位->target', '两者->target']:
    fr = to_img(dec_clamp(cases[k]))
    eb, ei, r = boundary_profile((fr - tgt_frames).mean(-1))
    print(f'{k:<16}{eb:>12.3f}{ei:>12.3f}{r:>10.3f}')


# ================= 出图：分位裁剪口径的视觉对照 =================
rows = {
    'target': tgt_frames,
    '模型原样(clamp)': mod_px,
    '模型原样(robust)': to_img(dec_robust(cases['模型原样'])),
    '振幅->target': to_img(dec_clamp(cases['振幅->target'])),
    '相位->target': to_img(dec_clamp(cases['相位->target'])),
}
cols = [0, 10, 20]
Hh, Ww = tgt_frames.shape[1:3]
canvas = np.zeros((len(rows) * Hh, len(cols) * Ww, 3), np.float32)
for r, (name, fs) in enumerate(rows.items()):
    for c, i in enumerate(cols):
        canvas[r * Hh:(r + 1) * Hh, c * Ww:(c + 1) * Ww] = fs[i]
Image.fromarray(canvas.astype(np.uint8)).save('outputs_real/cmp_ablation.png')
print()
print('[SAVED] outputs_real/cmp_ablation.png')
print('  行序: ' + ' / '.join(rows.keys()))
