# -*- coding: utf-8 -*-
"""决定性验证：d_model 截断到底吃掉多少画质？

背景：2026-10-09 记录写「D=64 26.70 dB / D=125 全保留 26.75 dB，几乎无差别」，
并据此得出「表征不是瓶颈，问题全在模型侧」。
但同一行记录里 patch=8/D=485 那条写着「同样 14.84 dB（手算漏了 *10.0，
真实 codec 26.70）」——**这个 26.70 是套用 D=64 的修正值，不是实测**。
所以「D=125 无差别」从未被真实测过。

本脚本用与训练完全一致的口径，逐一实测，不做任何手算。
"""
import sys
sys.path.insert(0, './src')
import numpy as np
import torch
from codec.gif_codec import load_gif_frames, resample_frames, decode_field_to_frames

PATCH, WIN, TS, IMG, GRID = 4, 5, 6, 128, 32
NORM = 10.0

NOTO = 'D:/AetherGIF-V1/data/gif_raw/noto/'
GIFS = [
    (NOTO + '8-ball_1f3b1.gif', '台球(v2500原素材)'),
    (NOTO + '100_1f4af.gif', '数字100'),
    (NOTO + 'airplane_2708_fe0f.gif', '飞机'),
    (NOTO + 'admission-tickets_1f39f_fe0f.gif', '门票'),
    (NOTO + 'alarm-clock_23f0.gif', '闹钟'),
    (NOTO + 'Scorpio_264f.gif', '星座'),
    ('D:/AetherGIF-V1/泛化素材/3D87B14FBB731ED3A948AC6DD9EA6EBC.gif', '★新素材头像'),
]


def grid_ratio(arr):
    b, i = [], []
    for f in arr:
        d = np.abs(np.diff(f, axis=1))
        xs = np.arange(d.shape[1])
        m = (xs % PATCH) == (PATCH - 1)
        b.append(d[:, m].mean()); i.append(d[:, ~m].mean())
    return float(np.mean(b)), float(np.mean(i)), float(np.mean(b)) / max(np.mean(i), 1e-9)


def encode_D(path, d_model):
    """复刻 encode_gif_to_field，但把 d_model 作为变量。"""
    T_target = TS * WIN
    frames = load_gif_frames(path, (IMG, IMG))
    frames = resample_frames(frames, T_target)
    x = frames.astype(np.float32) / 127.5 - 1.0
    xt = x.reshape(TS, WIN, GRID, PATCH, GRID, PATCH, 3)
    blocks = xt.transpose(0, 2, 4, 1, 3, 5, 6).reshape(
        TS * GRID * GRID, WIN, PATCH * PATCH * 3)
    flat = torch.from_numpy(np.ascontiguousarray(blocks))
    spec = torch.fft.rfft2(flat, dim=(-2, -1)) / NORM
    spec = spec.reshape(flat.shape[0], -1)
    full = spec.shape[-1]
    if full < d_model:
        spec = torch.cat([spec, torch.zeros(spec.shape[0], d_model - full,
                                            dtype=spec.dtype)], dim=-1)
    z = spec[:, :d_model].contiguous()
    return torch.cat([z.real, z.imag], dim=-1).float()


def decode(field, d_model, normalize=False):
    return np.stack(decode_field_to_frames(
        field, TS, GRID, PATCH, WIN, IMG, d_model, normalize=normalize)).astype(np.float32)


def psnr(a, b):
    return 10 * np.log10(255.0 ** 2 / max(((a - b) ** 2).mean(), 1e-12))


FULL = WIN * (PATCH * PATCH * 3 // 2 + 1)   # 5 * 25 = 125

print("=" * 108)
print(f"d_model 截断实测（PATCH={PATCH}, WIN={WIN} → rfft2 真实网格 {WIN}×"
      f"{PATCH*PATCH*3//2+1} = {FULL} 系数/块）")
print("=" * 108)
print(f"{'素材':<20}{'D':>5}{'std':>8}{'边界':>8}{'内部':>8}{'网格比':>8}"
      f"{'PSNR(往返)':>12}{'丢弃能量':>10}")
print("-" * 108)

summary = {}
for path, name in GIFS:
    src = load_gif_frames(path, (IMG, IMG)).astype(np.float32)
    src = resample_frames(src, TS * WIN).astype(np.float32)
    sb, si, sr = grid_ratio(src)
    print(f"{name:<20}{'素材':>5}{src.std():>8.1f}{sb:>8.2f}{si:>8.2f}{sr:>8.3f}"
          f"{'—':>12}{'—':>10}")
    # 各 D 的往返
    for D in (64, 100, FULL):
        fld = encode_D(path, D)
        rec = decode(fld, D)
        b, i, r = grid_ratio(rec)
        # 丢弃能量占比
        fld_full = encode_D(path, FULL)
        spec_all = torch.fft.rfft2(
            torch.from_numpy(np.ascontiguousarray(
                src.reshape(TS, WIN, GRID, PATCH, GRID, PATCH, 3)
                .transpose(0, 2, 4, 1, 3, 5, 6)
                .reshape(-1, WIN, PATCH * PATCH * 3))), dim=(-2, -1)) / NORM
        e = (spec_all.abs() ** 2).mean(0)
        tot = e.sum().item()
        kept = np.zeros((WIN, PATCH * PATCH * 3 // 2 + 1), dtype=bool)
        idx = np.arange(D)
        kept[(idx // (PATCH * PATCH * 3 // 2 + 1)),
              (idx % (PATCH * PATCH * 3 // 2 + 1))] = True
        ke = sum(e[h][kept[h]].sum().item() for h in range(WIN) if kept[h].any())
        p = psnr(rec, src)
        print(f"{'':<20}{D:>5}{rec.std():>8.1f}{b:>8.2f}{i:>8.2f}{r:>8.3f}"
              f"{p:>11.2f}dB{100*(1-ke/tot):>9.1f}%")
        summary.setdefault(D, []).append((name, p, r, 100 * (1 - ke / tot)))
    print("-" * 108)

print()
print("=" * 108)
print("汇总：平均往返 PSNR 与平均网格比")
print("=" * 108)
print(f"{'D':>6}{'保留%':>8}{'平均PSNR':>12}{'平均网格比':>12}{'网格比/素材':>14}")
base_r = []
for D in (64, 100, FULL):
    v = summary[D]
    mp = np.mean([x[1] for x in v])
    mr = np.mean([x[2] for x in v])
    keep = 100 * (1 - np.mean([x[3] for x in v]) / 100)
    print(f"{D:>6}{keep:>7.0f}%{mp:>11.2f}dB{mr:>12.3f}")
print()
print(f"素材自身平均网格比 = {np.mean([grid_ratio(resample_frames(load_gif_frames(p,(IMG,IMG)),TS*WIN).astype(np.float32))[2] for p,_ in GIFS]):.3f}")
