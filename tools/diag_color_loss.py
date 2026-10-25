# -*- coding: utf-8 -*-
"""颜色丢失病灶诊断（只读）。

【结构性推断】
编码链：GIF[T,H,W,3] → 每个 patch 取 win×4×4×3 = 5×48 维 → 沿 (win, 48) 做 rfft2
        → 得 [N, 5, 25] 复值，d_model=125 恰好覆盖 5×25 **全部频点，无截断**。

其中 48 = 4×4 空间像素 × 3 颜色通道，**颜色被卷进 FFT 的最后一位**：
    索引 = ((py*4+px)*3 + ch)
沿这 48 维做 rfft 时，ch 方向的变化周期 = 3，对应频率 ≈ 48/3 = 16。
→ **三通道差异（=颜色的本质）主要落在 w≈16 的高频区，而不是 w=0 的 DC。**
而 DC(w=0) 是 48 个值的总和，三通道差异在这里**互相抵消**。

所以推断：模型若只学好低频（w 小），结构能出来、颜色必然丢失。

本脚本验证三件事：
  A. target 的频域能量沿 w 的分布，以及"颜色差异能量"落在哪些 w
  B. 生成场在同一批 w 上的能量，与 target 逐桶对比（找出缺口最大的频段）
  C. 空间域直接对比：target 与生成的 R/G/B 通道差异幅度

用法：
  C:/Python312/python.exe tools/diag_color_loss.py
  C:/Python312/python.exe tools/diag_color_loss.py --ckpt <另一份.pt>
"""
import os
import sys
import math
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
V4200 = PROJ / "docx" / "实验记录" / "versions" / "v4200_多模态CLIP产品版"

for p in [str(PROJ), str(PROJ / "deps312"), str(PROJ / "int8_deps")]:
    sys.path.insert(0, p)
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass

for k, v in dict(COND_INIT_PC="1", COND_PC_RANK="8", COND_GROUPS="5",
                 COND_LOWRANK="16", PHASE_ACT="arctan", RES_LIMIT="0.5",
                 W2_NORM_MAX="3.0", KAPPA_MAX="6.0", FFT_FFTW="1",
                 IDEAL_CLAMP=str(math.pi)).items():
    os.environ.setdefault(k, v)

sys.path.insert(0, str(V4200))
sys.path.insert(0, str(V4200 / "src"))

import numpy as np
import torch
from PIL import Image

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import (encode_gif_to_field, load_gif_frames,
                             resample_frames, decode_field_to_frames)
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
SDIM = PATCH * PATCH * 3      # 48
WFFT = SDIM // 2 + 1          # 25

NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]


def field_to_pix(field, normalize=True):
    """[N,2D] -> [T,H,W,3] uint8，与 decode_field_to_frames 同口径。"""
    z = field[:, :D] + 1j * field[:, D:]
    N = z.shape[0]
    z = z * 10.0
    flat = torch.fft.irfft2(z.reshape(N, WIN, WFFT), s=(WIN, SDIM),
                            dim=(-2, -1)).real
    flat = flat.reshape(N, WIN * SDIM)
    if normalize:
        lo, hi = flat.amin(), flat.amax()
        flat = (flat - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    else:
        flat = flat.clamp(-1.2, 1.2)
    blocks = flat.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)
    x = xt.reshape(TS * WIN, IMG, IMG, 3)
    return ((x + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)


def field_energy_by_w(field):
    """[N,2D] -> 每个空间频点 w 的平均幅值（对 h 与 N 求均值）。"""
    z = field[:, :D] + 1j * field[:, D:]          # [N, 125]
    zn = z.reshape(-1, WIN, WFFT)                  # [N, 5, 25]
    amp = zn.abs().mean(dim=(0, 1))                # [25]
    return amp


def color_energy_by_w(field):
    """只统计"通道间差异"的能量，按 w 分桶。

    做法：把每个 patch 解码回 [win, 4, 4, 3]，减去通道均值（去掉亮度、只留色差），
    再沿同样的 (win, 48) 做 rfft，得到"颜色差异"的频域能量分布。
    """
    z = field[:, :D] + 1j * field[:, D:]
    N = z.shape[0]
    z = z * 10.0
    flat = torch.fft.irfft2(z.reshape(N, WIN, WFFT), s=(WIN, SDIM),
                            dim=(-2, -1)).real      # [N, 5, 48]
    v = flat.reshape(N, WIN, PATCH * PATCH, 3)
    v = v - v.mean(dim=-1, keepdim=True)            # 去掉亮度 -> 纯色差
    v = v.reshape(N, WIN, SDIM)
    spec = torch.fft.rfft(v, dim=-1)                # [N, 5, 25]
    return spec.abs().mean(dim=(0, 1))               # [25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(V4200 / "outputs_real" /
                                          "color4_resume_1023" / "guided_ckpt.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t, s = build_spacetime_index(TS, GRID, GRID, device)

    model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                           ffn_hidden=256, init_mode="init", n_modules=8,
                           mod_gamma=0.15).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    ck = {k: v for k, v in ck.items() if not k.startswith("qwen.")}
    r = m = model.load_state_dict(ck, strict=False)
    print(f"[CKPT] {Path(args.ckpt).parent.name}  missing={len(r.missing_keys)} "
          f"unexpected={len(r.unexpected_keys)}")
    model.eval()

    cv = np.load(V4200 / "data" / "clip_cache" / "cv3_full.npy")

    for i, nm in enumerate(NAMES):
        gp = V4200 / "data" / "gif_raw" / "noto" / nm
        tgt = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
        raw = resample_frames(load_gif_frames(str(gp), (IMG, IMG)), TS * WIN)

        c = torch.from_numpy(cv[i:i + 1]).to(device).float()
        with torch.inference_mode():
            a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=c, use_cond_field=True)
        # 生成场拼回 [N, 2D]
        z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
        gen = torch.cat([z.real, z.imag], dim=-1).float().cpu()

        print("\n" + "=" * 74)
        print(f"[{i}] {nm}")
        print("=" * 74)

        amp_t = field_energy_by_w(tgt.cpu())
        amp_g = field_energy_by_w(gen)
        print(f"{'w':>3} | {'target振幅':>11} | {'生成振幅':>11} | {'比值':>7}")
        print("-" * 46)
        for w in range(WFFT):
            rt = amp_t[w].item()
            rg = amp_g[w].item()
            ratio = rg / max(rt, 1e-9)
            flag = "  <<< 缺口大" if ratio < 0.35 else ""
            print(f"{w:3d} | {rt:11.5f} | {rg:11.5f} | {ratio:7.3f}{flag}")

        ct = color_energy_by_w(tgt.cpu())
        cg = color_energy_by_w(gen)
        top_t = torch.topk(ct, 5).indices.tolist()
        print(f"\n[颜色差异能量] target 最集中的 5 个 w = {sorted(top_t)}")
        print(f"{'w':>3} | {'target色差能量':>15} | {'生成色差能量':>15} | {'比值':>7}")
        print("-" * 58)
        for w in sorted(top_t)[:8]:
            rt, rg = ct[w].item(), cg[w].item()
            print(f"{w:3d} | {rt:15.6f} | {rg:15.6f} | {rg / max(rt, 1e-9):7.3f}")

        # 空间域直方
        pg = field_to_pix(gen)
        sat_t = float((raw.astype(np.float32).max(-1) - raw.astype(np.float32).min(-1)).mean())
        sat_g = float((pg.astype(np.float32).max(-1) - pg.astype(np.float32).min(-1)).mean())
        print(f"\n[空间域] 素材 sat={sat_t:.2f}  生成 sat={sat_g:.2f}  "
              f"({sat_g / max(sat_t, 1e-9) * 100:.1f}% of 素材)")


if __name__ == "__main__":
    main()
