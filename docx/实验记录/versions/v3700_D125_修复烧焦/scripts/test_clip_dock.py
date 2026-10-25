# -*- coding: utf-8 -*-
"""v3700 对接 jina-clip-v2 双塔 —— 第二步：训练验证「是否还能还原图片」。

核心问题（用户问）：接入 jina-clip-v2 图文双塔后，模型是否还能还原图片？

做法（最小侵入，零改 model.py / 物理核 / codec）：
  v3700 的 AetherGIFModel.forward 已有 `text_embs` 直传口（text_embs is not None
  时直接 [B,896] 喂给 4 个物理调制头）。所以不碰模型，只在外部加一个
  CondProjector(2049→896)，把 CLIP 双塔条件投影成 896 维塞进这个口。

loss 口径 = 完全复刻 v3700 overfit.py（loss_amp + loss_ph + 高频增益 + Nyquist 径向），
保证唯一变量是「条件入口」，与 v3700 基线（无条件重建 PSNR 22.46dB clamp）同口径可比。

用法：
  python scripts/test_clip_dock.py            # 默认 250 步，CLIP 条件，素材=台球
  python scripts/test_clip_dock.py 1          # 1 步冒烟
  python scripts/test_clip_dock.py 250 8-ball_1f3b1.gif
"""
import os
import sys
import math
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
from PIL import Image

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index

# 与 v3700 完全一致的编解码口径
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
H_FFT, W_FFT = WIN, (PATCH * PATCH * 3) // 2 + 1   # 5 × 25

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
GIF = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
AMP_P = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

# v3700 同款治烧焦 loss 组件（高频增益 + Nyquist 径向）
HF_SEGS = [(12, 19), (19, 25)]
GAIN_TGT, GAIN_W, HUBER_DELTA = 1.0, 0.5, 0.5
RAD_LO_FRAC, RAD_CAP, RAD_W = 0.80, 1.2, 1.0

OUT = "outputs_real/clip_dock.gif"
CACHE = ROOT / "clip_cache"
device = "cuda"
CLIP_EMB_DIM = 1024


def _decodable(a_, th_):
    """纯 torch 解码到 [T*WIN, IMG, IMG, 3]，保持梯度（复刻 overfit.py）。"""
    z = torch.complex(a_ * torch.cos(th_), a_ * torch.sin(th_))       # [N,D]
    N = z.shape[0]
    z = z * 10.0
    flat = torch.fft.irfft2(z.reshape(N, WIN, W_FFT), s=(WIN, PATCH * PATCH * 3),
                            dim=(-2, -1)).real
    flat = flat.reshape(N, WIN * PATCH * PATCH * 3)
    blocks = flat.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)
    return xt.reshape(TS * WIN, IMG, IMG, 3)


def main():
    # ---------- 素材 target field ----------
    gp = (ROOT / "data" / "gif_raw" / "noto" / GIF)
    field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    t, s = build_spacetime_index(TS, GRID, GRID, device)
    target = field.unsqueeze(0)
    re, im = target[..., :D], target[..., D:]
    a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
    th_tar = torch.atan2(im, re)
    print(f"[DATA] {GIF}  N={target.shape[1]}  a_tar mean={a_tar.mean().item():.5f}")

    # ---------- CLIP 条件 → 直连 2049 ----------
    cond = np.load(CACHE / "cond.npy")                       # [2, 2049]
    # 按文件名定位是哪个样本（0=8-ball, 1=3D87..）
    sample_idx = 0 if GIF.startswith("8-ball") else 1
    cv = torch.from_numpy(cond[sample_idx:sample_idx + 1]).float().to(device)  # [1,2049]

    # 两个样本的 CLIP 条件是否真的不同（区分度自检）
    c0 = torch.from_numpy(cond[0]).float()
    c1 = torch.from_numpy(cond[1]).float()
    cos_cond = float(torch.nn.functional.cosine_similarity(c0, c1, dim=0))
    print(f"[COND] 两样本条件余弦相似度 = {cos_cond:.4f}（<0.9 说明双塔确实分出区别）")

    # ---------- 模型（use_qwen=False，与 v3700 一致，不加载 Qwen）----------
    model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    # ---------- Nyquist 环预计算（复刻 overfit.py）----------
    _yy, _xx = np.indices((IMG, IMG))
    _rmax = float(np.sqrt((IMG // 2) ** 2 + (IMG // 2) ** 2))
    RAD_R = torch.from_numpy(
        (np.sqrt((_yy - IMG // 2) ** 2 + (_xx - IMG // 2) ** 2) >= RAD_LO_FRAC * _rmax).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        _tg = _decodable(a_tar[0], th_tar[0]).mean(dim=-1)
        _Fg = torch.fft.fft2(_tg - _tg.mean(dim=(1, 2), keepdim=True))
        RING_TAR = (_Fg.abs().pow(2) * RAD_R).sum(dim=(1, 2)).mean().clamp(min=1e-12)

    torch.manual_seed(1234)   # 固定初始条件，与 v3700 同

    print(f"[RUN] {STEPS} 步 CLIP 条件训练  lr={LR}  GIF={GIF}")
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        # CLIP 条件 2049 维 → 直接喂 text_embs 直传口
        text_embs = cv                                        # [1, 2049]
        a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s,
                         batch_size=1, text_embs=text_embs)

        pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
        la = torch.log1p(a.clamp(min=1e-8))
        lt = torch.log1p(a_tar.clamp(min=1e-8))
        w = (a_tar / a_tar.max().clamp(min=1e-8)).pow(AMP_P)
        # 高频增益（复刻）
        a_bar_out = a.clamp(min=1e-8).mean(dim=0)
        a_bar_tar = a_tar.clamp(min=1e-8).mean(dim=0)
        loss_gain, gain_seg = torch.zeros((), device=a.device), []
        for lo, hi in HF_SEGS:
            idx = torch.tensor([c for c in range(D) if lo <= (c % W_FFT) < hi], device=a.device)
            go = (a_bar_out[idx].mean() / a_bar_tar[idx].mean().clamp(min=1e-12)).clamp(0.1, 20.0)
            excess = (go - GAIN_TGT).clamp(min=0.0)
            huber = torch.where(excess <= HUBER_DELTA, 0.5 * excess.pow(2) / HUBER_DELTA,
                                excess - 0.5 * HUBER_DELTA)
            loss_gain = loss_gain + huber
            gain_seg.append(go.detach())
        loss_gain = loss_gain / len(HF_SEGS)
        # Nyquist 径向（复刻）
        frames = _decodable(a[0], th[0])
        gray = frames.mean(dim=-1)
        Fg = torch.fft.fft2(gray - gray.mean(dim=(1, 2), keepdim=True))
        ring_out = (Fg.abs().pow(2) * RAD_R).sum(dim=(1, 2)).mean()
        rad_ratio = ring_out / RING_TAR.clamp(min=1e-12)
        loss_rad = torch.clamp(rad_ratio - RAD_CAP, min=0.0)

        loss_amp = (w * (la - lt).pow(2)).mean()
        w_ph = (a_tar ** 2).clamp(min=1e-8)
        loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
        loss = loss_amp + loss_ph + GAIN_W * loss_gain + RAD_W * loss_rad
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == 1 or step == STEPS:
            with torch.no_grad():
                c = torch.cos(pe)
                cos = c.mean().item()
                wcos = (w_ph * c).sum().item() / w_ph.sum().item()
            print(f"[{step:4d}] loss={loss.item():.4f} amp={loss_amp.item():.4f} "
                  f"phase={loss_ph.item():.4f} cos={cos:+.4f} wcos={wcos:+.4f}", flush=True)

    # ---------- 出图 + 定量 PSNR（clamp 口径，与 v3700 可比）----------
    model.eval()
    with torch.inference_mode():
        text_embs = cv
        a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s,
                         batch_size=1, text_embs=text_embs)
    z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    frames = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D,
                                             normalize=True)).astype(np.uint8)
    pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    pil[0].save(OUT, save_all=True, append_images=pil[1:], loop=0, duration=100)

    # clamp 口径 PSNR
    with torch.inference_mode():
        zt = (a_tar[0] * torch.exp(1j * th_tar[0])).to(torch.complex64)
        ft = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()
    arr_t = np.stack(decode_field_to_frames(ft, TS, GRID, PATCH, WIN, IMG, D,
                                            normalize=False)).astype(np.float32)
    arr_m = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D,
                                            normalize=False)).astype(np.float32)
    psnr = 10 * math.log10(255.0 ** 2 / max(((arr_m - arr_t) ** 2).mean(), 1e-12))
    print(f"[CMP] PSNR(clamp)={psnr:.2f} dB  （v3700 无条件基线 = 22.46 dB）")
    print(f"[GEN] {OUT}")


if __name__ == "__main__":
    main()