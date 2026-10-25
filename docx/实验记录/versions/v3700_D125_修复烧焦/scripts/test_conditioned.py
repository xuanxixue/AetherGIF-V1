# -*- coding: utf-8 -*-
"""
条件化物理核心训练脚本

流程：
1. 加载 GIF 素材 → target_field
2. CLIP 条件 c (2049 维) → ConditionModulator → prototypes (16, 2*D)
3. 物理核驱动：DriftField 的吸引中心由 prototypes 决定
4. 训练：loss 计算用 target_field 重建（所以 targets 还是真实场）
   但 DriftField 的 proto 使用条件原型，而不是固定 prototypes 或 target
5. 推理：生成 GIF 用条件原型作为吸引中心

物理核公式完全不变，只是参数来源变为条件化。

实现：
- ConditionModulator: cond → prototypes
- DriftField.forward(targets, proto): 优先用 proto
- PhysicsBlock.forward(targets, proto): 传递 proto
- AetherGIFModel.forward: 集成 ConditionModulator，传递 proto

PSNR 作为重建质量指标
"""
import os
import sys
import math
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# 环境设置
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from physics.condition_mod import ConditionModulator

# 编解码口径
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
GIF = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4

OUT = "outputs_real/conditioned_dock.gif"
CACHE = ROOT / "clip_cache"
device = "cuda"

# NYQUIST 环预计算
HF_SEGS = [(12, 19), (19, 25)]
GAIN_TGT, GAIN_W, HUBER_DELTA = 1.0, 0.5, 0.5
RAD_LO_FRAC, RAD_CAP, RAD_W = 0.80, 1.2, 1.0


def _decodable(a_, th_):
    """与 v3700 一致的解码器（保持梯度）"""
    z = torch.complex(a_ * torch.cos(th_), a_ * torch.sin(th_))
    N = z.shape[0]
    z = z * 10.0
    flat = torch.fft.irfft2(z.reshape(N, WIN, -1),
                            s=(WIN, PATCH * PATCH * 3),
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

    # ---------- CLIP 条件 ----------
    cond = np.load(CACHE / "cond.npy")
    sample_idx = 0 if GIF.startswith("8-ball") else 1
    cv = torch.from_numpy(cond[sample_idx:sample_idx + 1]).float().to(device)  # [1, 2049]

    # ---------- 条件调制器 ----------
    cond_mod = ConditionModulator(cond_dim=2049, d_model=D, n_prototypes=16, hidden_dim=256)
    cond_mod = cond_mod.to(device)
    cond_mod.train()

    # ---------- 物理模型 ----------
    model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)
    model.train()

    # ---------- 优化器 ----------
    opt = torch.optim.AdamW(list(model.parameters()) + list(cond_mod.parameters()), lr=LR)

    # ---------- 参数量 ----------
    print(f"[MODEL] 物理核参数：{sum(p.numel() for p in model.parameters()):,}")
    print(f"[COND_MOD] 条件调制器参数：{sum(p.numel() for p in cond_mod.parameters()):,}")

    # ---------- Nyquist 环预计算 ----------
    _yy, _xx = np.indices((IMG, IMG))
    _rmax = float(np.sqrt((IMG // 2) ** 2 + (IMG // 2) ** 2))
    RAD_R = torch.from_numpy(
        (np.sqrt((_yy - IMG // 2) ** 2 + (_xx - IMG // 2) ** 2) >= RAD_LO_FRAC * _rmax).astype(np.float32)
    ).to(device)
    with torch.no_grad():
        _tg = _decodable(a_tar[0], th_tar[0]).mean(dim=-1)
        _Fg = torch.fft.fft2(_tg - _tg.mean(dim=(1, 2), keepdim=True))
        RING_TAR = (_Fg.abs().pow(2) * RAD_R).sum(dim=(1, 2)).mean().clamp(min=1e-12)

    torch.manual_seed(1234)

    print(f"[RUN] {STEPS} 步条件训练  lr={LR}  GIF={GIF}")

    for step in range(1, STEPS + 1):
        opt.zero_grad()

        # 条件 → 物理参数
        proto = cond_mod(cv)  # [1, 16, 2*D]
        # 计算基础 Omega
        omega_base = model._init_omega(t, s)  # [1, N, D]

        # 条件偏移 Omega (简化：用 text_embs 路径，但这里没有 text_embs 了，所以直接传 const)
        # 真正的条件调制器应该输出 omega，但这里我们用固定 omega 偏移以示区别
        # 为了保持一致性，我们先绕过 omega，用固定 omega（但 proto 已经变了）
        # 实际上，我们要改 model 以支持直接传入 proto，并且在 block.py 中传递 proto

        # 这里简化：直接调用 model 的 forward，让它内部用 proto 作为条件
        # 但 model.forward 需要 text_embs 才能得到 omega_bias 等，所以我们当前不通过 model.forward
        # 而是手动定制的前向检查

        # 初始化噪声场
        a0 = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
        theta0 = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi

        a, theta = a0, theta0
        for i, blk in enumerate(model.pmnncore.blocks):
            tf = model.pmnncore.time_field.forward(i, device)

            # 注意：我们这里手动调用物理块的 forward，并传入 proto
            # 需要修改物理块的 forward 接口以支持 proto
            a, theta, d = blk(
                a, theta, omega_base, t, s,
                kappa=tf["kappa"],
                T=tf["T"],
                omega_d=tf["omega_d"],
                targets=target,  # 训练时 targets 是真实场（用于计算直接吸引分量）
                proto=proto,     # 条件原型，用于 DriftField 的吸引中心
            )

        # Loss 计算（复刻 v3700）
        pe = torch.remainder(theta - th_tar + math.pi, 2 * math.pi) - math.pi
        la = torch.log1p(a.clamp(min=1e-8))
        lt = torch.log1p(a_tar.clamp(min=1e-8))
        w = (a_tar / a_tar.max().clamp(min=1e-8)).pow(0.5)

        # 高频增益
        a_bar_out = a.clamp(min=1e-8).mean(dim=0)
        a_bar_tar = a_tar.clamp(min=1e-8).mean(dim=0)
        loss_gain = torch.zeros((), device=a.device)
        for lo, hi in HF_SEGS:
            idx = torch.tensor([c for c in range(D) if lo <= (c % 25) < hi], device=a.device)
            go = (a_bar_out[idx].mean() / a_bar_tar[idx].mean().clamp(min=1e-12)).clamp(0.1, 20.0)
            excess = (go - GAIN_TGT).clamp(min=0.0)
            huber = torch.where(excess <= HUBER_DELTA, 0.5 * excess.pow(2) / HUBER_DELTA,
                                excess - 0.5 * HUBER_DELTA)
            loss_gain = loss_gain + huber
        loss_gain = loss_gain / len(HF_SEGS)

        # Nyquist 径向
        frames = _decodable(a[0], theta[0])
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

    # 出图 + PSNR
    model.eval()
    cond_mod.eval()
    with torch.inference_mode():
        proto = cond_mod(cv)
        a, theta = a0, theta0
        for i, blk in enumerate(model.pmnncore.blocks):
            tf = model.pmnncore.time_field.forward(i, device)
            a, theta, d = blk(
                a, theta, omega_base, t, s,
                kappa=tf["kappa"],
                T=tf["T"],
                omega_d=tf["omega_d"],
                targets=None,  # 生成时 targets 为 None，使用 proto
                proto=proto,
            )
        z = (a[0] * torch.exp(1j * theta[0])).to(torch.complex64)
        fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
        frames = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D,
                                                 normalize=True)).astype(np.uint8)
        pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
        pil[0].save(OUT, save_all=True, append_images=pil[1:], loop=0, duration=100)

        # 计算 PSNR
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
