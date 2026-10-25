# -*- coding: utf-8 -*-
"""
条件生成模型 v2：条件直接驱动 DriftField（无重建）

核心：DriftField 接受条件 cond，将条件嵌入到吸引中心计算中。
目标场不再是 MLP 生成，而是 DriftField 内部根据条件计算。

物理核不变，DriftField 新增条件分支。
"""
import os
import sys
import math
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torch.nn as nn

# 环境设置
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from physics.drift_field import DriftField
from physics.standing_wave_ffn import StandingWaveFFN
from physics.thermodynamics import Thermodynamics
from physics.block import PhysicsBlock
from physics.time_field import TimeField
from physics.condition_mod import ConditionModulator

# 编解码口径
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
HF_SEGS = [(12, 19), (19, 25)]
GAIN_TGT, GAIN_W, HUBER_DELTA = 1.0, 0.5, 0.5
RAD_LO_FRAC, RAD_CAP, RAD_W = 0.80, 1.2, 1.0

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
GIF = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4

OUT = "outputs_real/cond_gen_v2.gif"
CACHE = ROOT / "clip_cache"
device = "cuda"


class CondGenModelV2(torch.nn.Module):
    """条件生成模型 v2：条件直接调制 DriftField"""

    def __init__(self, cond_dim=2049, d_model=D, n_blocks=12):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks

        # 条件调制器：cond → 原型块
        self.cond_mod = ConditionModulator(cond_dim=cond_dim, d_model=d_model, n_prototypes=16, hidden_dim=256)

        # 物理核心组件
        self.time_field = TimeField(n_blocks=n_blocks)
        self.blocks = nn.ModuleList([
            PhysicsBlock(
                d_model=d_model,
                ffn_hidden=256,
                lambda_t=0.5,
                lambda_s=3.0,
                temporal_window=3,
                dt=0.1,
                n_prototypes=16,
                T_ref=20.0,
                X_ref=32.0,
                Y_ref=32.0,
                a_mu=None,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, cond, t_coord, s_coord):
        """
        cond: [B, cond_dim]
        """
        B = cond.shape[0]

        # 条件 → 原型
        proto = self.cond_mod(cond)  # [B, 16, 2D]

        # 噪声场
        N = t_coord.shape[0]
        a0 = torch.exp(torch.randn(B, N, self.d_model, device=cond.device) * 0.8 - 2.5)
        theta0 = torch.rand(B, N, self.d_model, device=cond.device) * 2 * math.pi - math.pi

        # 基础 Omega
        omega_base = self._init_omega(t_coord, s_coord)

        # 物理演化
        a, theta = a0, theta0
        for i, blk in enumerate(self.blocks):
            tf = self.time_field.forward(i, cond.device)
            # 关键：传入 proto 作为 DriftField 的吸引中心
            a, theta, d = blk(
                a, theta, omega_base, t_coord, s_coord,
                kappa=tf["kappa"],
                T=tf["T"],
                omega_d=tf["omega_d"],
                targets=None,  # 生成模式
                proto=proto,
            )

        return a, theta, None

    def _init_omega(self, t_coord, s_coord):
        d_idx = torch.arange(self.d_model, device=t_coord.device).view(1, -1)
        base = 0.4 * torch.sin(
            s_coord[..., 0:1] * 12.9898
            + s_coord[..., 1:2] * 78.233
            + t_coord.unsqueeze(-1) * 37.719
            + d_idx * 4.1
        )
        return base.unsqueeze(0)


def _decodable(a_, th_):
    """与 v3700 一致的解码器"""
    z = torch.complex(a_ * torch.cos(th_), a_ * torch.sin(th_))
    N = z.shape[0]
    z = z * 10.0
    W_FFT = (PATCH * PATCH * 3) // 2 + 1
    flat = torch.fft.irfft2(z.reshape(N, WIN, W_FFT),
                            s=(WIN, PATCH * PATCH * 3),
                            dim=(-2, -1)).real
    flat = flat.reshape(N, WIN * PATCH * PATCH * 3)
    blocks = flat.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)
    return xt.reshape(TS * WIN, IMG, IMG, 3)


def main():
    # 素材 target field
    gp = ROOT / "data" / "gif_raw" / "noto" / GIF
    field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    t, s = build_spacetime_index(TS, GRID, GRID, device)
    target = field.unsqueeze(0)
    re, im = target[..., :D], target[..., D:]
    a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
    th_tar = torch.atan2(im, re)
    print(f"[DATA] {GIF}  N={target.shape[1]}  a_tar mean={a_tar.mean().item():.5f}")

    # CLIP 条件
    cond = np.load(CACHE / "cond.npy")
    sample_idx = 0 if GIF.startswith("8-ball") else 1
    cv = torch.from_numpy(cond[sample_idx:sample_idx + 1]).float().to(device)  # [1,2049]

    # 模型
    model = CondGenModelV2(cond_dim=2049, d_model=D, n_blocks=12)
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    # 参数量
    print(f"[MODEL] 总参数量：{sum(p.numel() for p in model.parameters()):,}")

    # Nyquist 环
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

    print(f"[RUN] {STEPS} 步条件生成 v2 训练  lr={LR}  GIF={GIF}")

    for step in range(1, STEPS + 1):
        opt.zero_grad()

        # 条件 → 物理演化（生成模式）
        a, theta, _ = model(cv, t, s)

        # 这里没有 target_field，我们需要用重建 loss：a, theta 与真实 a_tar, th_tar 比较
        # 但因为没有 target_field 驱动，实际需要利用真实场作为监督

        # Loss 计算（与重建口径一致）
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
    with torch.inference_mode():
        a, theta, _ = model(cv, t, s)
        z = (a[0] * torch.exp(1j * theta[0])).to(torch.complex64)
        fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
        frames = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D,
                                                 normalize=True)).astype(np.uint8)
        pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
        pil[0].save(OUT, save_all=True, append_images=pil[1:], loop=0, duration=100)

        # PSNR
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
