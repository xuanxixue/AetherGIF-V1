# -*- coding: utf-8 -*-
"""逐层栈衰减诊断：force_init=target 后，逐 block 看 PSNR/幅度corr/相位cos 衰减，
定位是哪个算子（omega 驱动 / kappa 同步 / FFN幅度res / FFN相位res）在破坏信息。

用法：C:/Python312/python.exe -u scripts/diag_stack_decay.py
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir():
        break
    if _PROJ.parent == _PROJ:
        break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))
_DEPS = _PROJ / "deps312"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
for k, v in dict(COND_GROUPS="5", COND_LOWRANK="16", PHASE_ACT="arctan",
                 RES_LIMIT="0.5", W2_NORM_MAX="3.0", KAPPA_MAX="6.0",
                 FFT_FFTW="1").items():
    os.environ.setdefault(k, v)

import numpy as np
import torch
import torch.nn.functional as F
from codec.gif_codec import (encode_gif_to_field, decode_field_to_frames,
                             load_gif_frames, resample_frames)
from model.aether_gif_model import AetherGIFModel
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"
gif_dir = ROOT / "data" / "gif_raw" / "noto"
CKPT = ROOT / (os.environ.get("DIAG_CKPT") or "outputs_real/fix_r4/cond3_ckpt.pt")

t, s = build_spacetime_index(TS, GRID, GRID, device)


def psnr(a, b):
    return 10 * math.log10(255.0 ** 2 / max(
        ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(), 1e-12))


def decode(a, th):
    fld = torch.cat([a[0] * torch.cos(th[0]), a[0] * torch.sin(th[0])],
                    -1).cpu().float()
    return np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


gp = gif_dir / "8-ball_1f3b1.gif"
f0 = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
a_tar = torch.sqrt(f0[..., :D] ** 2 + f0[..., D:] ** 2 + 1e-12).to(device)
th_tar = torch.atan2(f0[..., D:], f0[..., :D]).to(device)
raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
ref = resample_frames(raw, TS * WIN).astype(np.float32)

sd = torch.load(CKPT, map_location=device)
sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
m.load_state_dict(sd, strict=False)
m.eval()

# 直接解码 target 的对照
print("=" * 84)
print(f"  逐层栈衰减诊断  {CKPT.name}")
print("=" * 84)
img_ref = decode(a_tar.unsqueeze(0), th_tar.unsqueeze(0))
print(f"  [对照] 直接解码 target（不走栈）PSNR = {psnr(img_ref, ref):.2f} dB")

# 手动逐 block 走栈，force_init=target，drift_target=target
with torch.no_grad():
    a0 = a_tar.unsqueeze(0).clone()
    th0 = th_tar.unsqueeze(0).clone()
    # 与 forward 一致：omega 确定性 + 可学习 bias（用 text_param 兜底）
    omega_base = m._init_omega(t, s)
    # 无 cond → text_param
    te = m.text_param.expand(1, -1).to(device)
    omega_bias = m.omega_bias_head(te)
    drift_targets = m.drift_target_head(te)
    kappa_scale = torch.sigmoid(m.kappa_scale_head(te)).squeeze(-1)
    T_scale = torch.sigmoid(m.T_scale_head(te)).squeeze(-1)
    omega = omega_base + omega_bias.unsqueeze(1)
    kappa = (2.0 * kappa_scale + 0.0).view(-1, 1, 1)
    T = (0.5 * T_scale + 0.0).view(-1, 1, 1)
    mod_id = None
    if m.n_modules > 0:
        gi = torch.arange(len(t), device=device) // m.n_modules
        mod_id = gi % m.n_modules

    a, theta = a0, th0
    targets = f0.unsqueeze(0)  # [1, N, 2D]
    print(f"\n  {'layer':>5}{'PSNR':>8}{'corr(loga)':>12}{'cos(θ)':>10}{'|a|std':>9}{'Δθ/层':>8}")
    prev_th = th0
    for i in range(m.n_blocks):
        tf = m.pmnncore.time_field.forward(i, device)
        _k_eff = (tf["kappa"] * kappa).clamp(max=6.0)
        a, theta, _ = m.pmnncore.blocks[i](
            a, theta, omega, t, s,
            kappa=_k_eff, T=tf["T"] * T, omega_d=tf["omega_d"],
            targets=targets, protos=None, mod_id=mod_id,
            cond_anchor=None, anchor_weight=0.3)
        if i in (0, 1, 2, 3, 5, 7, 9, 11):
            img = decode(a, theta)
            # corr(log a) 去 DC
            la = torch.log(a[0, :, 1:].clamp(min=1e-8))
            lt = torch.log(a_tar[:, 1:].clamp(min=1e-8))
            la = la - la.mean(); lt = lt - lt.mean()
            corr = float((la * lt).sum() / (la.norm() * lt.norm() + 1e-9))
            c_ph = float(torch.cos(theta[0, :, 1:] - th_tar[:, 1:]).mean())
            dth = float(torch.remainder(
                theta - prev_th + math.pi, 2 * math.pi).sub(math.pi).abs().mean())
            print(f"  {i:>5d}{psnr(img, ref):>8.2f}{corr:>+12.4f}{c_ph:>+10.4f}"
                  f"{a[0, :, 1:].std():>9.4f}{dth:>8.3f}")
            prev_th = theta

print("\n" + "=" * 84)
print("  判读：")
print("  若 PSNR 前几层就暴跌 → 栈前段(浅层)在破坏")
print("  cos(θ) 掉得快 → 相位(ω驱动/κ同步)在破坏；corr(loga) 掉得快 → 幅度(FFN res)在破坏")
print("  对照：物理栈残留越大，越需要修栈而不是修为多 loss")
print("=" * 84)