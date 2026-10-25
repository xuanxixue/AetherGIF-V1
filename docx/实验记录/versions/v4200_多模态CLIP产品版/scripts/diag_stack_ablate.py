# -*- coding: utf-8 -*-
"""栈内破坏源消融：force_init=target，逐个关掉栈组件，看 PSNR 恢复多少。

对照目标：不走栈 = 51.08 dB。关掉某组件后 PSNR 越接近 51，该组件破坏越大。
组件：
  - FFN 幅度 res（log_res 置 0）
  - FFN 相位 res（ph_res 置 0）
  - Kuramoto 同步（kappa 置 0）
  - drift 相位吸引（g 置 0）
  - OU 回复力（a_gamma 固定 0.5 + a_mu 不动）

用法：C:/Python312/python.exe -u scripts/diag_stack_ablate.py
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
for k, v in dict(COND_GROUPS="5", COND_LOWRANK="16", PHASE_ACT="arctan",
                 RES_LIMIT="0.5", W2_NORM_MAX="3.0", KAPPA_MAX="6.0",
                 FFT_FFTW="1").items():
    os.environ.setdefault(k, v)

import numpy as np
import torch
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


def build():
    m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=256, init_mode="init", n_modules=8,
                       mod_gamma=0.15).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    # 所有实验统一固定 a_gamma=0.5（隔离 gamma 失稳这个已知因素）
    for i in range(12):
        m.pmnncore.blocks[i].a_gamma.data.copy_(torch.tensor(0.5))
    return m


def manual_stack(m, disable=()):
    """手动走栈，返回最终 (a, theta)。disable 是组件名集合。"""
    omega_base = m._init_omega(t, s)
    te = m.text_param.expand(1, -1).to(device)
    omega_bias = m.omega_bias_head(te)
    kappa_scale = torch.sigmoid(m.kappa_scale_head(te)).squeeze(-1)
    T_scale = torch.sigmoid(m.T_scale_head(te)).squeeze(-1)
    omega = omega_base + omega_bias.unsqueeze(1)
    kappa = (2.0 * kappa_scale + 0.0).view(-1, 1, 1)
    T = (0.5 * T_scale + 0.0).view(-1, 1, 1)
    mod_id = None
    if m.n_modules > 0:
        gi = torch.arange(len(t), device=device) // m.n_modules
        mod_id = gi % m.n_modules

    a, theta = a_tar.unsqueeze(0).clone(), th_tar.unsqueeze(0).clone()
    targets = f0.unsqueeze(0)
    for i in range(m.n_blocks):
        tf = m.pmnncore.time_field.forward(i, device)
        blk = m.pmnncore.blocks[i]
        # drift
        g = blk.drift(a, theta, targets, None, mod_id=mod_id,
                      cond_anchor=None, anchor_weight=0.3)
        if "drift" in disable:
            g = torch.zeros_like(g)
        # kuramoto
        theta_new, _ = blk.kuramoto(theta, a, omega, g, 
                                     tf["kappa"] * kappa, tf["T"] * T, t, s)
        # FFN
        ffn_out = blk.ffn(a, theta_new, tf["omega_d"], t, s)
        log_res, ph_res = ffn_out.chunk(2, dim=-1)
        if "amp_res" in disable:
            log_res = torch.zeros_like(log_res)
        if "ph_res" in disable:
            ph_res = torch.zeros_like(ph_res)
        # 幅度
        _rl = float(os.environ.get("RES_LIMIT", "0.5"))
        res = _rl * torch.tanh(log_res / _rl)
        log_a = torch.log(a.clamp(min=1e-8))
        log_a_new = log_a + res - blk.a_gamma * (log_a - blk.a_mu.view(1, 1, -1))
        log_a_new = log_a_new.clamp(-20.0, 5.0)
        a = torch.exp(log_a_new)
        # 相位
        theta = theta_new + ph_res.clamp(-1.5, 1.5)
    return a, theta


print("=" * 76)
print(f"  栈内破坏源消融（a_gamma 统一固定 0.5） {CKPT.name}")
print("=" * 76)
print(f"  [基准] 不走栈直接解码 target        PSNR = 51.08 dB")

cases = [
    ("① 全开（仅修 gamma）", ()),
    ("② 关 FFN幅度res", ("amp_res",)),
    ("③ 关 FFN相位res", ("ph_res",)),
    ("④ 关 drift", ("drift",)),
    ("⑤ 关 FFN幅度+相位res", ("amp_res", "ph_res")),
    ("⑥ 关 drift+相位res", ("drift", "ph_res")),
    ("⑦ 全关（只剩 OU 递推）", ("amp_res", "ph_res", "drift")),
]
for tag, dis in cases:
    m = build()
    with torch.no_grad():
        a, th = manual_stack(m, dis)
    img = decode(a, th)
    p = psnr(img, ref)
    cph = float(torch.cos(th[0, :, 1:] - th_tar[:, 1:]).mean())
    astd = float(a[0, :, 1:].std())
    print(f"  {tag:24s} PSNR={p:6.2f} dB  cos(θ)={cph:+.4f}  幅度std={astd:.4f}")

print("=" * 76)