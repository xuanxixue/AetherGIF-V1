# -*- coding: utf-8 -*-
"""验证 a_gamma 修复对物理栈保真度的作用（零训练，直接改权重前向）。

对照三种 a_gamma 处理，看 force_init=target 后物理栈的 PSNR：
  ① 原样（当前，含负值）
  ② clamp 非负（min=0）
  ③ 全部重置 0.3（初值）
  ④ clamp 到固定 γ_lvl=0.5（记忆铁律推荐：γ 应固定非可训练）

用法：C:/Python312/python.exe -u scripts/diag_gamma_fix.py
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
    return m


def run(tag, gamma_fn):
    m = build()
    with torch.no_grad():
        for i in range(12):
            g = float(sd[f"pmnncore.blocks.{i}.a_gamma"])
            g2 = gamma_fn(g)
            m.pmnncore.blocks[i].a_gamma.copy_(torch.tensor(g2))
        a, th, _ = m(texts=None, target_field=f0.unsqueeze(0), t_coord=t,
                     s_coord=s, batch_size=1,
                     force_init=(a_tar.unsqueeze(0), th_tar.unsqueeze(0)))
    img = decode(a, th)
    p = psnr(img, ref)
    # 幅度 std 终态
    astd = float(a[0, :, 1:].std())
    cph = float(torch.cos(th[0, :, 1:] - th_tar[:, 1:]).mean())
    print(f"  {tag:34s} PSNR={p:6.2f} dB  幅度std={astd:.4f}  cos(θ)={cph:+.4f}")
    return p


print("=" * 88)
print(f"  a_gamma 修复对照（零训练，force_init=target） {CKPT.name}")
print("=" * 88)
print(f"  [对照] 直接解码 target 不走栈                PSNR=51.08 dB")
run("① 原样（含负 a_gamma）", lambda g: g)
run("② clamp 非负 (max(0,g))", lambda g: max(0.0, g))
run("③ 重置 0.3（初值）", lambda g: 0.3)
run("④ 固定 γ_lvl=0.5（记忆铁律）", lambda g: 0.5)
run("⑤ clamp 到 [0.1, 1.0]", lambda g: min(1.0, max(0.1, g)))
print("=" * 88)
print("  判读：哪种处理让 PSNR 离 51 最近，就用哪种。")
print("  若④固定 γ 最好 → 按记忆铁律把 a_gamma 改回非可训练常量。")
print("=" * 88)