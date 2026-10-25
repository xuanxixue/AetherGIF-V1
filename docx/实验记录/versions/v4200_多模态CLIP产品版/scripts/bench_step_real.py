# -*- coding: utf-8 -*-
"""实测当前架构单步耗时（分段 forward/backward/step），判断是否换页拖慢。

回答「150 步到底要多久」——不再靠猜。
输出：forward s / backward s / optimizer s / 单步合计 s，×150 步预估。
"""
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir(): break
    if _PROJ.parent == _PROJ: break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))
_DEPS = _PROJ / "deps312"
if _DEPS.is_dir(): sys.path.insert(0, str(_DEPS))
try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] torch_accel_shim 加载失败：{ex}")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
for k, v in dict(COND_GROUPS="5", COND_LOWRANK="16", PHASE_ACT="arctan",
                 RES_LIMIT="0.5", W2_NORM_MAX="3.0", KAPPA_MAX="6.0",
                 FFT_FFTW="1").items():
    os.environ.setdefault(k, v)

import numpy as np
import torch
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from train.spectral_loss import loss_hf_guard, loss_lo_floor, hf_ratio

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
device = "cuda" if torch.cuda.is_available() else "cpu"
gif_dir = ROOT / "data" / "gif_raw" / "noto"
t, s = build_spacetime_index(TS, GRID, GRID, device)
f0 = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"), TS, GRID, PATCH,
                         WIN, IMG, D).to(device)
a_tar = torch.sqrt(f0[..., :D] ** 2 + f0[..., D:] ** 2 + 1e-12)
th_tar = torch.atan2(f0[..., D:], f0[..., :D])
target = torch.cat([a_tar[None] * torch.cos(th_tar),
                    a_tar[None] * torch.sin(th_tar)], -1)

m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
m.train()
params = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=3e-4)

print(f"device={device}  参数量={sum(p.numel() for p in params):,}")

# warmup
for _ in range(2):
    a, th, _d = m(texts=None, target_field=target, t_coord=t, s_coord=s,
                  batch_size=1, cond_vec=None, use_cond_field=False)
    loss = a.pow(2).mean()
    loss.backward()
    opt.zero_grad()

torch.cuda.synchronize() if device == "cuda" else None
N = 5
tf = tb = ts_ = 0.0
for _ in range(N):
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    a, th, _d = m(texts=None, target_field=target, t_coord=t, s_coord=s,
                  batch_size=1, cond_vec=None, use_cond_field=False)
    loss = a.pow(2).mean()
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize() if device == "cuda" else None
    t2 = time.perf_counter()
    opt.step()
    opt.zero_grad()
    torch.cuda.synchronize() if device == "cuda" else None
    t3 = time.perf_counter()
    tf += t1 - t0; tb += t2 - t1; ts_ += t3 - t2

per = (tf + tb + ts_) / N
print(f"  forward       = {tf/N:.3f} s")
print(f"  backward      = {tb/N:.3f} s")
print(f"  optimizer     = {ts_/N:.3f} s")
print(f"  单步合计       = {per:.3f} s")
print(f"  → 150 步 ≈ {per*150/60:.1f} 分钟 = {per*150/3600:.2f} 小时")