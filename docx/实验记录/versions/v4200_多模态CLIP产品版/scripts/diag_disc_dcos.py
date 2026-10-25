# -*- coding: utf-8 -*-
"""探针 d2：模拟训练时 disc 副项的计算口径，看 cos 到底多少。

训练里 dth = diag["dtheta"]（含 FFT 旁路修正，[B,N,D]），
_dt2 = cond_init 直接返回（不含 FFT 修正，[B,N,G] 再 expand）。
两者口径不同 → 副项 cos 可能与探针 1（都走 forward）不同。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("FFT_FFTW", "1")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import sys
import math
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
if (_PROJ / "deps312").is_dir():
    sys.path.insert(0, str(_PROJ / "deps312"))
_VENV = _PROJ / ".venv" / "Lib" / "site-packages"
if _VENV.is_dir():
    sys.path.append(str(_VENV))

try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] shim: {ex}")

import numpy as np
import torch

from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
CKPT = ROOT / "outputs_real" / "fft4k" / "cond3_ckpt_1500_0744.pt"
CLIP_CACHE = ROOT / "data" / "clip_cache"

torch.set_num_threads(4)
device = "cpu"

rsd = torch.load(CKPT, map_location="cpu")
rsd = {k: v for k, v in rsd.items() if not k.startswith("qwen.")}
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=256, init_mode="init",
                       n_modules=8, mod_gamma=0.15).to(device)
model.load_state_dict(rsd, strict=False)
model.eval()

cv_full = np.load(CLIP_CACHE / "cv3_full.npy")
t, s = build_spacetime_index(TS, GRID, GRID, device)

text3 = np.load(CLIP_CACHE / "text3.npy")
first3 = np.load(CLIP_CACHE / "first3.npy")
gif3v = np.load(CLIP_CACHE / "gif3_video.npy")


def build_one_cv(src_i, has):
    z = np.zeros((1, 1024), dtype=np.float32)
    from data.clip_condition import CLIP_EMB_DIM
    z = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
    tt = text3[src_i:src_i + 1] if has[0] else z
    ff = first3[src_i:src_i + 1] if has[1] else z
    gg = gif3v[src_i:src_i + 1] if has[2] else z
    return build_cond_vector3(tt, ff, gg, has)


COMBOS = [("三路全给", (1, 1, 1)), ("文生", (1, 0, 0)), ("图生", (0, 1, 0)),
          ("GIF本身", (0, 0, 1)), ("图文生", (1, 1, 0)), ("文+GIF", (1, 0, 1)),
          ("图+GIF", (0, 1, 1))]

with torch.inference_mode():
    print("\n== 训练口径复刻：dth(含FFT旁路) vs _dt2(纯cond_init) ==")
    for name, has in COMBOS:
        cv0 = torch.from_numpy(build_one_cv(0, has)).to(device)
        cv1 = torch.from_numpy(build_one_cv(1, has)).to(device)
        # dth: 完整 forward（含 FFT 旁路）
        _, _, dg0 = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                          batch_size=1, cond_vec=cv0, use_cond_field=True)
        dth0 = dg0.get("dtheta")
        _, _, dg1 = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                          batch_size=1, cond_vec=cv1, use_cond_field=True)
        dth1 = dg1.get("dtheta")
        # _dt2: cond_init 直接返回（训练里另一素材的口径）
        dt2, _, _ = model.cond_init(cv1, return_parts=True, n_pos=len(t))
        dt2d = model.cond_init.expand_to_d(dt2) if dt2.shape != dth0.shape else dt2

        def cos(x, y):
            a, b = x.reshape(-1), y.reshape(-1)
            return float(a @ b / (a.norm() * b.norm() + 1e-12))

        print(f"  {name:5s}: cos(dth0,dth1)={cos(dth0, dth1):+.4f}  "
              f"cos(dth0,dt2d)={cos(dth0, dt2d):+.4f}  "
              f"shapes={tuple(dth0.shape)}/{tuple(dt2d.shape)}")
