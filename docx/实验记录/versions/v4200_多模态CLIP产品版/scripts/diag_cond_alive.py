# -*- coding: utf-8 -*-
"""探针：判别损失到底死在哪一层（CPU，只读 ckpt，不碰训练）。

三件事：
  1. h = proj(cv) 的跨素材 cos（判别项的直接对象）
  2. dtheta 的跨素材 cos（判别分离度透过低秩后剩多少）
  3. 各路 wcos（生成相位对齐实况）
对照 DISC_TGT_COS=0.60 门限，判断 relu 是否死掉。
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

# ★ torch_accel_shim：与训练脚本同款，patch transformers/torch 兼容性
try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] torch_accel_shim 加载失败：{ex}")

import numpy as np
import torch

from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
CKPT = Path(os.environ.get("PREV_CKPT",
         str(ROOT / "outputs_real" / "fft4k" / "cond3_ckpt_1500_0744.pt")))
CLIP_CACHE = ROOT / "data" / "clip_cache"

torch.set_num_threads(4)
device = "cpu"

rsd = torch.load(CKPT, map_location="cpu")
rsd = {k: v for k, v in rsd.items() if not k.startswith("qwen.")}
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=256, init_mode="init",
                       n_modules=8, mod_gamma=0.15).to(device)
rret = model.load_state_dict(rsd, strict=False)
print(f"[CKPT] {CKPT.name} missing={len(rret.missing_keys)} "
      f"unexpected={len(rret.unexpected_keys)}")
model.eval()

cv_full = np.load(CLIP_CACHE / "cv3_full.npy")
t, s = build_spacetime_index(TS, GRID, GRID, device)

with torch.inference_mode():
    # 1) h = proj(cv) 跨素材 cos（判别项直接对象）
    cv0 = torch.from_numpy(cv_full[0:1]).to(device)
    cv1 = torch.from_numpy(cv_full[1:1 + 1]).to(device)
    h0 = model.cond_init.proj(cv0).reshape(-1)
    h1 = model.cond_init.proj(cv1).reshape(-1)
    cos_h = float(h0 @ h1 / (h0.norm() * h1.norm()))
    print(f"\n[1] h=proj(cv) 跨素材 cos = {cos_h:.4f}   "
          f"DISC_TGT_COS 门限 = 0.60")
    print(f"    relu(cos-0.60) = {max(cos_h - 0.60, 0.0):.4f}  "
          f"{'← 判别项已死(无梯度)' if cos_h <= 0.60 else '← 判别项应激活'}")

    # 2) dtheta 跨素材 cos（透过低秩后剩多少分离度）
    dt0, _, _ = model.cond_init(cv0, return_parts=True, n_pos=len(t))
    dt1, _, _ = model.cond_init(cv1, return_parts=True, n_pos=len(t))
    f0, f1 = dt0.reshape(-1), dt1.reshape(-1)
    cos_dt = float(f0 @ f1 / (f0.norm() * f1.norm()))
    print(f"[2] dtheta 跨素材 cos = {cos_dt:.4f}   "
          f"（1.0=完全同向=通用场；0=无关；<0=分开）")

    # 3) wcos（与目标相位对齐）两素材
    from codec.gif_codec import encode_gif_to_field
    for i, nm in enumerate(["8号球", "前辈前辈"]):
        gp = ROOT / "data" / "gif_raw" / "noto" / (
            "8-ball_1f3b1.gif" if i == 0 else
            "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif")
        f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D)
        a_t = torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12)
        th_t = torch.atan2(f[..., D:], f[..., :D])
        cv = torch.from_numpy(cv_full[i:i + 1]).to(device)
        a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                         batch_size=1, cond_vec=cv, use_cond_field=True)
        pe = torch.remainder(th - th_t + math.pi, 2 * math.pi) - math.pi
        w_ph = (a_t ** 2).clamp(min=1e-8)
        wcos = float((w_ph * torch.cos(pe)).sum() / w_ph.sum())
        print(f"[3] {nm:6s} wcos = {wcos:+.4f}")
