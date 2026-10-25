# -*- coding: utf-8 -*-
"""逐层保真诊断：相位/振幅从第几层开始崩，崩的速率。

恒等初态（force_init=素材场）+ 三路全给条件，hook 每个 block 输出：
  - cos(Δθ) 未加权 / a_tar² 加权（相位对齐）
  - |Δθ| wrap 均值
  - corr(log a_layer, log a_tar)（振幅包络对齐）
  - 每层 a 的能量占比逐 h

判读：
  - 第 1~2 层就崩 → ω 旋转 / kuramoto 步长主导（动力学过快）
  - 中段缓降、末段崩 → 深层 ph_res 指数放大（第二十一死穴路径）
  - 全程缓降 → OU 回复力 / μ 主导
"""
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
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")
os.environ.setdefault("COND_LOWRANK", "16")
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = Path(os.environ.get("CKPT", str(ROOT / "outputs_real" / "fix_sep_500" / "cond3_ckpt.pt")))
print("=" * 78)
print(f"逐层保真诊断  CKPT={CKPT.parent.name}/{CKPT.name}")

gif_dir = ROOT / "data" / "gif_raw" / "noto"
gp = gif_dir / "8-ball_1f3b1.gif"
if not gp.exists():
    gp = ROOT.parent.parent / "data" / "gif_raw" / "noto" / "8-ball_1f3b1.gif"
f_tar = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
a_tar = torch.sqrt(f_tar[..., :D] ** 2 + f_tar[..., D:] ** 2 + 1e-12)
th_tar = torch.atan2(f_tar[..., D:], f_tar[..., :D])
if a_tar.dim() == 2:
    a_tar, th_tar = a_tar.unsqueeze(0), th_tar.unsqueeze(0)

CLIP_CACHE = ROOT / "data" / "clip_cache"
text3 = np.load(CLIP_CACHE / "text3.npy")
first3 = np.load(CLIP_CACHE / "first3.npy")
gif3v = np.load(CLIP_CACHE / "gif3_video.npy")
one_cv = torch.from_numpy(build_cond_vector3(
    text3[0:1], first3[0:1], gif3v[0:1], (1, 1, 1))).to(device)

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()
t, s = build_spacetime_index(TS, GRID, GRID, device)

# ---------------------------------------------------------------- hooks
recs = []


def make_hook(idx):
    def hook(_mod, _inp, out):
        # out = (a_new, theta_new, sync_diag)
        a_o, th_o = out[0], out[1]
        with torch.no_grad():
            d = th_o - th_tar                      # cos 周期，不用 wrap
            cos_u = torch.cos(d).mean().item()
            w = a_tar.pow(2).clamp(min=1e-12)
            cos_w = float((w * torch.cos(d)).sum() / w.sum())
            dwrap = torch.remainder(d + torch.pi, 2 * torch.pi) - torch.pi
            dabs = dwrap.abs().mean().item()
            with torch.no_grad():
                la = torch.log(a_o.clamp(min=1e-8))
                lt = torch.log(a_tar.clamp(min=1e-8))
                la_c = la - la.mean()
                lt_c = lt - lt.mean()
                ac = float((la_c * lt_c).sum()
                           / (la_c.norm() * lt_c.norm() + 1e-12))
                e = a_o[0].pow(2).reshape(-1, WIN, W_FFT).mean(0)
                p = e.sum(1)
                hp = (p / p.sum()).cpu().numpy()
            recs.append(dict(i=idx, cos_u=cos_u, cos_w=cos_w, dabs=dabs,
                             ac=ac, hp=hp))
    return hook


_blocks = model.pmnncore.blocks
for _i, _b in enumerate(_blocks):
    _b.register_forward_hook(make_hook(_i))

with torch.inference_mode():
    a_out, th_out, _ = model(texts=None, target_field=f_tar, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=one_cv, use_cond_field=True,
                             force_init=(a_tar, th_tar))

# 初态（第 0 层之前）
with torch.no_grad():
    d = th_tar - th_tar
print(f"{'层':>4}{'cos(Δθ)':>11}{'cosW':>9}{'|Δθ|':>8}{'corr(loga)':>12}{'逐h占比':>34}")
print("-" * 78)
print(f"{'初态':>4}{'+1.000':>11}{'+1.000':>9}{'0.000':>8}{'1.000':>12}"
      f"  {np.round(recs[0]['hp'] * 100, 0) if recs else '-'}")
for r in recs:
    print(f"{r['i']:>4}{r['cos_u']:>+11.3f}{r['cos_w']:>+9.3f}{r['dabs']:>8.3f}"
          f"{r['ac']:>+12.3f}  {np.round(r['hp'] * 100, 0)}")
print("-" * 78)
print("判读：cos(Δθ) 从哪层开始掉到 0 附近 = 相位信息从哪层被洗掉")
print("=" * 78)
