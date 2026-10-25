# -*- coding: utf-8 -*-
"""形状约定探测：field / a / th / fld 到底是什么形状（2026-10-18）"""
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

import torch
import numpy as np
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import CLIP_EMB_DIM
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"
TAG = os.environ.get("DIAG_TAG") or "cond3_e4_arctan_g12_3000"
_FFN_H = int(os.environ.get("FFN_HIDDEN", "512"))

gp = ROOT / "data" / "gif_raw" / "noto" / "8-ball_1f3b1.gif"
f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D)
print(f"[field]      shape={tuple(f.shape)}   (期望 [{TS*GRID*GRID}, {2*D}])")

# ★ 正确约定：field [N, 2D] → a [N, D]（每 patch D 个频率系数）
a_t = torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12)
th_t = torch.atan2(f[..., D:], f[..., :D])
print(f"[target a]   shape={tuple(a_t.shape)}")
z = (a_t * torch.exp(1j * th_t)).to(torch.complex64)
fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
print(f"[fld]        shape={tuple(fld.shape)}")
rec = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False))
print(f"[codec 往返] shape={rec.shape}  dtype={rec.dtype}")

# 模型输出形状
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=_FFN_H, init_mode="init",
                       n_modules=8, mod_gamma=0.15).to(device)
sd = torch.load(ROOT / "outputs_real" / TAG / "cond3_ckpt.pt", map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()
cc = ROOT / "data" / "clip_cache"
text3 = np.load(cc / "text3.npy"); first3 = np.load(cc / "first3.npy")
gif3v = np.load(cc / "gif3_video.npy")
cv = np.concatenate([text3[:1], first3[:1], gif3v[:1], np.ones((1, 3), np.float32)], 1)
t, s = build_spacetime_index(TS, GRID, GRID, device)
with torch.inference_mode():
    a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                     batch_size=1, cond_vec=torch.from_numpy(cv).to(device),
                     use_cond_field=True)
print(f"[model a]    shape={tuple(a.shape)}")
print(f"[model th]   shape={tuple(th.shape)}")
