# -*- coding: utf-8 -*-
"""零 ω 探针：自由旋转是不是结构杀手。

在 diag_pure_dynamics 基础上，把 model._init_omega 替换为返回全零，
其余不变（初态=目标场、cond 关、无 target）。
判读：PSNR 大涨（→20dB+）→ 自由旋转是主凶手，修法=降 ω 幅度/加可学补偿。
"""
import os, sys, math
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir(): break
    if _PROJ.parent == _PROJ: break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))
_DEPS = _PROJ / "deps312"
if _DEPS.is_dir(): sys.path.insert(0, str(_DEPS))
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception: pass
os.environ.setdefault("COND_GROUPS", "5"); os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan"); os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0"); os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1"); os.environ.setdefault("IDEAL_CLAMP", str(math.pi))
os.environ.setdefault("COND_INIT_PC", "1"); os.environ.setdefault("COND_PC_RANK", "8")
import numpy as np, torch
try:
    from torch_accel_shim import install_full; install_full()
except Exception as ex: print(f"[WARN] {ex}")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = Path(os.environ.get("CKPT", str(ROOT / "outputs_real" / "guid4_long" / "guided_ckpt.pt")))
GIF_NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]
print("=" * 74)
print(f"[零 ω 探针] ckpt = {CKPT.parent.name}/{CKPT.name}")

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd, strict=False); model.eval()

# ★ monkey-patch：omega 全零（并记下原始 omega 的旋转幅度）
t, s = build_spacetime_index(TS, GRID, GRID, device)
_omg_real = model._init_omega(t, s)
print(f"原始 omega: |mean|={_omg_real.abs().mean():.4f}  |max|={_omg_real.abs().max():.4f}"
      f"  12 层累积旋转幅度上限 = {(_omg_real.abs().max()*12):.2f} rad")
model._init_omega = lambda t_coord, s_coord: torch.zeros(
    1, len(t_coord), D, device=device)

gif_dir = ROOT / "data" / "gif_raw" / "noto"
for gi, gn in enumerate(GIF_NAMES):
    gp = gif_dir / gn
    if not gp.exists():
        gp = ROOT.parent.parent / "data" / "gif_raw" / "noto" / gn
    f_tar = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    a_tar = torch.sqrt(f_tar[..., :D] ** 2 + f_tar[..., D:] ** 2 + 1e-12)
    th_tar = torch.atan2(f_tar[..., D:], f_tar[..., :D])
    if a_tar.dim() == 2: a_tar, th_tar = a_tar.unsqueeze(0), th_tar.unsqueeze(0)
    z = (a_tar[0] * torch.exp(1j * th_tar[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    fr_tar = np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
    with torch.inference_mode():
        a_out, th_out, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                                 batch_size=1, cond_vec=None, use_cond_field=False,
                                 force_init=(a_tar, th_tar))
    zo = (a_out[0] * torch.exp(1j * th_out[0])).to(torch.complex64)
    fldo = torch.cat([zo.real, zo.imag], dim=-1).float().cpu()
    fr_out = np.stack(decode_field_to_frames(
        fldo, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
    n = min(len(fr_out), len(fr_tar))
    mse = ((fr_out[:n] - fr_tar[:n]) ** 2).mean()
    p = 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))
    print(f"[src{gi}] 零ω 保持: PSNR = {p:6.2f} dB   相位 |Δ| = {(th_out-th_tar).abs().mean():.4f} rad")
print("=" * 74)
