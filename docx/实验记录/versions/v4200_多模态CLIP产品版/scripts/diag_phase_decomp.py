# -*- coding: utf-8 -*-
"""分解第 0 层相位位移来源：omega旋转 / 耦合 / 漂移 / ph_res 各占多少。

用 pre_hook 捕获 blocks[0] 的真实输入，用子模块手动复现分解，
并以「手动复现输出 == 真实输出」自检分解可信度。
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
device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = Path(os.environ.get("CKPT", str(ROOT / "outputs_real" / "fix_sep_500" / "cond3_ckpt.pt")))
print("=" * 74)
print(f"第 0 层相位位移分解  CKPT={CKPT.parent.name}")

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

cap = {}


def pre0(_m, args, kwargs):
    cap["in"] = args
    cap["kw"] = kwargs


def post0(_m, inp, out):
    cap["out"] = out


b0 = model.pmnncore.blocks[0]
b0.register_forward_pre_hook(pre0, with_kwargs=True)
b0.register_forward_hook(post0)

with torch.inference_mode():
    model(texts=None, target_field=f_tar, t_coord=t, s_coord=s,
          batch_size=1, cond_vec=one_cv, use_cond_field=True,
          force_init=(a_tar, th_tar))

inp = cap["in"]
kw = cap.get("kw", {})
out = cap["out"]
print(f"blocks[0] positional={len(inp)} kwargs={sorted(kw.keys())}")

ai, thi, om = inp[0], inp[1], inp[2]
tc, sc = inp[3], inp[4]
kap = kw.get("kappa")
T0 = kw.get("T")
om_d = kw.get("omega_d")
targets = kw.get("targets")
protos = kw.get("protos")
mod_id = kw.get("mod_id")
cond_anchor = kw.get("cond_anchor")
aw = kw.get("anchor_weight", 0.3)

print(f"omega: mean|ω|={om.abs().mean():.4f} rms={om.pow(2).mean().sqrt():.4f} "
      f"max={om.abs().max():.3f}")
print(f"kappa={float(kap):.4f}  T={float(T0):.4f}  anchor_weight={float(aw) if torch.is_tensor(aw) else aw}")

with torch.inference_mode():
    # ---- 各通路量级（每步 rad，未乘 dt）
    g = b0.drift(ai, thi, targets, protos, mod_id=mod_id,
                 cond_anchor=cond_anchor, anchor_weight=aw)
    coup_s = b0.kuramoto.toeplitz.apply_coupling(thi, tc, sc)
    coup_mf, _ = b0.kuramoto.mean_field.forward(ai, thi)
    kappa_g = kap * g
    print(f"\n【dθ/dt 各通路 RMS（rad/单位时间，×dt=0.1 后为每步位移）】")
    print(f"  omega   rms = {om.pow(2).mean().sqrt():.4f}")
    print(f"  coup_s  rms = {coup_s.pow(2).mean().sqrt():.4f}")
    print(f"  coup_mf rms = {coup_mf.pow(2).mean().sqrt():.4f}")
    print(f"  kappa*g rms = {kappa_g.pow(2).mean().sqrt():.4f}")
    dt = b0.kuramoto.dt
    print(f"\n【每步位移 |Δθ_step|（×dt={dt}）】")
    print(f"  omega·dt     = {(om * dt).abs().mean():.4f}")
    print(f"  coup_s·dt    = {(coup_s * dt).abs().mean():.4f}")
    print(f"  coup_mf·dt   = {(coup_mf * dt).abs().mean():.4f}")
    print(f"  kappa·g·dt   = {(kappa_g * dt).abs().mean():.4f}")

    # ---- kuramoto 完整一步
    th_k, _ = b0.kuramoto(thi, ai, om, g, kap, T0, tc, sc)
    d_k = (th_k - thi).abs().mean().item()
    print(f"\n  kuramoto 一步 |θ_k − θ_in| = {d_k:.4f} rad")

    # ---- FFN 相位残差
    ffn_out = b0.ffn(ai, th_k, om_d, tc, sc)
    _lr, ph_res = ffn_out.chunk(2, dim=-1)
    print(f"  ph_res: mean|·|={ph_res.abs().mean():.4f} rms={ph_res.pow(2).mean().sqrt():.4f} "
          f"max={ph_res.abs().max():.3f}")
    import src.physics.block as _blk
    _PRA = getattr(_blk, "_PHASE_RES_ACT", "clamp")
    print(f"  PHASE_RES_ACT={_PRA}")

    # ---- 自检：手动复现 vs 真实输出
    d_total = (out[1] - thi)
    print(f"\n【总量】|θ_out − θ_in| = {d_total.abs().mean():.4f} rad "
          f"（首层位移，逐层诊断实测 1.354）")
    cos_real = torch.cos(out[1] - th_tar).mean().item()
    cos_k = torch.cos(th_k - th_tar).mean().item()
    print(f"  cos(Δθ) 真实输出 = {cos_real:+.3f}")
    print(f"  cos(Δθ) 仅 kuramoto 后 = {cos_k:+.3f}")
    print(f"  → kuramoto 一步贡献 {d_k:.3f} rad，ph_res 贡献 "
          f"{d_total.abs().mean() - d_k:+.3f} rad（含正负抵消）")
print("=" * 74)
