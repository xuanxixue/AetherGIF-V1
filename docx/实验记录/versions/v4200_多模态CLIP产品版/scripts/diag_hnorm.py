# -*- coding: utf-8 -*-
"""治本模拟：若 h 归一化（std→~1），_pre 会到多少？arctan 梯度复活吗？

治本方案候选：
  C1) proj 后半段加 LayerNorm(h) -> h ~ N(0,1)
  C2) proj 输出加固定 gain 归一：h = h / detach(h.std())
模拟 C2：用 fix_r2 ckpt 的 h（std=23.48），除以 std 后再过 to_phase，
看 _pre 量级与梯度的变化（不重新训练，直接对现有权重做增益变换）。
"""
import os, sys, math
from pathlib import Path
import numpy as np, torch

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
try:
    from torch_accel_shim import install_full; install_full()
except Exception: pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
for k, v in dict(COND_GROUPS="5", COND_LOWRANK="16", PHASE_ACT="arctan",
                 RES_LIMIT="0.5", W2_NORM_MAX="3.0", KAPPA_MAX="6.0",
                 FFT_FFTW="1").items():
    os.environ.setdefault(k, v)

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"
gif_dir = ROOT / "data" / "gif_raw" / "noto"
t, s = build_spacetime_index(TS, GRID, GRID, device)
f = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"), TS, GRID, PATCH,
                        WIN, IMG, D).to(device)
a_tar = torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12)
th_tar = torch.atan2(f[..., D:], f[..., :D])
CKPT = ROOT / "outputs_real" / "fix_r2" / "cond3_ckpt_150_0048.pt"
sd = torch.load(CKPT, map_location=device)
sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
m.load_state_dict(sd, strict=False); m.eval()
cc = ROOT / "data" / "clip_cache"
cv = torch.from_numpy(build_cond_vector3(
    np.load(cc / "text3.npy")[0:1], np.load(cc / "first3.npy")[0:1],
    np.load(cc / "gif3_video.npy")[0:1], (1, 1, 1))).to(device)

def cos_ph(w, th_f, th_t):
    pe = torch.remainder(th_f - th_t.detach() + math.pi, 2 * math.pi) - math.pi
    return float((w.detach() ** 2 * torch.cos(pe)).sum() / (w.detach() ** 2).sum())

torch.manual_seed(0)
with torch.no_grad():
    h = m.cond_init.proj(cv)                      # [B,512]
    print("=" * 80)
    print("  h 归一化治本模拟（fix_r2 ckpt）")
    print("=" * 80)
    print(f"  h 原始: mean_abs={h.abs().mean():.2f} std={h.std():.2f} max={h.max():.2f}")
    hstd = h.std().item()
    for gain_label, hh in [("h 原始", h), ("h/std", h / hstd),
                           ("h/2σ", h / (2*hstd)), ("h/3σ", h / (3*hstd))]:
        _pre = m.cond_init.to_phase(hh).reshape(-1)
        sat = ((_pre.abs() > 5).float().mean() * 100).item()
        g = 2.0 / (1.0 + _pre ** 2)
        thS = torch.arctan(_pre) * 2.0             # ×2 已是修复后正确值域
        thS_g = thS.reshape(1, len(t), 5)
        thS_D = m.cond_init.expand_to_d(thS_g)
        c = cos_ph(a_tar.unsqueeze(0), thS_D, th_tar)
        print(f"  {gain_label:8s}: _pre mean_abs={_pre.abs().mean():5.1f} "
              f"饱和率={sat:5.1f}% 梯度均值={g.mean():.2e} 梯度p50={g.median():.2e} "
              f"cos={c:+.4f}")
    print()
    print("  ★ 治本判据：h 归一后 _pre 应降到 <10 量级、饱和率<50%、梯度均值≥1e-2。")
