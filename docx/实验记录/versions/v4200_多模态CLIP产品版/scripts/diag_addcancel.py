# -*- coding: utf-8 -*-
"""加法坍缩探针：CLIP路 θ̂ 与 FFT路 θ̂ 到底是不是反相抵消。

加载 fix_r1 ckpt，三路全给 src0，no_grad 前向，
输出四条相位绝对对齐度 + 两路 θ̂ 之间的余弦：
  cos_clip = cos(展开D的θ̂_clip, th_tar)
  cos_fft  = cos(展开D的θ̂_fft,  th_tar)
  cos_both = cos(θ̂_clip+θ̂_fft, th_tar)   ← 实际训练初态（相对 θ0）
  cos_inter= cos(θ̂_clip, θ̂_fft)          ← 两路彼此反相？<0 即抵消
关键：若 cos_inter 显著 <0，则两路在逐点相加时相互抵消，
      这就是「单独 FFT 0.33 / 相加 0.075」坍缩的机制。
"""
import os, sys, math
from pathlib import Path
import numpy as np, torch

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
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")

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

CKPT = ROOT / "outputs_real" / "fix_r1" / "cond3_ckpt_300_2008.pt"
sd = torch.load(CKPT, map_location=device)
sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
m.load_state_dict(sd, strict=False)
m.eval()

cc = ROOT / "data" / "clip_cache"
cv = torch.from_numpy(build_cond_vector3(
    np.load(cc / "text3.npy")[0:1], np.load(cc / "first3.npy")[0:1],
    np.load(cc / "gif3_video.npy")[0:1], (1, 1, 1))).to(device)


def cos_ph(w, th_f, th_t):
    """a² 加权相位余弦。"""
    pe = torch.remainder(th_f - th_t.detach() + math.pi, 2 * math.pi) - math.pi
    return float((w.detach() ** 2 * torch.cos(pe)).sum() /
                 (w.detach() ** 2).sum())


torch.manual_seed(0)
with torch.no_grad():
    th0 = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi
    a0 = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)

    # CLIP 路 θ̂
    dth_c, dln_c, ip = m.cond_init(cv, return_parts=True, n_pos=len(t),
                                   theta0=th0, a0=a0)
    th_hat_clip = ip["theta_hat"]                 # [B,N,G]
    th_hat_clip_D = m.cond_init.expand_to_d(th_hat_clip)   # [B,N,D]
    dth_cD = m.cond_init.expand_to_d(dth_c)

    # FFT 路 θ̂（dth = atan → (−π/2,π/2)，是 θ̂ 的原始预测）
    fdth, fdla, _ = m.fft_bypass(f.unsqueeze(0), return_parts=True)
    # FFT 路的 dteil 直接用 atan 的 θ̂ 当 target 相位预测
    fdth_D = m.fft_bypass.expand_to_d(fdth)

    print("=" * 78)
    print("  加法坍缩探针（fix_r1 ckpt，三路全给 src0）")
    print("=" * 78)
    print(f"  cos(θ̂_clip_D, θ_tar)   = {cos_ph(a_tar.unsqueeze(0), th_hat_clip_D, th_tar):+.4f}")
    print(f"  cos(θ̂_fft_D,  θ_tar)   = {cos_ph(a_tar.unsqueeze(0), fdth_D, th_tar):+.4f}")
    # 两路 θ̂ 彼此余弦
    _x = th_hat_clip_D.reshape(-1)
    _y = fdth_D.reshape(-1)
    _n = _x.norm() * _y.norm()
    c_inter = float(_x @ _y / _n) if _n > 1e-8 else float("nan")
    print(f"  cos(θ̂_clip, θ̂_fft)      = {c_inter:+.4f}   ← <0 即反相抵消")
    # 相加后的综合
    comb = dth_cD + fdth_D
    print(f"  cos(dθ_clip+dθ_fft, tar)= {cos_ph(a_tar.unsqueeze(0), comb, th_tar):+.4f}")
    print()
    print("  ★ 结论：若 cos_inter 强负 → 两路 θ̂ 反相，逐点相加必然抵消。")
    print(f"  ★ 若由此可判：FFT 路独立 0.33 是**可达上限**，CLIP 路在拖后腿。")
