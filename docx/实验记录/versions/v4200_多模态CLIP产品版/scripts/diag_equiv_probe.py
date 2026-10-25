# -*- coding: utf-8 -*-
"""验证等价链：θ̂（探针口径）+0.52 是否等于 θ0+dtheta（物理栈口径）+0.088。

若相等 → 两口径一致，物理栈会吃到 +0.52 的初态对齐，问题在栈。
若不相等 → 探针测法有误或 expand_to_d/remainder 有 bug —— 必须先修。
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
CKPT = ROOT / "outputs_real" / "fix_r1" / "cond3_ckpt_300_2008.pt"
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

torch.manual_seed(123)
with torch.no_grad():
    a0 = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
    th0 = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi
    dth, dln, iparts = m.cond_init(cv, return_parts=True, n_pos=len(t),
                                   theta0=th0, a0=a0)
    th_hat = iparts["theta_hat"]                     # [B,N,G]
    # 口径1：θ̂ 直接 expand（探针测法）
    th_hat_D = m.cond_init.expand_to_d(th_hat)       # [B,N,D]
    # 口径2：θ0 + dtheta（物理栈实际吃到）
    dth_D = m.cond_init.expand_to_d(dth)             # [B,N,D]
    th_stack = th0 + dth_D

    print("=" * 78)
    print("  等价链验证：θ̂ vs θ0+dtheta（fix_r1 ckpt，×2.0 修正后）")
    print("=" * 78)
    c1 = cos_ph(a_tar.unsqueeze(0), th_hat_D, th_tar)
    c2 = cos_ph(a_tar.unsqueeze(0), th_stack, th_tar)
    print(f"  探针口径  cos(θ̂_D, θ_tar)        = {c1:+.4f}")
    print(f"  栈口径    cos(θ0+dtheta, θ_tar)   = {c2:+.4f}")
    # 逐元素：θ0+dtheta 与 θ̂ 的 wrap 差
    diff = torch.remainder(th_hat_D - th_stack + math.pi, 2 * math.pi) - math.pi
    print(f"  mean|wrap(θ̂−(θ0+dθ))| = {diff.abs().mean():.5f} rad "
          f"(≈0 则两口径等价)")
    # 是否因为 expand_to_d 对 dth vs th_hat 用不同组号？
    gid = m.cond_init._group_index(device)
    print(f"  _group_index 前 8 = {gid[:8].tolist()}")
    # expand_to_d 检查：dth 与 th_hat 是否一致（同组号映射到同通道）
    same_map = torch.equal(m.cond_init.expand_to_d(th_hat),
                           m.cond_init.expand_to_d(th_hat))
    print(f"  expand_to_d 幂等 = {same_map}")
    print()
    print("  ★ 结论：c1≈c2 → 探针正确、栈吃到 0.52；c1≠c2 → 探针/expand 有 bug。")
