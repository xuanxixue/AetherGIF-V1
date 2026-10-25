# -*- coding: utf-8 -*-
"""验证修复方向：对 _pre 做前置增益归一，能否让 arctan 回到线性区、梯度复活。

候选方案（纯结构归一，不注入 target 先验）：
  S1) arctan(_pre / K) * 2   —— K 固定，把 ±771 压回线性区
  S2) tanh(_pre / K) * π     —— tanh 有界，值域 (−1,1)*π=(−π,π)，线性区更宽
对比梯度：dθ̂/d_pre 应恢复到 ~0.1~1 量级（而非 1e-5）。
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
    h = m.cond_init.proj(cv)
    _pre = m.cond_init.to_phase(h).reshape(-1)
    print("=" * 80)
    print("  前置增益归一修复方向验证（fix_r2 ckpt）")
    print("=" * 80)
    print(f"  _pre 量级: mean_abs={_pre.abs().mean():.0f} max={_pre.abs().max():.0f}")
    print(f"  饱和率(|_pre|>5): {((_pre.abs()>5).float().mean()*100):.1f}%")
    print(f"  {'方案':22s} | {'梯度均值':>10s} | {'梯度p50':>10s} | θ̂ 分布 | cos(θ̂,tar)")
    # 现状
    thA = torch.arctan(_pre) * math.pi
    gA = (math.pi/(1+_pre**2))
    print(f"  {'A 旧 arctan*π(无归一)':22s} | {gA.mean():.2e} | {gA.median():.2e} | "
          f"std={thA.std():.1f} | {cos_ph(a_tar.unsqueeze(0), thA.reshape(1,-1)[:, :len(t)][:, :D] if False else thA.reshape(1,-1)[:,:125], th_tar):+.4f}")
    # 候选 S0: arctan(_pre/20)*2
    for K in [10, 20, 50]:
        thS = torch.arctan(_pre / K) * 2.0
        gS = (2.0/K) / (1 + (_pre/K)**2)
        # θ̂ 与 target 的 cos（需 reshape 成 [B,N,G]→expand D）
        thS_g = thS.reshape(1, len(t), 5)
        thS_D = m.cond_init.expand_to_d(thS_g)
        c = cos_ph(a_tar.unsqueeze(0), thS_D, th_tar)
        print(f"  {'S0 arctan(_pre/'+str(K)+')*2':22s} | {gS.mean():.2e} | {gS.median():.2e} | "
              f"std={thS.std():.1f} | {c:+.4f}")
    # 候选 S1: tanh(_pre/K)*π
    for K in [10, 20, 50]:
        thS = torch.tanh(_pre / K) * math.pi
        gS = (math.pi/K) * (1 - torch.tanh(_pre/K)**2)
        thS_g = thS.reshape(1, len(t), 5)
        thS_D = m.cond_init.expand_to_d(thS_g)
        c = cos_ph(a_tar.unsqueeze(0), thS_D, th_tar)
        print(f"  {'S1 tanh(_pre/'+str(K)+')*π':22s} | {gS.mean():.2e} | {gS.median():.2e} | "
              f"std={thS.std():.1f} | {c:+.4f}")
    print()
    print("  ★ 判断：梯度均值应回升到 ≥1e-2 且 cos 维持 +0.5 附近，方案才有效。")
