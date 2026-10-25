# -*- coding: utf-8 -*-
"""深度探针：为什么 cond_init 150 步学不动？

测三件事：
  1. h = proj(cond) 的**量级**（若 ±900，是 h 爆炸，不是 to_phase 权重问题）
  2. A/B 方案下 θ̂ 的实际**分布**（是否被 arctan 饱和压成二值）
  3. arctan 饱和区的**梯度**：dθ̂/d_pre = 2/(1+_pre²)，_pre 大时趋近 0 → 梯度死亡
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
print(f"device={device}")
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
    h_proj0 = m.cond_init.proj[0](cv)             # Linear(6147,512)
    h_silu0 = torch.nn.functional.silu(h_proj0)   # SiLU
    h_proj2 = m.cond_init.proj[2](h_silu0)        # Linear(512,512)
    h = torch.nn.functional.silu(h_proj2)         # 最终 h
    _pre = m.cond_init.to_phase(h)                # [B,30720]
    print("=" * 80)
    print("  cond_init 学不动深探针（fix_r2 ckpt）- 逐层 h 量级")
    print("=" * 80)
    def lv(x, tag):
        print(f"  {tag:22s} 形状={tuple(x.shape)} min={x.min():+.2f} max={x.max():+.2f} "
              f"mean_abs={x.abs().mean():.2f} std={x.std():.2f}")
    lv(cv, "cv(cond 输入)")
    lv(h_proj0, "proj.0 Linear 出")
    lv(h_silu0, "SiLU 后")
    lv(h_proj2, "proj.2 Linear 出")
    lv(h, "最终 h(经SiLU)")
    print(f"  → 爆发层定位：哪一层 mean_abs 从 0.0x 跳到几十")
    flat = _pre.reshape(-1)
    print(f"  _pre(to_phase出) 量级: min={flat.min():+.0f} max={flat.max():+.0f} "
          f"mean_abs={flat.abs().mean():.0f}")
    # arctan 饱和区梯度：d(arctan*2)/d_pre = 2/(1+_pre²)
    g = 2.0 / (1.0 + flat ** 2)
    print(f"  arctan*2 梯度 dθ̂/d_pre: 均值={g.mean():.2e} "
          f"p50={g.median():.2e} p5={g.quantile(0.05):.2e}")
    print(f"  → 饱和率(|_pre|>5): {((flat.abs()>5).float().mean()*100):.1f}%  "
          f"(这些位置梯度<0.077)")
    # A/B θ̂ 分布
    thA = (torch.arctan(flat) * math.pi)
    thB = (torch.arctan(flat) * 2.0)
    def stats(x, tag):
        print(f"  {tag:16s} θ̂: mean={x.mean():+.3f} std={x.std():.3f} "
              f"|值>π比例={((x.abs()>math.pi).float().mean()*100):.1f}%")
    stats(thA, "A arctan*π(旧)")
    stats(thB, "B arctan*2(新)")
    print()
    # 关键：h 爆炸是 proj 权重问题还是 cond 输入问题？
    print(f"  proj.0.weight std={m.cond_init.proj[0].weight.std():.3f} "
          f"(cond_dim=6147)")
    print(f"  proj.2.weight std={m.cond_init.proj[2].weight.std():.3f}")
    print(f"  cv(cond) 量级: mean_abs={cv.abs().mean():.2f} std={cv.std():.2f}")
    print()
    print("  ★ 结论：若 h 爆炸(±900) → 是 proj 无归一化所致；θ̂ 被 arctan 压成二值，"
          "饱和区梯度死亡 → cond_init 学不动。")
