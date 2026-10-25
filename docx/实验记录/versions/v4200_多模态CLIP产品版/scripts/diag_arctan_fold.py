# -*- coding: utf-8 -*-
"""探针：验证 cond_init θ̂ 头的 arctan(_pre)*π 值域折叠是否是负相关元凶。

问题假设：
  theta_hat = arctan(_pre) * math.pi      (cond_init.py:248)
  arctan 值域 = (−π/2, +π/2)（半圆，宽 π）
  ×π 后 = (−π²/2, +π²/2) ≈ (−4.93, +4.93)，**超出全圆 (−π, π]**。
  当 _pre 较大时 arctan 渐近饱和到 ±π/2，×π → ±4.93，wrap 后翻转成 ∓1.35
  —— 模型以为在增大正相位，实际被 wrap 成负相位 → 相位方向反转。

对比：
  A) arctan(_pre)*π（现状）
  B) arctan(_pre)*2（值域正好 (−π, π)，全圆不折叠）
  C) atan2(sin_pre, cos_pre) 双通道（数学上唯一连续全圆表示，理论最优）

本探针输出 fix_r1 ckpt 下三种方案的 cos(θ̂, θ_tar) 与折叠率，判断修复方向。
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
    pe = torch.remainder(th_f - th_t.detach() + math.pi, 2 * math.pi) - math.pi
    return float((w.detach() ** 2 * torch.cos(pe)).sum() /
                 (w.detach() ** 2).sum())


torch.manual_seed(0)
with torch.no_grad():
    h = m.cond_init.proj(cv)                       # [B,hidden]
    # 取出低秩头原始输出 _pre（θ̂ 通路）
    tp = m.cond_init.to_phase
    _pre = tp(h)                                   # [B, n_osc*n_groups]
    B = _pre.shape[0]
    G = m.cond_init.n_groups
    N = len(t)
    print("=" * 78)
    print("  arctan θ̂ 头值域折叠探针（fix_r1 ckpt，三路全给 src0）")
    print("=" * 78)
    print(f"  to_phase 结构: {tp}")
    print(f"  _pre 形状[B,n_osc*G] = [{B},{_pre.shape[1]}], G={G}, N={N}")
    # 统计 _pre 分布
    flat = _pre.reshape(-1)
    print(f"  _pre 幅值: min={flat.min():+.3f} max={flat.max():+.3f} "
          f"mean_abs={flat.abs().mean():.3f} std={flat.std():.3f}")
    # 折叠率：arctan(flat)*π 中 |值|>π 的比例
    thA_raw = torch.arctan(flat) * math.pi          # 现状
    foldA = (thA_raw.abs() > math.pi).float().mean().item()
    # arctan(flat)*2
    thB_raw = torch.arctan(flat) * 2.0
    foldB = (thB_raw.abs() > math.pi).float().mean().item()
    print(f"  A) arctan*π : |θ̂|>π 折叠率 = {foldA*100:.1f}%  "
          f"(值域 [{-math.pi**2/2:+.2f},{math.pi**2/2:+.2f}])")
    print(f"  B) arctan*2 : |θ̂|>π 折叠率 = {foldB*100:.1f}%  "
          f"(值域 [{-math.pi:+.2f},{math.pi:+.2f}])")
    print()

    # 三种方案各自 reshape 成 [B,N,G] → expand D → 与 target 比 cos
    def expand(theta_g):
        """[B,N,G] -> [B,N,D]（复用 cond_init.expand_to_d）"""
        return m.cond_init.expand_to_d(theta_g)

    for tag, scaler in [("A arctan*π", math.pi), ("B arctan*2", 2.0)]:
        th_hat = (torch.arctan(_pre) * scaler).reshape(B, N, G)
        th_D = expand(th_hat)
        c = cos_ph(a_tar.unsqueeze(0), th_D, th_tar)
        print(f"  {tag:12s}: cos(θ̂, θ_tar) = {c:+.4f}")
    # C) 双通道 atan2：用 _pre 实数部、_pre*0 虚部 = atan2(0, x) 无意义。
    #    需真正双通道头才能测，故此处仅说明，不展开。
    print()
    print("  ★ 关键：A 折叠率 >0 即证明 arctan*π 把部分相位翻转（正变负）。")
    print("    修复 = 乘 2（值域全圆）且开启 PH_ABS_W（锚定绝对相位）。")
