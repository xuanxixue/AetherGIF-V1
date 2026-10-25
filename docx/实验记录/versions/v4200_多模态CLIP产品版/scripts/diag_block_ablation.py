# -*- coding: utf-8 -*-
"""★★ block 内三算子消融：结构在layer 1 的哪一步被摧毁（2026-10-18）

【前一轮的决定性数据】
  喂入 target 的完美振幅+相位，跑物理核：
      layer 1  2.33 dB   ← 一步就崩
      layer12  1.60 dB
  比噪声基线（13 dB）还差 11 dB。**结构在第一个 block 就被摧毁。**

【本脚本】逐个关掉 block 内的算子，看哪个是凶手
  PhysicsBlock 内部顺序（src/physics/block.py）：
      g       = self.drift(...)        # 漂移场
      θ_new   = self.kuramoto(...)     # Kuramoto 同步
      ffn_out = self.ffn(...)          # 共振驻波 FFN
      a_new   = a * exp(log_res)       # 振幅乘性更新
      θ_new  += ph_res                 # 相位残差

  消融配置：
      full        全部开（复现 2.33 dB）
      no_drift    drift 置零
      no_kuramoto θ 直接跳过Kuramoto（只留 FFN 相位残差）
      no_ffn      FFN 输出置零（只留 Kuramoto）
      only_ffn    只有 FFN（drift=0, θ0 直接进 FFN）
"""
import gc
import math
import os
import random
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
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

gif_dir = ROOT / "data" / "gif_raw" / "noto"
f = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"),
                        TS, GRID, PATCH, WIN, IMG, D).to(device)
target = f.unsqueeze(0)                    # [1, N, 2D]
a_tar = target[..., :D]
th_tar = torch.atan2(target[..., D:], target[..., :D])
t, s = build_spacetime_index(TS, GRID, GRID, device)

cc = ROOT / "data" / "clip_cache"
cv = torch.from_numpy(build_cond_vector3(
    np.load(cc / "text3.npy")[0:1], np.load(cc / "first3.npy")[0:1],
    np.load(cc / "gif3_video.npy")[0:1], (1, 1, 1))).to(device)

torch.manual_seed(0); random.seed(0); np.random.seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                   init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
m.eval()


def psnr_t(mse):
    return float(-10.0 * torch.log10(mse.clamp(min=1e-20)))


# 公共参数
with torch.no_grad():
    emb = m._fuse_three_way(cv)
    omega = m._init_omega(t, s) + m.omega_bias_head(emb).unsqueeze(1)
    gi = torch.arange(len(t), device=device) // m.n_modules
    mod_id = gi % m.n_modules
    kappa = (2.0 * torch.sigmoid(m.kappa_scale_head(emb)).squeeze(-1)).view(-1, 1, 1)
    T = (0.5 * torch.sigmoid(m.T_scale_head(emb)).squeeze(-1)).view(-1, 1, 1)
blk = m.pmnncore.blocks[0]
tf = m.pmnncore.time_field.forward(0, device)

print("=" * 96)
print("★★ block 0 内三算子消融（输入 = target 完美振幅+相位）")
print("=" * 96)
print(f"  输入 PSNR = {psnr_t((a_tar[0] - a_tar[0]).pow(2).mean()):.2f} dB（自比，恒等）")


@torch.no_grad()
def run(mode):
    a = a_tar.clone()
    th = th_tar.clone()
    # drift
    if mode in ("no_drift", "only_ffn"):
        g = torch.zeros_like(th)
    else:
        g = blk.drift(a, th, None, None, mod_id=mod_id,
                      cond_anchor=th[..., :1], anchor_weight=1.0)
    # kuramoto
    if mode in ("no_kuramoto", "only_ffn"):
        th_new = th
    else:
        th_new, _ = blk.kuramoto(th, a, omega, g, tf["kappa"] * kappa,
                                 tf["T"] * T, t, s)
    # ffn
    if mode in ("no_ffn", "no_kuramoto"):
        return a, th_new
    omega_d = tf["omega_d"]
    ffn_out = blk.ffn(a, th_new, tf["omega_d"], t, s)
    log_res, ph_res = ffn_out.chunk(2, dim=-1)
    a_new = a * torch.exp(log_res.clamp(-6, 3))
    th_new = th_new + ph_res.clamp(-1.5, 1.5)
    return a_new, th_new


print("\n" + "-" * 96)
print("[1] 各配置下的输出 PSNR（target 自比为 ∞，噪声基线 ≈13 dB）")
print("-" * 96)
res = {}
for mode, desc in [("full", "全部开（复现 baseline）"),
                   ("no_drift", "关漂移场"),
                   ("no_kuramoto", "关 Kuramoto"),
                   ("no_ffn", "关 FFN"),
                   ("only_ffn", "只留 FFN")]:
    a_out, th_out = run(mode)
    p = psnr_t((a_out[0] - a_tar[0]).pow(2).mean())
    # 相位质量
    c = float(torch.cos(th_out - th_tar).mean())
    res[mode] = p
    print(f"  {desc:<22} 振幅 PSNR = {p:>7.2f} dB   θ cos = {c:+.4f}")

print("\n" + "-" * 96)
print("[2] 判决：谁是凶手")
print("-" * 96)
base = res["full"]
for mode, desc in [("no_drift", "关漂移场"), ("no_kuramoto", "关 Kuramoto"),
                   ("no_ffn", "关 FFN"), ("only_ffn", "只留 FFN")]:
    d = res[mode] - base
    tag = "★★ 显著改善" if d > 3.0 else("改善" if d > 0.5 else "无明显影响")
    print(f"  {desc:<14} {res[mode]:>7.2f} dB   Δ={d:+.2f} dB   {tag}")

print("\n" + "-" * 96)
print("[3] ★ 决定性追问：单算子能否「保持」target（identity 能力）")
print("   若 Kuramoto 拿完美 θ 输入却输出乱θ → Kuramoto 在洗掉相位结构")
print("-" * 96)
a = a_tar.clone()
th = th_tar.clone()
g = blk.drift(a, th, None, None, mod_id=mod_id,
              cond_anchor=th[..., :1], anchor_weight=1.0)
print(f"  drift 输出 g 的量级：mean|g| = {float(g.abs().mean()):.4f}")
th_k, _ = blk.kuramoto(th, a, omega, g, tf["kappa"] * kappa,
                        tf["T"] * T, t, s)
print(f"  Kuramoto(完美θ) → cos(θ_out, θ_tar) = "
      f"{float(torch.cos(th_k - th_tar).mean()):+.4f}")
print(f"    θ偏移 |Δθ| 均值 = {float((th_k - th_tar).abs().mean()):.4f} rad")
ffn_out = blk.ffn(a, th_k, tf["omega_d"], t, s)
log_res, ph_res = ffn_out.chunk(2, dim=-1)
print(f"  FFN log_res: mean={float(log_res.mean()):+.4f} "
      f"std={float(log_res.std()):.4f} min={float(log_res.min()):+.2f} "
      f"max={float(log_res.max()):+.2f}")
print(f"    → exp 后振幅乘子 mean={float(log_res.clamp(-6,3).exp().mean()):.4f}"
      f"（=1 表示振幅被原样保留）")
a_ffn = a * torch.exp(log_res.clamp(-6, 3))
print(f"  FFN 后振幅 PSNR = {psnr_t((a_ffn[0] - a_tar[0]).pow(2).mean()):.2f} dB")
print(f"  FFN ph_res: mean={float(ph_res.mean()):+.4f} "
      f"std={float(ph_res.std()):.4f}")
print("=" * 96)