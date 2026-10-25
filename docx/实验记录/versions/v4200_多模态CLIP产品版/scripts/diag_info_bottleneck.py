# -*- coding: utf-8 -*-
"""★★ 决定性诊断：信息瓶颈到底在哪（2026-10-18）

【前面所有排查的盲点】
  我们一直在查「梯度方向是否一致」「梯度噪声从哪来」，但梯度是
  **有信息的**（block9-11 余弦 +0.65）。真正的问题可能不是优化器
  走不动，而是**目标本身不可达**：模型的前向里根本没有足够信息
  去重建这张图。

【新发现的硬瓶颈（代码级）】
  forward 里 `drift_target = None`（2026-10-16 的修正，为了不抄答案），
  所以 target **完全不进前向**。模型唯一的信息来源是 cond_init：
      a0= a0 * exp(dlna)      dlna : [B, N, 1]  ← 只 1 个标量/位置
      theta0 = theta0 + dtheta dtheta: [B, N, 1]  ← 只 1 个标量/位置
  两者都通过 unsqueeze 广播到 [B, N, D]，**D=125 个频率通道共享
  完全相同的空间形状**。

  → 后果（致命）：
    1. cond_init 只能给出「整张图统一的亮度/相位偏移」，无法表达
       任何**空间结构**（轮廓在哪、纹理在哪）。
    2. 目标场的空间结构必须由 FFN+Kuramoto 从 a0/θ0 的随机噪声里
       长出来 —— 但 a0/θ0 的随机种子与 target 无关，而 ω 又是
       确定性的，于是同一个条件下模型永远给不出该条件的正确空间结构。
    3. 交叉训练下更糟：同一张图有 7 种条件组合，模型学到的是
       「条件无关的通用噪声纹理」→ **这正是「全是噪声」的字面成因**。

【本脚本测三件事，全部秒级】
  [1] 信息瓶颈量化：cond_init 输出的 dlna/dtheta 在D 维上是否真的
      完全相同（若是，说明空间结构通路不存在）
  [2] 同条件两次前向是否给出相同结果（决定性：若相同但≠target，
      则模型是「确定性错答案」；若不同，则连确定性都没有）
  [3] ★ 空间结构可表达性上界：把 target 的 log幅度/相位直接当作
      cond_init 的输出（即人工喂最优初值），看前向能否重建。
      这一条直接测出「物理核+FFN 有没有能力重建」，把
      「信息瓶颈」和「架构能力不足」彻底分开。
"""
import gc
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

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
device = "cuda" if torch.cuda.is_available() else "cpu"

gif_dir = ROOT / "data" / "gif_raw" / "noto"
NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]
fields, th_tars = [], []
for nm in NAMES:
    f = encode_gif_to_field(str(gif_dir / nm), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]).unsqueeze(0))
target = torch.stack(fields, 0)
th_tar = torch.stack(th_tars, 0)
a_tar = target[..., :D]
t, s = build_spacetime_index(TS, GRID, GRID, device)

cc = ROOT / "data" / "clip_cache"
text3 = np.load(cc / "text3.npy")
first3 = np.load(cc / "first3.npy")
gif3v = np.load(cc / "gif3_video.npy")


def build_cv(src):
    return torch.from_numpy(build_cond_vector3(
        text3[src:src + 1], first3[src:src + 1], gif3v[src:src + 1],
        (1, 1, 1))).to(device)


torch.manual_seed(0); random.seed(0); np.random.seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                   init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
m.eval()
cv = build_cv(0)

print("=" * 96)
print("★★ 信息瓶颈诊断 device={}".format(device))
print("=" * 96)

# ================================================================
print("\n" + "-" * 96)
print("[1] cond_init 输出的形状与D 维一致性")
print("-" * 96)
with torch.no_grad():
    dtheta, dlna, _ = m._cond_init(cv, len(t))
print(f"  dtheta.shape = {tuple(dtheta.shape)}   dlna.shape = {tuple(dlna.shape)}")
print(f"  N={len(t)}  D={D}")
if dlna.shape[-1] == 1:
    print("  ★ dlna 最后一维=1 → 广播到 D 维时**125 个通道完全相同**")
    print("    ⇒ cond_init 只能表达「整图统一偏移」，无法表达任何空间结构")
    # 直接量化：把 dlna 广播后看通道间方差
    br = dlna.expand(-1, -1, D)
    print(f"    广播后跨通道 std = {float(br.std(dim=-1).mean()):.3e}"
          f"（=0 即完全相同，实测确认）")

# ================================================================
print("\n" + "-" * 96)
print("[2] 同条件两次前向的确定性 + 与 target 的差距")
print("-" * 96)
outs = []
with torch.no_grad():
    for i in range(2):
        torch.manual_seed(1234)                 # 固定 a0/θ0 的随机种子
        a, th, _ = m(texts=None, target_field=None, t_coord=t, s_coord=s,
                     batch_size=1, cond_vec=cv, use_cond_field=True)
        outs.append((a.clone(), th.clone()))
a0_, t0_ = outs[0]
a1_, t1_ = outs[1]
print(f"  两次前向 a逐位相同 = {torch.equal(a0_, a1_)}")
print(f"  两次前向 θ 逐位相同 = {torch.equal(t0_, t1_)}")
# 与 target 比
la_p = torch.log1p(a0_.clamp(min=1e-8))
la_t = torch.log1p(a_tar[0:1].clamp(min=1e-8))
rel_amp = float((la_p - la_t).abs().mean() / la_t.abs().mean())
print(f"  log1p 振幅相对误差 = {rel_amp:.4f}（0=完美）")


def psnr_t(mse):
    return float(-10.0 * torch.log10(mse.clamp(min=1e-20)))


mse = (a0_[0] - a_tar[0]).pow(2).mean()
print(f"  振幅 PSNR vs target = {psnr_t(mse):.2f} dB")

# ================================================================
print("\n" + "-" * 96)
print("[3] ★ 空间结构可表达性上界：人工喂最优初值能否重建")
print("-" * 96)
print("  3a. target场直接当 a0/θ0（绕过一切物理），应≈无损")
with torch.no_grad():
    print(f"      直接取 target: PSNR = "
          f"{psnr_t((a_tar[0] - a_tar[0]).pow(2).mean()):.2f} dB（自比，恒等）")

print("  3b. ★ 关键实验：a0=target 振幅、θ0=target 相位，跑完 12 层物理核")
print("      （ω 仍确定性、drift 仍为 None，只让物理核+FFN 演化）")
print("      若结果≈无损 → 架构有能力重建，瓶颈纯在初始化的信息量")
print("      若结果≈噪声 → 物理核本身在摧毁结构，架构能力不足")


def forward_from_init(a_init, th_init):
    """用给定初值跑完整物理演化（复制 forward 的演化段）。"""
    omega_base = m._init_omega(t, s)
    om_b = m.omega_bias_head(m._fuse_three_way(cv))
    om_d = m.drift_target_head(m._fuse_three_way(cv))
    omega = omega_base + om_b.unsqueeze(1)
    mod_id = None
    if m.n_modules > 0:
        gi = torch.arange(len(t), device=device) // m.n_modules
        mod_id = gi % m.n_modules
    ks = torch.sigmoid(m.kappa_scale_head(m._fuse_three_way(cv))).squeeze(-1)
    Ts = torch.sigmoid(m.T_scale_head(m._fuse_three_way(cv))).squeeze(-1)
    kappa = (2.0 * ks + 0.0).view(-1, 1, 1)
    T = (0.5 * Ts + 0.0).view(-1, 1, 1)
    a, th = a_init, th_init
    for i in range(m.n_blocks):
        tf = m.pmnncore.time_field.forward(i, device)
        a, th, _ = m.pmnncore.blocks[i](
            a, th, omega, t, s, kappa=tf["kappa"] * kappa, T=tf["T"] * T,
            omega_d=tf["omega_d"], targets=None, protos=None,
            mod_id=mod_id, cond_anchor=th[..., :1], anchor_weight=1.0)
    return a, th


with torch.no_grad():
    ae, the = forward_from_init(a_tar[0:1].clone(), th_tar[0].clone())
mse_e = (ae[0] - a_tar[0]).pow(2).mean()
print(f"      12 层物理核后 PSNR = {psnr_t(mse_e):.2f} dB"
      f"   （target 自比=∞，噪声基线≈13）")
lp_e = torch.log1p(ae[0].clamp(min=1e-8))
lp_t = torch.log1p(a_tar[0].clamp(min=1e-8))
print(f"      log1p 相对误差 = "
      f"{float((lp_e - lp_t).abs().mean() / lp_t.abs().mean()):.4f}")

print("\n  3c. 分档：不同层数后还剩多少（看结构是在哪一层被摧毁的）")
with torch.no_grad():
    omega_base = m._init_omega(t, s)
    om_b = m.omega_bias_head(m._fuse_three_way(cv))
    omega = omega_base + om_b.unsqueeze(1)
    gi = torch.arange(len(t), device=device) // m.n_modules
    mod_id = gi % m.n_modules
    ks = torch.sigmoid(m.kappa_scale_head(m._fuse_three_way(cv))).squeeze(-1)
    Ts = torch.sigmoid(m.T_scale_head(m._fuse_three_way(cv))).squeeze(-1)
    kappa = (2.0 * ks + 0.0).view(-1, 1, 1)
    T = (0.5 * Ts + 0.0).view(-1, 1, 1)
    a, th = a_tar[0:1].clone(), th_tar[0].clone()
    for i in range(m.n_blocks):
        tf = m.pmnncore.time_field.forward(i, device)
        a, th, _ = m.pmnncore.blocks[i](
            a, th, omega, t, s, kappa=tf["kappa"] * kappa, T=tf["T"] * T,
            omega_d=tf["omega_d"], targets=None, protos=None,
            mod_id=mod_id, cond_anchor=th[..., :1], anchor_weight=1.0)
        if i + 1 in (0, 1, 2, 3, 5, 7, 9, 11):
            p = psnr_t((a[0] - a_tar[0]).pow(2).mean())
            print(f"      layer{i+1:>2d}: PSNR = {p:>7.2f} dB")
print("=" * 96)