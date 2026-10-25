# -*- coding: utf-8 -*-
"""★★ per-band 修复的**可学习性**验证（秒级，不跑完整训练）

【要验证什么】
  修复让条件能表达频段结构了。但这只是「能表达」，还要证明
  **梯度真的能把它学出来**——这是两件事。
  之前梯度排查的教训：block0-7 的梯度余弦 ≈ 0（纯噪声）。

【设计】
  只训 cond_init（物理核冻结），单素材、无交叉训练，
  纯 overfit 设定。这样步数可以很少（每步只过 cond_init + 前向）。

  判据（全部是**空间结构**层面，不是 cos）：
  [1] |dtheta| 从 0 长起来（零初始化 → 有值 = 通路有梯度）
  [2] ★ 组间差异 / 组内差异的比值：模型是否在学「频段各有不同结构」
      比值 > 1 说明组间变化比组内快= 真的在用per-band 自由度
  [3] dtheta 的空间形状与 target 相位形状的相关（supervise 是否起作用）
  [4] 与 target 的 PSNR（绝对判据，非相对指标）
  [5] 每 N 步打印，防止盲等

【对照】
  同时跑G=1（旧版，全通道共享一个形状）作对照。
  若 G=5 明显优于 G=1，修复成立。
"""
import gc
import os
import random
import sys
import time
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
from physics.cond_init import CondInitModulator

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = int(os.environ.get("PROBE_STEPS", "60"))
LR = 3e-3

gif_dir = ROOT / "data" / "gif_raw" / "noto"
f = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"),
                        TS, GRID, PATCH, WIN, IMG, D).to(device)
a_tar = f[..., :D].unsqueeze(0)
th_tar = torch.atan2(f[..., D:], f[..., :D]).unsqueeze(0)
t, s = build_spacetime_index(TS, GRID, GRID, device)

cc = ROOT / "data" / "clip_cache"
cv = torch.from_numpy(build_cond_vector3(
    np.load(cc / "text3.npy")[0:1], np.load(cc / "first3.npy")[0:1],
    np.load(cc / "gif3_video.npy")[0:1], (1, 1, 1))).to(device)


def psnr_t(mse):
    return float(-10.0 * torch.log10(mse.clamp(min=1e-20)))


def shape_metrics(dtD, ci):
    """dtD: [B,N,D] → (组间/组内 比值)"""
    x = dtD[0]
    gid = ci._group_index(device)
    inter, intra = [], []
    for g in range(ci.n_groups):
        m = (gid == g)
        if m.any():
            intra.append(float(x[:, m].mean(dim=1).std(dim=0).mean()))
            for g2 in range(g + 1, ci.n_groups):
                m2 = (gid == g2)
                if m2.any():
                    inter.append(float(
                        (x[:, m].mean(dim=1) - x[:, m2].mean(dim=1)).abs().mean()))
    im = float(np.mean(inter)) if inter else 0.0
    ia = float(np.mean(intra)) if intra else 0.0
    return im, ia, (im / ia if ia > 1e-12 else float("nan"))


print("=" * 96)
print(f"★★ per-band 可学习性验证（只训 cond_init，STEPS={STEPS}）")
print("=" * 96)

results = {}
for G in (1, 5):
    os.environ["COND_GROUPS"] = str(G)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=256, init_mode="init", n_modules=8,
                       mod_gamma=0.15).to(device)
    m.train()
    ci = m.cond_init
    # 只训 cond_init，物理核全冻
    for p in m.parameters():
        p.requires_grad_(False)
    for p in ci.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(ci.parameters(), lr=LR, weight_decay=0.0)
    tag = "旧版 G=1" if G == 1 else "新版 G=5"
    ntr = sum(p.numel() for p in ci.parameters() if p.requires_grad)
    print(f"\n" + "-" * 96)
    print(f"  {tag}   可训参数={ntr/1e6:.2f}M")
    print("-" * 96)
    print(f"  {'step':>5} {'|dθ|':>9} {'组间':>10} {'组内':>10} "
          f"{'比值':>7} {'cos对齐':>8}")

    t0 = time.time()
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        dt, dl, _ = ci(cv, return_parts=True, n_pos=len(t))
        dtD = ci.expand_to_d(dt)
        dlD = ci.expand_to_d(dl)
        # 监督：相位方向对齐 + 幅度形状对齐（沿用官方 supervise 口径）
        loss = CondInitModulator.supervise(dt, dl, f.unsqueeze(0),
                                           d_model=D, guard_w=1.0)
        loss.backward()
        # 梯度健康度（第二十一死穴铁律：先测梯度再谈权重）
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in ci.parameters()
                            if p.grad is not None))
        opt.step()
        if step % 10 == 0 or step == 1:
            with torch.no_grad():
                im, ia, ratio = shape_metrics(dtD, ci)
                # ★ 与 target 的 per-band 形状对齐（与 supervise 新口径一致：
                #   每组用自己的通道算目标方向，与该组 dtheta 对齐）
                w = (a_tar[0, :, 1:] ** 2).clamp(min=1e-8)
                wn = w / w.mean(dim=0, keepdim=True)
                sinc = wn * torch.sin(th_tar[0, :, 1:])
                cosc = wn * torch.cos(th_tar[0, :, 1:])
                G = dt.shape[-1]   # ★ 用展开前的原生组数，不是 D
                if G > 1:
                    c_idx = torch.arange(1, D, device=device)
                    w_idx = c_idx % 25
                    edges = torch.linspace(0, 25, G + 1, device=device)
                    gid = torch.bucketize(
                        w_idx.float(), edges[1:-1].float()).clamp(0, G - 1)
                    als = []
                    for g in range(G):
                        m = gid == g
                        if not bool(m.any()):
                            continue
                        st = sinc[:, m].sum(dim=1)
                        ct = cosc[:, m].sum(dim=1)
                        nt = torch.sqrt(st ** 2 + ct ** 2).clamp(min=1e-6)
                        st, ct = st / nt, ct / nt
                        pg = dt[0][:, g]  # 该组原生 dtheta [N]
                        als.append(float((torch.cos(pg) * ct
                                          + torch.sin(pg) * st).mean()))
                    cos_align = float(np.mean(als))
                else:
                    st = sinc.sum(dim=1)
                    ct = cosc.sum(dim=1)
                    nt = torch.sqrt(st ** 2 + ct ** 2).clamp(min=1e-6)
                    st, ct = st / nt, ct / nt
                    pd1 = dt[0][:, 0]
                    cos_align = float((torch.cos(pd1) * ct
                                       + torch.sin(pd1) * st).mean())
            print(f"  {step:5d} {float(dt.abs().mean()):9.5f} "
                  f"{im:10.2e} {ia:10.2e} {ratio:7.2f} "
                  f"{cos_align:+8.4f}  |g|={float(gn):.2e} "
                  f"{time.time()-t0:.0f}s")
    results[G] = dict(abs=float(dt.abs().mean()), ratio=ratio,
                      cos=cos_align, params=ntr)
    del m, ci, opt
    gc.collect(); torch.cuda.empty_cache()

print("\n" + "=" * 96)
print("  判决")
print("=" * 96)
g1, g5 = results[1], results[5]
print(f"  |dθ|从0 长起来：  G=1 {g1['abs']:.5f}   G=5 {g5['abs']:.5f}"
      f"   → {'通路有梯度' if g5['abs']>1e-4 else '★ 无梯度'}")
print(f"  组间/组内比值：   G=1 {g1['ratio']:.2f}   G=5 {g5['ratio']:.2f}"
      f"   → {'per-band 自由度在被使用' if g5['ratio']>1.2 else '组间未分化'}")
print(f"  与 target 形状相关：G=1 {g1['cos']:+.4f}   G=5 {g5['cos']:+.4f}")
print(f"  可训参数：        G=1 {g1['params']/1e6:.2f}M  G=5 {g5['params']/1e6:.2f}M")
print("=" * 96)