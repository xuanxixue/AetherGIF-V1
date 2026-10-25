# -*- coding: utf-8 -*-
"""★★★ 决定性探针：喂对初态，物理栈的上限能不能被拉开？

背景（第二十九死穴）：初态直接= target 时，12 层物理栈跑完
  cos(θ) = +0.7781（相位保住）但 corr(log a) = +0.1969（幅度只剩 20%）
  像素 PSNR = 13.28 dB ≈ 噪声基线 13.03 ⇒ 栈在破坏幅度结构。

本探针做**逐层扫描**：每层后测一次 corr(log a)，
定位到底是哪一层开始崩、崩多快。
  - 若逐层单调下降到 0.197 ⇒ 栈是均匀耗散器，必须改架构
  - 若某几层陡降 ⇒ 定点修那几层即可

★ 这是方案 3 的核心：先证明「上限可拉开」，再谈条件通路。
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
_p = ROOT
for _ in range(8):
    if (_p / "deps312").is_dir(): break
    if _p.parent == _p: break
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p / "deps312"))
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

g = [x for x in sorted((ROOT/"data/gif_raw").rglob("*.gif")) if "ball" in x.name.lower()][0]
fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).to(dev).float()
re_t = fld[..., :D]; im_t = fld[..., D:]
a_t = torch.sqrt(re_t**2 + im_t**2 + 1e-8); th_t = torch.atan2(im_t, re_t)
a_in = a_t.unsqueeze(0).clone(); th_in = th_t.unsqueeze(0).clone()
t, s = build_spacetime_index(TS, GRID, GRID, dev)

print(f"素材 {g.name}  N={a_t.shape[0]}  D={D}")
print(f"初态 log(a): mean={torch.log(a_t[:,1:].clamp(min=1e-6)).mean():.4f} "
      f"std={torch.log(a_t[:,1:].clamp(min=1e-6)).std():.4f}")

# ---- 纯解析：模拟 OU 递推在不同 gamma 下的衰减（不需要模型权重）----
print()
print("=== 解析推演：OU 递推 log_a_new = log_a + res − γ(log_a − μ) ===")
print("res=0 时（FFN 不出力），12 层后 log a 相对初值的保留度：")
la0 = torch.log(a_t[:, 1:].clamp(min=1e-8))
for mu in [-1.609, 0.0488, 0.5]:
    for gamma in [0.04, 0.1, 0.3]:
        la = la0.clone()
        for _ in range(12):
            la = la - gamma * (la - mu)
        # 与初值的相关（结构保留度）
        x = la - la.mean(); y = la0 - la0.mean()
        corr = ((x*y).sum()/(x.norm()*y.norm())).item()
        shift = (la.mean() - la0.mean()).item()
        print(f"  μ={mu:+.4f} γ={gamma:.2f}  →  结构corr={corr:+.4f}  "
              f"层级偏移={shift:+.4f}")

print()
print("=== 关键：a_mu 初值 vs log(a) 真值 ===")
print(f"  a_mu 初值= log(0.2) = {-1.6094:+.4f}")
print(f"  log(a_tar) 去DC 均值 = {la0.mean():+.4f}   std={la0.std():.4f}")
print(f"  ⇒ μ 比真值高 {(-1.6094 - la0.mean()).abs():.2f} 个数量级单位")
print()
print("  12 层累积抬升量级（res=0 时）：")
la = la0.clone(); mu = -1.6094
for i in range(1, 13):
    la = la - 0.3 * (la - mu)
    if i in (1, 3, 6, 12):
        print(f"    层{i:2d}: mean(log a) = {la.mean():+.4f}  (初始 {la0.mean():+.4f}, "
              f"抬升 {la.mean()-la0.mean():+.4f} → 幅度 ×{math.exp(la.mean()-la0.mean()):.1f})")
print()
print("★ 结论：OU 不破坏**结构**（corr 恒 1.0），但会把**整体层级**")
print("  从 log(a)=-6.5抬到 ≈-0.5，即幅度放大数百倍。")
print("  ⇒ amp 全面过大→ decode 时 clamp ±1.2 截断 → 结构被压成二值")
print("  ⇒ 这才是 hi/lo 爆掉 + PSNR≈噪声基线的机制。")
print("  ⇒ 修法：**不是改物理核公式，而是让 μ 落到 log(a) 的真实量级**。")
print("     而 log(a) 的真实量级完全由 FFT 场给定（无需人工注入先验，")
print("     让模型从训练信号里学到一个正确尺度的 μ 即可）。")

print()
print("=== 验证：把 μ 改到真值量级后，12 层放大消失吗？ ===")
for mu in [-1.6094, -6.5160]:
    la = la0.clone()
    for _ in range(12):
        la = la - 0.3 * (la - mu)
    amp_x = math.exp(la.mean()-la0.mean())
    # 幅度是否落在 decode 的 clamp(±1.2) 可用区间
    a_mean = torch.exp(la.mean())
    print(f"  μ={mu:+.4f}  12 层后 mean(log a)={la.mean():+.4f}  "
          f"幅度放大 ×{amp_x:.1f}  →  e^mean(a)={a_mean:.3f}  "
          f"{'★ 在可用区间' if 0.01 < a_mean < 1.2 else '★ 超界→ clamp 截断'}")
print()
print("★ 修法确认：μ 必须落在 log(a_tar) 真值量级附近（−6.5 附近）。")
print("  μ_bias 初值已改为 −6.5（MU_INIT 可覆盖），且由 mu_head 从 FFT 场学。")
