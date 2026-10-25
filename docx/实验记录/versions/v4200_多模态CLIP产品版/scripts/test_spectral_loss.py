# -*- coding: utf-8 -*-
"""A/B 频谱 loss 单元验证（2026-10-18）

必须验的四件事（不然加进训练只是猜）：
 1. loss_spec_shape 对「高频能量超标」敏感（会给出显著 loss）
 2. loss_spec_shape 对「高频能量不足」也给 loss（但 guard 单侧不罚）
 3. ★ 抗整体缩放作弊：a_pred *= 0.5 时 loss_spec_shape 几乎不变
    （占比是形状量）→ 这是 B 项相对绝对 MSE 的核心优势
 4. loss_hf_guard 单侧：ratio 低于 cap 时严格为 0
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from train.spectral_loss import (band_profile, hf_ratio, loss_hf_guard,
                                 loss_spec_shape, BAND_EDGES)

torch.manual_seed(0)
WIN, W_FFT = 5, 25
B, N = 1, 6144
D = WIN * W_FFT


def make_tar():
    """构造重尾 target：DC 主导 + 高频极小（复现真实 FFT 系数分布）。"""
    a = torch.rand(B, N, D) * 0.002 + 1e-4          # 高频基底噪声
    a[:, :, 0] = 1.0 + torch.rand(B, N)             # w=0 直流巨大
    a[:, :, 1] = 0.5 + torch.rand(B, N) * 0.1
    return a


def noisy(tar, hi_amp=1.0, seed=1):
    """在target 上叠加高频随机噪声（模拟当前 23 倍 hi/lo 的病灶）。"""
    g = torch.Generator().manual_seed(seed)
    hi = torch.rand(B, N, WIN, W_FFT, generator=g) * hi_amp
    hi[:, :, :, 0] = 0.0                            # 不动 DC
    return (tar.reshape(B, N, WIN, W_FFT) + hi).reshape(B, N, D)


tar = make_tar()
perfect = tar.clone()

print("=" * 84)
print("[1] 真实 target 自身的谱形状")
p = band_profile(tar)[0]
for i in range(len(BAND_EDGES) - 1):
    print(f"    band{i} w∈[{BAND_EDGES[i]:2d},{BAND_EDGES[i+1]:2d}): {p[i]:.6e}")
print(f"    hi/lo ratio = {hf_ratio(tar).item():.6e}")
print(f"    loss_spec_shape(完美重建) = {loss_spec_shape(perfect, tar).item():.3e}  <-- 应≈0")
print("-" * 84)
print("[2] 叠加高频噪声后，两项 loss 是否报警")
for hi_amp in (0.0, 0.01, 0.05, 0.2):
    a = noisy(tar, hi_amp)
    ls = loss_spec_shape(a, tar).item()
    lg = loss_hf_guard(a, tar).item()
    print(f"    hi_amp={hi_amp:5.3f}  hi/lo={hf_ratio(a).item():.4e}  "
          f"loss_spec={ls:.4e}  loss_guard={lg:.4e}")
print("-" * 84)
print("[3] ★ 抗整体缩放作弊（B 项核心优势）")
for k in (0.25, 0.5, 1.0, 2.0):
    a = perfect * k
    print(f"    a*={k:4.2f}  loss_spec_shape={loss_spec_shape(a, tar).item():.4e}  "
          f"（恒为 0 = 对总量完全不敏感）")
print("    对照：普通 log1p MSE 在同场景下 →")
for k in (0.25, 0.5, 1.0, 2.0):
    a = perfect * k
    mse = (torch.log1p(a.clamp(min=1e-8)) - torch.log1p(tar.clamp(min=1e-8))).pow(2).mean()
    print(f"      a*={k:4.2f}  MSE={mse.item():.4e}  ← 会被整体缩放骗")
print("-" * 84)
print("[4] loss_hf_guard 单侧性（slack=0.5 → cap = r_tar*1.5）")
cap = hf_ratio(tar).item() * 1.5
print(f"    cap = {cap:.4f}")


def make_with_ratio(r):
    """构造 hi/lo 实测恰为 r 的 a：DC 幅度 1，高频每格幅度 sqrt(r/格数)。"""
    n_hi = WIN * (W_FFT - 1)
    amp = math.sqrt(r / n_hi)
    hi = torch.full((B, N, WIN, W_FFT), amp)
    hi[:, :, :, 0] = 0.0
    a = hi.reshape(B, N, D)
    a[:, :, 0] = 1.0
    return a


for r in (cap * 0.3, cap * 0.99, cap * 1.01, cap * 3.0):
    a = make_with_ratio(r)
    meas = hf_ratio(a).item()
    g = loss_hf_guard(a, tar).item()
    flag = "OK" if ((meas <= cap and g == 0.0) or (meas > cap and g > 0.0)) else "!!"
    print(f"    目标ratio={r:7.4f}  实测={meas:7.4f}  guard={g:.4e}  {flag}")
print("=" * 84)