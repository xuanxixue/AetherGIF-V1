# -*- coding: utf-8 -*-
"""绝对幅度 + 相位 loss 单元验证（2026-10-18，梯度口径）

★ 口径修正（第一版错在哪）：
  之前用「loss 值的大小」比较灵敏度，是错的。真正决定「学不学得动」
  的是**梯度**，不是 loss 值。loss_ph 在高频相位全错时值很大（2.75），
  但那是 DC 的误差贡献；它对**高频相位参数**的梯度才是关键。
  本版直接对模型输出求梯度，测每个频段实际拿到多少梯度。

要验的：
 1. 梯度口径下，旧 loss_ph 对高频相位的梯度 vs 新 loss_phase_band
 2. per-band 绝对幅度项能抓住「形状对但绝对幅度爆」的病灶
 3. loss_phase_band_floor 单侧
 4. 全部 loss 在完美重建时 = 0
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from train.spectral_loss import (BAND_EDGES, loss_band_abs, loss_band_abs_cap,
                                 loss_phase_band, loss_phase_band_floor,
                                 loss_spec_shape, hf_ratio)

torch.manual_seed(0)
WIN, W_FFT = 5, 25
B, N, D = 1, 6144, WIN * W_FFT
NB = len(BAND_EDGES) - 1
DC_COLS = BAND_EDGES[1]
print(f"bands={NB} edges={BAND_EDGES}低频列数={DC_COLS}")

# ---------- 真实分布的 target：DC 占绝对主导（实测97.2%） ----------
def make_tar(dc_frac=0.97):
    a = torch.zeros(B, N, D)
    # 高频 part：重尾，总能量 = (1-dc_frac)
    hi_total = 1.0 - dc_frac
    scale = [f * hi_total for f in (0.60, 0.25, 0.10, 0.03, 0.02)]
    for i in range(1, NB):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        # 幅度 ∝ sqrt(能量份额)
        a[:, :, lo:hi] = math.sqrt(scale[i] / (hi - lo) * N) * 0.02
    # DC part
    lo, hi = BAND_EDGES[0], BAND_EDGES[1]
    a[:, :, lo:hi] = math.sqrt(dc_frac / (hi - lo) * N) * 0.02
    return a


tar = make_tar()
e = (tar.reshape(B, N, WIN, W_FFT) ** 2)
lo_e = e[:, :, :DC_COLS].sum().item()
hi_e = e[:, :, DC_COLS:].sum().item()
print("=" * 92)
print(f"[1] target 自检：低频能量占比 = {lo_e/(lo_e+hi_e)*100:.2f}%（实测应≈97%）")
print(f"    hi/lo = {hf_ratio(tar).item():.5f}")
print("=" * 92)

# ================= 1. 梯度口径：DC 淹没 =================
print("[2] ★★ 梯度口径：旧 loss_ph vs 新 loss_phase_band")
th_tar = torch.rand(B, N, D) * 2 * math.pi - math.pi
g = torch.Generator().manual_seed(7)
# 病灶：**只有高频相位错**，DC 相位完全正确
noise = (torch.rand(B, N, D, generator=g) * 2 - 1) * math.pi
th_bad = th_tar.clone()
th_bad[:, :, DC_COLS:] = torch.remainder(
    th_tar[:, :, DC_COLS:] + noise[:, :, DC_COLS:] + math.pi, 2 * math.pi) - math.pi

w_ph = (tar ** 2).clamp(min=1e-8)


def grad_by_band(loss_fn, th_var):
    """按频段统计 dL/d(th) 的 RMS —— 这就是每个频段实际拿到的梯度。"""
    loss = loss_fn(th_var)
    gr, = torch.autograd.grad(loss, th_var, retain_graph=False)
    gr = gr.reshape(B, N, WIN, W_FFT)
    return [gr[:, :, :, BAND_EDGES[i]:BAND_EDGES[i + 1]].pow(2).mean().sqrt().item()
            for i in range(NB)]


th_v = th_bad.clone().requires_grad_(True)
old_band = grad_by_band(lambda x: (w_ph * torch.remainder(
    x - th_tar + math.pi, 2 * math.pi) - math.pi).pow(2).sum() / w_ph.sum(), th_v)
th_v2 = th_bad.clone().requires_grad_(True)
new_band = grad_by_band(lambda x: loss_phase_band(x, th_tar), th_v2)

print(f"{'band':<8}{'w范围':<12}{'旧loss_ph梯度':>16}{'新phase_band梯度':>18}{'倍数':>10}")
for i in range(NB):
    lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
    ratio = new_band[i] / max(old_band[i], 1e-30)
    print(f"  band{i:<3}{f'w[{lo},{hi})':<12}{old_band[i]:>16.3e}"
          f"{new_band[i]:>18.3e}{ratio:>9.0f}x")
hi_old = max(old_band[1:])
hi_new = max(new_band[1:])
print(f"  → 高频段梯度最大者：旧={hi_old:.3e}  新={hi_new:.3e}"
      f"  **新 loss 给高频的梯度是旧的 {hi_new/max(hi_old,1e-30):.0f} 倍**")
print(f"  → DC 梯度占比：旧 loss中 band0 占 "
      f"{old_band[0]/sum(old_band)*100:.2f}%；新 loss 中band0 = 0（skip_dc）")
print("-" * 92)

# ================= 2. 绝对幅度项 =================
print("[3] 形状对但绝对幅度爆 —— 两个 loss 的分工")
# 保持 hi/lo 不变，只放大高频绝对量（真实病灶：w1-5 217889×、w6+ 98587×）
r = hi_e / lo_e
# 要保持 hi/lo 不变：放大后 hi=k·hi, lo 不变 → k·hi/(lo+k·hi)=hi/lo=r
# 解得 k = r·lo / (hi·(1−r))；r→0 时 k→0，改用直接给一个固定放大倍数
K = 1e5
boom = tar.clone()
boom[:, :, DC_COLS:] = tar[:, :, DC_COLS:] * math.sqrt(K)
print(f"    病灶：高频能量放大 {K:.1e} 倍（幅度×{math.sqrt(K):.0f}）")
print(f"    hi/lo {hf_ratio(tar).item():.6f} -> {hf_ratio(boom).item():.6f}")
print(f"    loss_spec_shape  = {loss_spec_shape(boom, tar).item():.6f}"
      f"  ← 能报警，但已饱和到 max 量级（log 比值 ~ln(1e5)，无法区分 1e5 与 1e8）")
print(f"    loss_band_abs    = {loss_band_abs(boom, tar).item():.6f}"
      f"  ← 线性量级，随放大倍数单调递增，可当**回归目标**用")
print(f"    loss_band_abs_cap= {loss_band_abs_cap(boom, tar, cap_mult=3.0).item():.6f}")
print(f"    完美重建：spec={loss_spec_shape(tar,tar).item():.1e}"
      f"  band_abs={loss_band_abs(tar,tar).item():.1e}"
      f"  cap={loss_band_abs_cap(tar,tar).item():.1e}")
print("-" * 92)

# ================= 3. floor 单侧 =================
print("[4] loss_phase_band_floor 单侧性")
for nm, thx in [("完美", th_tar),
                ("半随机", torch.remainder(th_tar + 0.5 * noise + math.pi, 2*math.pi) - math.pi),
                ("全随机", torch.remainder(noise + math.pi, 2*math.pi) - math.pi)]:
    cb = torch.cos(torch.remainder(thx - th_tar + math.pi, 2*math.pi) - math.pi)
    cbm = torch.stack([cb.reshape(B, N, WIN, W_FFT)[:, :, :,
                       BAND_EDGES[i]:BAND_EDGES[i+1]].mean()
                       for i in range(1, NB)]).mean()
    print(f"    {nm:6s} bandcos={cbm.item():+.4f}  floor={loss_phase_band_floor(thx, th_tar).item():.4f}")
print("=" * 92)
print("[5] 全部 loss 在完美重建时归零")
for nm, fn in [("spec_shape", lambda: loss_spec_shape(tar, tar)),
               ("band_abs", lambda: loss_band_abs(tar, tar)),
               ("band_cap", lambda: loss_band_abs_cap(tar, tar)),
               ("phase_band", lambda: loss_phase_band(th_tar, th_tar)),
               ("phase_floor", lambda: loss_phase_band_floor(th_tar, th_tar))]:
    print(f"    {nm:12s} = {fn().item():.3e}")
print("=" * 92)