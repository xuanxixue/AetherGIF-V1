# -*- coding: utf-8 -*-
"""秒级诊断：target 方向场对两素材有多相似？（第二十七步，根因定位）

【要回答的问题】
  discfix 已让条件通路分开两素材（cos_h −0.055），但生成的是「两个不同的
  通用场」而非「两个素材各自的场」——
  8号球的圆形轮廓完全没出现。

  嫌疑假设：**supervise 的目标方向场对两个素材高度相似**
  （都是「一张图的大致布局」），模型只要输出任意一个「图状场」
  就能降 loss ⇒ 它没有动力去学「这个圆形」vs「那个方形」。

【要测什么】
  [1] 两素材的 target 方向场之间的余弦（逐 per-band 组）
      → 若≈0.99，则假设成立，supervise 无法区分素材
  [2] target 幅度场 a_tar 的跨素材区分度（对照：幅度分得开吗？）
      → 若 a_tar 分得开、相位分不开 ⇒ 修相位监督
  [3] 低频轮廓（DC 与低频段）是否可分
      → 决定「内容级判别」该在哪个频段做
  [4] 修法可行性：用方向场的**低频子空间**做判别，测素材可分性

  ⚠ 必须区分「方向场不可分」与「方向场可分但 supervise 权重让低频主导」。
"""
import os
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

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
G = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
gif_dir = ROOT / "data" / "gif_raw" / "noto"
NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]

fs = [encode_gif_to_field(str(gif_dir / n), TS, GRID, PATCH, WIN, IMG, D).to(device)
      for n in NAMES]
a0, a1 = fs[0][..., :D], fs[1][..., :D]
t0, t1 = fs[0][..., D:], fs[1][..., D:]


def cosf(x, y):
    x, y = x.reshape(-1).float(), y.reshape(-1).float()
    return float(x @ y / (x.norm() * y.norm() + 1e-12))


print("=" * 88)
print(f"  [0] 素材: {NAMES[0]}  vs  {NAMES[1]}")
print("=" * 88)
print(f"  a_tar 跨素材cos = {cosf(a0, a1):+.6f}   ← 幅度场可分性")
print(f"  原始 fft 复数cos = {cosf(torch.complex(fs[0].reshape(-1), torch.zeros(1,device=device)), torch.complex(fs[1].reshape(-1), torch.zeros(1,device=device))):+.6f}")

print("\n" + "=" * 88)
print("  [1] ★ target 方向场的跨素材余弦（per-band，supervise 的口径）")
print("=" * 88)
c_idx = torch.arange(1, D, device=device)
w_idx = c_idx % W_FFT
edges = torch.linspace(0, W_FFT, G + 1, device=device)
gid = torch.bucketize(w_idx.float(), edges[1:-1].float()).clamp(0, G - 1)

# supervise 口径：功率加权后逐组求和归一
wt0 = (a0[..., 1:] ** 2).clamp(min=1e-8)
wt1 = (a1[..., 1:] ** 2).clamp(min=1e-8)
s0 = wt0 * torch.sin(t0[..., 1:])
c0 = wt1 * 0 + wt0 * torch.cos(t0[..., 1:])
s1 = wt1 * torch.sin(t1[..., 1:])
c1 = wt1 * torch.cos(t1[..., 1:])
print(f"  {'组':>3} | {'方向场cos':>12} | {'幅度cos':>12} | 判定")
print("  " + "-" * 52)
dir_coss, amp_coss = [], []
for g in range(G):
    m = gid == g
    if not bool(m.any()):
        continue
    st0, ct0 = s0[:, m].sum(1), c0[:, m].sum(1)
    st1, ct1 = s1[:, m].sum(1), c1[:, m].sum(1)
    n0 = torch.sqrt(st0 ** 2 + ct0 ** 2).clamp(min=1e-6)
    n1 = torch.sqrt(st1 ** 2 + ct1 ** 2).clamp(min=1e-6)
    dc = cosf(st0 / n0, st1 / n1)
    ac = cosf(a0[..., 1:][:, m], a1[..., 1:][:, m])
    dir_coss.append(dc)
    amp_coss.append(ac)
    v = "★不可分" if dc > 0.95 else ("弱" if dc > 0.8 else "可分")
    print(f"  {g:>3} | {dc:>+12.6f} | {ac:>+12.6f} | {v}")
print(f"  平均: 方向场cos={np.mean(dir_coss):+.6f}   幅度cos={np.mean(amp_coss):+.6f}")

print("\n" + "=" * 88)
print("  [2] 低频轮廓可分性（内容级判别该在哪做）")
print("=" * 88)
# 频段掩码：按 w 切
for lbl, sel in (("w=0 (DC)", c_idx % W_FFT == 0),
                 ("w 1-4 (最低频)", (c_idx % W_FFT >= 1) & (c_idx % W_FFT <= 4)),
                 ("w 5-12 (低频)", (c_idx % W_FFT >= 5) & (c_idx % W_FFT <= 12)),
                 ("w 13-24 (高频)", c_idx % W_FFT >= 13)):
    if not bool(sel.any()):
        continue
    ac = cosf(a0[..., 1:][:, sel], a1[..., 1:][:, sel])
    st0, ct0 = s0[:, sel].sum(1), c0[:, sel].sum(1)
    st1, ct1 = s1[:, sel].sum(1), c1[:, sel].sum(1)
    n0 = torch.sqrt(st0 ** 2 + ct0 ** 2).clamp(min=1e-6)
    n1 = torch.sqrt(st1 ** 2 + ct1 ** 2).clamp(min=1e-6)
    dc = cosf(st0 / n0, st1 / n1)
    print(f"  {lbl:>16} 幅度cos={ac:+.4f}   方向场cos={dc:+.4f}")

print("\n" + "=" * 88)
print("  [3] 修法可行性：低频轮廓的**跨素材分类余量**")
print("=" * 88)
# 若只用 w<5 的低频，两素材的轮廓方向差多少？>0.3 就有判别空间
sel = (c_idx % W_FFT <= 4)
st0, ct0 = s0[:, sel].sum(1), c0[:, sel].sum(1)
st1, ct1 = s1[:, sel].sum(1), c1[:, sel].sum(1)
n0 = torch.sqrt(st0 ** 2 + ct0 ** 2).clamp(min=1e-6)
n1 = torch.sqrt(st1 ** 2 + ct1 ** 2).clamp(min=1e-6)
v0, v1 = st0 / n0, ct1 / n1
print(f"  低频方向余弦 = {cosf(v0, v1):+.6f}")
print(f"  低频幅度余弦 = {cosf(a0[..., 1:][:, sel], a1[..., 1:][:, sel]):+.6f}")

print("\n" + "=" * 88)
print("  [4] 判决")
print("=" * 88)
dbar = float(np.mean(dir_coss))
abar = float(np.mean(amp_coss))
print(f"  方向场平均cos = {dbar:+.4f}（supervise 的监督对象）")
print(f"  幅度场平均 cos = {abar:+.4f}（信息量参照）")
if dbar > 0.9 and abar < 0.9:
    print("  → ★★ **相位方向场不可分，但幅度场可分**。")
    print("     supervise 只监督相位方向 ⇒ 素材信息在相位侧被压掉。")
    print("     修法：给 cond_init 的**相位监督**加低频轮廓锚，")
    print("           或直接监督 dlna（幅度侧本来可分）。")
elif dbar > 0.9 and abar > 0.9:
    print("  → ★★ **相位和幅度都不可分**。素材差异在编解码阶段就没了，")
    print("     需回查 encode_gif_to_field / patch 化是否已丢信息。")
else:
    print("  → 方向场可分，问题不在 supervise 的口径，需查权重分布。")