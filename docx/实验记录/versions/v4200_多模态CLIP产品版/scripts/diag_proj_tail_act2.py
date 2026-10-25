# -*- coding: utf-8 -*-
"""秒级对照 v2：用**真实训练权重**测尾部激活（第二十六步 c 修正版）

【为什么 v1 无效】
  v1 用随机权重，五种尾部激活的 cos 全部=0.98582（完全相同）。
  ⇒ cos 对尾部激活不敏感，**对照设计本身无效**：
     随机 Linear 把任何输入映到近同一方向，激活改不改都一样。
  但真实 ckpt 里 proj[3] Linear2 裸输出 cos=0.675（区分度良好），
  加上 proj[4] SiLU 后 0.987 ⇒ 跳跃确实发生在激活处，只在真实权重下成立。

【v2 设计】
  直接取perband6 的真实权重，逐个替换尾部激活，测跨素材 cos。
  每个方案都用**同一份真实 Linear 权重**，只换激活 ⇒ 干净的对照。
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
import torch.nn as nn
import torch.nn.functional as Fn

from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D = 125
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["COND_GROUPS"] = "5"
os.environ.setdefault("COND_LOWRANK", "16")

CKPT = os.environ.get("CKPT", "outputs_real/perband6/cond3_ckpt.pt")
torch.manual_seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
sd = torch.load(ROOT / CKPT, map_location=device)
m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("qwen.")},
                  strict=False)
m.eval()
ci = m.cond_init
print(f"[ckpt] {CKPT}")

cc = ROOT / "data" / "clip_cache"
T3 = np.load(cc / "text3.npy")
F3 = np.load(cc / "first3.npy")
G3 = np.load(cc / "gif3_video.npy")
c0 = torch.from_numpy(build_cond_vector3(T3[0:1], F3[0:1], G3[0:1], (1, 1, 1))).to(device)
c1 = torch.from_numpy(build_cond_vector3(T3[1:2], F3[1:2], G3[1:2], (1, 1, 1))).to(device)


def cosf(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    return float(x @ y / (x.norm() * y.norm() + 1e-12))


with torch.no_grad():
    h0 = ci.proj[3](Fn.silu(ci.proj[0](c0)))   # 到 Linear2 之前
    h1 = ci.proj[3](Fn.silu(ci.proj[0](c1)))
    p0 = ci.proj[3](Fn.silu(ci.proj[0](c0)))
    p1 = ci.proj[3](Fn.silu(ci.proj[0](c1)))
    # ci.proj 结构 = [Linear1, SiLU, Linear2, SiLU]（index 0..3）
    y_raw0, y_raw1 = ci.proj[3](ci.proj[1](ci.proj[0](c0))), \
                     ci.proj[3](ci.proj[1](ci.proj[0](c1)))

print(f"\n  真实权重下 Linear2 裸输出 cos = {cosf(y_raw0, y_raw1):+.6f}")
print(f"  真实权重下末端 SiLU 后  cos = {cosf(Fn.silu(y_raw0), Fn.silu(y_raw1)):+.6f}")

print("\n" + "=" * 84)
print("  真实权重 · 尾部激活对照")
print("=" * 84)
schemes = {
    "无激活（裸Linear2）": lambda x: x,
    "SiLU（现状）": lambda x: Fn.silu(x),
    "Tanh": lambda x: torch.tanh(x),
    "ReLU": lambda x: Fn.relu(x),
    "GELU": lambda x: Fn.gelu(x),
    "零均值+归一": lambda x: ((x - x.mean()) /
                        (x.pow(2).mean().sqrt() + 1e-6)),
}
print(f"  {'尾部激活':>18} | {'跨素材cos':>12} | 判定")
print("  " + "-" * 62)
best = (None, 2.0)
for nm, fn in schemes.items():
    c = cosf(fn(y_raw0), fn(y_raw1))
    v = "★区分度已死" if c > 0.95 else ("区分度弱" if c > 0.8 else "★区分度保留")
    print(f"  {nm:>18} | {c:>+12.6f} | {v}")
    if c < best[1]:
        best = (nm, c)

print("\n" + "=" * 84)
print("  ★ 区分度究竟丢在哪一段？（真实权重，沿 forward逐段测）")
print("=" * 84)
with torch.no_grad():
    segs = {}
    segs["cond_vec"] = (c0, c1)
    z0 = ci.proj[0](c0)
    z1 = ci.proj[0](c1)
    segs["proj[0] Linear1"] = (z0, z1)
    z0 = ci.proj[1](z0)
    z1 = ci.proj[1](z1)
    segs["proj[1] SiLU1"] = (z0, z1)
    z0 = ci.proj[2](z0)
    z1 = ci.proj[2](z1)
    segs["proj[2] Linear2"] = (z0, z1)
    z0 = ci.proj[3](z0)
    z1 = ci.proj[3](z1)
    segs["proj[3] SiLU2  = h"] = (z0, z1)
    # to_phase 的两层（Sequential: [0]=Linear(R), [1]=Linear(out)）
    a0 = ci.to_phase[0](z0)
    a1 = ci.to_phase[0](z1)
    segs["to_phase[0] (低秩R)"] = (a0, a1)
    b0 = ci.to_phase[1](a0)
    b1 = ci.to_phase[1](a1)
    segs["to_phase[1] (升维)"] = (b0, b1)
    d0, _, _ = ci(c0, return_parts=True)
    d1, _, _ = ci(c1, return_parts=True)
    segs["dtheta (最终)"] = (d0, d1)

for nm, (x, y) in segs.items():
    c = cosf(x, y)
    flag = "★" if c > 0.95 else (" " if c > 0.8 else "▼")
    print(f"  {flag} {nm:22s} cos = {c:+.6f}   dim={x.numel()}")

# ★ 修复方案验证：跳跃点在 proj[2] Linear2（输出被搬到正半轴），
#   它的下游 proj[3] 是 SiLU。把 proj[3] 换成 LayerNorm 试能否救回。
print("\n" + "=" * 84)
print("  ★ 修复方案：在跳跃点下游（proj[3]）换激活，看 h 的区分度")
print("=" * 84)
_y0 = ci.proj[2](ci.proj[1](ci.proj[0](c0)))
_y1 = ci.proj[2](ci.proj[1](ci.proj[0](c1)))
print(f"  {'proj[3] 换成':>18} | {'h 跨素材 cos':>14} | 判定")
print("  " + "-" * 60)
_fixes = {
    "SiLU（现状）": lambda x: Fn.silu(x),
    "Tanh": lambda x: torch.tanh(x),
    "ReLU": lambda x: Fn.relu(x),
    "LayerNorm": None,     # 单独处理
    "零均值+归一": lambda x: ((x - x.mean()) /
                       (x.pow(2).mean().sqrt() + 1e-6)),
}
_ln = nn.LayerNorm(_y0.shape[-1]).to(device)
for nm, fn in _fixes.items():
    if nm == "LayerNorm":
        u0, u1 = _ln(_y0), _ln(_y1)
    else:
        u0, u1 = fn(_y0), fn(_y1)
    c = cosf(u0, u1)
    v = "★已死" if c > 0.95 else ("偏弱" if c > 0.85 else "★已救回")
    print(f"  {nm:>18} | {c:>+14.6f} | {v}")

print("\n" + "=" * 84)
print("  判决")
print("=" * 84)
_c_proj = cosf(*segs["proj[3] SiLU2  = h"])
_c_pre = cosf(*segs["to_phase[0] (低秩R)"])
_c_out = cosf(*segs["to_phase[1] (升维)"])
_c_dt = cosf(*segs["dtheta (最终)"])
print(f"  h={_c_proj:+.4f} → to_phase[0]={_c_pre:+.4f} → "
      f"to_phase[1]={_c_out:+.4f} → dtheta={_c_dt:+.4f}")
if _c_pre > 0.95 and _c_proj < 0.9:
    print("  → ★★区分度死在**低秩第一层 to_phase[0]**（hidden→R）。")
    print("     低秩 16 迫使所有输出共用同一个 16 维子空间，"
          "而该子空间被训练成素材无关的通用方向。")
elif _c_out > 0.95 and _c_pre < 0.9:
    print("  → ★★ 区分度死在**升维层 to_phase[1]**（R→out_dim）。")
    print("     zero-init 让这层初始输出恒 0，训练中它学到的是「通用场方向」。")
else:
    print("  →区分度分布见上表，按最陡的一段定位。")