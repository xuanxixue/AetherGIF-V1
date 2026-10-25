# -*- coding: utf-8 -*-
"""秒级对照：proj 结尾激活的选择（第二十六步 c，修复方案验证）

【已定位的事实】（diag_cond_layerwise.py，训练后 ckpt）
  cond_vec      cos=0.632
  proj[0] Lin1  cos=0.719
  proj[1] SiLU1 cos=0.714
  proj[3] Lin2  cos=0.675← 区分度良好！
  proj[4] SiLU2 cos=0.987      ← ★ 跳跃点，区分度在这里死掉

【机理】
  SiLU(x)=x·σ(x) 是**单向**的：x≫0 时≈x，x≪0 时被压到≈0。
  proj[3] 输出的「素材差异分量」主要落在负半轴 ⇒ SiLU 把它们压平，
  两个素材的输出都只剩正向大分量 ⇒ 方向趋同（cos→0.99）。
  换句话说：**不是权重没学到素材，是学到的素材信息被激活函数销毁了**。

【要验证的修法】
  A. 保留末尾 SiLU（现状）→ 已知 cos=0.987
  B. 末尾换成 LayerNorm → 标准化到零均值单位方差，**反方向差异被保留**
  C. 末尾换成 Tanh → 有界对称，双向，理论上也应保留
  D. 直接把 Linear2 裸输出（无激活）→ 零假设

【附加】每种方案都测「未训练随机权重」与「训练后 ckpt」两态，
  确认修法在两态都成立（不是只在某态偶然）。
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

from data.clip_condition import build_cond_vector3

device = "cuda" if torch.cuda.is_available() else "cpu"
cc = ROOT / "data" / "clip_cache"
T3 = np.load(cc / "text3.npy")
F3 = np.load(cc / "first3.npy")
G3 = np.load(cc / "gif3_video.npy")

cvs = [torch.from_numpy(build_cond_vector3(T3[i:i+1], F3[i:i+1], G3[i:i+1],
                                           (1, 1, 1))).to(device) for i in (0, 1)]
c0, c1 = cvs
print(f"[cond] dim={c0.numel()}  原始跨素材 cos="
      f"{float((c0.reshape(-1)@c1.reshape(-1))/(c0.norm()*c1.norm())):+.6f}")

CD = c0.shape[-1]
HID = 512


def cosf(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    return float(x @ y / (x.norm() * y.norm() + 1e-12))


def probe(tag, tail: str):
    """用同一份 Linear1/SiLU1/Linear2 权重，只换尾部激活，看区分度。"""
    torch.manual_seed(7)
    lin1 = nn.Linear(CD, HID).to(device)
    lin2 = nn.Linear(HID, HID).to(device)
    with torch.no_grad():
        h0 = lin1(c0)
        h1 = lin1(c1)
        p0 = lin2(torch.nn.functional.silu(h0))
        p1 = lin2(torch.nn.functional.silu(h1))
        # 线性层输出（跳过尾部激活）
        y0, y1 = p0, p1
        if tail == "silu":
            y0, y1 = torch.nn.functional.silu(p0), torch.nn.functional.silu(p1)
        elif tail == "tanh":
            y0, y1 = torch.tanh(p0), torch.tanh(p1)
        elif tail == "layernorm":
            ln = nn.LayerNorm(HID).to(device)
            y0, y1 = ln(p0), ln(p1)
        elif tail == "none":
            y0, y1 = p0, p1
        elif tail == "identity_center":
            # 零均值 + 缩放（LayerNorm 的可学版本：只有仿射参数，无平移）
            y0 = p0 - p0.mean(dim=-1, keepdim=True)
            y1 = p1 - p1.mean(dim=-1, keepdim=True)
            s = (y0.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)
            t = (y1.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-6)
            y0, y1 = y0 / s, y1 / t
    return cosf(y0, y1)


print("\n" + "=" * 82)
print("  尾部激活对「跨素材区分度」的影响（随机权重，单素材对）")
print("=" * 82)
print(f"  {'尾部激活':>16} | {'跨素材cos':>12} | 判定")
print("  " + "-" * 60)
for tail, label in (("none", "Linear 裸输出"),
                    ("silu", "SiLU（现状）"),
                    ("tanh", "Tanh"),
                    ("layernorm", "LayerNorm"),
                    ("identity_center", "零均值+归一")):
    c = probe("", tail)
    verdict = "★区分度已死" if c > 0.95 else ("区分度弱" if c > 0.8 else "★区分度保留")
    print(f"  {label:>16} | {c:>+12.6f} | {verdict}")

print("\n" + "=" * 82)
print("  判决")
print("=" * 82)
c_silu = probe("", "silu")
c_ln = probe("", "layernorm")
c_tanh = probe("", "tanh")
print(f"  SiLU(现状)={c_silu:+.4f}   Tanh={c_tanh:+.4f}   LayerNorm={c_ln:+.4f}")
if c_ln < c_silu - 0.05:
    print("  → ★ LayerNorm 显著恢复区分度，可作修复方案。")
elif c_tanh < c_silu - 0.05:
    print("  → ★ Tanh 显著恢复区分度，可作修复方案。")
else:
    print("  →△ 单层改动无效，说明机理判断有误，需回到逐层实测复核。")