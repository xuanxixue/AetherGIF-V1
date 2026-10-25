# -*- coding: utf-8 -*-
"""秒级诊断：物理栈把结构洗在哪一层？（第二十七步 c，真凶定位）

【已排除】
  ✗ target 信息丢失：target 方向场两素材 cos=−0.049，素材信息完整
  ✗ 条件通路不区分：discfix 后 cos_dθ=−1.000，已完全分开
  ⇒ 剩下的唯一解释：**物理栈的演化过程把结构洗成了棋盘格**

【本诊断】
  逐层记录振子场状态，测三个量：
  [1] 空间结构度：θ 在空间维度上的方差占比（是否趋于均匀）
      → Kuramoto 全同步 ⇒ 空间方差→0 ⇒ 输出是常数场 ⇒ 棋盘格
  [2] 与 target 的保真度：cos(θ_layer, θ_tar)
      → 洗结构发生在第几层
  [3] 频率分布：能量是否堆到特定 w（棋盘格的频谱特征）
      → 棋盘格 = 特定频率的纯振荡

【三个配置对照】
  A. 全开（默认）
  B. 关 Kuramoto（只有 drift + FFN）
  C. 关FFN（只有 drift + Kuramoto）
  → 定位是谁在洗结构
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
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
device = "cuda" if torch.cuda.is_available() else "cpu"
gif_dir = ROOT / "data" / "gif_raw" / "noto"
f = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"),
                        TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
a_tar = f[..., :D]
th_tar = torch.atan2(f[..., D:], a_tar)

os.environ["COND_GROUPS"] = "5"
os.environ["COND_LOWRANK"] = "16"
os.environ.setdefault("PHASE_ACT", "arctan")
CKPT = os.environ.get("CKPT", "outputs_real/discfix/cond3_ckpt.pt")

torch.manual_seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
sd = torch.load(ROOT / CKPT, map_location=device)
m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("qwen.")},
                  strict=False)
m.eval()
print(f"[ckpt] {CKPT}")

cc = ROOT / "data" / "clip_cache"
import numpy as _np
from data.clip_condition import build_cond_vector3
cv = torch.from_numpy(build_cond_vector3(
    _np.load(cc / "text3.npy")[0:1], _np.load(cc / "first3.npy")[0:1],
    _np.load(cc / "gif3_video.npy")[0:1], (1, 1, 1))).to(device)

print("\n" + "=" * 92)
print("  [1] 逐块结构度（结构方差占比 + 与 target 保真）")
print("=" * 92)
print(f"  {'块':>3} | {'空间std':>9} | {'相位std':>8} | {'cos(θ,tar)':>11} | 判定")
print("  " + "-" * 58)

import math
torch.manual_seed(123)
with torch.no_grad():
    a = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
    th = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi
    om = m._init_omega(t, s)
    dth, dln, _ = m.cond_init(cv, return_parts=True, n_pos=len(t))
    dth = m.cond_init.expand_to_d(dth)
    dln = m.cond_init.expand_to_d(dln)
    a = a * torch.exp(dln)
    th = th + dth

    def rep(thx, tag, depth=0):
        # 空间std：θ 在 N 维度上的标准差均值（全同步 → 趋0）
        sp = float(thx.reshape(-1, D).std(dim=0).mean())
        # 相位std：绕圆周的整体离散（同步 → 大）
        ph = float(torch.std(torch.atan2(torch.sin(thx), torch.cos(thx))))
        # 与 target 的余弦（相位差的余弦均值）
        pe = torch.remainder(thx[0] - th_tar + math.pi, 2 * math.pi) - math.pi
        cs = float(torch.cos(pe).mean())
        v = "★结构已死" if sp < 0.15 else ("弱" if sp < 0.4 else "健康")
        print(f"  {tag:>3} | {sp:>9.4f} | {ph:>8.4f} | {cs:>+11.4f} | {v}")
        return sp

    rep(th, "init")
    blk = m.pmnncore.blocks
    tf_of = m.pmnncore.time_field
    for i, b in enumerate(blk):
        tf = tf_of(i, device)
        a, th, _d = b(
            a, th, om, t, s,
            kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
            targets=None, protos=None, mod_id=None,
            cond_anchor=None, anchor_weight=0.0,
        )
        rep(th, f"{i+1}")

print("\n" + "=" * 92)
print("  [2] 各块频率分布（棋盘格的频谱指纹）")
print("=" * 92)
# 重建最终场并看w 分布
print(f"  {'块':>3} | {'w=0':>8} {'w1-4':>8} {'w5-12':>8} {'w13-24':>8}")
print("  " + "-" * 50)

torch.manual_seed(123)
with torch.no_grad():
    a = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
    th = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi
    a = a * torch.exp(dln)
    th = th + dth
    cid = torch.arange(D, device=device)
    wid = cid % W_FFT
    msk = [wid == 0, (wid >= 1) & (wid <= 4), (wid >= 5) & (wid <= 12), wid >= 13]
    for i, b in enumerate(list(m.pmnncore.blocks) + [None]):
        if b is not None:
            tf = m.pmnncore.time_field(i, device)
            a, th, _d = b(
                a, th, om, t, s,
                kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
                targets=None, protos=None, mod_id=None,
                cond_anchor=None, anchor_weight=0.0,
            )
        e = (a[0] ** 2).mean(dim=0)
        tot = e.sum().clamp(min=1e-12)
        row = [float(e[k].sum() / tot) for k in msk]
        tag = str(i + 1) if b is not None else "end"
        print(f"  {tag:>3} | " + " ".join(f"{v:>8.4f}" for v in row))