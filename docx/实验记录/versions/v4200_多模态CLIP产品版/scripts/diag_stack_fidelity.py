# -*- coding: utf-8 -*-
"""秒级诊断：物理栈把正确的 dtheta 变成棋盘格了吗？（第二十七步 b）

【上一诊断的结论（证伪了我的假设）】
  target 方向场两素材 cos = −0.049（几乎正交，判别空间极大）
  ⇒ **素材信息在 target 里完全保留**，不是supervise 口径问题。
  条件通路也分得开（discfix: cos_dθ = −1.000）。
  但生成仍是棋盘格 ⇒ 问题在**物理栈的变换**。

【本诊断要回答】
  把**已知正确的 dtheta**（直接由 target 相位构造，不经任何训练）
  喂进物理栈，看12 层后输出是否还是 target。
  - 若输出 = target       ⇒ 物理栈保真，问题在dtheta 本身
  - 若输出 = 棋盘格/常数  ⇒ ★ 物理栈把结构洗掉了（真凶）
  这就是 diag_kernel_fidelity.py 的思路，但那次用的是「只测 drift」，
  且用的是官方targets=target 通路。这次要严格控制：
  **只改 dtheta 来源，其余完全一致**。
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

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"
gif_dir = ROOT / "data" / "gif_raw" / "noto"
NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]

os.environ["COND_GROUPS"] = "5"
os.environ["COND_LOWRANK"] = "16"
os.environ.setdefault("PHASE_ACT", "arctan")
CKPT = os.environ.get("CKPT", "outputs_real/discfix/cond3_ckpt.pt")

t, s = build_spacetime_index(TS, GRID, GRID, device)
fs = [encode_gif_to_field(str(gif_dir / n), TS, GRID, PATCH, WIN, IMG, D).to(device)
      for n in NAMES]

torch.manual_seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
sd = torch.load(ROOT / CKPT, map_location=device)
m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("qwen.")},
                  strict=False)
m.eval()
print(f"[ckpt] {CKPT}")


def cosf(x, y):
    x, y = x.reshape(-1).float(), y.reshape(-1).float()
    return float(x @ y / (x.norm() * y.norm() + 1e-12))


def stats(img, tag):
    """空间域统计：检测棋盘格（规则纹理）特征"""
    x = img.reshape(-1).float()
    u = (x - x.mean())
    # 相邻像素差：棋盘格会在每个 patch 边界出现规则跳变
    g = img.mean(dim=-1)          # 灰度 [H,W]
    dx = (g[:, 1:] - g[:, :-1]).abs().mean()
    dy = (g[1:] - g[:-1]).abs().mean()
    # 拉普拉斯能量：高频总量
    lap = torch.tensor(0.0)
    for _ in range(0):
        pass
    return dict(std=float(x.std()), gradx=float(dx), grady=float(dy))


print("\n" + "=" * 92)
print("  [1] 三组对照：同一物理栈，只改初始相位")
print("=" * 92)

# --- A. 正确初相位 = target 本身（dtheta=0，即 theta0 = th_tar）
resA = m(texts=None, target_field=fs[0][None], t_coord=t, s_coord=s,
         batch_size=1, cond_vec=None, use_cond_field=False,
         init_override=torch.atan2(fs[0][..., D:], fs[0][..., :D])[None])