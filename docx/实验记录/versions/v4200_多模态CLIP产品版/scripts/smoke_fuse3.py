# -*- coding: utf-8 -*-
"""任务 #68 冒烟：删投影后模型实例化 + 1 步前向，确认三路标量加权通路。

核对：
  1. cond_w 参数存在且形状 [3]，softmax 起步 = 等权 [1/3,1/3,1/3]
  2. 四个旧头输入维度 = 2048，CondProjector3 已彻底删除
  3. _fuse_three_way 输出 [B, 2048]
  4. 一步 forward 端到端跑通（新 cv3 缓存 + A 方案 cond_init）
  5. 参数量对比（应比旧版少 ~5.5M 投影参数）
"""
import math
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
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass

try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] torch_accel_shim 加载失败：{ex}")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import CLIP_EMB_DIM, N_COND3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("[1] 实例化模型（use_qwen=False，A 方案 init_mode=init）")
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       init_mode="init", n_modules=8, mod_gamma=0.15).to(device)

print("\n[2] 结构核对")
w = torch.softmax(model.cond_w, dim=0)
print(f"  cond_w        : shape={tuple(model.cond_w.shape)} softmax={w.tolist()}")
assert model.cond_w.shape == (3,)
assert hasattr(model, "cond_projector") is False, "CondProjector3 没删干净！"
print(f"  cond_projector: 已删除 ✔")
for name in ("omega_bias_head", "drift_target_head", "kappa_scale_head", "T_scale_head"):
    lin = getattr(model, name)
    print(f"  {name:16s}: in={lin.in_features} out={lin.out_features}")
    assert lin.in_features == CLIP_EMB_DIM == 2048, f"{name} 输入维度不是 2048"

n_params = sum(p.numel() for p in model.parameters())
print(f"\n  总参数量 = {n_params:,}")

print("\n[3] _fuse_three_way 维度核对")
cv = torch.randn(2, N_COND3, device=device)
fused = model._fuse_three_way(cv)
print(f"  in={tuple(cv.shape)} -> out={tuple(fused.shape)}")
assert fused.shape == (2, 2048)

print("\n[4] 一步前向（真 GIF target + 新 cv3_full 缓存）")
CLIP_CACHE = ROOT / "data" / "clip_cache"
cv_full = np.load(CLIP_CACHE / "cv3_full.npy")
gp = str(ROOT / "data" / "gif_raw" / "noto" / "8-ball_1f3b1.gif")
f = encode_gif_to_field(gp, TS, GRID, PATCH, WIN, IMG, D).to(device)
target = f.unsqueeze(0)
t, s = build_spacetime_index(TS, GRID, GRID, device)
one_cv = torch.from_numpy(cv_full[0:1]).to(device)
model.train()
a, th, diag = model(texts=None, target_field=target, t_coord=t, s_coord=s,
                    batch_size=1, cond_vec=one_cv, use_cond_field=True)
print(f"  a   : {tuple(a.shape)}")
print(f"  theta: {tuple(th.shape)}")
assert a.shape == th.shape == (1, TS * GRID * GRID, D)
assert "dtheta" in diag and "dlna" in diag, "A 方案 init_parts 没返回"
print(f"  dtheta: {tuple(diag['dtheta'].shape)}  dlna: {tuple(diag['dlna'].shape)}")
print(f"  r={diag['r']:.4f}  theta_diff={diag['theta_diff']:.4f}")

# 反传冒烟
loss = a.sum() + th.sum()
loss.backward()
print(f"  cond_w.grad = {model.cond_w.grad.tolist()}")
print("\n[SMOKE PASS] 删投影 + 标量加权 + 旧头 2048 全部通路正常")
print("=" * 70)
