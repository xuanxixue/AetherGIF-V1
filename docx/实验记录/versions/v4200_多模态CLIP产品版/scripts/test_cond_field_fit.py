# -*- coding: utf-8 -*-
"""最小对照实验：cond_field 单独拟合 target 场（不接物理核）。

目的：隔离两个变量，判定「300 步 sup 卡死」到底是
  (a) cond_field 结构表达力不够（27 基驻波画不出两个素材的谱），还是
  (b) 训练 setup 问题（lr 太低 / 与主 loss 耦合稀释梯度）。

做法：只优化 cond_field 参数，纯 supervise loss，两个素材 batch=2 一起喂。
  若 sup 能快速降到 ~0.1 且场差拉大 → 结构没问题，问题在训练 setup → 分组 lr。
  若 sup 卡在 1.x 不动 → 结构表达力瓶颈，需换基底/加容量。
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from physics.cond_field import CondFieldPredictor
from int8_qwen_encoder import Int8QwenEncoder
from data.clip_condition import build_cond_vector, CLIP_EMB_DIM

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "一个黑色的8号台球在绿色台球桌上滚动"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "一个黄道十二星座的水瓶座符号在深色背景上发光"),
]
gif_dir = ROOT / "data" / "gif_raw" / "noto"

print("[1] 编码条件 + target 场")
enc = Int8QwenEncoder(max_frames=8)
t_emb = enc.encode_texts([d for _, d in GIF_SPECS], batch_size=2)
i_emb = enc.encode_gif_paths([str(gif_dir / n) for n, _ in GIF_SPECS], batch_size=1)
del enc; import gc; gc.collect(); torch.cuda.empty_cache()
cv = torch.from_numpy(build_cond_vector(t_emb, i_emb, [1, 1])).to(device).float()

f0 = encode_gif_to_field(str(gif_dir / GIF_SPECS[0][0]), TS, GRID, PATCH, WIN, IMG, D).unsqueeze(0)
f1 = encode_gif_to_field(str(gif_dir / GIF_SPECS[1][0]), TS, GRID, PATCH, WIN, IMG, D).unsqueeze(0)
tgt = torch.cat([f0, f1], 0).to(device)   # [2, N, 2D]

t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)

print("\n[2] 纯 supervise 拟合（只优化 cond_field）")
for lr in [3e-4, 1e-3]:
    torch.manual_seed(1234)
    cf = CondFieldPredictor(cond_dim=CLIP_EMB_DIM * 2 + 1, d_model=D).to(device)
    opt = torch.optim.AdamW(cf.parameters(), lr=lr)
    print(f"\n=== lr={lr} ===")
    for step in range(1, 201):
        opt.zero_grad()
        field = cf(cv, t_coord, s_coord)
        sup = CondFieldPredictor.supervise(field, tgt, d_model=D)
        sup.backward()
        opt.step()
        if step % 40 == 0 or step == 1:
            with torch.no_grad():
                d = (field[0] - field[1]).abs().mean().item()
                ca = cf.chan_amp.data.std().item()
            print(f"  [{step:3d}] sup={sup.item():.4f} 场差={d:.4e} chan_amp_std={ca:.4f}")

print("\n[done] 若 sup 快速→0.1 且 chan_amp_std 明显升，则结构 OK，问题在训练 setup")
print("      若 sup 卡 1.x 且 chan_amp 不动，则 27 基驻波表达力不足，需换基底")