# -*- coding: utf-8 -*-
"""诊断 cond_field 条件分化——定位「300 步 sup loss 卡死、src1 坍缩到 A」的根因。

对照 TODO 文档 P0 的验收信号三连：
  1. 混合权重跨基 std：应 > 均匀基线 0.03634（之前卡在 0.00121）
  2. 不同 group 输出场两两平均差：应 > 1e-5（之前卡在 5.7e-06）
  3. chan_amp 的 std：应显著偏离初值 0.02（之前卡在 0.02015）

本脚本加载 cond2_ckpt.pt，用 int8 编码两个素材条件，检查：
  - 两个 cond_vec 的 pred_field 差异（场差）
  - to_chan 的 delta 幅度（零初始化是否压死）
  - chan_amp 各基 std（是否偏离初值 0.02）
  - 混合权重 w 跨基 std
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
sys.path.insert(0, str(_PROJ))               # 项目根（torch_accel_shim / int8_qwen_encoder）
_DEPS = _PROJ / "deps312"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))           # D 盘好版 transformers 5.19
# 剔除 C 盘残缺 transformers（4.38/4.44 互覆，缺 utils/versions.py）
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
from model.aether_gif_model import AetherGIFModel
from int8_qwen_encoder import Int8QwenEncoder
from data.clip_condition import build_cond_vector

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "一个黑色的8号台球在绿色台球桌上滚动"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "一个黄道十二星座的水瓶座符号在深色背景上发光"),
]
gif_dir = ROOT / "data" / "gif_raw" / "noto"
OUT_DIR = ROOT / "outputs_real" / "cond2"

print("[1] int8 编码两个素材条件")
enc = Int8QwenEncoder(max_frames=8)
t_emb = enc.encode_texts([d for _, d in GIF_SPECS], batch_size=2)
i_emb = enc.encode_gif_paths([str(gif_dir / n) for n, _ in GIF_SPECS], batch_size=1)
del enc; import gc; gc.collect(); torch.cuda.empty_cache()

ZERO = np.zeros_like(t_emb)
cv_both = build_cond_vector(t_emb, i_emb, [1, 1])   # 图文生，两条

print("\n[2] 加载模型 + ckpt")
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(OUT_DIR / "cond2_ckpt.pt", map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()

t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)

print("\n[3] cond_field 诊断")
cf = model.cond_field
with torch.no_grad():
    cv = torch.from_numpy(cv_both).to(device).float()
    field, parts = cf(cv, t_coord, s_coord, return_parts=True)  # [2, N, 2D]

    # 1. 场差
    diff = (field[0] - field[1]).abs().mean().item()
    print(f"[场差] 两 cond pred_field 绝对平均差 = {diff:.6e}  (>1e-5 才算分化)")

    # 2. 混合权重 w 跨基 std
    w = parts["w"]  # [2, 27]
    w_std = w.std(dim=1).mean().item()
    print(f"[w_std] softmax 混合权重跨基 std = {w_std:.5f}  (均匀基线 0.03634)")

    # 3. chan_amp std（应偏离初值 0.02）
    ca = cf.chan_amp.data  # [27, 250]
    print(f"[chan_amp] std = {ca.std().item():.5f}  (初值 0.02)")

    # 4. to_chan delta 幅度（零初始化是否压死）
    h = cf.w_act(cf.w_proj(cv))
    delta = cf.to_chan(h).view(2, cf.n_basis, 2 * D)
    delta = torch.tanh(delta) * 0.5
    print(f"[delta] to_chan 条件残差 abs_mean = {delta.abs().mean().item():.6f}  (0=零初始化没动)")

    # 5. 两个条件的 w 与 delta 是否真的有差异
    w_diff = (w[0] - w[1]).abs().mean().item()
    delta_diff = (delta[0] - delta[1]).abs().mean().item()
    print(f"[w差异] 两 cond 混合权重差异 = {w_diff:.6e}")
    print(f"[delta差异] 两 cond 通道调制差异 = {delta_diff:.6e}")

    # 6. supervise 对当前 field 的值（复算一次，确认 sup 卡在哪）
    from physics.cond_field import CondFieldPredictor
    # 需要 target，重新编码两个素材场
    f0 = encode_gif_to_field(str(gif_dir / GIF_SPECS[0][0]), TS, GRID, PATCH, WIN, IMG, D).unsqueeze(0).to(device)
    f1 = encode_gif_to_field(str(gif_dir / GIF_SPECS[1][0]), TS, GRID, PATCH, WIN, IMG, D).unsqueeze(0).to(device)
    tgt = torch.cat([f0, f1], 0)
    sup = CondFieldPredictor.supervise(field, tgt, d_model=D)
    print(f"[sup复算] 当前 supervise loss = {sup.item():.4f}")

print("\n[done]")