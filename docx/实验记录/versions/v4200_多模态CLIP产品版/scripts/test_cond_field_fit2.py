# -*- coding: utf-8 -*-
"""对照实验 2：区分「驻波基表达力不足」vs「优化速度不足」。

fit2 在 fit1（27 基、lr≤1e-3、200 步，sup 只降 5%）基础上加两个变量：
  - n_modes 3×3×3=27 → 4×4×4=64（基数量 ×2.4）
  - lr 上探到 3e-3
  - 步数 200 → 400

条件矩阵：
  A: 27 基  lr=3e-3   （只加 lr）
  B: 64 基  lr=1e-3   （只加基）
  C: 64 基  lr=3e-3   （两者都加）

判定：
  - B/C 显著快于 A → 基数量是主因 → cond_field 扩到 64 基
  - A 远快于 fit1 的 lr=1e-3 且 64 基无额外收益 → 纯 setup 问题 → 分组 lr
  - 都只缓慢降 → 更深问题（疑点：StandingWave3D 模式序号从 1 起，无 m=0
    DC 基，所有基三维度全振荡，难以表达缓变包络）→ 补 DC 基

★ 条件向量缓存：第一次 int8 编码后落盘 fit_cache.pt，
  之后该脚本不再需要 GPU 编码器（纯小张量拟合，CPU 也能跑），
  方便与 preencode_int8.py（占 GPU 7 小时）并存。
"""
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
from data.clip_condition import build_cond_vector, CLIP_EMB_DIM

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "一个黑色的8号台球在绿色台球桌上滚动"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "一个黄道十二星座的水瓶座符号在深色背景上发光"),
]
gif_dir = ROOT / "data" / "gif_raw" / "noto"
CACHE = ROOT / "outputs_real" / "fit_cache.pt"

print(f"[0] device={device}")
t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)

# ---------- 条件向量：优先缓存，否则 int8 编码后落盘 ----------
if CACHE.exists():
    blob = torch.load(CACHE, map_location="cpu")
    cv = blob["cv"].to(device).float()
    tgt = blob["tgt"].to(device)
    print(f"[1] 命中缓存 {CACHE}  cv={tuple(cv.shape)} tgt={tuple(tgt.shape)}")
else:
    from int8_qwen_encoder import Int8QwenEncoder
    print("[1] 首次运行：int8 GPU 编码条件 + target 场")
    enc = Int8QwenEncoder(max_frames=8)
    t_emb = enc.encode_texts([d for _, d in GIF_SPECS], batch_size=2)
    i_emb = enc.encode_gif_paths([str(gif_dir / n) for n, _ in GIF_SPECS], batch_size=1)
    del enc
    import gc; gc.collect(); torch.cuda.empty_cache()
    cv_np = build_cond_vector(t_emb, i_emb, [1, 1])
    cv = torch.from_numpy(cv_np).to(device).float()

    f0 = encode_gif_to_field(str(gif_dir / GIF_SPECS[0][0]), TS, GRID, PATCH, WIN, IMG, D).unsqueeze(0)
    f1 = encode_gif_to_field(str(gif_dir / GIF_SPECS[1][0]), TS, GRID, PATCH, WIN, IMG, D).unsqueeze(0)
    tgt = torch.cat([f0, f1], 0).to(device)   # [2, N, 2D]

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cv": cv.cpu(), "tgt": tgt.cpu()}, CACHE)
    print(f"[1] 已落盘缓存 {CACHE}（下次直接加载，不再需要 GPU 编码器）")

cond_dim = CLIP_EMB_DIM * 2 + 1
N_STEPS = 400
COND_ID = {27: (3, 3), 64: (4, 4)}

print("\n[2] 纯 supervise 拟合")
for n_basis, lr in [(27, 3e-3), (64, 1e-3), (64, 3e-3)]:
    mt, mxy = COND_ID[n_basis]
    torch.manual_seed(1234)
    cf = CondFieldPredictor(cond_dim=cond_dim, d_model=D,
                            n_modes_t=mt, n_modes_xy=mxy).to(device)
    opt = torch.optim.AdamW(cf.parameters(), lr=lr)
    print(f"\n=== n_basis={n_basis}({mt}x{mxy}) lr={lr} 参数={sum(p.numel() for p in cf.parameters()):,} ===")
    for step in range(1, N_STEPS + 1):
        opt.zero_grad()
        field = cf(cv, t_coord, s_coord)
        sup = CondFieldPredictor.supervise(field, tgt, d_model=D)
        sup.backward()
        opt.step()
        if step % 80 == 0 or step == 1:
            with torch.no_grad():
                d = (field[0] - field[1]).abs().mean().item()
                ca = cf.chan_amp.data.std().item()
            print(f"  [{step:3d}] sup={sup.item():.4f} 场差={d:.4e} chan_amp_std={ca:.4f}")

print("\n[done] 判定见文件头注释")
