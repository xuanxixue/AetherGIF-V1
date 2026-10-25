# -*- coding: utf-8 -*-
"""对照实验 3：验证「驻波基底补 DC 模式」能否解锁 cond_field 拟合。

fit2 结论：64 基仍只降 9.1%（sup 1.51→1.37），lr=3e-3 还发散；结构瓶颈。
疑点（fit2 §9.3）：StandingWave3D 模式序号从 1 起，无 m=0 DC 基，
所有基三维度全振荡，谱包络的缓变成分张不住。

本实验：mode_offset=0（m∈{0..3}，含 DC；基数量不变=64，最高频换成最低频）
  D: 64 基 offset=0 lr=1e-3   对比 fit2-B（同配置 offset=1 → sup 1.3734）
  E: 64 基 offset=0 lr=2e-3   探含 DC 后稳定性能否吃更高 lr

判定：
  D 明显快于 fit2-B（400 步 sup 跌破 ~1.2 或斜率显著更陡）→ DC 埇假设成立
      → cond_field 切 mode_offset=0，改 overfit_cond2 重训
  D 与 fit2-B 相当 → DC 无效 → 换思路（位置基底改可学低秩，
      或 supervise 改逐通道包络而弃逐位置形状）
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
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
from physics.kuramoto_sync import build_spacetime_index
from physics.cond_field import CondFieldPredictor
from data.clip_condition import CLIP_EMB_DIM

D, GRID, TS = 125, 32, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

CACHE = ROOT / "outputs_real" / "fit_cache.pt"
blob = torch.load(CACHE, map_location="cpu")
cv = blob["cv"].to(device).float()
tgt = blob["tgt"].to(device)
print(f"[0] device={device}  cv={tuple(cv.shape)} tgt={tuple(tgt.shape)}")

t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)
cond_dim = CLIP_EMB_DIM * 2 + 1

print("\n[1] 纯 supervise 拟合（mode_offset=0 含 DC 基）")
for lr in [1e-3, 2e-3]:
    torch.manual_seed(1234)
    cf = CondFieldPredictor(cond_dim=cond_dim, d_model=D,
                            n_modes_t=4, n_modes_xy=4,
                            mode_offset=0).to(device)
    opt = torch.optim.AdamW(cf.parameters(), lr=lr)
    print(f"\n=== 64基(offset=0, 含DC) lr={lr} ===")
    for step in range(1, 401):
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

print("\n[done] 对照基准 fit2-B（64基 offset=1 lr=1e-3 400步）: sup=1.3734")
