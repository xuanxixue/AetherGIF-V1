# -*- coding: utf-8 -*-
"""对照实验 4：确定「生成通路可承受的基数量」+ 验证 einsum 等价优化。

两件事一次跑清：
  F: 27基 offset=0 lr=2e-3 —— 若拟合够，生成通路无需动显存（27基 B=1 实测 2.7GB）
  G: 64基 offset=0 lr=2e-3 —— 复测 einsum("bkc,nk->bnc") 等价分解，
     应复现 fit3-E（1.1781）；若数值不符说明优化改错了

判定：
  F 与 G 差距小（<0.03）→ 取 27 基（显存最安全）
  F 明显差于 G → 取 64 基，生成通路吃 einsum 优化省下的显存
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

blob = torch.load(ROOT / "outputs_real" / "fit_cache.pt", map_location="cpu",
                   weights_only=True)
cv = blob["cv"].to(device).float()
tgt = blob["tgt"].to(device)
print(f"[0] device={device}  cv={tuple(cv.shape)} tgt={tuple(tgt.shape)}")

t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)
cond_dim = CLIP_EMB_DIM * 2 + 1

print("\n[1] 纯 supervise 拟合（mode_offset=0 含 DC 基，einsum 等价优化已生效）")
for n_basis, mt, mxy, lr in [(27, 3, 3, 2e-3), (64, 4, 4, 2e-3)]:
    torch.manual_seed(1234)
    cf = CondFieldPredictor(cond_dim=cond_dim, d_model=D,
                            n_modes_t=mt, n_modes_xy=mxy,
                            mode_offset=0).to(device)
    opt = torch.optim.AdamW(cf.parameters(), lr=lr)
    print(f"\n=== {n_basis}基(offset=0) lr={lr} ===")
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

print("\n[done] 对照基准：fit3-E（64基 offset=0 lr=2e-3 旧 einsum）sup=1.1781")
print("      若 G 复现 1.1781 → einsum 等价优化正确")
