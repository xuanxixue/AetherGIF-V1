# -*- coding: utf-8 -*-
"""容量诊断：物理核 vs 条件通路的参数分布，确认 FFN 容量扩充的落点。"""
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

import torch
from model.aether_gif_model import AetherGIFModel

m = AetherGIFModel(d_model=125, device="cpu", use_qwen=False, n_blocks=12,
                   init_mode="init", n_modules=8, mod_gamma=0.15)
groups = {"pmnncore(物理核)": 0, "cond_init(条件初始状态)": 0, "cond_field(旧B方案)": 0,
          "heads(四个旧头)": 0, "cond_w(三路标量)": 0, "other": 0}
for n, p in m.named_parameters():
    if n.startswith("pmnncore."):
        groups["pmnncore(物理核)"] += p.numel()
    elif n.startswith("cond_init."):
        groups["cond_init(条件初始状态)"] += p.numel()
    elif n.startswith("cond_field."):
        groups["cond_field(旧B方案)"] += p.numel()
    elif any(n.startswith(h) for h in ("omega_bias_head", "drift_target_head",
                                       "kappa_scale_head", "T_scale_head")):
        groups["heads(四个旧头)"] += p.numel()
    elif n.startswith("cond_w"):
        groups["cond_w(三路标量)"] += p.numel()
    else:
        groups["other"] += p.numel()
tot = sum(groups.values())
print("=" * 60)
for k, v in groups.items():
    print(f"  {k:26s}: {v:>12,}  ({v / tot * 100:5.1f}%)")
print(f"  {'总计':26s}: {tot:>12,}")

blk = m.pmnncore.blocks[0]
ffn_p = blk.ffn.w1.weight.numel() + blk.ffn.w2.weight.numel()
print(f"\n单块 StandingWaveFFN: w1{tuple(blk.ffn.w1.weight.shape)} "
      f"w2{tuple(blk.ffn.w2.weight.shape)} = {ffn_p:,} 参数")
print(f"hidden=256 → 512: 单块 {ffn_p:,} → {2 * 250 * 512 + 512 + 512 * 250 + 250:,}，"
      f"12 块增量 = {12 * (2 * 250 * 512 - ffn_p):,}")
print("=" * 60)
