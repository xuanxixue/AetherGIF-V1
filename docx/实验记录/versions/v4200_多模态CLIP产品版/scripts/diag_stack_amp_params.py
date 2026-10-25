# -*- coding: utf-8 -*-
"""定位物理栈幅度失控点：读取 ckpt 里 OU 递推( a_gamma / a_mu )与 FFN 输出层范数，
确认「幅度 std 在 layer7-9 暴涨 15 倍」的来源是哪个可学习参数走飞了。

用法：C:/Python312/python.exe -u scripts/diag_stack_amp_params.py
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
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
for k, v in dict(COND_GROUPS="5", COND_LOWRANK="16", PHASE_ACT="arctan",
                 RES_LIMIT="0.5", W2_NORM_MAX="3.0", KAPPA_MAX="6.0",
                 FFT_FFTW="1").items():
    os.environ.setdefault(k, v)

import torch

D = 125
device = "cpu"
CKPT = ROOT / (os.environ.get("DIAG_CKPT") or "outputs_real/fix_r4/cond3_ckpt.pt")

sd = torch.load(CKPT, map_location=device)
sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}

print("=" * 72)
print(f"  物理栈幅度参数体检  {CKPT.name}")
print("=" * 72)

# 1. per-block a_gamma（OU 回复力，可学习，初值 0.3）
print("\n【a_gamma】OU 回复力（初值 0.3）")
gammas = []
for i in range(12):
    k = f"pmnncore.blocks.{i}.a_gamma"
    v = sd.get(k)
    if v is not None:
        gammas.append(float(v))
        print(f"  block{i:2d}: a_gamma = {float(v):+.4f}")
    else:
        print(f"  block{i:2d}: (missing)")
if gammas:
    print(f"  → min={min(gammas):+.4f} max={max(gammas):+.4f} mean={sum(gammas)/len(gammas):+.4f}")

# 2. per-block a_mu（OU 平衡点，可学习，初值 log(0.2)=-1.609）
print("\n【a_mu】OU 平衡点（初值 log(0.2)=-1.609）")
mus = []
for i in range(12):
    k = f"pmnncore.blocks.{i}.a_mu"
    v = sd.get(k)
    if v is not None:
        vv = v.float()
        mus.append((float(vv.min()), float(vv.max()), float(vv.mean())))
        print(f"  block{i:2d}: a_mu min={vv.min():+.4f} max={vv.max():+.4f} "
              f"mean={vv.mean():+.4f} std={vv.std():.4f}")

# 3. FFN 输出层 w2 范数（第三十死穴约束目标 ≤3.0）
print("\n【ffn.w2 范数】第三十死穴约束目标 ≤3.0")
for i in range(12):
    k = f"pmnncore.blocks.{i}.ffn.w2.weight"
    v = sd.get(k)
    if v is not None:
        print(f"  block{i:2d}: |w2| = {float(v.norm()):.4f}")

# 4. 其他幅度相关：direct_weight
print("\n【其他】")
for k in ["pmnncore.blocks.0.drift.direct_weight"]:
    v = sd.get(k)
    if v is not None:
        print(f"  {k} = {float(v):+.4f}")

print("\n" + "=" * 72)
print("  判读：")
print("  a_gamma 学到接近 0 → OU 回复力失稳，res 累积失控（记忆里 γ→0.04 的捷径）")
print("  a_mu std 大 → 平衡点漂移，把幅度拉离 target 量级")
print("  w2 范数 > 3 → project_w2 约束失效（或未在本次训练生效）")
print("=" * 72)