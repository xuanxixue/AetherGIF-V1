# -*- coding: utf-8 -*-
"""CondInitPC 集成最小冒烟（不跑训练循环，只验证形状/显存/梯度通路）。

验证项：
  1. COND_INIT_PC=1 时模型实例化、参数量打印；
  2. 一次 forward（θ̂ 模式 + cond_vec + target_field 两种口径）不报错；
  3. _CondInitCls.supervise(逐通道) 有梯度回传到 cond_init；
  4. 显存峰值（4GB 约束的核心风险点）。
用法： python scripts/smoke_pc.py
"""
import os, sys, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ---- 与 train_guided.py 完全相同的引导块（deps312 + 屏蔽坏 transformers + shim）----
_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir():
        break
    if _PROJ.parent == _PROJ:
        break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))
_DEPS = _PROJ / "deps312"
_VENV = _PROJ / ".venv" / "Lib" / "site-packages"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
if _VENV.is_dir():
    sys.path.append(str(_VENV))
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
os.chdir(ROOT)

# ---- 与 run_sep_fix.cmd / train_guided.py 同口径（import 前设好） ----
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")
os.environ.setdefault("IDEAL_CLAMP", str(math.pi))
os.environ.setdefault("COND_INIT_PC", os.environ.get("COND_INIT_PC", "1"))
os.environ.setdefault("COND_PC_RANK", os.environ.get("COND_PC_RANK", "8"))

import torch
from model.aether_gif_model import AetherGIFModel
from physics.cond_init_pc import CondInitPC
from physics.kuramoto_sync import build_spacetime_index

D, TS, GRID = 125, 6, 32
N = TS * GRID * GRID          # 6144
N_COND3 = 6147

print("[smoke] building model ...")
model = AetherGIFModel(d_model=D, device="cuda", use_qwen=False, n_blocks=12,
                       ffn_hidden=256, init_mode="init",
                       n_modules=8, mod_gamma=0.15).cuda()
model.train()

_ci = model.cond_init
print(f"[smoke] cond_init class = {type(_ci).__name__}")
print(f"[smoke] rank = {getattr(_ci, 'rank', None)}")
_total = sum(p.numel() for p in model.parameters())
_ci_n = sum(p.numel() for p in _ci.parameters())
print(f"[smoke] params: total={_total/1e6:.2f}M  cond_init={_ci_n/1e6:.2f}M")

# ---- 构造假条件向量与 target 场 ----
cv = torch.randn(1, N_COND3).cuda()
cv = cv / cv.norm() * 10.0
t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, "cuda")

# target_field: [B, N, 2D]
rng = torch.Generator().manual_seed(0)
a_tar = (torch.rand(1, N, D, generator=rng) * 0.1).cuda()
ph_tar = (torch.rand(1, N, D, generator=rng) * 2 * math.pi - math.pi).cuda()
target_field = torch.cat([a_tar * torch.cos(ph_tar), a_tar * torch.sin(ph_tar)], dim=-1).cuda()

# ---- ① teacher 口径（target_field 给定）----
a, th, diag = model(texts=None, target_field=target_field, t_coord=t_coord,
                    s_coord=s_coord, batch_size=1, cond_vec=cv, use_cond_field=True)
print(f"[smoke] teacher fwd OK: a={tuple(a.shape)} th={tuple(th.shape)}")
for _k in ("dtheta", "dlna", "theta_hat", "a_hat"):
    _v = diag.get(_k)
    print(f"[smoke]   diag[{_k}] = {tuple(_v.shape) if _v is not None else None}")

# ---- ② dropout/推理口径（target_field=None）----
a2, th2, diag2 = model(texts=None, target_field=None, t_coord=t_coord,
                       s_coord=s_coord, batch_size=1, cond_vec=cv, use_cond_field=True)
print(f"[smoke] cond-only fwd OK: a={tuple(a2.shape)} th={tuple(th2.shape)}")
print(f"[smoke]   fft_dth present? {diag2.get('fft_dth') is not None} (应为 False)")

# ---- ③ supervise 梯度通路（两种类都验证）----
loss = CondInitPC.supervise(diag["dtheta"], diag["dlna"], target_field, d_model=D,
                            theta_hat=diag.get("theta_hat"), a_hat=diag.get("a_hat"))
loss.backward()
_g = [p.grad for p in model.cond_init.parameters() if p.grad is not None]
_gnorm = sum(float(g.norm() ** 2) for g in _g) ** 0.5
print(f"[smoke] supervise loss={float(loss):.4f}  cond_init grad_norm={_gnorm:.4f} (非零=通)")

if torch.cuda.is_available():
    _alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
    _resv = torch.cuda.max_memory_reserved() / 1024 ** 2
    print(f"[smoke] peak alloc={_alloc:.0f}MiB  peak reserved={_resv:.0f}MiB  (上限 4096MiB)")
print("[smoke] ALL_OK")
