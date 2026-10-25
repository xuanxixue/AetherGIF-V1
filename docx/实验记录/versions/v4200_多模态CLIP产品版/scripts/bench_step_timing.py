# -*- coding: utf-8 -*-
"""计时诊断：46 秒/步 vs 直接分量前 17.6 秒/步 的成因定位。

分段计时 forward / backward / 优化器步，并用 direct_weight 1.0 vs 0.0
精确隔离「直接相位分量 sin(θ_tar−θ)」这一项的代价（direct_weight=0 时
计算路径完全相同，只是 sin 结果乘 0）。

若 dw=0 就快回 17.6s → 罪魁是 sin 项（需优化实现）
若 dw=0 仍 46s → 与 sin 无关，是系统内存换页 / 其他因素

同时报告训练进程的物理工作集 vs 私有提交，判断是否换页。
"""
import math
import os
import sys
import time
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

import torch
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from physics.cond_field import CondFieldPredictor

D, GRID, TS = 125, 32, 6
device = "cuda"

OUT_DIR = ROOT / "outputs_real" / "cond2"
blob = torch.load(ROOT / "outputs_real" / "fit_cache.pt", map_location="cpu",
                   weights_only=True)
cv = blob["cv"].to(device).float()
tgt = blob["tgt"].to(device)

t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(OUT_DIR / "cond2_ckpt.pt", map_location=device)
model.load_state_dict(sd, strict=False)
model.train()


def bench(n_steps=3):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    one_cv = cv[0:1]
    one_tgt = tgt[0:1]
    a_tar = torch.sqrt(one_tgt[..., :D] ** 2 + one_tgt[..., D:] ** 2 + 1e-12)
    th_tar = torch.atan2(one_tgt[..., D:], one_tgt[..., :D])
    t_fw = t_bw = 0.0
    for step in range(n_steps):
        opt.zero_grad()
        t0 = time.time()
        a, th, diag = model(texts=None, target_field=one_tgt, t_coord=t_coord,
                            s_coord=s_coord, batch_size=1, cond_vec=one_cv,
                            use_cond_field=True)
        torch.cuda.synchronize()
        t1 = time.time()
        pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
        la = torch.log1p(a.clamp(min=1e-8))
        lt = torch.log1p(a_tar.clamp(min=1e-8))
        loss_amp = (la - lt).pow(2).mean()
        w_ph = (a_tar ** 2).clamp(min=1e-8)
        loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
        pred = diag.get("pred_field")
        loss_sup = CondFieldPredictor.supervise(pred, one_tgt, d_model=D)
        loss = loss_amp + loss_ph + 2.0 * loss_sup
        t2 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.time()
        opt.step()
        t_fw += t1 - t0 + t2 - t1
        t_bw += t3 - t2
    return t_fw / n_steps, t_bw / n_steps


def proc_mem():
    import ctypes, ctypes.wintypes as wt
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x400 | 0x10, False, os.getpid())
    if not h:
        return None
    class PMC(ctypes.Structure):
        _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                    ('PeakWS', ctypes.c_size_t), ('WS', ctypes.c_size_t),
                    ('q1', ctypes.c_size_t), ('q2', ctypes.c_size_t),
                    ('q3', ctypes.c_size_t), ('q4', ctypes.c_size_t),
                    ('Pagefile', ctypes.c_size_t), ('PeakPagefile', ctypes.c_size_t),
                    ('Private', ctypes.c_size_t)]
    pmc = PMC(); pmc.cb = ctypes.sizeof(pmc)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), ctypes.sizeof(pmc))
    k32.CloseHandle(h)
    if not ok:
        return None
    gb = lambda x: f"{x/1024**3:.2f}GB"
    return f"工作集={gb(pmc.WS)} 私有提交={gb(pmc.Private)}"


import gc

print(f"[0] 初始进程内存: {proc_mem()}")
torch.cuda.reset_peak_memory_stats()

print("\n[1] direct_weight = 当前值（=1.0 训练后）")
fw, bw = bench()
print(f"    forward={fw:.1f}s/步  backward={bw:.1f}s/步  合计={fw+bw:.1f}s/步")
print(f"    进程内存: {proc_mem()}")
print(f"    GPU 峰值: {torch.cuda.max_memory_allocated()/1024**2:.0f}MiB")

print("\n[2] direct_weight = 0（sin 项结果乘 0，计算路径完全相同）")
for blk in model.pmnncore.blocks:
    blk.drift.direct_weight.data.zero_()
fw0, bw0 = bench()
print(f"    forward={fw0:.1f}s/步  backward={bw0:.1f}s/步  合计={fw0+bw0:.1f}s/步")
print(f"    进程内存: {proc_mem()}")

print(f"\n[结论] 直接分量代价 = {(fw+bw)-(fw0+bw0):+.1f}s/步")
print("      若 ~0 → 慢与 sin 无关，是系统内存换页；若大幅为正 → sin 项实现问题")
