# -*- coding: utf-8 -*-
"""定位 corr(log a)=0.197 的真凶：FFN 残差 res 还是别的

已排除（对照实验 diag_heavy_tail_vs_ou.py）：
  ✗ 标量 OU μ：corr +1.0003（仿射变换不改结构）
  ✗ 收敛粒度：越细越差（per-channel 0.309 / per-band 0.233）
  ⇒ a_mu 初值量级也已被纠正（NORM 口径混用导致的误判）

本探针直接测：用真实权重跑初态=target，逐块记录 corr/log幅度/std，
定位到底是第几层、哪个环节开始崩。
"""
import sys, math
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
_p = ROOT
for _ in range(8):
    if (_p / "deps312").is_dir(): break
    if _p.parent == _p: break
    _p = _p.parent
sys.path.insert(0, str(_p))
sys.path.insert(0, str(_p))
try:
    from torch_accel_shim import install_full; install_full()
except Exception: pass
if (_p / "deps312").is_dir():
    sys.path.insert(0, str(_p / "deps312"))
# ★ 剔除残缺的 transformers（记忆坑：一次性脚本里 import 会炸）
import os as _os
for _bad in (r"C://Python312//Lib//site-packages//transformers",
             r"D://WorkBuddy//resources//app.asar.unpacked//resources//plugins"):
    sys.path = [x for x in sys.path
                if _os.path.normcase(_os.path.abspath(x)) != _os.path.normcase(_os.path.abspath(_bad))]
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

g = [x for x in sorted((ROOT/"data/gif_raw").rglob("*.gif")) if "ball" in x.name.lower()][0]
fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).to(dev).float()
re_t, im_t = fld[..., :D], fld[..., D:]
a_t = torch.sqrt(re_t**2 + im_t**2 + 1e-8); th_t = torch.atan2(im_t, re_t)
la_ref = torch.log(a_t[:, 1:].clamp(min=1e-8))
t, s = build_spacetime_index(TS, GRID, GRID, dev)

from model.aether_gif_model import AetherGIFModel
ck = torch.load(ROOT/"outputs_real/hat2_150/cond3_ckpt.pt", map_location=dev)
sd = {k: v for k, v in ck.items() if not k.startswith("qwen.")}
m = AetherGIFModel(d_model=D, device=dev, use_qwen=False, n_blocks=12,
                 ffn_hidden=256, init_mode='init', n_modules=8, mod_gamma=0.15).to(dev)
_w = sd['cond_init.to_phase.1.weight']
m.cond_init.n_groups = 5; m.cond_init.n_osc = _w.shape[0]//5
m.cond_init.to_phase[1] = torch.nn.Linear(_w.shape[1], _w.shape[0]).to(dev)
m.cond_init.to_amp[1] = torch.nn.Linear(_w.shape[1], sd['cond_init.to_amp.1.weight'].shape[0]).to(dev)
m.load_state_dict(sd, strict=False); m.eval()

with torch.no_grad():
    a, th, diag = m(["x"], fld.unsqueeze(0), t, s, batch_size=1,
                    use_cond_field=False,
                    force_init=(a_t.unsqueeze(0), th_t.unsqueeze(0)))

def _ndc(x):
    """去 DC：统一成 [N, D-1]。不假设输入是 [N,D] 还是 [B,N,D]。"""
    if x.dim() == 3: x = x[0]
    return x[:, 1:]

def cc(x, y):
    x = x - x.mean(); y = y - y.mean()
    return ((x*y).sum()/(x.norm()*y.norm()+1e-9)).item()

def _ndc(x):
    """去 DC：统一成 [N, D-1]。不假设输入是 [N,D] 还是 [B,N,D]。"""
    if x.dim() == 3: x = x[0]
    return x[:, 1:]

def cc(x, y):
    x = x - x.mean(); y = y - y.mean()
    return ((x*y).sum()/(x.norm()*y.norm()+1e-9)).item()

print("逐块扫描（初态 = target，绕开条件通路）:")
print(f"  初态: corr(log a)=+1.0000  a_std={a_t[:,1:].std():.4f}")
print()
print(f"{'层':>3} {'cos(θ)':>9} {'corr(log a)':>12} {'a_std':>9}")

with torch.no_grad():
    om = m._init_omega(t, s)
    a_c = a_t.unsqueeze(0).clone(); th_c = th_t.unsqueeze(0).clone()
    for i2, blk in enumerate(m.pmnncore.blocks):
        tf = m.pmnncore.time_field.forward(i2, dev)
        a_c, th_c, _ = blk(a_c, th_c, om, t, s,
                kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
                targets=None, protos=None, mod_id=None,
                cond_anchor=None, anchor_weight=0.3)
        print(f"{i2:>3} {torch.cos(_ndc(th_c)-_ndc(th_t)).mean():+9.4f} "
              f"{cc(torch.log(_ndc(a_c).clamp(min=1e-8)), la_ref):+12.4f} "
              f"{_ndc(a_c).std():9.4f}")
