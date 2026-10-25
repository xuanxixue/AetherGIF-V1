# -*- coding: utf-8 -*-
"""Kuramoto 相位演化各项的贡献分解

背景：消融实验里 cos(θ) 在**所有**配置下恒钉 +0.36，
说明相位破坏源完全独立于 FFN/OU。
本探针拆解 dtheta 的四项：omega / coupling_struct / coupling_mf / kappa*g
外加 omega_max 饱和，测各项 RMS 与对 cos 的影响。
"""
import sys, math
from pathlib import Path
import torch
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
_p = ROOT
for _ in range(8):
    if (_p / "deps312").is_dir(): break
    if _p.parent == _p: break
    _p = _p.parent
sys.path.insert(0, str(_p))
try:
    from torch_accel_shim import install_full; install_full()
except Exception: pass
if (_p / "deps312").is_dir(): sys.path.insert(0, str(_p / "deps312"))
import os as _os
_b = r"C://Python312//Lib//site-packages//transformers"
sys.path = [x for x in sys.path if _os.path.normcase(_os.path.abspath(x)) != _os.path.normcase(_os.path.abspath(_b))]

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

g = [x for x in sorted((ROOT/"data/gif_raw").rglob("*.gif")) if "ball" in x.name.lower()][0]
fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).to(dev).float()
re_t, im_t = fld[..., :D], fld[..., D:]
a_t = torch.sqrt(re_t**2+im_t**2+1e-8); th_t = torch.atan2(im_t, re_t)
t, s = build_spacetime_index(TS, GRID, GRID, dev)

ck = torch.load(ROOT/"outputs_real/hat2_150/cond3_ckpt.pt", map_location=dev)
sd = {k: v for k, v in ck.items() if not k.startswith("qwen.")}
m = AetherGIFModel(d_model=D, device=dev, use_qwen=False, n_blocks=12,
                 ffn_hidden=256, init_mode='init', n_modules=8, mod_gamma=0.15).to(dev)
_w = sd['cond_init.to_phase.1.weight']
m.cond_init.n_groups = 5; m.cond_init.n_osc = _w.shape[0]//5
m.cond_init.to_phase[1] = torch.nn.Linear(_w.shape[1], _w.shape[0]).to(dev)
m.cond_init.to_amp[1] = torch.nn.Linear(_w.shape[1], sd['cond_init.to_amp.1.weight'].shape[0]).to(dev)
m.load_state_dict(sd, strict=False); m.eval()

print("=== Kuramoto 内部参数 ===")
k0 = m.pmnncore.blocks[0].kuramoto
print(f"  omega_max = {k0.omega_max}   dt = {k0.dt}")
print(f"  use_langevin = {k0.use_langevin}  (eval 模式不注入噪声)")

print()
print("=== omega 分布（相位漂移的驱动力）===")
om = m._init_omega(t, s)
print(f"  omega RMS = {om.std():.4f}  min={om.min():.4f} max={om.max():.4f}")
print(f"  12 层累计dt·omega = {12*k0.dt*om.std():.4f} rad RMS")
print(f"  ⇒ 若 omega 是纯漂移，相位会累积平移 {12*k0.dt*om.mean():.4f} rad")

print()
print("=== 逐层：单独跑 Kuramoto（关 FFN 幅度残差与 ph_res）===")
def cos_of(th): return torch.cos(th[:,1:]-th_t[:,1:]).mean().item()

with torch.no_grad():
    # 只走 Kuramoto：monkeypatch掉 ffn 与 drift
    for blk in m.pmnncore.blocks:
        blk.ffn.register_forward_hook(lambda mod,i,o: torch.zeros_like(o))
        blk.a_gamma.data.zero_()
        if hasattr(blk.drift, 'w'): blk.drift.w.data.zero_()
    th = th_t.unsqueeze(0).clone(); a = a_t.unsqueeze(0).clone()
    print(f"  起始cos = {cos_of(th[0]):+.4f}")
    for i, blk in enumerate(m.pmnncore.blocks):
        tf = m.pmnncore.time_field.forward(i, dev)
        a, th, _ = blk(a, th, om, t, s,
                kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
                targets=None, protos=None, mod_id=None,
                cond_anchor=None, anchor_weight=0.3)
        print(f"  层{i:2d} cos={cos_of(th[0]):+.4f}  κ={tf['kappa']:.3f}")
