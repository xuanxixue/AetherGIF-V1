# -*- coding: utf-8 -*-
"""逐块测 FFN 幅度残差 res 的实际用量 + 权重规模

目的：解释「为什么只有后 4 层破坏结构」。
假设A：后 4 层 res 量级更大（权重学到了大扰动）
假设 B：后 4 层 res 量级正常，但方向与target 无关（纯乱动）
区分：若量级大→A；若量级小但 corr 低 → B（乱动）。
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

# 正确残差：dla_out = log a_new − (log a − γ(log a − μ))，即 res 的实际贡献
print(f"{'层':>3} {'res均值':>9} {'res_std':>9} {'res_max':>9} "
      f"{'w2范数':>9} {'a_gamma':>8} {'a_mu均值':>9}")
with torch.no_grad():
    om = m._init_omega(t, s)
    a_c = a_t.unsqueeze(0).clone(); th_c = th_t.unsqueeze(0).clone()
    for i, blk in enumerate(m.pmnncore.blocks):
        la0 = torch.log(a_c.clamp(min=1e-8))
        tf = m.pmnncore.time_field.forward(i, dev)
        a_n, th_n, d = blk(a_c, th_c, om, t, s,
                kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
                targets=None, protos=None, mod_id=None,
                cond_anchor=None, anchor_weight=0.3)
        la1 = torch.log(a_n.clamp(min=1e-8))
        ou_part = -blk.a_gamma * (la0 - blk.a_mu.view(1,1,-1))
        res_eff = la1 - (la0 + ou_part)          # 残差实际贡献
        w2 = blk.ffn.w2.weight
        print(f"{i:>3} {res_eff.mean():9.4f} {res_eff.std():9.4f} "
              f"{res_eff.abs().max():9.4f} {w2.norm():9.3f} "
              f"{blk.a_gamma.item():8.4f} {blk.a_mu.mean():9.4f}")
        a_c, th_c = a_n, th_n
