# -*- coding: utf-8 -*-
"""残差消融：物理栈里每个部件对corr/cos 的贡献

裁剪 w2 后 corr 只到 0.50（层7 时是 0.974），说明还有独立破坏源。
本探针逐个关掉部件，看谁在破坏：
  - res（FFN 幅度残差）
  - OU（回复力）
  - drift（漂移场引导）
  - ph_res（FFN 相位残差）
  - kuramoto 本身
定位到具体部件后再决定改哪个。
"""
import sys, math
from pathlib import Path
import torch
import torch.nn as nn
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

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
import numpy as np

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

g = [x for x in sorted((ROOT/"data/gif_raw").rglob("*.gif")) if "ball" in x.name.lower()][0]
fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).to(dev).float()
re_t, im_t = fld[..., :D], fld[..., D:]
a_t = torch.sqrt(re_t**2+im_t**2+1e-8); th_t = torch.atan2(im_t, re_t)
la_ref = torch.log(a_t[:, 1:].clamp(min=1e-8))
t, s = build_spacetime_index(TS, GRID, GRID, dev)

ck = torch.load(ROOT/"outputs_real/hat2_150/cond3_ckpt.pt", map_location=dev)
sd0 = {k: v for k, v in ck.items() if not k.startswith("qwen.")}

def build():
    m = AetherGIFModel(d_model=D, device=dev, use_qwen=False, n_blocks=12,
                     ffn_hidden=256, init_mode='init', n_modules=8, mod_gamma=0.15).to(dev)
    _w = sd0['cond_init.to_phase.1.weight']
    m.cond_init.n_groups = 5; m.cond_init.n_osc = _w.shape[0]//5
    m.cond_init.to_phase[1] = torch.nn.Linear(_w.shape[1], _w.shape[0]).to(dev)
    m.cond_init.to_amp[1] = torch.nn.Linear(_w.shape[1], sd0['cond_init.to_amp.1.weight'].shape[0]).to(dev)
    m.load_state_dict(sd0, strict=False); m.eval()
    return m

def cc(x, y):
    x = x - x.mean(); y = y - y.mean()
    return ((x*y).sum()/(x.norm()*y.norm()+1e-9)).item()

def run(m, tag, clip=3.0):
    with torch.no_grad():
        if clip:
            for blk in m.pmnncore.blocks:
                _inner = getattr(blk.ffn, 'inner', blk.ffn)
                n = _inner.w2.weight.norm()
                if float(n) > clip: _inner.w2.weight.mul_(clip/float(n))
        om = m._init_omega(t, s)
        a_c = a_t.unsqueeze(0).clone(); th_c = th_t.unsqueeze(0).clone()
        for i, blk in enumerate(m.pmnncore.blocks):
            tf = m.pmnncore.time_field.forward(i, dev)
            a_c, th_c, _ = blk(a_c, th_c, om, t, s,
                    kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
                    targets=None, protos=None, mod_id=None,
                    cond_anchor=None, anchor_weight=0.3)
    A, TH = a_c[0], th_c[0]
    c_ph = torch.cos(TH[:,1:]-th_t[:,1:]).mean().item()
    c_a = cc(torch.log(A[:,1:].clamp(min=1e-8)), la_ref)
    fld_o = torch.cat([A*torch.cos(TH), A*torch.sin(TH)], -1).cpu().float()
    fo = np.stack(decode_field_to_frames(fld_o,TS,GRID,PATCH,WIN,IMG,D,normalize=True)).astype(np.float64)
    ft = np.stack(decode_field_to_frames(fld.cpu().float(),TS,GRID,PATCH,WIN,IMG,D,normalize=True)).astype(np.float64)
    psnr = 10*math.log10(255.0**2/max(((fo-ft)**2).mean(),1e-12))
    print(f"  {tag:36s} cos(θ)={c_ph:+.4f} corr={c_a:+.4f} PSNR={psnr:6.2f}dB std={A[:,1:].std():.4f}")

print("=== 残差消融（w2 已裁到 ≤3.0）===")
run(build(), "基线（w2≤3.0）")

# 关 res
m = build()
with torch.no_grad():
    for blk in m.pmnncore.blocks: blk._no_res = True
    # 用 forward hook 把 ffn 输出清零
    for blk in m.pmnncore.blocks:
        blk.ffn.register_forward_hook(lambda mod,i,o: torch.zeros_like(o))
run(m, "关 FFN（res + ph_res 全关）")

# 只关幅度 res（保留相位）
class ZeroAmp(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner=inner
    def forward(self,*a,**k):
        o = self.inner(*a,**k)
        return torch.cat([torch.zeros_like(o[...,:o.shape[-1]//2]), o[..., o.shape[-1]//2:]], -1)
m = build()
with torch.no_grad():
    for blk in m.pmnncore.blocks:
        blk.ffn = ZeroAmp(blk.ffn)
run(m, "只关幅度 res（保留 ph_res）")

# 关 OU
m = build()
with torch.no_grad():
    for blk in m.pmnncore.blocks: blk.a_gamma.data.zero_()
run(m, "关 OU（a_gamma=0）")

# 关 drift
m = build()
with torch.no_grad():
    for blk in m.pmnncore.blocks: blk.drift.scale.data.zero_()
run(m, "关 drift（scale=0）")

# 全关
m = build()
with torch.no_grad():
    for blk in m.pmnncore.blocks:
        blk.ffn.register_forward_hook(lambda mod,i,o: torch.zeros_like(o))
        blk.a_gamma.data.zero_(); blk.drift.scale.data.zero_()
run(m, "全关（FFN+OU+drift）")
