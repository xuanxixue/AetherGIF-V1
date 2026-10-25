# -*- coding: utf-8 -*-
"""验证三个修法（RES_LIMIT / W2_NORM_MAX / KAPPA_MAX）综合效果

对照：同一 ckpt、同一素材，初态=target，逐档开关修法。
判据：corr(log a)、cos(θ)、像素 PSNR、std。
"""
import sys, math, os
from pathlib import Path
import numpy as np, torch
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

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
dev = "cuda" if torch.cuda.is_available() else "cpu"

g = [x for x in sorted((ROOT/"data/gif_raw").rglob("*.gif")) if "ball" in x.name.lower()][0]
fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).to(dev).float()
re_t, im_t = fld[..., :D], fld[..., D:]
a_t = torch.sqrt(re_t**2+im_t**2+1e-8); th_t = torch.atan2(im_t, re_t)
la_ref = torch.log(a_t[:, 1:].clamp(min=1e-8))
t, s = build_spacetime_index(TS, GRID, GRID, dev)

ck = torch.load(ROOT/"outputs_real/hat2_150/cond3_ckpt.pt", map_location=dev)
sd = {k: v for k, v in ck.items() if not k.startswith("qwen.")}

def build():
    m = AetherGIFModel(d_model=D, device=dev, use_qwen=False, n_blocks=12,
                     ffn_hidden=256, init_mode='init', n_modules=8, mod_gamma=0.15).to(dev)
    _w = sd['cond_init.to_phase.1.weight']
    m.cond_init.n_groups = 5; m.cond_init.n_osc = _w.shape[0]//5
    m.cond_init.to_phase[1] = torch.nn.Linear(_w.shape[1], _w.shape[0]).to(dev)
    m.cond_init.to_amp[1] = torch.nn.Linear(_w.shape[1], sd['cond_init.to_amp.1.weight'].shape[0]).to(dev)
    m.load_state_dict(sd, strict=False); m.eval()
    return m

def cc(x, y):
    x = x - x.mean(); y = y - y.mean()
    return ((x*y).sum()/(x.norm()*y.norm()+1e-9)).item()

def run(tag):
    m = build()
    with torch.no_grad():
        a, th, _ = m(["x"], fld.unsqueeze(0), t, s, batch_size=1,
                     use_cond_field=False,
                     force_init=(a_t.unsqueeze(0), th_t.unsqueeze(0)))
    A, TH = a[0], th[0]
    c_ph = torch.cos(TH[:,1:]-th_t[:,1:]).mean().item()
    c_a = cc(torch.log(A[:,1:].clamp(min=1e-8)), la_ref)
    fld_o = torch.cat([A*torch.cos(TH), A*torch.sin(TH)], -1).cpu().float()
    fo = np.stack(decode_field_to_frames(fld_o,TS,GRID,PATCH,WIN,IMG,D,normalize=True)).astype(np.float64)
    ft = np.stack(decode_field_to_frames(fld.cpu().float(),TS,GRID,PATCH,WIN,IMG,D,normalize=True)).astype(np.float64)
    psnr = 10*math.log10(255.0**2/max(((fo-ft)**2).mean(),1e-12))
    print(f"  {tag:44s} cos(θ)={c_ph:+.4f} corr={c_a:+.4f} PSNR={psnr:6.2f}dB std={A[:,1:].std():.4f}")

print(f"环境: RES_LIMIT={_os.environ.get('RES_LIMIT','(默认0.5)')} "
      f"W2_NORM_MAX={_os.environ.get('W2_NORM_MAX','(默认3.0)')} "
      f"KAPPA_MAX={_os.environ.get('KAPPA_MAX','(默认6.0)')}")
run("当前配置")
