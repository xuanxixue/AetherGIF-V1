# -*- coding: utf-8 -*-
"""验证修法：把后 4 层 ffn.w2 裁剪到前8 层的权重规模，物理栈还保不保结构？

根因（diag_res_by_block.py）：
  w2 范数从层0 的 1.72 **单调涨到**层11 的 14.50（8.4×）
  res_std 0.014 → 0.196（14×），末层 res_max 顶满 tanh 上限 2.0
  ⇒ 后 4 层在往幅度里灌巨量扰动，把结构冲掉。

本探针做**无需训练的直接干预**：
  把每层 w2 逐层缩放到统一范数（或只裁剪后 4 层），
  看 corr(log a) 能救回多少。这是修法有效性的上限测试。
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

base = torch.load(ROOT/"outputs_real/hat2_150/cond3_ckpt.pt", map_location=dev)
sd0 = {k: v for k, v in base.items() if not k.startswith("qwen.")}

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

def run(m, tag):
    with torch.no_grad():
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
    # 像素 PSNR
    fld_o = torch.cat([A*torch.cos(TH), A*torch.sin(TH)], -1).cpu().float()
    fr_o = decode_field_to_frames(fld_o, TS,GRID,PATCH,WIN,IMG,D, normalize=True)
    fr_t = decode_field_to_frames(fld.cpu().float(), TS,GRID,PATCH,WIN,IMG,D, normalize=True)
    fo = np.stack(fr_o).astype(np.float64); ft = np.stack(fr_t).astype(np.float64)
    psnr = 10*math.log10(255.0**2/max(((fo-ft)**2).mean(),1e-12))
    print(f"  {tag:34s} cos(θ)={c_ph:+.4f} corr(log a)={c_a:+.4f} "
          f"PSNR={psnr:6.2f}dB  std={A[:,1:].std():.4f}")
    return c_a, psnr

print("=== 干预实验：ffn.w2 权重范数裁剪（无需训练，直接缩放）===")
m0 = build(); run(m0, "原样（w2 范数 1.7→14.5）")

for target_norm in [10.0, 6.0, 3.0, 2.0, 1.72]:
    m = build()
    with torch.no_grad():
        for blk in m.pmnncore.blocks:
            n = blk.ffn.w2.weight.norm()
            if float(n) > target_norm:
                blk.ffn.w2.weight.mul_(target_norm / float(n))
    run(m, f"所有层 w2 裁到≤{target_norm:.2f}")

print()
print("=== 只裁后 4 层（层0-7 保持不动）===")
for target_norm in [6.0, 3.0, 2.0]:
    m = build()
    with torch.no_grad():
        for i, blk in enumerate(m.pmnncore.blocks):
            if i >= 8:
                n = blk.ffn.w2.weight.norm()
                if float(n) > target_norm:
                    blk.ffn.w2.weight.mul_(target_norm / float(n))
    run(m, f"仅层8-11 裁到≤{target_norm:.2f}")
