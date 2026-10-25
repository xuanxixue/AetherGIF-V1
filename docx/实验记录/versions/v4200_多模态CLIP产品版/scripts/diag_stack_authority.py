# -*- coding: utf-8 -*-
"""物理栈权威性探针（第二十九问）

受控实验：初态直接 = target 的 (a, θ)，看 12 层物理栈之后还剩多少信息。
  保留度高 ⇒ 栈不是瓶颈，瓶颈在「条件通路能否把初态预测到 target」
  保留度低 ⇒ 栈自己在破坏信息，必须先修栈

之前所有"物理栈无罪"的结论都是在**随机初态**下测的，
随机初态本就没结构，测不出栈的上限。这是第一次测上限。
"""
import sys, math, torch
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "gif_raw"
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

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(ROOT/"outputs_real/hat2_150/cond3_ckpt.pt", map_location=dev)
sd = {k: v for k, v in ck.items() if not k.startswith("qwen.")}
m = AetherGIFModel(d_model=D, device=dev, use_qwen=False, n_blocks=12,
                 ffn_hidden=256, init_mode='init',
                 n_modules=8, mod_gamma=0.15).to(dev)
# cond_init 的 n_osc 依赖 N，训练里是按 N=6144 重建的
_w = sd['cond_init.to_phase.1.weight']
N_POS = _w.shape[0] // 5# 训练用 COND_GROUPS=5
m.cond_init.n_groups = 5
m.cond_init.n_osc = N_POS
m.cond_init.to_phase[1] = torch.nn.Linear(_w.shape[1], _w.shape[0]).to(dev)
m.cond_init.to_amp[1] = torch.nn.Linear(_w.shape[1], sd['cond_init.to_amp.1.weight'].shape[0]).to(dev)
print(f'cond_init 重建: n_osc={N_POS} n_groups=5')
r = m.load_state_dict(sd, strict=False); m.eval()
print(f"载入权重 missing={len(r.missing_keys)} unexpected={len(r.unexpected_keys)}")

g = [x for x in sorted(DATA.rglob("*.gif")) if "ball" in x.name.lower()][0]
fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).to(dev).float()
print(f"素材 {g.name}  field={tuple(fld.shape)}")

re_t = fld[..., :D]; im_t = fld[..., D:]
a_t = torch.sqrt(re_t**2 + im_t**2 + 1e-8)
th_t = torch.atan2(im_t, re_t)
a_in = a_t.unsqueeze(0).clone(); th_in = th_t.unsqueeze(0).clone()
print(f"初态注入: a mean={a_in.mean():.5f} std={a_in.std():.5f} | θ mean={th_in.mean():.4f}")

t, s = build_spacetime_index(TS, GRID, GRID, dev)
print(f"坐标 t={tuple(t.shape)} s={tuple(s.shape)}")

with torch.no_grad():
    a_out, th_out, diag = m(["x"], fld.unsqueeze(0), t, s,
                batch_size=1, use_cond_field=False,
                force_init=(a_in, th_in))

print()
print("=== 物理栈后 vs target ===")
ao = a_out[0]; to = th_out[0]
# 相位一致性（去DC，DC 相位无意义）
c_ph = torch.cos((to[:,1:] - th_t[:,1:])).mean().item()
print(f"  cos(θ_out, θ_tar)        = {c_ph:+.4f}   （DC 除外）")
# 幅度：log 域相关
la_o = torch.log(ao[:,1:].clamp(min=1e-6)); la_t = torch.log(a_t[:,1:].clamp(min=1e-6))
la_o = la_o - la_o.mean(); la_t = la_t - la_t.mean()
corr = (la_o*la_t).sum()/(la_o.norm()*la_t.norm()+1e-9)
print(f"  corr(log a_out, log a_tar) = {corr.item():+.4f}")
print(f"  幅度 std: in={a_in.std():.5f} out={ao.std():.5f} tar={a_t.std():.5f}")
# 空间域：直接解码看 PSNR
fld_out = torch.cat([ao * torch.cos(to), ao * torch.sin(to)], dim=-1)
fld_out_c = fld_out.cpu().float()
fld_tar_c = fld.cpu().float()
fr_out = decode_field_to_frames(fld_out_c,
                                 TS, GRID, PATCH, WIN, IMG, D, normalize=False)
fr_tar = decode_field_to_frames(fld_tar_c,
                                TS, GRID, PATCH, WIN, IMG, D, normalize=False)
fo = np.stack(fr_out).astype(np.float32); ft = np.stack(fr_tar).astype(np.float32)
mse = ((fo-ft)**2).mean(); psnr = 10*math.log10(ft.max()**2/max(mse,1e-12))
print(f"  ★ 像素域 PSNR（初态就是target，物理栈跑完）= {psnr:.2f} dB")
print(f"    像素 std: out={fo.std():.2f}tar={ft.std():.2f}")
print()
if psnr > 30:
    print("★ 物理栈几乎无损 ⇒ 栈不是瓶颈，瓶颈在条件通路的预测能力")
elif psnr > 18:
    print("★ 物理栈有中等损耗 ⇒ 条件通路与栈都有责任")
else:
    print("★★ 物理栈在**破坏**信息 ⇒ 必须先修栈，调loss 无用")
