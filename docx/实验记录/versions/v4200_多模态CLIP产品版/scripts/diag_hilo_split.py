# -*- coding: utf-8 -*-
"""定位 fix_r3 的 hilo=2.99 是谁爆的：CLIP 路 / FFT 路 / 相加？

绝对判据里 hilo（高频/低频能量比）超标 = 高频噪声。修好梯度后
hilo 从 4.42→2.99 但仍远超标（素材 0.01）。本探针分路测：
  仅 CLIP 初态 /仅 FFT 初态 / 相加初态 → 各自 hilo + PSNR + ac1
判定：若某路单独就爆 → 是那支的问题；若相加才爆 → 是叠加。
"""
import os, sys, math
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir(): break
    if _PROJ.parent == _PROJ: break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))
_DEPS = _PROJ / "deps312"
if _DEPS.is_dir(): sys.path.insert(0, str(_DEPS))
try:
    from torch_accel_shim import install_full; install_full()
except Exception: pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
for k, v in dict(COND_GROUPS="5", COND_LOWRANK="16", PHASE_ACT="arctan",
                 RES_LIMIT="0.5", W2_NORM_MAX="3.0", KAPPA_MAX="6.0",
                 FFT_FFTW="1").items():
    os.environ.setdefault(k, v)

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"
gif_dir = ROOT / "data" / "gif_raw" / "noto"
t, s = build_spacetime_index(TS, GRID, GRID, device)
f0 = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"), TS, GRID, PATCH,
                         WIN, IMG, D).to(device)
a_tar = torch.sqrt(f0[..., :D] ** 2 + f0[..., D:] ** 2 + 1e-12)
th_tar = torch.atan2(f0[..., D:], f0[..., :D])

CKPT = ROOT / "outputs_real" / "fix_r3" / "cond3_ckpt_300_0157.pt"
sd = torch.load(CKPT, map_location=device)
sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
m.load_state_dict(sd, strict=False); m.eval()
cc = ROOT / "data" / "clip_cache"
cv = torch.from_numpy(build_cond_vector3(
    np.load(cc / "text3.npy")[0:1], np.load(cc / "first3.npy")[0:1],
    np.load(cc / "gif3_video.npy")[0:1], (1, 1, 1))).to(device)

def hilo(x):
    g = x.mean(axis=-1)
    F = np.abs(np.fft.fftshift(np.fft.fft2(
        g - g.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
    H, W_ = F.shape[1], F.shape[2]
    yy, xx = np.mgrid[0:H, 0:W_]
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W_ // 2) ** 2)
    return float(F[:, r > H * 0.35].sum() / max(F[:, r < H * 0.15].sum(), 1e-12))

def metrics(a, th, tag, is_mat=False):
    if is_mat:
        fld = torch.cat([a * torch.cos(th), a * torch.sin(th)], -1).cpu().float()
    else:
        fld = torch.cat([a * torch.cos(th), a * torch.sin(th)], -1)[0].cpu().float()
    fr = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D,
                                         normalize=True)).astype(np.float64)
    ft = np.stack(decode_field_to_frames(f0.cpu().float(), TS, GRID, PATCH,
                                         WIN, IMG, D, normalize=True)).astype(np.float64)
    g = fr.mean(axis=-1)
    gg = g - g.mean(axis=(1, 2), keepdims=True)
    num = (gg[:, 1:] * gg[:, :-1]).mean(); den = (gg * gg).mean() + 1e-12
    ac1 = float(num / den)
    psnr = 10 * math.log10(255.0 ** 2 / max(((fr - ft) ** 2).mean(), 1e-12))
    print(f"  {tag:20s} PSNR={psnr:6.2f}dB  ac1={ac1:+.3f}  hilo={hilo(fr):.3f}")
    return hilo(fr)

print("=" * 76)
print(f"  fix_r3 分路 hilo 定位（{CKPT.name}）")
print("=" * 76)
metrics(a_tar, th_tar, "素材(基准)", is_mat=True)
torch.manual_seed(0)
with torch.no_grad():
    a0 = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
    th0 = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi
    # CLIP 路
    dth, dln, iparts = m.cond_init(cv, return_parts=True, n_pos=len(t),
                                   theta0=th0, a0=a0)
    dth_D = m.cond_init.expand_to_d(dth); dln_D = m.cond_init.expand_to_d(dln)
    a_c = a0 * torch.exp(dln_D); th_c = th0 + dth_D
    metrics(a_c, th_c, "仅CLIP初态")
    # FFT 路
    fdth, fdla, _ = m.fft_bypass(f0.unsqueeze(0), return_parts=True)
    fdth_D = m.fft_bypass.expand_to_d(fdth); fdla_D = m.fft_bypass.expand_to_d(fdla)
    a_f = a0 * torch.exp(fdla_D); th_f = th0 + fdth_D
    metrics(a_f, th_f, "仅FFT初态")
    # 相加
    a_s = a_c * torch.exp(fdla_D); th_s = th_c + fdth_D
    metrics(a_s, th_s, "相加初态")
    # FFT 路幅度单独看（dla 的 log 值量级）
    print(f"\n  dlna(CLIP) mean_abs={dln_D.abs().mean():.3f}")
    print(f"  dla(FFT)  mean_abs={fdla_D.abs().mean():.3f}")