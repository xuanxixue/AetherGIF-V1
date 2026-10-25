# -*- coding: utf-8 -*-
"""相位监督口径矩阵探针（方向2 的变体筛选，**不需要训练**）。

问题：当前 loss_ph_abs 用 w_norm 逐通道归一权重压制高频，
      导致 |pe| 中位 1.51 rad 卡在 IDEAL_CLAMP 饱和区、与条件无关。
本探针在**同一份 fix_r3 权重**下扫描多种相位监督口径，
输出每种口径的：loss 值/ 梯度量级 / 目标方向相关 / DC与非DC 的压力分配。
用于**训练前**筛掉明显无效的组合，训练后直接进矩阵。

口径候选：
  W0 当前      : w_norm = (a_tar²/ch均值) 逐通道归一（现状，高频被压）
  W1 sqrt      : w = sqrt(a_tar²)=a_tar      （缓和压制）
  W2 uniform   : w = 1（逐通道均匀，高频与DC 同权）
  W3 log       : w = 1/(1+log1p(a_tar))     （反重尾，压制 DC 更狠）
  W4 dc_off    : w_norm 且 DC 置0（完全排除 DC）
  W5 band      : per-band 归一（每 band 内均匀，band 间按能量）
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

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
device = "cuda" if torch.cuda.is_available() else "cpu"
gif_dir = ROOT / "data" / "gif_raw" / "noto"
t, s = build_spacetime_index(TS, GRID, GRID, device)

CKPT = ROOT / "outputs_real" / "fix_r3" / "cond3_ckpt_300_0157.pt"
sd = torch.load(CKPT, map_location=device)
sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
m.load_state_dict(sd, strict=False); m.eval()

cc = ROOT / "data" / "clip_cache"
f0 = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"), TS, GRID, PATCH,
                         WIN, IMG, D).to(device)
a_tar = torch.sqrt(f0[..., :D] ** 2 + f0[..., D:] ** 2 + 1e-12)
th_tar = torch.atan2(f0[..., D:], f0[..., :D])
at = a_tar.unsqueeze(0)
cv = torch.from_numpy(np.load(cc / "cv3_full.npy")[0:1]).to(device).float()
target = torch.cat([at * torch.cos(th_tar)[None], at * torch.sin(th_tar)[None]], -1)

print("=" * 96)
print("  相位监督口径矩阵探针（fix_r3 权重，同权重下比较口径，**非训练**）")
print("=" * 96)

with torch.no_grad():
    a, th, diag = m(texts=None, target_field=target, t_coord=t, s_coord=s,
                    batch_size=1, cond_vec=cv, use_cond_field=True)
    # 条件通路的 θ̂（训练监督对象）—— supervise 里 pd_g = theta_hat
    th_hat = diag.get("theta_hat")                     # [B,N,G]
    th_hat_D = m.cond_init.expand_to_d(th_hat)          # [B,N,D]
    a0 = diag.get("theta0")
    # supervise 的目标：ideal = th_ch - (theta0 + dtheta)，theta0+dtheta ≡ th_hat (mod 2pi)
    ideal_all = torch.remainder(th_tar[None] - th_hat_D + math.pi, 2 * math.pi) - math.pi

# 构造各种权重（[N,D]，只监督非 DC）
a_sq = (a_tar ** 2).clamp(min=1e-8)                    # [N,D]
w_ch = a_sq[:, 1:]
def report(tag, w):
    """w: [N,D-1] 权重。ideal 非DC。"""
    ide = ideal_all[0][:, 1:]
    wl = w.detach()
    # clamp 前后
    for cl_name, cl in [("clamp=pi/2", math.pi / 2), ("clamp=pi", math.pi)]:
        v = (wl * ide.clamp(min=-cl, max=cl).pow(2)).mean()
        sat = (ide.abs() > cl).float().mean()
        # DC 与非DC 的梯度压力占比
        dcp = (wl * ide[:, 0:1].clamp(min=-cl, max=cl).pow(2)).mean()
        ncp = (wl[:, 1:] * ide[:, 1:].clamp(min=-cl, max=cl).pow(2)).mean()
        ratio = float(dcp / (ncp + 1e-12))
        print(f"  {tag:10s} {cl_name:10s} loss={float(v):.4f}饱和={sat*100:4.1f}%  "
              f"DC压力/非DC={ratio:.4f}  非DC权重和={float(wl[:,1:].sum()):.3e}")
    print()

w_norm = w_ch / w_ch.mean(dim=0, keepdim=True).clamp(min=1e-8)   # W0 现状
report("W0当前", w_norm)
report("W1sqrt", a_tar[:, 1:].clamp(min=1e-8))
report("W2均匀", torch.ones_like(w_ch))
report("W3log", 1.0 / (1.0 + torch.log1p(a_tar[:, 1:].clamp(min=1e-8))))
w4 = w_norm.clone(); w4[:, 0] = 0.0
report("W4去DC", w4)
# W5: per-band 归一（band 内均匀）
gid = (torch.arange(1, D, device=device) % W_FFT)
edges = torch.linspace(0, W_FFT, 6, device=device)
gidb = torch.bucketize(gid.float(), edges[1:-1].float()).clamp(0, 4)
w5 = torch.zeros_like(w_ch)
for g in range(5):
    msk = (gidb == g)
    if bool(msk.any()):
        w5[:, msk] = 1.0
report("W5band内", w5)
print("  ★ 判读：DC压力/非DC 比值越小越好（现状 DC 独占会把高频梯度淹没）；")
print("    loss 量级应在 0.1~5（过大会压垮其余 loss，见 phabs2k 教训）。")
print("    饱和率越低越好（clamp 饱和 = 梯度推不动）。")
