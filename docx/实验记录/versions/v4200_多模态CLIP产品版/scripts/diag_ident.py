# -*- coding: utf-8 -*-
"""恒等测试：12 层物理栈对「正确初态」的保持能力（分叉诊断）。

三路对比（同一 ckpt、同一素材 src0、三路全给条件）：
  L0 恒等基线  ：a0=素材场，**不经模型**，直接 decode（≈ codec 往返 51 dB）
  L1 恒等初态  ：force_init=(a_tar, th_tar) → 12 层演化 → 输出
                 ★ 高分 = 物理栈能保持，锅在初态（条件通路重建初态能力不足）
                 ★ 低分 = 物理栈把「已经正确的场」搅散 = 结构性问题
  L2 普通重建  ：随机初态 + target_field（已知 ~13.9 dB）
  L3 生成      ：随机初态 + 无 target

顺带分频段看 L1 演化后 h/w 能量分布是否被摊平。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
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
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")
os.environ.setdefault("COND_LOWRANK", "16")
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = Path(os.environ.get("CKPT", str(ROOT / "outputs_real" / "fix_hspec_150" / "cond3_ckpt.pt")))
GIF_NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]
print("=" * 74)
print(f"CKPT={CKPT.parent.name}/{CKPT.name}  device={device}")

gif_dir = ROOT / "data" / "gif_raw" / "noto"
gp = gif_dir / GIF_NAMES[0]
if not gp.exists():
    gp = ROOT.parent.parent / "data" / "gif_raw" / "noto" / GIF_NAMES[0]
f_tar = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
a_tar = torch.sqrt(f_tar[..., :D] ** 2 + f_tar[..., D:] ** 2 + 1e-12)
th_tar = torch.atan2(f_tar[..., D:], f_tar[..., :D])
if a_tar.dim() == 2:
    a_tar, th_tar = a_tar.unsqueeze(0), th_tar.unsqueeze(0)

CLIP_CACHE = ROOT / "data" / "clip_cache"
text3 = np.load(CLIP_CACHE / "text3.npy")
first3 = np.load(CLIP_CACHE / "first3.npy")
gif3v = np.load(CLIP_CACHE / "gif3_video.npy")
cv = build_cond_vector3(text3[0:1], first3[0:1], gif3v[0:1], (1, 1, 1))
one_cv = torch.from_numpy(cv).to(device)

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()
t, s = build_spacetime_index(TS, GRID, GRID, device)


def decode_frames(a, th):
    z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    return np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


def psnr_to(ref):
    def _p(fr):
        n = min(len(fr), len(ref))
        mse = ((fr[:n] - ref[:n]) ** 2).mean()
        return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))
    return _p


def hprof(a):
    """[1,N,125] -> 逐 h 能量占比（对 N 求均值的能量）"""
    e = a[0].pow(2).reshape(-1, WIN, W_FFT).mean(0)      # [5,25]
    p = e.sum(1)
    return (p / p.sum()).cpu().numpy()


fr_tar = decode_frames(a_tar, th_tar)
print(f"L0 恒等基线（不经模型）: 帧数={len(fr_tar)}  PSNR=∞（自洽）")

runs = {}
with torch.inference_mode():
    # L1 恒等初态：force_init 绕开随机初态
    a1, th1, _ = model(texts=None, target_field=f_tar, t_coord=t, s_coord=s,
                       batch_size=1, cond_vec=one_cv, use_cond_field=True,
                       force_init=(a_tar, th_tar))
    runs["L1 恒等初态(经12层)"] = (a1, th1)
    # L2 普通重建
    a2, th2, _ = model(texts=None, target_field=f_tar, t_coord=t, s_coord=s,
                       batch_size=1, cond_vec=one_cv, use_cond_field=True)
    runs["L2 普通重建"] = (a2, th2)
    # L3 生成
    a3, th3, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                       batch_size=1, cond_vec=one_cv, use_cond_field=True)
    runs["L3 生成"] = (a3, th3)

P = psnr_to(fr_tar)
hp_tar = hprof(a_tar)
print(f"\n素材逐h能量占比 = {np.round(hp_tar * 100, 1)}")
print("-" * 74)
print(f"{'路':<22}{'PSNR':>9}{'逐h占比':>36}")
print("-" * 74)
for nm, (a, th) in runs.items():
    fr = decode_frames(a, th)
    hp = hprof(a)
    print(f"{nm:<22}{P(fr):>8.2f}dB  {np.round(hp * 100, 1)}")
print("-" * 74)
print("判读：L1 高(>40dB) → 锅在初态（条件通路）；"
      "L1 低(<20dB) → 物理栈把正确场搅散 = 结构问题")
print("=" * 74)
