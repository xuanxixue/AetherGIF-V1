# -*- coding: utf-8 -*-
"""纯物理栈保持力探针（2026-10-22）。

在 guid4 收敛 ckpt 上，把初态**强制成目标场本身**（force_init=(a_tar, θ_tar)），
且**关闭一切条件通路**（cond_vec=None → cond_init 不跑、fft 旁路不跑），
target_field=None → 漂移场也不拿答案。

判读：
  输出 ≈ 目标（PSNR 高、wcos≈1）→ 12 层动力学是结构保持的，
      瓶颈在「条件头能否生成正确初态」（→ 容量/低秩问题）；
  输出被搅散（PSNR≈噪声）→ 动力学本身摧毁结构（→ 架构问题）。
老版 diag_ident 的 L1 不纯：use_cond_field=True 时条件头的 dθ/dlna 会
污染被强制的初态，本探针排除该混杂。
"""
import os
import sys
import math
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
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")
os.environ.setdefault("IDEAL_CLAMP", str(math.pi))
# 与 guid4 训练同口径：PC 头（仅影响模块形状，本探针不用它出力）
os.environ.setdefault("COND_INIT_PC", "1")
os.environ.setdefault("COND_PC_RANK", "8")

import numpy as np
import torch

try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] torch_accel_shim 加载失败：{ex}")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = Path(os.environ.get("CKPT", str(ROOT / "outputs_real" / "guid4_long" / "guided_ckpt.pt")))
GIF_NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]

print("=" * 74)
print(f"[纯物理栈保持力] ckpt = {CKPT.parent.name}/{CKPT.name}  device={device}")

gif_dir = ROOT / "data" / "gif_raw" / "noto"


def psnr_img(a_img, b_img):
    mse = ((a_img.astype(np.float32) - b_img.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def wcos_of(a, th, a_t, th_t):
    """与 teacher 探针同口径的频谱包络余弦（log 幅度，对 N 求均值）。"""
    la = torch.log1p(a[0]); lt = torch.log1p(a_t[0])
    la = la - la.mean(0, keepdim=True); lt = lt - lt.mean(0, keepdim=True)
    return float((la * lt).sum() / (la.norm() * lt.norm() + 1e-12))


model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(CKPT, map_location=device)
_r = model.load_state_dict(sd, strict=False)
print(f"[LOAD] missing={len(_r.missing_keys)} unexpected={len(_r.unexpected_keys)}")
model.eval()
t, s = build_spacetime_index(TS, GRID, GRID, device)

for gi, gn in enumerate(GIF_NAMES):
    gp = gif_dir / gn
    if not gp.exists():
        gp = ROOT.parent.parent / "data" / "gif_raw" / "noto" / gn
    f_tar = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    a_tar = torch.sqrt(f_tar[..., :D] ** 2 + f_tar[..., D:] ** 2 + 1e-12)
    th_tar = torch.atan2(f_tar[..., D:], f_tar[..., :D])
    if a_tar.dim() == 2:
        a_tar, th_tar = a_tar.unsqueeze(0), th_tar.unsqueeze(0)

    # 目标帧（参考）
    z = (a_tar[0] * torch.exp(1j * th_tar[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    fr_tar = np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)

    with torch.inference_mode():
        # 纯动力学：初态=目标，cond 全关，无 target
        a_out, th_out, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                                 batch_size=1, cond_vec=None, use_cond_field=False,
                                 force_init=(a_tar, th_tar))
    zo = (a_out[0] * torch.exp(1j * th_out[0])).to(torch.complex64)
    fldo = torch.cat([zo.real, zo.imag], dim=-1).float().cpu()
    fr_out = np.stack(decode_field_to_frames(
        fldo, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
    n = min(len(fr_out), len(fr_tar))
    p = psnr_img(fr_out[:n], fr_tar[:n])
    wc = wcos_of(a_out, th_out, a_tar, th_tar)
    # 也测恒等基线（不经模型）
    p0 = psnr_img(fr_tar[:n], fr_tar[:n])
    print(f"[src{gi} {gn}]  纯物理栈保持: PSNR = {p:6.2f} dB   wcos = {wc:+.4f}")
    print(f"          幅度 std: 目标 {a_tar.std():.4f} → 输出 {a_out.std():.4f}"
          f"   相位 |Δ| 均值 = {(th_out-th_tar).abs().mean():.4f} rad")

print("=" * 74)
print("判读：PSNR ≫ 噪声基线(13.03/7.44) 且 wcos≈1 → 动力学保持结构，瓶颈在条件头容量；")
print("      PSNR ≈ 噪声或 wcos 崩 → 动力学自身摧毁结构，须改架构。")
