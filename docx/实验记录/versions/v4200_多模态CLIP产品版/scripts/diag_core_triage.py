# -*- coding: utf-8 -*-
"""核心三分诊断：一次性定位「生成噪声」到底卡在哪一层。

三个决定性数字：
  A. 物理栈上限   force_init=(a_tar,th_tar) 走 12 层栈 → PSNR
     高 = 栈无损，问题在条件通路；低 = 先修栈。
  B. cond_init 单独（生成口径，FFT 旁路缺席）→ wcos + PSNR
     这是生成的**真实**条件来源。低 = cond_init 没学会。
  C. FFT 旁路单独（训练时它吃 target_field=teacher）→ 分路 wcos + PSNR
     高 = teacher 扛住了训练 phase loss，抢走 cond_init 的梯度信号。

假设（待验证）：
  训练时 FFT 旁路 direct 看到 target 答案，phase loss 大部分由它贡献；
  cond_init 的梯度被「联合输出已接近 target」稀释，实际没学会表达相位场。
  生成时 target_field=None → FFT 旁路缺席 → 只剩弱 cond_init → 噪声。

用法：
  C:/Python312/python.exe -u scripts/diag_core_triage.py
  DIAG_CKPT=outputs_real/fix_r4/cond3_ckpt.pt 指定 ckpt
"""
import math
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
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
for k, v in dict(COND_GROUPS="5", COND_LOWRANK="16", PHASE_ACT="arctan",
                 RES_LIMIT="0.5", W2_NORM_MAX="3.0", KAPPA_MAX="6.0",
                 FFT_FFTW="1").items():
    os.environ.setdefault(k, v)

import numpy as np
import torch
from codec.gif_codec import (encode_gif_to_field, decode_field_to_frames,
                             load_gif_frames, resample_frames)
from model.aether_gif_model import AetherGIFModel
from physics.kuramoto_sync import build_spacetime_index
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"
gif_dir = ROOT / "data" / "gif_raw" / "noto"
cc = ROOT / "data" / "clip_cache"
CKPT = ROOT / (os.environ.get("DIAG_CKPT") or "outputs_real/fix_r4/cond3_ckpt.pt")

t, s = build_spacetime_index(TS, GRID, GRID, device)


def psnr(a, b):
    return 10 * math.log10(255.0 ** 2 / max(
        ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(), 1e-12))


def dec(a, th, is_mat=False):
    if is_mat:
        fld = torch.cat([a * torch.cos(th), a * torch.sin(th)], -1).cpu().float()
    else:
        fld = torch.cat([a[0] * torch.cos(th[0]), a[0] * torch.sin(th[0])],
                        -1).cpu().float()
    return np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


# 加载素材
fields, a_tars, th_tars, refs = [], [], [], []
for name in ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]:
    gp = gif_dir / name
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D)
    fields.append(f.to(device))
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12).to(device))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]).to(device))
    raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
    refs.append(resample_frames(raw, TS * WIN).astype(np.float32))

sd = torch.load(CKPT, map_location=device)
sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
m.load_state_dict(sd, strict=False)
m.eval()

text3 = np.load(cc / "text3.npy")
first3 = np.load(cc / "first3.npy")
gif3v = np.load(cc / "gif3_video.npy")
cv = torch.from_numpy(build_cond_vector3(text3[0:1], first3[0:1], gif3v[0:1],
                                         (1, 1, 1))).to(device)
f0 = fields[0]
a_tar = a_tars[0]
th_tar = th_tars[0]
ref = refs[0]

print("=" * 88)
print(f"  核心三分诊断  {CKPT.name}")
print("=" * 88)

# ---------- A. 物理栈上限 ----------
print("\n【A. 物理栈上限】初态=target，走 12 层栈")
with torch.no_grad():
    a_out, th_out, _ = m(texts=None, target_field=f0.unsqueeze(0), t_coord=t,
                         s_coord=s, batch_size=1,
                         force_init=(a_tar.unsqueeze(0), th_tar.unsqueeze(0)))
img = dec(a_out, th_out)
print(f"   初态已是 target，栈后 PSNR = {psnr(img, ref):.2f} dB  "
      f"（>30 无损 / >18 中等 / <18 栈在破坏）")

# ---------- B. cond_init 单独（生成口径） ----------
print("\n【B. cond_init 单独（生成口径，无 FFT 旁路）】")
with torch.no_grad():
    a0 = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
    th0 = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi
    dth, dln, iparts = m.cond_init(cv, return_parts=True, n_pos=len(t),
                                   theta0=th0, a0=a0)
    dth_D = m.cond_init.expand_to_d(dth)
    dln_D = m.cond_init.expand_to_d(dln)
    a_c = a0 * torch.exp(dln_D)
    th_c = th0 + dth_D
    # 初态直接解码（不经栈）
    img = dec(a_c, th_c)
    pe = torch.remainder(th_c - th_tar.unsqueeze(0) + math.pi, 2 * math.pi) - math.pi
    wph = (a_tar.unsqueeze(0) ** 2).clamp(min=1e-8)
    wcos = float((wph * torch.cos(pe)).sum() / wph.sum())
    print(f"   cond_init 初态 wcos={wcos:+.4f}  初态 PSNR={psnr(img, ref):.2f} dB")
    # 走栈
    a_st, th_st, _ = m(texts=None, target_field=None, t_coord=t, s_coord=s,
                       batch_size=1, cond_vec=cv, use_cond_field=True)
    img2 = dec(a_st, th_st)
    print(f"   cond_init 走栈后   PSNR={psnr(img2, ref):.2f} dB")

# ---------- C. FFT 旁路单独 ----------
print("\n【C. FFT 旁路单独（训练 teacher 口径，吃 target_field）】")
with torch.no_grad():
    fdth, fdla, _ = m.fft_bypass(f0.unsqueeze(0), return_parts=True)
    fdth_D = m.fft_bypass.expand_to_d(fdth)
    fdla_D = m.fft_bypass.expand_to_d(fdla)
    a_f = a0 * torch.exp(fdla_D)
    th_f = th0 + fdth_D
    img = dec(a_f, th_f)
    pe = torch.remainder(th_f - th_tar.unsqueeze(0) + math.pi,
                         2 * math.pi) - math.pi
    wcos = float((wph * torch.cos(pe)).sum() / wph.sum())
    print(f"   FFT 旁路初态 wcos={wcos:+.4f}  初态 PSNR={psnr(img, ref):.2f} dB")

# ---------- D. 联合（训练口径） vs 生成口径 对照 ----------
print("\n【D. 训练口径联合 = cond_init + FFT旁路】")
with torch.no_grad():
    a_s = a_c * torch.exp(fdla_D)
    th_s = th_c + fdth_D
    img = dec(a_s, th_s)
    pe = torch.remainder(th_s - th_tar.unsqueeze(0) + math.pi,
                         2 * math.pi) - math.pi
    wcos = float((wph * torch.cos(pe)).sum() / wph.sum())
    # 注意：训练时真正走的是「完整 forward」且 FFT 旁路吃 target，这里近似
    print(f"   CLIP+FFT 初态 wcos={wcos:+.4f}  初态 PSNR={psnr(img, ref):.2f} dB")

print("\n" + "=" * 88)
print("  判读：")
print("  A 高 + B 低 + C 高  → FFT 旁路(teacher)扛住训练，cond_init 没学会 → 生成噪声")
print("  A 低               → 物理栈在破坏，先修栈")
print("  C 也低             → 连 teacher 都学不动，问题在栈或旁路容量")
print("=" * 88)