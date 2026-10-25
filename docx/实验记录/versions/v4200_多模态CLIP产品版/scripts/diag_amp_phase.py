# -*- coding: utf-8 -*-
"""决定性诊断：空间域噪声到底来自**振幅**还是**相位**？（2026-10-18）

【背景】A/B 频谱 loss 已把模型振幅谱形状治对（训练 hi/lo比 0.21、
守卫归零、spec 1.08），空间域 sobel 也从 28.31 回到 6.09（素材 6.97），
**但 PSNR 仍是 13.5 ≈ 噪声基线 13.03，空间域 hilo 反而 6.22（素材 0.01）**。
→ 振幅治好了，图还是噪声 → **噪声主因不在振幅**。

【最强嫌疑：相位】训练日志 phase loss 从 step1 到 step100 完全不动
（2.7653 → 2.7481 → 2.7599），wcos 恒定 ~0.22 → 相位基本没学到，
一直是随机游走。随机相位 × 正确振幅 = 噪声图。

【判决实验：四路交叉组合】把模型的 a / th 与 target 的 a / th 两两组合，
各自解码，看 PSNR 落在哪：
 ①模型a  + 模型th   = 当前生成            （基线）
 ②模型a  + target th                   →若好，说明振幅可用、**相位是主凶**
 ③target a + 模型th                    → 若差，说明模型相位是纯噪声
 ④target a + target th                 = codec 天花板（对照）
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

import numpy as np
import torch

from codec.gif_codec import (encode_gif_to_field, decode_field_to_frames,
                             load_gif_frames, resample_frames)
from model.aether_gif_model import AetherGIFModel
from physics.kuramoto_sync import build_spacetime_index
from data.clip_condition import CLIP_EMB_DIM

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "8号球"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "表情包"),
]
gif_dir = ROOT / "data" / "gif_raw" / "noto"
NEED = TS * WIN

TAG = os.environ.get("DIAG_TAG") or "ab2_probe_150"
_FFN_H = int(os.environ.get("FFN_HIDDEN", "512"))


def psnr(a, b):
    return 10 * math.log10(255.0 ** 2
                           / max(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(), 1e-12))


def ac1(x):
    a = x - x.mean()
    return float((a[..., :-1] * a[..., 1:]).sum() / max((a * a).sum(), 1e-12))


def sobel(x):
    g = x.mean(axis=-1)
    gx = np.abs(np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) / 2
    gy = np.abs(np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0)) / 2
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def dec(a, th):
    z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    return np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


fields, a_tars, th_tars, refs = [], [], [], []
for name, _ in GIF_SPECS:
    gp = gif_dir / name
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D)
    fields.append(f.to(device))
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12).to(device))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]).to(device))
    raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
    refs.append(resample_frames(raw, NEED).astype(np.float32))

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=_FFN_H, init_mode="init",
                       n_modules=8, mod_gamma=0.15).to(device)
sd = torch.load(ROOT / "outputs_real" / TAG / "cond3_ckpt.pt", map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()

cc = ROOT / "data" / "clip_cache"
text3 = np.load(cc / "text3.npy")
first3 = np.load(cc / "first3.npy")
gif3v = np.load(cc / "gif3_video.npy")
cv_full = np.concatenate([text3, first3, gif3v, np.ones((2, 3), np.float32)], 1)
t, s = build_spacetime_index(TS, GRID, GRID, device)

print("=" * 96)
print(f"[振幅/相位归因] {TAG}")
print("=" * 96)
print(f"{'素材':<6}{'组合':<26}{'PSNR':>9}{'ac1':>9}{'sobel':>9}")
print("-" * 96)

with torch.inference_mode():
    for i in range(2):
        cv = torch.from_numpy(cv_full[i:i + 1]).to(device)
        a_m, th_m, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=cv, use_cond_field=True)
        a_t = a_tars[i].unsqueeze(0)
        th_t = th_tars[i].unsqueeze(0)
        ref = refs[i]

        # 振幅层面直接对比
        rel = (a_m / a_t.clamp(min=1e-8))
        print(f"  [素材{i}振幅诊断] a_pred/a_tar: "
              f"DC列均值={rel[0,:,0].mean():.4f} "
              f"w1-5均值={rel[0,:,1:6].mean():.4f} "
              f"w6+均值={rel[0,:,6:].mean():.4f}")
        # 相位层面
        pe = torch.remainder(th_m - th_t + math.pi, 2 * math.pi) - math.pi
        wph = (a_t ** 2).clamp(min=1e-8)
        wcos = float((wph * torch.cos(pe)).sum() / wph.sum())

        cases = [
            ("①模型a + 模型th（当前）", a_m, th_m),
            ("②模型a + target th", a_m, th_t),
            ("③target a + 模型th", a_t, th_m),
            ("④target a + target th（天花板）", a_t, th_t),
        ]
        for nm, aa, tt in cases:
            img = dec(aa, tt)
            print(f"{'素材'+str(i):<6}{nm:<26}{psnr(img, ref):>9.2f}"
                  f"{ac1(img):>+9.3f}{sobel(img):>9.2f}")
        print(f"       相位 wcos={wcos:+.4f}  平均相位误差={pe.abs().mean().item():.4f} rad")
        print("-" * 96)
print("  判读：若②远高于① → **相位是主凶**；若③接近噪声 → 模型相位确为随机。")
print("=" * 96)