# -*- coding: utf-8 -*-
"""teacher 模式重建天花板探针 —— 判断 wcos≈0.5 的结构性瓶颈对应多差的质量。

【问题】
fix_r3/r4 训练到 teacher wcos≈+0.50~0.51（训练 keep 步），但**条件评测**
（target_field=None）只有 10.35 dB ≈ 噪声。两个混杂的问题：
  ① 训练/推理口径错配（dropout 治这个）；
  ② 结构天花板：fft_bypass 与 cond_init **同为 G=5 per-group 输出**
     （125 通道 → 5 组相位/幅度），teacher 通路自己就被 25× 瓶颈卡住。
本探针只测 ②：给 fix_r4 ckpt 喂 target_field（teacher 全开），
看它重建出什么质量。

【判据】
- 若 teacher 重建 PSNR 远高于 codec 天花板（51.7dB）以下但仍可观
  （比如 20+ dB）→ 天花板没顶死，dropout 有意义；
- 若 teacher 重建也只有 ~13dB（≈噪声）→ **瓶颈在容量**，
  dropout 再怎么训也到不了「3D-FFT 还原效果」，必须扩容（per-channel 头）。

【运行】
cd v4200 根目录
  C:/Python312/python.exe -u scripts/diag_teacher_ceiling.py [CKPT]
默认 CKPT=outputs_real/fix_r4/cond3_ckpt.pt
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
_VENV = _PROJ / ".venv" / "Lib" / "site-packages"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
if _VENV.is_dir():
    sys.path.append(str(_VENV))
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass

try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] torch_accel_shim 加载失败：{ex}")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ★ 与 run_sep_fix.cmd 同口径（import 前设置）
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")
os.environ.setdefault("IDEAL_CLAMP", str(math.pi))

import numpy as np
import torch

from codec.gif_codec import (encode_gif_to_field, decode_field_to_frames,
                              load_gif_frames, resample_frames)
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1
_CONTENT_MASK = torch.arange(D) % W_FFT == 0
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = sys.argv[1] if len(sys.argv) > 1 else "outputs_real/fix_r4/cond3_ckpt.pt"

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "8号球图标"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "前辈前辈表情包"),
]


def psnr(a_img, b_img):
    mse = ((a_img.astype(np.float32) - b_img.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def ac1(x):
    a = x - x.mean()
    return float((a[..., :-1] * a[..., 1:]).sum() / max((a * a).sum(), 1e-12))


def sobel_mag(x):
    g = x.mean(axis=-1)
    gx = np.abs(np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) / 2
    gy = np.abs(np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0)) / 2
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def hilo(x):
    g = x.mean(axis=-1)
    F = np.abs(np.fft.fftshift(np.fft.fft2(
        g - g.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
    H, W_ = F.shape[1], F.shape[2]
    yy, xx = np.mgrid[0:H, 0:W_]
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W_ // 2) ** 2)
    return float(F[:, r > H * 0.35].sum() / max(F[:, r < H * 0.15].sum(), 1e-12))


print("=" * 78)
print(f"[teacher 天花板探针] ckpt = {CKPT}")
gif_dir = ROOT / "data" / "gif_raw" / "noto"
fields, a_tars, th_tars, raw_imgs = [], [], [], []
for name, _ in GIF_SPECS:
    gp = gif_dir / name
    if not gp.exists():
        alt = ROOT.parent.parent / "data" / "gif_raw" / "noto" / name
        gp = alt if alt.exists() else None
    if gp is None or not gp.exists():
        raise FileNotFoundError(name)
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]))
    _raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
    raw_imgs.append(resample_frames(_raw, TS * WIN).astype(np.float32))

target = torch.stack(fields, 0)
a_tar = torch.stack(a_tars, 0)
th_tar = torch.stack(th_tars, 0)
t, s = build_spacetime_index(TS, GRID, GRID, device)

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=256, init_mode="init",
                       n_modules=8, mod_gamma=0.15).to(device)
_rp = (ROOT / CKPT) if not Path(CKPT).is_absolute() else Path(CKPT)
if not _rp.exists():
    raise FileNotFoundError(_rp)
_rsd = torch.load(_rp, map_location=device, weights_only=True)
_rsd = {k: v for k, v in _rsd.items() if not k.startswith("qwen.")}
_rret = model.load_state_dict(_rsd, strict=False)
print(f"[LOAD] missing={len(_rret.missing_keys)} unexpected={len(_rret.unexpected_keys)}")
model.eval()

# ★ 条件向量（与训练 keep 步同口径：cond_vec 必须给，否则 fft_bypass/cond_init
#   整个分支都不执行，测的就不是 teacher 模式了）
CLIP_CACHE = ROOT / "data" / "clip_cache"
cv_full = np.load(CLIP_CACHE / "cv3_full.npy")   # [2, N_COND3]
assert cv_full.shape[1] == 6147

OUT = ROOT / "outputs_real" / "teacher_ceil"
OUT.mkdir(parents=True, exist_ok=True)
from PIL import Image

print("\n两种前向口径 × 两素材（全部 no_grad，cond_vec=三路全给）：")
print("  teacher = target_field 给定（fft 旁路 + cond_init 全开 = 训练 keep 步口径）")
print("  cond    = target_field=None（纯条件 = 评测口径）")
print("-" * 96)
print(f"{'口径':<8}{'素材':<6}{'PSNR':>8}{'wcos':>9}{'wcos_c':>9}{'ac1':>8}{'sobel':>8}{'hilo':>8}{'素材hilo':>9}")
print("-" * 96)
rows = {}
with torch.inference_mode():
    for mode in ("teacher", "cond"):
        for i in range(2):
            one_cv = torch.from_numpy(cv_full[i:i + 1]).to(device)
            a, th, _ = model(texts=None,
                             target_field=(target[i:i + 1] if mode == "teacher" else None),
                             t_coord=t, s_coord=s, batch_size=1,
                             cond_vec=one_cv, use_cond_field=True)
            z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
            fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
            gen = np.stack(decode_field_to_frames(
                fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
            _pe = torch.remainder(th[0].cpu() - th_tar[i].cpu() + math.pi,
                                  2 * math.pi) - math.pi
            _wp = (a_tar[i].cpu() ** 2).clamp(min=1e-8)
            wcos = float((_wp * torch.cos(_pe)).sum() / _wp.sum())
            wcos_c = float((_wp * torch.cos(_pe))[..., _CONTENT_MASK].sum()
                           / _wp[..., _CONTENT_MASK].sum())
            ps = psnr(gen, raw_imgs[i])
            rows[(mode, i)] = (ps, wcos, wcos_c, ac1(gen), sobel_mag(gen), hilo(gen))
            # 存 GIF（teacher 重建 = 这架构能给出的最好重建）
            if mode == "teacher":
                pil = [Image.fromarray(np.ascontiguousarray(f)) for f in gen.astype(np.uint8)]
                pil[0].save(OUT / f"teacher_src{i}.gif", save_all=True,
                            append_images=pil[1:], loop=0, duration=100)
            print(f"{mode:<8}{i:<6}{ps:>8.2f}{wcos:>+9.4f}{wcos_c:>+9.4f}"
                  f"{ac1(gen):>+8.3f}{sobel_mag(gen):>8.2f}{hilo(gen):>8.2f}"
                  f"{hilo(raw_imgs[i]):>9.2f}")
print("-" * 96)
print("\n素材基准：PSNR=∞（自己）；噪声基线见下")
for i in range(2):
    rng = np.random.default_rng(0)
    ref = raw_imgs[i]
    n = np.clip(rng.normal(ref.mean(), ref.std(), ref.shape), 0, 255).astype(np.float32)
    print(f"  素材{i} 噪声基线 PSNR = {psnr(n, ref):.2f} dB（低于此 = 图就是噪声）")
print(f"\n判读：teacher PSNR ≫ 噪声基线且可观（>20dB）→ 天花板没顶死，dropout 有意义；")
print(f"      teacher PSNR ≈ 噪声 → 容量瓶颈，必须 per-channel 扩容（500M 预算充足）。")
print(f"GIF -> {OUT}")
print("=" * 78)
