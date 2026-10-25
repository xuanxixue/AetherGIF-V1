# -*- coding: utf-8 -*-
"""fft4k 中途预览：从某个 ckpt 快照出一张 GIF，全程 CPU，不碰训练。

口径与 overfit_cond3.py 的 eval 段完全一致：
  - 模型构造 d_model=125 / n_blocks=12 / ffn_hidden=256 / init_mode=init
    / n_modules=8 / mod_gamma=0.15
  - 前向 model(texts=None, target_field=None, t_coord, s_coord,
              batch_size=1, cond_vec=..., use_cond_field=True)
  - cond_vec 直接用 data/clip_cache/cv3_*.npy（eval 同源缓存）
  - 视觉 GIF 用 normalize=True；指标用 normalize=False 对齐 raw 素材

用法：
  python scripts/preview_fft4k.py                      # 默认 full x src0
  PREV_CKPT=outputs_real/fft4k/cond3_ckpt_step600.pt python scripts/preview_fft4k.py
  PREV_COND=text PREV_SRC=0 python scripts/preview_fft4k.py

★ 强制 CPU：CUDA_VISIBLE_DEVICES=-1 在 import torch 之前生效，
  进程根本看不到显卡，物理上不可能与后台训练抢占显存。
★ 只读 step%600 的独立快照，不读每 100 步覆盖的主 ckpt（避免读写竞态）。
"""
import os

# ---- 必须在 import torch 之前 ----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # 强制 CPU，零干扰训练
os.environ.setdefault("FFT_FFTW", "1")        # 与训练一致：FFT 旁路开启
os.environ.setdefault("KAPPA_MAX", "6.0")
# ★ 与训练命令完全一致的条件通路环境变量（ckpt 形状依赖）
#   不设会 size mismatch：to_phase.1.weight [30720,16] vs [6144,16]
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import sys
import math
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ★ 与 overfit_cond3.py 完全一致的路径逻辑（deps312/.venv/盘根上溯）
#   C:\Python312 自带的 transformers 损坏（缺 utils.versions），
#   必须靠这里的上溯把可用的 transformers 顶到 sys.path 前面。
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

# ★ torch_accel_shim：训练脚本同款，patch transformers/torch 兼容性
try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] torch_accel_shim 加载失败：{ex}")

import numpy as np
import torch
from PIL import Image

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames, load_gif_frames, resample_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

# ---------------- 编解码口径（与 overfit_cond3.py 一致） ----------------
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1  # 25

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "8号球"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "前辈前辈"),
]
CKPT = Path(os.environ.get("PREV_CKPT",
           str(ROOT / "outputs_real" / "fft4k" / "cond3_ckpt_step600.pt")))
OUT = ROOT / "outputs_real" / "fft4k" / "preview"
OUT.mkdir(parents=True, exist_ok=True)

# 五种条件与 eval 段同源（overfit_cond3.py 的 CONDS 表）
_CV_MAP = {
    "full": ("三路全给", "cv3_full.npy"),
    "text": ("文生", "cv3_text.npy"),
    "first": ("图生", "cv3_first.npy"),
    "gif": ("GIF本身", "cv3_gif.npy"),
    "tf": ("图文生", "cv3_tf.npy"),
}
COND = os.environ.get("PREV_COND", "full")
SRC = int(os.environ.get("PREV_SRC", "0"))
assert COND in _CV_MAP, f"PREV_COND 只支持 {list(_CV_MAP)}"
assert SRC in (0, 1)

CLIP_CACHE = ROOT / "data" / "clip_cache"


def psnr(a_img, b_img):
    mse = ((a_img.astype(np.float32) - b_img.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def ac1(x):
    a = x - x.mean()
    return float((a[..., :-1] * a[..., 1:]).sum() / max((a * a).sum(), 1e-12))


def hilo(x):
    g = x.mean(axis=-1)
    F = np.abs(np.fft.fftshift(np.fft.fft2(
        g - g.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
    H, W_ = F.shape[1], F.shape[2]
    yy, xx = np.mgrid[0:H, 0:W_]
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W_ // 2) ** 2)
    return float(F[:, r > H * 0.35].sum() / max(F[:, r < H * 0.15].sum(), 1e-12))


def main():
    torch.set_num_threads(4)
    device = "cpu"
    print(f"[PREVIEW] device={device}  ckpt={CKPT.name}  cond={COND}  src={SRC}")

    # ---------- 载入 ckpt（口径与 RESUME 段完全一致） ----------
    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    rsd = torch.load(CKPT, map_location="cpu")
    rsd = {k: v for k, v in rsd.items() if not k.startswith("qwen.")}

    # ---------- 构造模型（与训练同参数） ----------
    model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                           ffn_hidden=int(os.environ.get("FFN_HIDDEN", "256")),
                           init_mode=os.environ.get("INIT_MODE", "init"),
                           n_modules=int(os.environ.get("N_MODULES", "8")),
                           mod_gamma=float(os.environ.get("MOD_GAMMA", "0.15"))).to(device)
    rret = model.load_state_dict(rsd, strict=False)
    print(f"[CKPT] missing={len(rret.missing_keys)} unexpected={len(rret.unexpected_keys)}")
    if rret.missing_keys:
        print(f"      missing keys: {rret.missing_keys[:8]}{'...' if len(rret.missing_keys) > 8 else ''}")
    model.eval()

    # ---------- 素材 ----------
    raw_imgs = []
    for name, tag in GIF_SPECS:
        gp = ROOT / "data" / "gif_raw" / "noto" / name
        _raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
        raw_imgs.append(resample_frames(_raw, TS * WIN).astype(np.float32))
        # 顺手存一份素材对照 GIF
        _arr = np.clip(raw_imgs[-1], 0, 255).astype(np.uint8)
        _pil = [Image.fromarray(np.ascontiguousarray(f)) for f in _arr]
        _pil[0].save(OUT / f"target_{tag}.gif", save_all=True,
                     append_images=_pil[1:], loop=0, duration=100)

    # ---------- 条件向量（eval 同源缓存） ----------
    _name, _cvf = _CV_MAP[COND]
    cv = np.load(CLIP_CACHE / _cvf)
    one = torch.from_numpy(cv[SRC:SRC + 1]).to(device)

    # ---------- 前向（同 eval 段） ----------
    t, s = build_spacetime_index(TS, GRID, GRID, device)
    with torch.inference_mode():
        a, th, diag = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                            batch_size=1, cond_vec=one, use_cond_field=True)

    # ---------- 视觉 GIF（normalize=True，可看） ----------
    z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    frames = np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=True)).astype(np.uint8)
    pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    gif_path = OUT / f"preview_{COND}_src{SRC}.gif"
    pil[0].save(gif_path, save_all=True, append_images=pil[1:], loop=0, duration=100)

    # ---------- 指标（normalize=False 对齐 raw，与训练验收同口径） ----------
    gen = np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
    tag = GIF_SPECS[SRC][1]
    ps_self = psnr(gen, raw_imgs[SRC])
    ps_other = psnr(gen, raw_imgs[1 - SRC])
    # 噪声基线（同 std 高斯），PSNR 低于它就是噪声
    rng = np.random.default_rng(0)
    ref = raw_imgs[SRC]
    nb = psnr(np.clip(rng.normal(ref.mean(), ref.std(), ref.shape), 0, 255).astype(np.float32), ref)

    print(f"\n[结果] {_name}@src{SRC}（{tag}）")
    print(f"  vs_自己素材 PSNR = {ps_self:6.2f} dB")
    print(f"  vs_另一素材 PSNR = {ps_other:6.2f} dB   （>自己 = 认对人）")
    print(f"  噪声基线（同std） = {nb:6.2f} dB   ← 低于它就是噪声")
    print(f"  ac1 = {ac1(gen):.3f}（真实图像 >0.5，纯噪声 ≈0）")
    print(f"  hilo = {hilo(gen):.3f}（素材 0.01~0.05，白噪声 ≈8）")
    print(f"\n  GIF: {gif_path}")
    print(f"  素材对照: {OUT / f'target_{tag}.gif'}")


if __name__ == "__main__":
    main()
