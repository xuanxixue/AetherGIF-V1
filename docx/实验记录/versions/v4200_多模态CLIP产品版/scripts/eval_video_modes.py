# -*- coding: utf-8 -*-
"""正确口径的条件评测：7 种条件全测，视频模式注入 3D-FFT 旁路。

修 train_guided.py 评测段的两个口径错：
  1. 视频向量模式（GIF本身/文+GIF/图+GIF/三路全给）应同时走 3D-FFT 旁路
     （模型内 _fft_cond_cache 通路）——视频在手，调用方算得出源 3D-FFT。
     纯生成模式（文生/图生/图文生）无视频 → 无旁路 → 真生成。
  2. 7 种条件组合全测（原评测漏了 文+GIF / 图+GIF）。

分组判据：
  视频再生组 PSNR 应显著 > 纯生成组（旁路携带结构信息）；
  若两组几乎相同 → 旁路信息没被用上（动力学瓶颈，与第三十七死穴一致）。
顺带量化「颜色不明显」：帧均值/饱和度生成 vs 素材。
"""
import os, sys, math
from pathlib import Path
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
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception: pass
os.environ.setdefault("COND_GROUPS", "5"); os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan"); os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0"); os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1"); os.environ.setdefault("IDEAL_CLAMP", str(math.pi))
os.environ.setdefault("COND_INIT_PC", "1"); os.environ.setdefault("COND_PC_RANK", "8")
import numpy as np, torch
try:
    from torch_accel_shim import install_full; install_full()
except Exception as ex: print(f"[WARN] {ex}")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from codec.gif_codec import (load_gif_frames, decode_field_to_frames,
                              resample_frames, encode_gif_to_field)
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1
device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = Path(os.environ.get("CKPT", str(ROOT / "outputs_real" / "guid4_long" / "guided_ckpt.pt")))
print("=" * 86)
print(f"[7 条件正确口径评测] ckpt = {CKPT.parent.name}/{CKPT.name}")

CLIP_CACHE = ROOT / "data" / "clip_cache"
text3 = np.load(CLIP_CACHE / "text3.npy")
first3 = np.load(CLIP_CACHE / "first3.npy")
gif3v = np.load(CLIP_CACHE / "gif3_video.npy")
GIF_SPECS = [("8-ball_1f3b1.gif", "8号球图标"),
             ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "前辈前辈表情包")]
gif_dir = ROOT / "data" / "gif_raw" / "noto"
fields, raw_imgs = [], []
for name, _ in GIF_SPECS:
    gp = gif_dir / name
    if not gp.exists():
        gp = ROOT.parent.parent / "data" / "gif_raw" / "noto" / name
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    _raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
    raw_imgs.append(resample_frames(_raw, TS * WIN).astype(np.float32))

target = torch.stack(fields, 0)

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd, strict=False); model.eval()
t, s = build_spacetime_index(TS, GRID, GRID, device)


def build_cv(src_i, has):
    z = np.zeros((1, 2048), dtype=np.float32)
    return build_cond_vector3(
        text3[src_i:src_i + 1] if has[0] else z,
        first3[src_i:src_i + 1] if has[1] else z,
        gif3v[src_i:src_i + 1] if has[2] else z, has).astype(np.float32)


# 7 种条件：name, has(文/图/视频)
COMBOS = [
    ("文生",   (1, 0, 0), False),
    ("图生",   (0, 1, 0), False),
    ("图文生", (1, 1, 0), False),
    ("GIF本身",  (0, 0, 1), True),
    ("文+GIF", (1, 0, 1), True),
    ("图+GIF", (0, 1, 1), True),
    ("三路全给", (1, 1, 1), True),
]


def psnr(a_img, b_img):
    mse = ((a_img.astype(np.float32) - b_img.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def sat(x):  # 饱和度近似：每像素 max-min 通道差均值
    return float((x.max(-1) - x.min(-1)).mean())


print("-" * 86)
print(f"{'条件':<8}{'模式':<10}{'src':<5}{'vs_A':>8}{'vs_B':>8}{'选对':<7}"
      f"{'wcos_c':>8}{'帧均值':>8}{'帧饱和':>8}{'噪声基线':>9}")
print("-" * 86)
stats = {}
with torch.inference_mode():
    for name, has, use_bypass in COMBOS:
        for src_i in range(2):
            cv = torch.from_numpy(build_cv(src_i, has)).to(device)
            # ★ 视频模式注入源 3D-FFT 旁路；纯生成模式不注入（cache 置空）
            if use_bypass:
                model._fft_cond_cache = target[src_i:src_i + 1]
            else:
                model._fft_cond_cache = None
            a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=cv, use_cond_field=True)
            z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
            fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
            gen = np.stack(decode_field_to_frames(
                fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
            ps = [psnr(gen, raw_imgs[k]) for k in range(2)]
            # wcos_c（content 层）
            a_t = torch.sqrt(target[src_i, ..., :D] ** 2 + target[src_i, ..., D:] ** 2 + 1e-12).cpu()
            th_t = torch.atan2(target[src_i, ..., D:], target[src_i, ..., :D]).cpu()
            pe = torch.remainder(th[0].cpu() - th_t + math.pi, 2 * math.pi) - math.pi
            wp = (a_t ** 2).clamp(min=1e-8)
            mask = torch.arange(D) % W_FFT == 0
            wc = float((wp * torch.cos(pe))[:, mask].sum() / wp[:, mask].sum())
            ref = raw_imgs[src_i]
            won = "✓" if ps[src_i] > ps[1 - src_i] else "✗"
            mode = "视频再生" if use_bypass else "纯生成"
            print(f"{name:<8}{mode:<10}src{src_i:<3}{ps[0]:>7.2f}dB{ps[1]:>7.2f}dB  {won:<6}"
                  f"{wc:>+8.3f}{gen.mean():>8.1f}{sat(gen):>8.1f}{psnr(gen, ref):>8.2f}dB")
            stats.setdefault(mode, []).append((name, src_i, ps[src_i], wc))
print("-" * 86)
# 素材自身的颜色基准
for i, r in enumerate(raw_imgs):
    print(f"[素材基准 src{i}] 帧均值={r.mean():.1f}  饱和度={sat(r):.1f}")
print()
for mode, lst in stats.items():
    avg = sum(x[2] for x in lst) / len(lst)
    print(f"{mode} 组平均 PSNR(自素材) = {avg:.2f} dB  ({len(lst)} 个样本)")
vg = sum(x[2] for x in stats.get("视频再生", [])) / max(len(stats.get("视频再生", [])), 1)
pg = sum(x[2] for x in stats.get("纯生成", [])) / max(len(stats.get("纯生成", [])), 1)
print(f"\n★ 视频再生({vg:.2f}) − 纯生成({pg:.2f}) = {vg - pg:+.2f} dB")
print("判读：差值显著为正 → 旁路信息被用上（设计语义成立）；≈0 → 旁路白给"
      "（动力学瓶颈依旧，与第三十七死穴一致）。")
print("=" * 86)
