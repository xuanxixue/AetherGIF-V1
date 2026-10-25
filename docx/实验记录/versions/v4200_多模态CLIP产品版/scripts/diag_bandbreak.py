# -*- coding: utf-8 -*-
"""分频段爆破诊断：定位「低频塌陷」到底塌在振幅还是相位。

三路对比（同一 ckpt、同一素材）：
  A) 素材场 a_tar/th_tar（codec encode）
  B) 生成场 a_gen/th_gen（target_field=None，三路全给条件）
  C) 重建场（target_field 前向，应 ≈ A，验证前向链路本身没坏）

诊断口径：
  - 训练口径：patch 级频谱 [N,125] -> (h=time_freq 0~4, w=spatial_freq 0~24)
    逐 (h,w) 格点：能量 E=mean_N(a^2)、未加权 cos(dθ)、a_tar^2 加权 cos
  - 图像口径：decode 后 2D FFT 径向频段能量占比（素材 E_lo 应 >95%）

解答的三个问题：
  1) 振幅通路：E_gen/E_tar 分频段 —— 低频是塌了还是高频爆了？
  2) 相位通路：DC/低频相位对齐没（加权 vs 未加权差多少）？
  3) 重建 vs 生成：前向链路本身是否忠实（C≈A 则锅在生成侧/条件侧）
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
# ★ 架构环境变量必须与训练脚本（run_sep_fix.cmd）完全一致，否则
#   to_phase/to_amp 维度对不上（COND_GROUPS=5 → out=6144×5=30720）。
#   这些只影响架构/前向，不影响训练 loss 权重。
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
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1          # 25
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = Path(os.environ.get("CKPT", str(ROOT / "outputs_real" / "fix_clamp_150" / "cond3_ckpt.pt")))
SRC = int(os.environ.get("SRC", "0"))
GIF_NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]

print("=" * 74)
print(f"CKPT={CKPT.name}  SRC={SRC}  device={device}")

# ---------------------------------------------------------------- 素材
gif_dir = ROOT / "data" / "gif_raw" / "noto"
gp = gif_dir / GIF_NAMES[SRC]
if not gp.exists():
    alt = ROOT.parent.parent / "data" / "gif_raw" / "noto" / GIF_NAMES[SRC]
    gp = alt if alt.exists() else None
if gp is None or not gp.exists():
    raise FileNotFoundError(GIF_NAMES[SRC])
f_tar = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)   # [1,N,2D]
a_tar = torch.sqrt(f_tar[..., :D] ** 2 + f_tar[..., D:] ** 2 + 1e-12)           # [N,125]
th_tar = torch.atan2(f_tar[..., D:], f_tar[..., :D])
# ★ encode_gif_to_field 返回 [N,2D]（无 batch 维），模型前向返回 [1,N,D]。
#   统一成 [1,N,125]，否则会把 D=125 误当 N、广播错位。
if a_tar.dim() == 2:
    a_tar = a_tar.unsqueeze(0)
    th_tar = th_tar.unsqueeze(0)
    f_tar = f_tar.unsqueeze(0)
Ntot = a_tar.shape[1]
print(f"素材 {GIF_NAMES[SRC]}: N={Ntot}  (a shape={tuple(a_tar.shape)})")

# ---------------------------------------------------------------- 条件（三路全给）
CLIP_CACHE = ROOT / "data" / "clip_cache"
text3 = np.load(CLIP_CACHE / "text3.npy")
first3 = np.load(CLIP_CACHE / "first3.npy")
gif3v = np.load(CLIP_CACHE / "gif3_video.npy")
cv = build_cond_vector3(text3[SRC:SRC + 1], first3[SRC:SRC + 1], gif3v[SRC:SRC + 1], (1, 1, 1))
one_cv = torch.from_numpy(cv).to(device)

# ---------------------------------------------------------------- 模型
model = AetherGifModel = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(CKPT, map_location=device)
_r = model.load_state_dict(sd, strict=False)
print(f"[load] missing={len(_r.missing_keys)} unexpected={len(_r.unexpected_keys)}")
model.eval()
t, s = build_spacetime_index(TS, GRID, GRID, device)

with torch.inference_mode():
    a_gen, th_gen, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=one_cv, use_cond_field=True)
    a_rec, th_rec, _ = model(texts=None, target_field=f_tar, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=one_cv, use_cond_field=True)

for _nm, _a in [("gen", a_gen), ("rec", a_rec)]:
    if torch.isnan(_a).any():
        print(f"[WARN] a_{_nm} 含 NaN: {torch.isnan(_a).sum().item()} 个")

# ---------------------------------------------------------------- 工具
def grids(a, th):
    """[1,N,125] -> E[5,25], TH[5,25](相位差均值汇总用)"""
    A = a[0].reshape(-1, WIN, W_FFT)
    TH = th[0].reshape(-1, WIN, W_FFT)
    E = A.pow(2).mean(0)                                   # [5,25] 每格点平均能量
    return E.cpu().numpy(), TH

E_tar, TH_tar = grids(a_tar, th_tar)
E_gen, TH_gen = grids(a_gen, th_gen)
E_rec, TH_rec = grids(a_rec, th_rec)

def phase_cos(a_ref, th_ref, th_x):
    """未加权 cos(dθ) 与 a_ref^2 加权 cos(dθ)，逐 (h,w)"""
    pe = torch.remainder(th_x - th_ref + math.pi, 2 * math.pi) - math.pi     # [1,N,125]
    pe = pe[0].reshape(-1, WIN, W_FFT)
    w = a_ref[0].reshape(-1, WIN, W_FFT).pow(2).clamp(min=1e-12)
    cos_u = torch.cos(pe).mean(0)                                            # 未加权
    cos_w = (w * torch.cos(pe)).sum(0) / w.sum(0)                            # 能量加权
    cos_w = cos_w.clamp(-1.0, 1.0)     # 权重极小处除法会爆炸，钳到合法域
    return cos_u.cpu().numpy(), cos_w.cpu().numpy()

cosU_gen, cosW_gen = phase_cos(a_tar, th_tar, th_gen)
cosU_rec, cosW_rec = phase_cos(a_tar, th_tar, th_rec)

SYM = " .:-=+*#%@"


def mat_ratio(num, den, lo=-6, hi=6):
    """log2(num/den) 符号矩阵"""
    out = []
    for h in range(WIN):
        row = ""
        for w in range(W_FFT):
            v = math.log2(max(num[h, w], 1e-18) / max(den[h, w], 1e-18))
            v = max(lo, min(hi, v))
            row += SYM[int((v - lo) / (hi - lo) * 9.999)]
        out.append(f"h{h} |{row}|")
    return "\n".join(out)


def mat_cos(c):
    """cos 矩阵：-1~1 -> 符号"""
    out = []
    for h in range(WIN):
        row = ""
        for w in range(W_FFT):
            v = max(-1.0, min(1.0, float(c[h, w])))
            row += SYM[int((v + 1) / 2 * 9.999)]
        out.append(f"h{h} |{row}|")
    return "\n".join(out)


def band_split(E):
    """按 w 切低频(w<6)/高频(w>=6) 能量占比；再按 h 切时间频段"""
    lo = E[:, :6].sum()
    hi = E[:, 6:].sum()
    tot = E.sum() + 1e-18
    per_h = E.sum(1) / tot
    return lo / tot, hi / tot, per_h


# ---------------------------------------------------------------- [1] 振幅能量比
print("\n" + "=" * 74)
print("[1] 振幅能量比 log2(E_gen / E_tar)  —— 生成 vs 素材（'.'=÷4, '@'=×64）")
print("    列 = 空间频率 w（0=直流，承载 97% 能量），行 = 时间频率 h")
print(mat_ratio(E_gen, E_tar))
bt, bg, bh = band_split(E_tar)
gt, gg, gh = band_split(E_gen)
rt, rg, rh = band_split(E_rec)
print(f"    素材：E_lo(w<6)={bt * 100:6.2f}%  E_hi={bg * 100:6.2f}%  逐h={np.round(bh * 100, 1)}")
print(f"    生成：E_lo(w<6)={gt * 100:6.2f}%  E_hi={gg * 100:6.2f}%  逐h={np.round(gh * 100, 1)}")
print(f"    重建：E_lo(w<6)={rt * 100:6.2f}%  E_hi={rg * 100:6.2f}%  逐h={np.round(rh * 100, 1)}")

print("\n[1b] log2(E_rec / E_tar)  —— 重建 vs 素材（前向链路忠实度）")
print(mat_ratio(E_rec, E_tar))

# ---------------------------------------------------------------- [2] 相位
print("\n" + "=" * 74)
print("[2a] 未加权 cos(Δθ)：生成 vs 素材（'.'≈0 随机, '@'≈+1 对齐, ' '≈-1 反相）")
print(mat_cos(cosU_gen))
print("[2b] 能量加权 cos(Δθ) cosW（a_tar^2 加权，只看大振幅系数）")
print(mat_cos(cosW_gen))
print("[2c] 重建 未加权 cos(Δθ)（若前向忠实应几乎全 @）")
print(mat_cos(cosU_rec))

def COS(c, sel):
    return float(c[sel].mean())

print("\n[2d] 关键格点 cos 值")
print(f"    {'格点':<14}{'E_tar占比':>10}{'E_gen/E_tar':>13}{'cosU_gen':>10}{'cosW_gen':>10}")
for h in range(WIN):
    for w in [0, 1, 5, 6, 12, 24]:
        share = E_tar[h, w] / (E_tar.sum() + 1e-18)
        if share < 1e-4 and not (h == 0 and w == 0):
            continue
        r = E_gen[h, w] / max(E_tar[h, w], 1e-18)
        print(f"    (h{h},w{w:<3}){share * 100:>9.2f}%{r:>13.3f}"
              f"{COS(cosU_gen, (h, w)):>+10.3f}{COS(cosW_gen, (h, w)):>+10.3f}")

# ---------------------------------------------------------------- [3] 图像口径
print("\n" + "=" * 74)
print("[3] 图像口径：decode 后 2D FFT 径向能量占比")


def decode_frames(a, th):
    z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    return np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


def radial(frames):
    """返回 (E_lo r<0.15H, E_mid, E_hi r>0.35H) 占比 + ac1 + std"""
    g = frames.mean(-1)                                     # [T,H,W]
    F = np.abs(np.fft.fftshift(np.fft.fft2(
        g - g.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
    H, W_ = F.shape[1], F.shape[2]
    yy, xx = np.mgrid[0:H, 0:W_]
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W_ // 2) ** 2)
    tot = F.sum()
    e_lo = F[:, r < H * 0.15].sum() / tot
    e_hi = F[:, r >= H * 0.35].sum() / tot
    e_mid = 1 - e_lo - e_hi
    d = frames[1:] - frames[:-1]
    ac1 = float((d[:-1].reshape(len(d) - 1, -1) * d[1:].reshape(len(d) - 1, -1)).sum()
                / (d[:-1].std() * d[1:].std() * d[:-1].size + 1e-12))
    return dict(lo=float(e_lo), mid=float(e_mid), hi=float(e_hi),
                ac1=ac1, std=float(frames.std()))


fr_gen = decode_frames(a_gen, th_gen)
fr_rec = decode_frames(a_rec, th_rec)
z = (a_tar[0] * torch.exp(1j * th_tar[0])).to(torch.complex64)
fr_tar = np.stack(decode_field_to_frames(
    torch.cat([z.real, z.imag], dim=-1).float().cpu(), TS, GRID, PATCH, WIN, IMG, D,
    normalize=False)).astype(np.float32)

for nm, fr in [("素材(codec)", fr_tar), ("生成", fr_gen), ("重建", fr_rec)]:
    st = radial(fr)
    print(f"    {nm:<12} E_lo={st['lo'] * 100:6.2f}%  E_mid={st['mid'] * 100:6.2f}%"
          f"  E_hi={st['hi'] * 100:6.2f}%  ac1={st['ac1']:+.3f}  std={st['std']:6.2f}"
          f"  帧数={len(fr)}")

# PSNR 交叉
def psnr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mse = ((a - b) ** 2).mean()
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


print(f"\n    PSNR 生成 vs 素材: {psnr(fr_gen, fr_tar):.2f} dB")
print(f"    PSNR 重建 vs 素材: {psnr(fr_rec, fr_tar):.2f} dB")
print("=" * 74)
