# -*- coding: utf-8 -*-
"""绝对质量诊断：生成图到底是不是噪声（2026-10-18）

★ 为什么必须做这个（用户直觉纠偏）：
  margin/wcos 全是**相对**指标。"src0 判对、src1 差一点判错"在
  全噪声图上也成立（两个噪声谁离"目标噪声"更近是有定义的）。
  margin 从 −2.59 → −1.40 的"进步"完全可能发生在图像不可视区间。
  → 必须报**绝对**空间域指标 + **噪声基线**对照。

判据（决定性）：
  若 PSNR(生成, 素材) ≈ PSNR(随机噪声, 素材)
  则生成 = 噪声，wcos/margin 的任何改善都是噪声里的对齐。
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
sys.path.insert(0, str(_PROJ))     # torch_accel_shim 在 _PROJ 下（丢了会静默失败）
_DEPS = _PROJ / "deps312"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
# ★ C 盘残缺 transformers 必须从 sys.path 剔除（deps312 才是完整版）
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

import random
import numpy as np
import torch
from PIL import Image

from codec.gif_codec import (encode_gif_to_field, decode_field_to_frames,
                             load_gif_frames, resample_frames)
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import CLIP_EMB_DIM

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(0); np.random.seed(0); torch.manual_seed(0)

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "8号球"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "表情包"),
]

TAG = os.environ.get("DIAG_TAG") or "cond3_e4_arctan_g12_3000"
CKPT = ROOT / "outputs_real" / TAG / "cond3_ckpt.pt"


# ------------------------------------------------------------ 指标工具
def stats(x):
    """x: [T,H,W,3] float 0~255 → 空间域统计"""
    f = x.reshape(-1, 3).astype(np.float64)
    lum = x.mean(axis=-1)
    return {
        "std": float(x.std()),
        "p1": float(np.percentile(f, 1)),
        "p50": float(np.percentile(f, 50)),
        "p99": float(np.percentile(f, 99)),
        "mean": float(f.mean()),
        "lum_std": float(lum.std()),
        "t_std_min": float(min(fr.std() for fr in x)),
        "t_std_max": float(max(fr.std() for fr in x)),
    }


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def autocorr1d(x, axis=-1):
    """相邻像素相关系数：噪声≈0，图像>0.5。用切片不用 roll（roll 会环绕）"""
    a = x - x.mean()
    if axis == -1:
        num = (a[..., :-1] * a[..., 1:]).sum()
    else:
        num = (a[..., :-1, :] * a[..., 1:, :]).sum()
    den = (a * a).sum()
    return float(num / max(den, 1e-12))


def sobel_mag(x):
    """Sobel 梯度幅值均值：锐度"""
    g = x.mean(axis=-1)
    gx = np.abs(np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) / 2
    gy = np.abs(np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0)) / 2
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def spectrum_ratio(x):
    """空间频谱能量占比：高/低频比。图像 = 能量集中低频；白噪声 = 平坦"""
    g = x.mean(axis=-1)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
    H, W_ = F.shape[1], F.shape[2]
    cy, cx = H // 2, W_ // 2
    yy, xx = np.mgrid[0:H, 0:W_]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    lo = F[:, r < H * 0.15].sum()
    hi = F[:, r > H * 0.35].sum()
    return float(hi / max(lo, 1e-12))


def report(tag, x, ref, extra=""):
    s = stats(x)
    print(f"  {tag:22s} PSNR={psnr(x, ref):6.2f}  std={s['std']:6.2f}  "
          f"lum_std={s['lum_std']:6.2f}  p1/p50/p99={s['p1']:5.1f}/{s['p50']:5.1f}/{s['p99']:5.1f}  "
          f"帧std=[{s['t_std_min']:.1f},{s['t_std_max']:.1f}]  "
          f"ac1={autocorr1d(x):+.3f}  sobel={sobel_mag(x):5.2f}  hi/lo={spectrum_ratio(x):.2f} {extra}")


# ------------------------------------------------------------ 素材 + 噪声基线
print("=" * 110)
print("[1] 素材原图 + 噪声基线（决定性对照）")
gif_dir = ROOT / "data" / "gif_raw" / "noto"
NEED = TS * WIN          # decode_field_to_frames 输出 [TS*WIN, IMG, IMG, 3]
refs = []
for name, desc in GIF_SPECS:
    gp = gif_dir / name
    if not gp.exists():
        gp = ROOT.parent.parent / "data" / "gif_raw" / "noto" / name
    raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
    # ★★ 参考帧必须与 encode_gif_to_field **完全同口径**（2026-10-18 修正）：
    #   encode 内部走 resample_frames(frames, TS*WIN) = linspace 线性插值。
    #   此前这里在 raw.shape[0] >= NEED 时用了 arange(NEED)（取前 30 帧），
    #   → 与 encode 时序错位，素材1 错位 PSNR 仅 12.01 dB，
    #     误报成「codec 天花板只有 12 dB」。修正后真实天花板 51.69/52.08 dB
    #     （D=125 = 5×25 全保留，零截断）。
    refs.append(resample_frames(raw, NEED).astype(np.float32))
    print(f"  {name}: 原素材 {raw.shape} -> 对齐(linspace) {refs[-1].shape}")

for i, (name, desc) in enumerate(GIF_SPECS):
    print(f"\n  --- 素材{i}: {desc} ---")
    report("素材原图", refs[i], refs[i], "(自比=∞)")

    # 基线 1：均匀随机噪声（0~255）
    noise_u = np.random.uniform(0, 255, refs[i].shape).astype(np.float32)
    report("均匀随机噪声", noise_u, refs[i])

    # 基线 2：匹配素材 std 的白噪声
    noise_m = np.random.normal(refs[i].mean(), refs[i].std(), refs[i].shape).astype(np.float32)
    noise_m = np.clip(noise_m, 0, 255)
    report("同std高斯噪声", noise_m, refs[i])

    # 基线 3：匹配素材 std 的平滑噪声（低通白噪声）——最像"糊的噪声"
    lo = np.random.normal(0, 1, refs[i].shape).astype(np.float32)
    F = np.fft.fft2(lo, axes=(1, 2))
    H, W_ = lo.shape[1], lo.shape[2]
    yy, xx = np.mgrid[0:H, 0:W_]
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W_ // 2) ** 2)
    F[:, r > H * 0.10] = 0                   # 只留低频（3D 数组要带 [:, mask]）
    lo_s = np.real(np.fft.ifft2(F, axes=(1, 2))).astype(np.float32)
    lo_s = lo_s * (refs[i].std() / max(lo_s.std(), 1e-8)) + refs[i].mean()
    report("低通噪声(糊)", np.clip(lo_s, 0, 255), refs[i])

    # 基线 4：素材自身的两帧互比（"同素材不同帧"的 PSNR 上限参照）
    if TS >= 2:
        report("素材第0帧 vs 第1帧", refs[i][:1], refs[i][1:2], "(同素材帧间差)")

# ------------------------------------------------------------ codec 天花板
print("\n" + "=" * 110)
print("[2] codec 天花板（FFT 往返，架构上限）")
fields = []
for name, _ in GIF_SPECS:
    gp = gif_dir / name
    if not gp.exists():
        gp = ROOT.parent.parent / "data" / "gif_raw" / "noto" / name
    fields.append(encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D))
for i in range(2):
    f = fields[i]                                   # [N, 2D]
    a = torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12)   # [N, D]
    th = torch.atan2(f[..., D:], f[..., :D])
    z = (a * torch.exp(1j * th)).to(torch.complex64)  # [N, D]（不是 a[0]）
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    rec = np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
    report(f"codec 天花板 i{i}", rec, refs[i])

# ------------------------------------------------------------ 模型生成
print("\n" + "=" * 110)
print(f"[3] 模型生成（{TAG}）")
_FFN_H = int(os.environ.get("FFN_HIDDEN", "512"))
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=_FFN_H, init_mode="init",
                       n_modules=8, mod_gamma=0.15).to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()

clip_cache = ROOT / "data" / "clip_cache"
text3 = np.load(clip_cache / "text3.npy")
first3 = np.load(clip_cache / "first3.npy")
gif3v = np.load(clip_cache / "gif3_video.npy")
SEG = CLIP_EMB_DIM
cv_full = np.concatenate([text3, first3, gif3v,
                          np.ones((2, 3), np.float32)], 1)

from physics.kuramoto_sync import build_spacetime_index
t, s = build_spacetime_index(TS, GRID, GRID, device)

with torch.inference_mode():
    # ---- 重建模式（给 target_field）：定位"是 loss/频谱问题还是条件映射问题" ----
    # 若重建也高频爆炸 → 模型/频谱问题；若重建干净 → 条件→初始态映射丢了细节
    tgt = torch.from_numpy(fields[0].numpy()).unsqueeze(0).to(device)
    a_r, th_r, _ = model(texts=None, target_field=tgt, t_coord=t, s_coord=s,
                         batch_size=1, cond_vec=torch.from_numpy(cv_full[0:1]).to(device),
                         use_cond_field=True)
    z = (a_r[0] * torch.exp(1j * th_r[0])).to(torch.complex64)
    fld_r = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    gen_r = np.stack(decode_field_to_frames(
        fld_r, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
    report("★重建模式|src0", gen_r, refs[0], "(给 target)")

    for cond_name, sel in [("三路全给", None)]:
        for src_i in range(2):
            cv = torch.from_numpy(cv_full[src_i:src_i + 1]).to(device)
            a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=cv, use_cond_field=True)
            z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
            fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
            gen = np.stack(decode_field_to_frames(
                fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
            report(f"{cond_name}|src{src_i}", gen, refs[src_i])

# ------------------------------------------------------------ 结论
print("\n" + "=" * 110)
print("[4] 判决（对照噪声基线）")
print("  规则：生成 PSNR 落在[同std高斯噪声, 均匀随机噪声]区间 → 噪声；")
print("        明显高于噪声基线 → 真有信号；低于 codec 天花板 → 物理核损失。")
print("  ★ wcos/margin 是相对指标，全噪声图上同样成立 —— 只能作辅助判据。")
