# -*- coding: utf-8 -*-
"""阶段一：物理版 VAE（latent codec）自编码重建验证。

【目标（用户 2026-10-21 最终指令）】
    生成质量达到 3D-FFT 还原效果。本阶段先单独验证「物理潜编解码器」
    本身的重建天花板：GIF -> 3DFFT场 -> 物理潜 -> 重建场 -> GIF，
    不碰文本、不加噪、不接生成器。

【验收判据】
    ceiling = decode(encode(target)) vs 原始素材 的 PSNR（codec 天花板）
    rec     = decode(latent) 同口径 PSNR（训练后）
    gap     = ceiling − rec。gap<=1dB 判「达到 3D-FFT 还原效果」；
    gap<=3dB 判「接近」。基线参照：当前生成模型只有 ~9.6dB。
    附带判据：重建的每(时间频率h×频段)能量谱与 target 逐项一致
    （本设计里组能量恒精确，该项只用于暴露实现 bug）。

【旋钮】（全部 env，默认给最物理最压缩的配置）
    STEPS       命令行参数1，默认 400
    LR          1e-3
    HIDDEN      展布网络宽度 64
    N_BANDS     空间频率频段数 5
    H_SPLIT     1=按(频段×时间频率)分组（L=85）；0=只按频段（L=25）
    N_MODES     每组相位偏离傅里叶模式数（0=纯序参量；保真度逃生阀）
    PHYS_SPREAD 1=Kuramoto 相干度-展布关系；0=自由
    CONS_W      序参量一致性正则 0.05
    AMP_P       幅度 loss 软化指数 0.5

【运行】
    cd 到 v4200 根目录后
      C:/Python312/python.exe -u scripts/train_latent_codec.py [STEPS]
      OVERFIT_TAG=xxx ...           # 产物落 outputs_real/latent_codec_<tag>/
"""
import math
import os
import sys
import time
import json
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

import numpy as np
import torch
from PIL import Image

from codec.gif_codec import (encode_gif_to_field, decode_field_to_frames,
                             load_gif_frames, resample_frames)
from physics.latent_codec import PhysicalLatentCodec

# ---------------------------------------------------------------- 口径
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1  # 25

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
LR = float(os.environ.get("LR", "1e-3"))
HIDDEN = int(os.environ.get("HIDDEN", "64"))
N_BANDS = int(os.environ.get("N_BANDS", "5"))
H_SPLIT = int(os.environ.get("H_SPLIT", "1"))
N_MODES = int(os.environ.get("N_MODES", "0"))
PHYS_SPREAD = int(os.environ.get("PHYS_SPREAD", "1"))
CONS_W = float(os.environ.get("CONS_W", "0.05"))
AMP_P = float(os.environ.get("AMP_P", "0.5"))
PH_W = float(os.environ.get("PH_W", "1.0"))
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "8号球图标"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "前辈前辈表情包"),
]

OUT_DIR = ROOT / "outputs_real" / ("latent_codec_" + (os.environ.get("OVERFIT_TAG") or "default"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
if os.environ.get("OVERFIT_TAG"):
    print(f"[OUT] OVERFIT_TAG={os.environ['OVERFIT_TAG']} -> {OUT_DIR}")


def psnr(a_img, b_img):
    mse = ((a_img.astype(np.float32) - b_img.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def field_to_pixels(field):
    """[N,2D] 场 -> [T,H,W,3] float32（normalize=False 严格互逆口径）。"""
    return np.stack(decode_field_to_frames(
        field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


def field_to_gif(field, path):
    frames = np.stack(decode_field_to_frames(
        field, TS, GRID, PATCH, WIN, IMG, D, normalize=True)).astype(np.uint8)
    pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:], loop=0, duration=100)
    return len(pil)


# ---------------------------------------------------------------- 素材
print("=" * 78)
print(f"[1] 加载 {len(GIF_SPECS)} 张素材 -> 频域场（D={D}）")
gif_dir = ROOT / "data" / "gif_raw" / "noto"
fields, raw_imgs = [], []
for name, desc in GIF_SPECS:
    gp = gif_dir / name
    if not gp.exists():
        raise FileNotFoundError(gp)
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    _raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
    raw_imgs.append(resample_frames(_raw, TS * WIN).astype(np.float32))
    print(f"    {name}: N={f.shape[0]} a_mean={f[..., :D].abs().mean():.5f}")

target = torch.stack(fields, 0)                       # [2, N, 2D]
a_tar, th_tar = PhysicalLatentCodec.polar(target, D)

# codec 天花板（全保留 125 系数的 3D-FFT 往返）
print("\n[2] codec 天花板（3D-FFT 往返）")
ceil_imgs = []
for i in range(2):
    ci = field_to_pixels(target[i].cpu())
    ceil_imgs.append(ci)
    print(f"    素材{i}: ceiling vs raw PSNR = {psnr(ci, raw_imgs[i]):6.2f} dB")

# ---------------------------------------------------------------- codec
print(f"\n[3] 物理潜编解码器：H_SPLIT={H_SPLIT}, N_BANDS={N_BANDS}, "
      f"N_MODES={N_MODES}, PHYS_SPREAD={PHYS_SPREAD}, HIDDEN={HIDDEN}")
codec = PhysicalLatentCodec(D, win=WIN, n_bands=N_BANDS, h_split=H_SPLIT,
                            hidden=HIDDEN, n_modes=N_MODES,
                            phys_spread=PHYS_SPREAD).to(device)
print(f"    latent_dim={codec.latent_dim}（相对 250 压缩 {250.0/codec.latent_dim:.1f}x）"
      f" 参数={codec.param_count()}")

with torch.no_grad():
    lat = codec.encode(target)                        # [2, N, L] 零参数纯物理
    print(f"    latent: mean={lat.mean():.4f} std={lat.std():.4f} "
          f"min={lat.min():.3f} max={lat.max():.3f}")
    # ---- 零训练体检：每组序参量 r 分布 + 能量占比
    o = 2 * WIN
    rs, es = [], []
    for i in range(codec.n_groups):
        r = lat[..., o].mean().item()
        logE = lat[..., o + 2]
        E = torch.expm1(logE)
        rs.append(r)
        es.append(E.mean().item())
        o += 3 + (2 * N_MODES if N_MODES else 0)
    rs = np.array(rs); es = np.array(es)
    print(f"    序参量 r: mean={rs.mean():.3f} min={rs.min():.3f} max={rs.max():.3f}")
    print(f"    组能量 E: mean={np.mean(es):.5f} min={np.min(es):.5f} max={np.max(es):.5f}")
    if H_SPLIT:
        print("    r 矩阵（行=频段, 列=时间频率 h）:")
        for g in range(N_BANDS):
            row = " ".join(f"{rs[g * WIN + h]:.3f}" for h in range(WIN))
            print(f"      band{g}: {row}")
    # ---- content-only 基线（texture 全零）：texture 能量决定的天花板下限
    rec_co = codec.decode(codec.zero_texture(lat))
    ps = [psnr(field_to_pixels(rec_co[i].cpu()), raw_imgs[i]) for i in range(2)]
    print(f"    [content-only 基线] PSNR = {ps[0]:6.2f} / {ps[1]:6.2f} dB"
          f"（texture=2.8% 能量全部丢弃时的下限）")
    # ---- 未训练解码器
    rec0 = codec.decode(lat)
    ps0 = [psnr(field_to_pixels(rec0[i].cpu()), raw_imgs[i]) for i in range(2)]
    print(f"    [未训练解码器]   PSNR = {ps0[0]:6.2f} / {ps0[1]:6.2f} dB")

# ---------------------------------------------------------------- 训练
print(f"\n[4] 训练解码器 {STEPS} 步（全批量两素材，AdamW lr={LR}，CONS_W={CONS_W}）")
opt = torch.optim.AdamW(codec.parameters(), lr=LR)
hist = []
t0 = time.time()
for step in range(1, STEPS + 1):
    rec = codec.decode(lat)
    a_rec, th_rec = PhysicalLatentCodec.polar(rec, D)
    loss_amp, loss_ph, pe, wcos = PhysicalLatentCodec.field_loss(
        a_rec, th_rec, a_tar, th_tar, amp_p=AMP_P)
    loss = loss_amp + PH_W * loss_ph
    if CONS_W > 0:
        loss = loss + CONS_W * codec.order_consistency(lat, rec)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 50 == 0 or step == 1:
        with torch.no_grad():
            rec_e = codec.decode(lat)
            ps = [psnr(field_to_pixels(rec_e[i].cpu()), raw_imgs[i]) for i in range(2)]
        print(f"  [{step:4d}] loss={float(loss):.5f} amp={float(loss_amp):.5f} "
              f"ph={float(loss_ph):.5f} wcos={float(wcos):+.4f} "
              f"PSNR={ps[0]:6.2f}/{ps[1]:6.2f} dB  [{time.time()-t0:.0f}s]", flush=True)
        hist.append({"step": step, "loss": float(loss), "psnr": ps})

# ---------------------------------------------------------------- 验收
print(f"\n[5] 验收：重建 PSNR vs codec 天花板")
with torch.no_grad():
    rec_final = codec.decode(lat).cpu()
verdict = {}
print(f"{'对象':<24}{'vs raw':>12}{'vs ceiling':>12}{'gap':>8}{'判决':>10}")
print("-" * 66)
for i in range(2):
    ceil_p = psnr(ceil_imgs[i], raw_imgs[i])
    rec_p = psnr(field_to_pixels(rec_final[i]), raw_imgs[i])
    rec_c = psnr(field_to_pixels(rec_final[i]), ceil_imgs[i])
    gap = ceil_p - rec_p
    if gap <= 1.0:
        v = "达到"
    elif gap <= 3.0:
        v = "接近"
    else:
        v = "不足"
    verdict[f"src{i}"] = dict(ceiling=ceil_p, rec=rec_p, gap=gap, verdict=v)
    print(f"{'素材' + str(i):<24}{rec_p:>10.2f}dB{rec_c:>10.2f}dB{gap:>8.2f}{v:>10}")

# 能量谱一致性自检（组能量恒精确，偏了就是实现 bug）
with torch.no_grad():
    a_r, _ = PhysicalLatentCodec.polar(rec_final.to(device), D)
    e_r = (a_r ** 2).mean(dim=1)     # [2, D]
    e_t = (a_tar ** 2).mean(dim=1)
    rel = ((e_r - e_t).abs() / e_t.clamp(min=1e-12)).mean().item()
print(f"    逐通道能量相对偏差 mean = {rel:.4f}（设计上组内总量精确，"
      f"该值应很小；大 = 实现 bug）")

# ---------------------------------------------------------------- 产物
for i in range(2):
    field_to_gif(target[i].cpu(), str(OUT_DIR / f"target_src{i}.gif"))
    field_to_gif(rec_final[i], str(OUT_DIR / f"recon_src{i}.gif"))
torch.save(codec.state_dict(), OUT_DIR / "latent_codec.pt")
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as fp:
    json.dump({
        "config": dict(D=D, WIN=WIN, N_BANDS=N_BANDS, H_SPLIT=H_SPLIT,
                       N_MODES=N_MODES, PHYS_SPREAD=PHYS_SPREAD,
                       HIDDEN=HIDDEN, STEPS=STEPS, LR=LR, CONS_W=CONS_W),
        "latent_dim": codec.latent_dim, "params": codec.param_count(),
        "verdict": verdict, "energy_rel_dev": rel,
        "history": hist,
    }, fp, ensure_ascii=False, indent=2)
print(f"\n[产物] {OUT_DIR}  (target/recon GIF, ckpt, summary.json)")
print(f"[耗时] {time.time()-t0:.0f}s")
