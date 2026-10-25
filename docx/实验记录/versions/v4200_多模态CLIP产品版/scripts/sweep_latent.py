# -*- coding: utf-8 -*-
"""阶段一扫参：物理潜的「保真度-压缩率」曲线。

400 步正式跑已证明 N_MODES=0 的平台在 PSNR 26.5/16.4（gap 25/36 dB），
瓶颈是相位（loss_ph 0.0997 ≫ loss_amp 0.0013）。本脚本扫
    H_SPLIT × N_MODES × PHYS_SPREAD
每组 250 步（400 步跑显示 50 步即平台，250 足够区分配置），
输出「L / PSNR / gap」总表 + 最优配置的逐组相位残差热图。

【判据】用户要求"达到 3DFFT 还原效果"(ceiling≈51.7dB)：
    gap≤1dB 视为达标；本阶段先找「可达多高 / 需要多大 L」。
"""
import math
import os
import sys
import time
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
for _p in (_PROJ / "deps312", _PROJ / ".venv" / "Lib" / "site-packages"):
    if _p.is_dir():
        # ★ deps312 插前；.venv 必须 append（cp313 的 numpy 会遮蔽 cp312 的）
        if "deps312" in str(_p):
            sys.path.insert(0, str(_p))
        else:
            sys.path.append(str(_p))
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

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.latent_codec import PhysicalLatentCodec

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
STEPS = int(os.environ.get("SWEEP_STEPS", "250"))
LR = float(os.environ.get("LR", "1e-3"))
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]
OUT_DIR = ROOT / "outputs_real" / ("latent_sweep_" + (os.environ.get("OVERFIT_TAG") or "default"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def psnr(a_img, b_img):
    mse = ((a_img.astype(np.float32) - b_img.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def field_to_pixels(field):
    return np.stack(decode_field_to_frames(
        field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)


# ---------------------------------------------------------------- 数据（一次加载）
print("=" * 90)
fields, raw_imgs, ceil_imgs = [], [], []
for n in GIF_NAMES:
    gp = ROOT / "data" / "gif_raw" / "noto" / n
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    from codec.gif_codec import load_gif_frames, resample_frames
    raw = load_gif_frames(str(gp), resize_to=(IMG, IMG))
    raw_imgs.append(resample_frames(raw, TS * WIN).astype(np.float32))
    ceil_imgs.append(field_to_pixels(f.cpu()))
target = torch.stack(fields, 0)
a_tar, th_tar = PhysicalLatentCodec.polar(target, D)
for i in range(2):
    print(f"    素材{i}: ceiling vs raw = {psnr(ceil_imgs[i], raw_imgs[i]):.2f} dB")
CEIL = [psnr(ceil_imgs[i], raw_imgs[i]) for i in range(2)]
lat_full = None   # h_split=1 的 latent 复用（r/ψ/logE 与 n_modes 无关）

# ---------------------------------------------------------------- 配置
CONFIGS = []
for hs in (1, 0):
    if hs == 1:
        for nm in (0, 1, 2, 3, 4):
            CONFIGS.append((hs, nm, 1))
    else:
        for nm in (0, 4, 8, 12):
            CONFIGS.append((hs, nm, 1))
CONFIGS.append((1, 2, 0))    # PHYS_SPREAD=0 对照

print(f"\n扫参：{len(CONFIGS)} 配置 × {STEPS} 步（全批量两素材，AdamW lr={LR}）")
print(f"{'h_split':>8}{'n_modes':>9}{'phys':>6}{'L':>6}{'压缩':>7}"
      f"{'PSNR0':>8}{'PSNR1':>8}{'gap0':>7}{'gap1':>7}{'loss_ph':>9}{'判决':>7}")
print("-" * 96)

results = []
t0 = time.time()
for ci, (hs, nm, ps_) in enumerate(CONFIGS):
    torch.manual_seed(1234)
    codec = PhysicalLatentCodec(D, win=WIN, n_bands=5, h_split=hs,
                                hidden=64, n_modes=nm, phys_spread=ps_).to(device)
    with torch.no_grad():
        lat = codec.encode(target)
    opt = torch.optim.AdamW(codec.parameters(), lr=LR)
    for step in range(1, STEPS + 1):
        rec = codec.decode(lat)
        a_rec, th_rec = PhysicalLatentCodec.polar(rec, D)
        loss_amp, loss_ph, pe, wcos = PhysicalLatentCodec.field_loss(
            a_rec, th_rec, a_tar, th_tar, amp_p=0.5)
        loss = loss_amp + loss_ph + 0.05 * codec.order_consistency(lat, rec)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        rec_f = codec.decode(lat).cpu()
        a_r, th_r = PhysicalLatentCodec.polar(rec_f.to(device), D)
        _, loss_ph_e, _, wcos_e = PhysicalLatentCodec.field_loss(
            a_r, th_r, a_tar, th_tar, amp_p=0.5)
        ps = [psnr(field_to_pixels(rec_f[i]), raw_imgs[i]) for i in range(2)]
    gaps = [CEIL[i] - ps[i] for i in range(2)]
    tag = "达标" if max(gaps) <= 1.0 else ("接近" if max(gaps) <= 3.0 else "不足")
    print(f"{hs:>8}{nm:>9}{ps_:>6}{codec.latent_dim:>6}"
          f"{250.0/codec.latent_dim:>6.1f}x{ps[0]:>8.2f}{ps[1]:>8.2f}"
          f"{gaps[0]:>7.2f}{gaps[1]:>7.2f}{float(loss_ph_e):>9.5f}{tag:>7}", flush=True)
    results.append(dict(h_split=hs, n_modes=nm, phys=ps_, L=codec.latent_dim,
                        psnr=ps, gaps=gaps, loss_ph=float(loss_ph_e)))
    if max(gaps) == min(max(r_["gaps"]) for r_ in results):
        torch.save(codec.state_dict(), OUT_DIR / "best_codec.pt")

# ---------------------------------------------------------------- 最优配置逐组残差
best = min(results, key=lambda r_: (max(r_["gaps"]), r_["L"]))
print(f"\n[最优] h_split={best['h_split']} n_modes={best['n_modes']} phys={best['phys']}"
      f" L={best['L']} PSNR={best['psnr']} gap={best['gaps']}")
if best["h_split"] == 1 and best["n_modes"] > 0:
    print("  （n_modes>0 时逐组残差分解意义有限，略）")
else:
    torch.manual_seed(1234)
    codec = PhysicalLatentCodec(D, win=WIN, n_bands=5, h_split=best["h_split"],
                                hidden=64, n_modes=best["n_modes"],
                                phys_spread=best["phys"]).to(device)
    with torch.no_grad():
        lat = codec.encode(target)
    # 重训最优配置（快）
    opt = torch.optim.AdamW(codec.parameters(), lr=LR)
    for step in range(1, STEPS + 1):
        rec = codec.decode(lat)
        a_r_, th_r_ = PhysicalLatentCodec.polar(rec, D)
        la, lp, _, _ = PhysicalLatentCodec.field_loss(a_r_, th_r_, a_tar, th_tar, 0.5)
        (la + lp + 0.05 * codec.order_consistency(lat, rec)).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        rec = codec.decode(lat)
        a_r_, th_r_ = PhysicalLatentCodec.polar(rec, D)
        # 逐组 a² 加权相位残差（组内 = 该频段×时间频率）
        print("  逐组相位残差（a²加权 1-cos，行=频段, 列=时间频率h）：")
        o = 2 * WIN
        for g in range(5):
            row = []
            for h in range(WIN):
                if best["h_split"]:
                    chs = codec._grp(g * WIN + h)
                else:
                    chs = codec._grp(g)
                w2 = (a_tar[..., chs] ** 2).clamp(min=1e-12)
                pe = torch.remainder(th_r_[..., chs] - th_tar[..., chs]
                                     + math.pi, 2 * math.pi) - math.pi
                row.append(float((w2 * (1 - torch.cos(pe))).sum() / w2.sum()))
            print("    band%d: " % g + " ".join(f"{v:.3f}" for v in row))

import json
with open(OUT_DIR / "sweep.json", "w", encoding="utf-8") as fp:
    json.dump(dict(ceil=CEIL, steps=STEPS, results=results, best=best),
              fp, ensure_ascii=False, indent=2)
print(f"\n[产物] {OUT_DIR}/sweep.json  耗时 {time.time()-t0:.0f}s")
