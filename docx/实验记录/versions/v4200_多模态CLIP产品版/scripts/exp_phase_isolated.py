# -*- coding: utf-8 -*-
"""★ 隔离实验：只学相位、冻结振幅 —— 相位到底可学不可学？（2026-10-18）

【为什么必须先做这个】
  diag_amp_phase.py 已判决相位是主凶（素材0 相位 wcos=−0.456，比随机还差）。
  但**在动loss 之前必须先确认「相位在当前架构下能不能学」**：
  若架构层面相位就不可学，改 loss 调参全是白费（与「参考帧bug」同一教训 ——
  先验证前提，再优化细节）。

【实验设计】三组对照，各150 步，同种子同素材同条件序列：
  P0冻结振幅 + 旧 loss_ph（对照组，预期学不动→ 证明问题在 loss）
  P1  冻结振幅 + 新 loss_phase_band（验证分组监督能否救）
  P2  冻结振幅 + 新 loss + floor 守卫（兜底：直接罚「低于随机」）

【冻结振幅的实现】
  把 a 的梯度彻底切断：a_pred = a.detach()（不参与 loss_amp/频域各项），
  但**相位仍由物理核从 a 读取**（耦合关系保留，只是不更新振幅通路）。
  这样 wcos 的变化只能来自相位通路的可学性，与振幅无关。

【判读】
  · P1/P2 的 phase_band loss 若显著下降且 wcos 转正 → **相位可学**，
    之前的 wcos≈0.22/ −0.456 纯粹是 loss 权重问题 → 直接上loss 即可
  · 若P1/P2 的 loss 几乎不动 → **架构层面相位不可学**（FFN 相位头 ±1.5 clamp
    或 Kuramoto 吸引子锁死），必须改架构，调参无意义
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

import random
import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from physics.cond_init import CondInitModulator
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import CLIP_EMB_DIM, build_cond_vector3
from train.spectral_loss import loss_phase_band, loss_phase_band_floor

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1
STEPS = int(os.environ.get("PSTEPS", "150"))
LR = 3e-4
COND_LR = 1e-3
GEN_W = 2.0
GUARD_W = float(os.environ.get("GUARD_W", "12.0"))
FFN_HIDDEN = int(os.environ.get("FFN_HIDDEN", "512"))
PB_W = float(os.environ.get("PB_W", "1.0"))        # 新 loss_phase_band 权重
PF_W = float(os.environ.get("PF_W", "1.0"))        # floor 守卫权重
MODE = os.environ.get("PMODE", "P1")              # P0 / P1 / P2
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_SPECS = [("8-ball_1f3b1.gif", "8号球"),
             ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "前辈前辈表情包")]
OUT_DIR = ROOT / "outputs_real" / f"phase_iso_{MODE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

gif_dir = ROOT / "data" / "gif_raw" / "noto"
fields, a_tars, th_tars = [], [], []
for name, _ in GIF_SPECS:
    f = encode_gif_to_field(str(gif_dir / name), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]))
target = torch.stack(fields, 0)
a_tar = torch.stack(a_tars, 0)
th_tar = torch.stack(th_tars, 0)
t, s = build_spacetime_index(TS, GRID, GRID, device)

cc = ROOT / "data" / "clip_cache"
text3 = np.load(cc / "text3.npy")
first3 = np.load(cc / "first3.npy")
gif3v = np.load(cc / "gif3_video.npy")


def build_cv(src_i, has):
    z = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
    return build_cond_vector3(text3[src_i:src_i + 1] if has[0] else z,
                              first3[src_i:src_i + 1] if has[1] else z,
                              gif3v[src_i:src_i + 1] if has[2] else z, has)


print("=" * 96)
print(f"[相位隔离实验] MODE={MODE}  STEPS={STEPS}  "
      f"(PB_W={PB_W} PF_W={PF_W} GUARD_W={GUARD_W})")
print(f"  P0=旧 loss_ph | P1=新 phase_band | P2=新 phase_band + floor 守卫")
print(f"  ★ 三组均**冻结振幅**（loss_amp 权重=0，a 不接收梯度更新）")
print("=" * 96)

torch.manual_seed(1234)
random.seed(1234)
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=FFN_HIDDEN, init_mode="init",
                       n_modules=8, mod_gamma=0.15).to(device)
_cond = [p for n, p in model.named_parameters()
         if n.startswith("cond_field.") or n.startswith("cond_init.")]
_other = [p for n, p in model.named_parameters()
          if not (n.startswith("cond_field.") or n.startswith("cond_init."))]
opt = torch.optim.AdamW([{"params": _other, "lr": LR},
                         {"params": _cond, "lr": COND_LR}], lr=LR)
model.train()

COMBOS = [((1, 1, 1)), ((1, 0, 0)), ((0, 1, 0)), ((0, 0, 1)),
          ((1, 1, 0)), ((1, 0, 1)), ((0, 1, 1))]
hist = []
t_start = time.time()

for step in range(1, STEPS + 1):
    opt.zero_grad()
    src = random.randrange(2)
    has = COMBOS[random.randrange(len(COMBOS))]
    one_cv = torch.from_numpy(build_cv(src, has)).to(device)
    one_tgt = target[src:src + 1]
    a, th, diag = model(texts=None, target_field=one_tgt, t_coord=t, s_coord=s,
                        batch_size=1, cond_vec=one_cv, use_cond_field=True)
    th_t = th_tar[src:src + 1]
    pe = torch.remainder(th - th_t + math.pi, 2 * math.pi) - math.pi
    w_ph = (a_tar[src:src + 1] ** 2).clamp(min=1e-8)

    # === 相位 loss：三组唯一差异 ===
    old_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
    new_ph = loss_phase_band(th, th_t, win=WIN, w_fft=W_FFT)
    new_fl = loss_phase_band_floor(th, th_t, win=WIN, w_fft=W_FFT)
    if MODE == "P0":
        loss_ph = old_ph
    elif MODE == "P1":
        loss_ph = PB_W * new_ph
    else:                                    # P2
        loss_ph = PB_W * new_ph + PF_W * new_fl

    dth, dln = diag.get("dtheta"), diag.get("dlna")
    loss_gen = CondInitModulator.supervise(dth, dln, one_tgt, d_model=D,
                                           w_ph=w_ph, guard_w=GUARD_W) \
        if dth is not None else torch.zeros((), device=device)

    # ★ 冻结振幅：只保留相位项（+ gen 项保 cond 通路有梯度）
    loss = loss_ph + GEN_W * loss_gen
    loss.backward()
    opt.step()

    if step % 25 == 0 or step == 1:
        with torch.no_grad():
            wcos = float((w_ph * torch.cos(pe)).sum() / w_ph.sum())
            # per-band wcos（看高频段是否真的在学）
            peb = torch.remainder(th - th_t + math.pi, 2 * math.pi) - math.pi
            cb = torch.cos(peb).reshape(1, -1, WIN, W_FFT)
            band_cos = [float(cb[:, :, :, B:min(B + 6, W_FFT)].mean())
                        for B in (0, 6, 12, 18, 22)]
        hist.append((step, loss.item(), wcos, float(new_ph)))
        print(f"  [{step:4d}] src={src} loss={loss.item():.4f} "
              f"old_ph={old_ph.item():.4f} new_ph={new_ph.item():.4f} "
              f"floor={new_fl.item():.4f} wcos={wcos:+.4f} "
              f"band_cos=[{band_cos[0]:+.3f},{band_cos[1]:+.3f},"
              f"{band_cos[2]:+.3f},{band_cos[3]:+.3f},{band_cos[4]:+.3f}] "
              f"({time.time()-t_start:.0f}s)", flush=True)

sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
torch.save(sd, OUT_DIR / "phase_iso_ckpt.pt")
print(f"\n[判读] MODE={MODE}")
if len(hist) >= 2:
    f_, l0, w0, p0 = hist[0]
    _, l1, w1, p1 = hist[-1]
    print(f"    new_ph: {p0:.4f} -> {p1:.4f}  ({'下降 ✓ 可学' if p1 < p0*0.9 else '几乎不动 ✗ 可能不可学'})")
    print(f"    wcos  : {w0:+.4f} -> {w1:+.4f}  ({'转正/改善 ✓' if w1 > w0 else '未改善 ✗'})")
    print(f"    耗时  : {time.time()-t_start:.0f}s")
print(f"    产物: {OUT_DIR}")
print("=" * 96)