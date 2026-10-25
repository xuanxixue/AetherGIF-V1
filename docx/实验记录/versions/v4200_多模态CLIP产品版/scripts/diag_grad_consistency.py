# -*- coding: utf-8 -*-
"""★★ 决定性诊断：相位为什么有梯度却学不动？—— 梯度方向一致性（2026-10-18）

【已排除的原因（都实测过，不要重走）】
  ✗ clamp 饱和：|ph_res|max 在 block11=39.5，但饱和率仅 0.46%，
    实测 clamp vs tanh 软饱和的 dL/dθ 梯度几乎相同（1.597e-6 vs 1.600e-6）
  ✗ 梯度断流：121/121 参数张量全部有非零梯度，最小非零 3.6e-05
  ✗ 梯度分布失衡：旧 loss_ph 把 97.82% 相位梯度压在 DC 频段
    （已由 per-band 分组监督修复，高频梯度提升 14~21 倍）
  → 结论：梯度**存在、通畅、分布正常**，但**学不动**。

【本脚本测的最后一件事】
  梯度**方向一致性**：相邻两步梯度余弦。
  · ≈ +1 → 方向一致，梯度能积累成有效下降
  · ≈0/负 → 纯随机行走，无论步长多大都不下降
  实测（现行交叉训练）：+0.185 / +0.007 / −0.036，**平均 +0.052 ≈ 0**。
  这就是「有梯度却学不动」的唯一解释。

【对照实验】固定单目标（src0 + 三路全给）vs 现行随机交叉：
  若固定后余弦显著转正 → **交叉训练（随机素材 × 7 种条件组合）
  把loss 曲面搅乱**。修法 = 梯度累积 / 先单目标后交叉 / 降交叉频率。
"""
import gc
import math
import os
import random
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
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import CLIP_EMB_DIM, build_cond_vector3
from train.spectral_loss import loss_phase_band

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ.setdefault("PHASE_RES_ACT", "soft2")

gif_dir = ROOT / "data" / "gif_raw" / "noto"
NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]
fields, th_tars = [], []
for nm in NAMES:
    f = encode_gif_to_field(str(gif_dir / nm), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]).unsqueeze(0))
target = torch.stack(fields, 0)
th_tar = torch.stack(th_tars, 0)
t, s = build_spacetime_index(TS, GRID, GRID, device)

cc = ROOT / "data" / "clip_cache"
text3 = np.load(cc / "text3.npy")
first3 = np.load(cc / "first3.npy")
gif3v = np.load(cc / "gif3_video.npy")
COMBOS = [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]


def build_cv(src, has):
    z = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
    return torch.from_numpy(build_cond_vector3(
        text3[src:src + 1] if has[0] else z,
        first3[src:src + 1] if has[1] else z,
        gif3v[src:src + 1] if has[2] else z, has)).to(device)


def make_model():
    m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                       init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
    m.train()
    return m


def grad_cos(model, sampler, n=6, label=""):
    """相邻两步梯度余弦的平均值。"""
    gs, ls = [], []
    for _ in range(n):
        model.zero_grad(set_to_none=True)
        src, has = sampler()
        a, th, _ = model(texts=None, target_field=target[src:src + 1],
                         t_coord=t, s_coord=s, batch_size=1,
                         cond_vec=build_cv(src, has), use_cond_field=True)
        l = loss_phase_band(th, th_tar[src:src + 1], win=WIN, w_fft=W_FFT)
        l.backward()
        gs.append(torch.cat([p.grad.reshape(-1) for p in model.parameters()
                             if p.grad is not None]).detach().clone())
        ls.append(float(l))
    cs = [float(torch.dot(gs[i - 1], gs[i]) /
                (gs[i - 1].norm() * gs[i].norm()).clamp(min=1e-30))
          for i in range(1, n)]
    mcs = sum(cs) / len(cs)
    print(f"  {label:<34} 余弦={mcs:+.4f}  (各步 {[f'{c:+.3f}' for c in cs]})")
    print(f"  {'':34} loss={[f'{v:.5f}' for v in ls]}")
    del gs
    gc.collect(); torch.cuda.empty_cache()
    return mcs


print("=" * 96)
print("★★ 梯度方向一致性 —— 有梯度却学不动 的唯一解释")
print("=" * 96)
print(f"  PHASE_RES_ACT={os.environ.get('PHASE_RES_ACT')}  "
      f"FFN_HIDDEN={os.environ.get('FFN_HIDDEN','512')}")
print("-" * 96)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

m = make_model()
print("[1] 现行训练分布（随机素材 × 随机 7 条件组合）")
r_cross = grad_cos(m, lambda: (random.randrange(2), random.choice(COMBOS)),
                   label="随机交叉训练")
del m; gc.collect(); torch.cuda.empty_cache()

print("[2] 对照组：固定单目标（src0 + 三路全给）")
m = make_model()
r_fix = grad_cos(m, lambda: (0, (1, 1, 1)), label="固定单目标")
del m; gc.collect(); torch.cuda.empty_cache()

print("[3] 对照组：固定素材，随机条件（拆开看条件组合的贡献）")
m = make_model()
r_src = grad_cos(m, lambda: (0, random.choice(COMBOS)), label="固定src0 + 随机条件")
del m; gc.collect(); torch.cuda.empty_cache()

print("[4] 对照组：随机素材，固定条件（拆开看素材切换的贡献）")
m = make_model()
r_has = grad_cos(m, lambda: (random.randrange(2), (1, 1, 1)), label="随机素材 + 固定三路")
del m; gc.collect(); torch.cuda.empty_cache()

print("=" * 96)
print("  汇总（梯度余弦，越接近 +1 越好学）")
print(f"    随机交叉训练        : {r_cross:+.4f}")
print(f"    固定单目标          : {r_fix:+.4f}   ← 对照")
print(f"    固定素材+随机条件   : {r_src:+.4f}")
print(f"    随机素材+固定条件   : {r_has:+.4f}")
print("-" * 96)
cond_part = r_src - r_cross
mat_part = r_has - r_cross
print(f"  条件组合切换的贡献   : {cond_part:+.4f}")
print(f"  素材切换的贡献: {mat_part:+.4f}")
if r_fix > max(r_cross, 0.3):
    print(f"\n  ★ 判决：**交叉训练是梯度混乱的主因**（{r_cross:+.4f} → {r_fix:+.4f}）")
    print("    随机切素材 + 随机切 7 种条件组合，让每步的loss 曲面都不同，")
    print("    梯度方向逐步漂移 → 等效随机行走 → 无论梯度多大都不下降。")
    print("    修法方向：① 梯度累积（K步平均后再更新）")
    print("          ② 两段式：先固定单目标把相位学会，再放开交叉")
    print("          ③ 降低交叉频率 + 加大 batch")
else:
    print("\n  → 交叉训练不是主因（固定后余弦仍低），需另找原因。")
print("=" * 96)