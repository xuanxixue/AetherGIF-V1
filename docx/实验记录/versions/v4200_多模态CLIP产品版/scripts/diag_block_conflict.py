# -*- coding: utf-8 -*-
"""★★ 第二十二死穴候选：块间梯度冲突（2026-10-18）

【排除链（已测实）】
  · 前向确定（S0：a/θ 三次逐位相同）
  · 梯度存在（121/121 非零）、分布合理、频段均衡（loss_phase_band 后）
  · 交叉训练不是主因（固定单目标 −0.076 比随机 +0.0095更差）
  · 物理核参数不是主因（只训 FFN 相位头输出层，余弦仍 +0.0526）

【剩下的唯一可测假设】
  12 个 block 的梯度在参数空间里**几乎两两正交甚至相互抵消**：
  净梯度 = Σ block 梯度，长度只有 Σ|block| 的 ~1/√12，
  且每步各block 的贡献权重在翻转 → 净方向逐步随机 → 梯度余弦 ≈ 0。

【为什么这个假设能解释全部观测】
  ① 梯度存在、通畅、分布正常（每 block 单独看都健康）
  ② 余弦 ≈ 0（净方向被块间冲突打乱）
  ③ 降 lr 没用（缩短步长不改变方向随机性）
  ④ 提高 lr 到 1e-4 后 std 守卫才开始起作用
     （步长够大才能越过抵消的"势谷"，这正是低 lr 时模型走捷径的原因）

【本脚本测什么】
  [1] 跨步同block 的余弦（块内时间一致性）
  [2] 同一步跨 block 的余弦矩阵（块间空间一致性）+ 其分布
  [3] 抵消比：|Σg_i| / Σ|g_i|（=1 完全不冲突，~1/√12 完全正交冲突）
  [4] ★ 决定性验证：把各 block 梯度按"沿上一步净梯度的符号"对齐后再合并，
      重算相邻步余弦。若显著转正 → 冲突假设成立，且给出了现成修法。
"""
import gc
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
from data.clip_condition import build_cond_vector3
from train.spectral_loss import loss_phase_band

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
NSTEP = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

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


def build_cv(src):
    return torch.from_numpy(build_cond_vector3(
        text3[src:src + 1], first3[src:src + 1], gif3v[src:src + 1],
        (1, 1, 1))).to(device)


torch.manual_seed(0); random.seed(0); np.random.seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                   init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
m.train()
NB = len(m.pmnncore.blocks)
# 统一观测载体：每个 block 的 w2.weight 梯度（振幅+相位两路都含）
KEYS = [f"pmnncore.blocks.{i}.ffn.w2.weight" for i in range(NB)]
LOOKUP = dict(m.named_parameters())


def collect(src=0):
    m.zero_grad(set_to_none=True)
    a, th, _ = m(texts=None, target_field=target[src:src + 1],
                 t_coord=t, s_coord=s, batch_size=1,
                 cond_vec=build_cv(src), use_cond_field=True)
    l = loss_phase_band(th, th_tar[src:src + 1], win=WIN, w_fft=W_FFT)
    l.backward()
    G = [LOOKUP[k].grad.detach().reshape(-1).clone() for k in KEYS]
    return torch.stack(G), float(l)          # [NB, per_block_dim]


print("=" * 96)
print(f"★★ 块间梯度冲突诊断  device={device}  NB={NB}")
print("=" * 96)

steps, loss = [], []
for k in range(NSTEP):
    G, l = collect(0)
    steps.append(G)
    loss.append(l)
    print(f"  step{k}  loss={l:.5f}  |g| 各block min/med/max = "
          f"{float(G.norm(dim=1).min()):.3e} / "
          f"{float(G.norm(dim=1).median()):.3e} / "
          f"{float(G.norm(dim=1).max()):.3e}")


def cos(u, v):
    return float(torch.dot(u, v) / (u.norm() * v.norm()).clamp(min=1e-30))


print("\n" + "-" * 96)
print("[1] 跨步同 block 的余弦（块内时间一致性，adjacent steps）")
print("-" * 96)
intra = []
for i in range(NB):
    cs = [cos(steps[k][i], steps[k + 1][i]) for k in range(NSTEP - 1)]
    intra.append(float(np.mean(cs)))
    print(f"  block{i:<3} 余弦均值={np.mean(cs):+.4f}各步="
          f"{[f'{x:+.3f}' for x in cs]}")
print(f"  → 块内时间一致性均值 = {np.mean(intra):+.4f}")

print("\n" + "-" * 96)
print("[2] 同一步跨 block 的余弦矩阵（块间空间一致性）")
print("-" * 96)
for k in [0, NSTEP - 1]:
    Cm = np.zeros((NB, NB))
    for i in range(NB):
        for j in range(NB):
            Cm[i, j] = cos(steps[k][i], steps[k][j])
    off = Cm[~np.eye(NB, dtype=bool)]
    print(f"  step{k}:非对角均值={off.mean():+.4f} 中位={np.median(off):+.4f} "
          f"最小={off.min():+.4f} 最大={off.max():+.4f} "
          f"负比例={(off < 0).mean()*100:.1f}%")
    print("     " + "".join(f"{j:>7d}" for j in range(NB)))
    for i in range(NB):
        print(f"  {i:<3d} " + "".join(f"{Cm[i, j]:>+7.3f}" for j in range(NB)))

print("\n" + "-" * 96)
print("[3] 抵消比 |Σg_i| / Σ|g_i|（1=完全不冲突，1/√12≈0.29=完全正交）")
print("-" * 96)
for k in range(NSTEP):
    G = steps[k]
    net = G.sum(0)
    ratio = float(net.norm() / G.norm(dim=1).sum())
    print(f"  step{k}: |Σg|={float(net.norm()):.4e}  Σ|g_i|="
          f"{float(G.norm(dim=1).sum()):.4e}  比值={ratio:.4f}"
          f"  (理想随机 {1.0/ np.sqrt(NB):.4f})")

print("\n" + "-" * 96)
print("[4] ★ 决定性验证：块间符号对齐后合并，重算相邻步余弦")
print("-" * 96)
raw_cs = [torch.cat(list(steps[k])).reshape(-1) for k in range(NSTEP)]
raw_cs = [cos(raw_cs[k], raw_cs[k + 1]) for k in range(NSTEP - 1)]
print(f"  原始净梯度余弦={[f'{x:+.3f}' for x in raw_cs]}  "
      f"均值={np.mean(raw_cs):+.4f}")

al_cs = []
prev_dir = None
for k in range(NSTEP):
    G = steps[k]
    if prev_dir is None:
        merged = G.sum(0)
    else:
        # 各 block 投影到上一步净方向，按符号对齐（PCGrad 的简化版：
        # 只对齐符号，不做梯度投影）
        pr = G @ prev_dir                      # [NB]
        sgn = torch.where(pr < 0, -1.0, 1.0)
        merged = (G * sgn[:, None]).sum(0)
    d = merged / merged.norm().clamp(min=1e-30)
    al_cs.append(d)
    prev_dir = d
al_cos = [float(torch.dot(al_cs[k], al_cs[k + 1])) for k in range(NSTEP - 1)]
print(f"  对齐后余弦    ={[f'{x:+.3f}' for x in al_cos]}  "
      f"均值={np.mean(al_cos):+.4f}")

print("\n" + "-" * 96)
print("[5] 对齐对步长的要求：|Σg| 在对齐前后")
print("-" * 96)
for k in range(NSTEP):
    G = steps[k]
    net_raw = float(G.sum(0).norm())
    pr = G @ al_cs[max(k - 1, 0)]
    sgn = torch.where(pr < 0, -1.0, 1.0)
    net_al = float((G * sgn[:, None]).sum(0).norm())
    print(f"  step{k}:原始={net_raw:.4e}  对齐后={net_al:.4e}  "
          f"放大={net_al/max(net_raw,1e-30):.2f}x")
print("=" * 96)