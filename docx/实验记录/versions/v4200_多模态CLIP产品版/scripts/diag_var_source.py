# -*- coding: utf-8 -*-
"""★★ 相位梯度方差来源定位（2026-10-18，v2干净版）

【已确立的事实】
  · 前向完全确定（S0：a/θ 三次逐位相同）
  · 梯度存在（121/121 非零）、分布合理、频段均衡（loss_phase_band 后）
  · 但相邻步梯度余弦 ≈ 0（随机交叉 +0.0095，固定单目标 −0.076）
    → 等效随机行走，参数更新在噪声上做布朗运动

【v1 的 bug（已修）】
  v1 把掩码挂在 `m._grad_masks`（dict of Parameter）。实测三组结果完全相同、
  ntr=147 走了 fallback → 掩码根本没生效。根因是 nn.Module.__setattr__
  对 dict 的处理 + 以 Parameter 作 dict key 不可靠。
  v2 改为**模块级全局字典 GRAD_MASKS，key 用参数名字符串**，并加断言。

【测什么】
  S1 三种可训范围的梯度余弦对比：
     full          全参数（现行）
     kuramoto_only 只训 FFN 相位头输出层 + w1（掩掉振幅头与其余全部）
     ffn_only      只训 FFN 相位头输出层（w2 的后 D 行 + b2 后 D）
  判读：
     ffn_only 余弦显著转正 → 方差来自物理核（Kuramoto/drift）反向传播路径
     三组都 ≈0            → 方差不是可训范围问题，梯度本身在目标面上乱走
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
from data.clip_condition import build_cond_vector3
from train.spectral_loss import loss_phase_band

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
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

# ★ 模块级全局掩码：{参数名: mask Tensor}
GRAD_MASKS = {}


def build_cv(src):
    return torch.from_numpy(build_cond_vector3(
        text3[src:src + 1], first3[src:src + 1], gif3v[src:src + 1],
        (1, 1, 1))).to(device)


def make_model(mode="full"):
    GRAD_MASKS.clear()
    m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                       init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
    m.train()
    for nm, p in m.named_parameters():
        p.requires_grad_(True)

    if mode in ("kuramoto_only", "ffn_only"):
        for nm, p in m.named_parameters():
            GRAD_MASKS[nm] = torch.zeros_like(p)      # 默认全屏蔽
        for i, b in enumerate(m.pmnncore.blocks):
            pre = f"pmnncore.blocks.{i}.ffn."
            # w2 输出 2D：log_res, ph_res = ffn_out.chunk(2, dim=-1)
            # → 后 D 行是相位残差
            GRAD_MASKS[pre + "w2.weight"][-D:, :] = 1.0
            GRAD_MASKS[pre + "w2.bias"][-D:] = 1.0
            if mode == "kuramoto_only":
                GRAD_MASKS[pre + "w1.weight"][:] = 1.0
                GRAD_MASKS[pre + "w1.bias"][:] = 1.0
        nz = sum(int(v.abs().sum()) for v in GRAD_MASKS.values())
        nent = sum(v.numel() for v in GRAD_MASKS.values())
        assert nent > 0 and 0 < nz < nent, f"掩码异常 nz={nz} nent={nent}"
        print(f"    [掩码OK] {len(GRAD_MASKS)}张量/{nent}元素，"
              f"{nz} 个放行({100.0*nz/nent:.2f}%)")
    return m


def grads_of(model, src=0):
    model.zero_grad(set_to_none=True)
    a, th, _ = model(texts=None, target_field=target[src:src + 1],
                     t_coord=t, s_coord=s, batch_size=1,
                     cond_vec=build_cv(src), use_cond_field=True)
    l = loss_phase_band(th, th_tar[src:src + 1], win=WIN, w_fft=W_FFT)
    l.backward()
    gs = []
    for nm, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        msk = GRAD_MASKS.get(nm)
        if msk is not None:
            g = g * msk
        if g.abs().sum() <= 0:
            continue
        gs.append(g.reshape(-1))
    if not gs:
        return torch.zeros(1, device=device), float(l), None
    g = torch.cat(gs)
    return g.detach().clone(), float(l), th.detach().clone()


def cos_seq(gp, n=4):
    cs = []
    for i in range(1, n):
        cs.append(float(torch.dot(gp[i - 1], gp[i]) /
                        (gp[i - 1].norm() * gp[i].norm()).clamp(min=1e-30)))
    return sum(cs) / len(cs), cs


print("=" * 96)
print("★★ 梯度方差来源定位 v2（掩码已修）")
print("=" * 96)
print(f"  PHASE_RES_ACT={os.environ.get('PHASE_RES_ACT', 'clamp')}  device={device}")

print("\n" + "-" * 96)
print("[S1] 梯度方向一致性 —— 三种可训范围（固定 src0，三路全给条件）")
print("-" * 96)
res = {}
for mode, desc in [("full", "全参数（现行）"),
                   ("kuramoto_only", "只训 FFN 相位头+w1"),
                   ("ffn_only", "只训 FFN 相位头输出层")]:
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    m = make_model(mode)
    gp, ls = [], []
    for k in range(5):
        g, l, _ = grads_of(m, 0)
        gp.append(g); ls.append(l)
    c, cs = cos_seq(gp, 5)
    nd = gp[0].numel()
    res[mode] = c
    print(f"  {desc:<26} 余弦={c:+.4f}  各步={[f'{x:+.3f}' for x in cs]}")
    print(f"  {'':26} loss={[f'{v:.5f}' for v in ls]}  梯度维度={nd}")
    del m, gp
    gc.collect(); torch.cuda.empty_cache()

print("-" * 96)
print("  判决：")
if res.get("ffn_only", -1) > 0.3:
    print(f"    ★ **FFN 相位头单独可学（{res['ffn_only']:+.4f}）"
          f"→ 方差来自物理核（Kuramoto/drift）的反向路径**")
elif res.get("kuramoto_only", -1) > 0.3:
    print(f"    ★ FFN 头+w1 可学（{res['kuramoto_only']:+.4f}）"
          f"→ 方差来自窄化输出层或物理核")
elif abs(res.get("full", 0) - res.get("ffn_only", 0)) > 0.15:
    print(f"    掩码生效且范围有影响：full={res['full']:+.4f} vs "
          f"ffn_only={res['ffn_only']:+.4f}（差 "
          f"{res['full']-res['ffn_only']:+.4f}）")
else:
    print(f"    三组余弦几乎相同 → 可训范围不是方差来源，"
          f"梯度在整个目标面上随机行走")
print("=" * 96)