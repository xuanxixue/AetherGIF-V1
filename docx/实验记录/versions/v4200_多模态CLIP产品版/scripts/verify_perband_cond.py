# -*- coding: utf-8 -*-
"""★★ 第二十三死穴修复验收：per-band 条件调制的表达力探针（2026-10-18）

【修复内容】
  cond_init 的 to_phase/to_amp 从输出 n_osc（→[B,N,1] 广播全D 通道）
  改为输出 n_osc × n_groups（按w 列分箱），并加 expand_to_d 按组展开。
  旧设计假设「各频率共享同一套空间形状」被实测证伪：
  广播后跨通道 std = 0.000e+00，且 FFT 每通道振幅均值跨 727 倍。

【本脚本验证三件事（都不训练，秒级）】
  [1] 形状与参数量：forward 能否跑通，参数/显存增量
  [2] ★ 表达力：随机权重下，per-band 调制是否真的产生了
      **组间不同、组内相同**的空间结构（这是修复是否生效的唯一判据）
      旧版此项必然全零差异
  [3] 条件区分度：两个不同条件的调制量余弦（应显著低于 1）
  [4] drift_field 崩溃是否修复（targets=None + cond_anchor=None 走原型分支）
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

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
N_GROUPS = int(os.environ.get("COND_GROUPS", "5"))
device = "cuda" if torch.cuda.is_available() else "cpu"

gif_dir = ROOT / "data" / "gif_raw" / "noto"
f = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"),
                        TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)

cc = ROOT / "data" / "clip_cache"
text3 = np.load(cc / "text3.npy")
first3 = np.load(cc / "first3.npy")
gif3v = np.load(cc / "gif3_video.npy")


def cv_of(src):
    return torch.from_numpy(build_cond_vector3(
        text3[src:src + 1], first3[src:src + 1], gif3v[src:src + 1],
        (1, 1, 1))).to(device)


print("=" * 96)
print(f"★★ 第二十三死穴修复验收：per-band 条件调制  COND_GROUPS={N_GROUPS}")
print("=" * 96)

for G in (1, N_GROUPS):
    os.environ["COND_GROUPS"] = str(G)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    # 让输出头非零，才能测表达力（zero_init=True 时恒为 0测不出东西）
    m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=512, init_mode="init", n_modules=8,
                       mod_gamma=0.15).to(device)
    m.eval()
    ci = m.cond_init
    tag = "旧版(n_groups=1)" if G == 1 else f"新版(n_groups={G})"
    nparam = sum(p.numel() for p in ci.parameters())
    with torch.no_grad():
        # 兼容 Sequential(低秩) 与 Linear(全秩)：零初始化最后一层
        for _head in (ci.to_phase, ci.to_amp):
            _last = _head[-1] if hasattr(_head, "__getitem__") else _head
            torch.nn.init.normal_(_last.weight, std=0.02)
            if _last.bias is not None:
                torch.nn.init.normal_(_last.bias, std=0.02)
        dt, dl, _ = ci(cv_of(0), return_parts=True, n_pos=len(t))
        dtD = ci.expand_to_d(dt)
        dlD = ci.expand_to_d(dl)
    print(f"\n" + "-" * 96)
    print(f"  {tag}")
    print("-" * 96)
    print(f"  cond_init 参数量 = {nparam/1e6:.2f}M   "
          f"总参数 = {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    print(f"  dtheta原始形状 = {tuple(dt.shape)}  → 展开后 = {tuple(dtD.shape)}")

    # [2] ★ 组间差异 vs 组内差异
    x = dtD[0]                                   # [N, D]
    gid = ci._group_index(device)                # [D]
    # 组间：不同组在同一批通道上的空间形状差异
    inter = []
    for g1 in range(ci.n_groups):
        for g2 in range(g1 + 1, ci.n_groups):
            m1 = (gid == g1)
            m2 = (gid == g2)
            if m1.any() and m2.any():
                a1 = x[:, m1].mean(dim=1)
                a2 = x[:, m2].mean(dim=1)
                inter.append(float((a1 - a2).abs().mean()))
    intra = float(x.std(dim=1).mean())
    imean = float(np.mean(inter)) if inter else 0.0
    print(f"  组内（同一组跨位置）std = {intra:.6e}")
    print(f"  组间（两组空间形状差异）均值 = {imean:.6e}"
          f"   ← ★ 修复生效则此值 > 0")
    if not inter or imean < 1e-9:
        print("    ⇒ 全零：这是**旧行为**，125 通道共享同一形状（无空间结构）")
    else:
        print("    ⇒ ★ 组间确实产生了不同空间结构，per-band 通路已打通")

    # 组内一致性：同组内任意两通道的空间形状是否一致
    same_ok = []
    for g in range(ci.n_groups):
        msk = (gid == g).nonzero().flatten()
        if msk.numel() < 2:
            continue
        A = x[:, msk[0]]
        B = x[:, msk[1]]
        same_ok.append(float((A - B).abs().max()))
    if same_ok:
        print(f"  组内两通道最大差= {max(same_ok):.3e}（≈0 = 组内共享，"
              f"省参数设计成立）")

    # [3] 条件区分度
    with torch.no_grad():
        dt2, _, _ = ci(cv_of(1), return_parts=True, n_pos=len(t))
        dt2D = ci.expand_to_d(dt2)
    cs = float(torch.nn.functional.cosine_similarity(
        dtD.reshape(-1), dt2D.reshape(-1), dim=0))
    print(f"  两条件调制量余弦 = {cs:+.4f}  （<1 说明条件可区分）")
    del m, x
    gc.collect(); torch.cuda.empty_cache()

print("\n" + "-" * 96)
print("[4] drift_field 原型分支崩溃是否修复")
print("-" * 96)
torch.manual_seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=512, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
m.eval()
a = torch.rand(1, len(t), D, device=device) * 0.5
th = torch.rand(1, len(t), D, device=device) * 6.28 - 3.14
blk = m.pmnncore.blocks[0]
gi = torch.arange(len(t), device=device) // m.n_modules
mod_id = gi % m.n_modules
try:
    with torch.no_grad():
        g = blk.drift(a, th, None, None, mod_id=mod_id,
                      cond_anchor=None, anchor_weight=0.0)
    print(f"  ✓ 修复：targets=None + cond_anchor=None 正常返回 "
          f"g.shape={tuple(g.shape)}")
except Exception as e:
    print(f"  ✗ 仍崩溃：{type(e).__name__}: {e}")

print("\n" + "-" * 96)
print("[5] 全模型前向冒烟测试（含 per-band 调制）")
print("-" * 96)
try:
    with torch.no_grad():
        aa, tt, dg = m(texts=None, target_field=None, t_coord=t, s_coord=s,
                       batch_size=1, cond_vec=cv_of(0), use_cond_field=True)
    print(f"  ✓ 前向通过  a={tuple(aa.shape)} θ={tuple(tt.shape)}")
    print(f"    a 的 std = {float(aa.std()):.4f}  "
          f"跨通道 std（空间结构是否被表达）= "
          f"{float(aa[0].std(dim=1).mean()):.6e}")
except Exception as e:
    import traceback
    print(f"  ✗ 前向失败：{type(e).__name__}: {e}")
    traceback.print_exc()
print("=" * 96)