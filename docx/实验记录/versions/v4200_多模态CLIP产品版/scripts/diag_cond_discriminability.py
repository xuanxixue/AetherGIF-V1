# -*- coding: utf-8 -*-
"""秒级探针：条件向量到底能不能区分素材？（第二十六步，2026-10-18）

【为什么查这个】
  perband6训练日志里 dθcos 全程精确 +1.000。
  该指标 = **两个不同素材在同一条件下 dtheta 的余弦**，
  衡量的是「条件通路能否区分素材」——
  1.000 意味着两素材得到几乎同一个dtheta，
  即**条件通路输出与素材无关**。
  这比 dθ塌缩更早一层：连输入都没区分开。

【要查什么】
  [1] 两个素材的 CLIP 条件向量（text/first/gif 三路）逐路余弦
  [2] 经proj 后的 hidden 余弦
  [3] 经 to_phase 后的 dtheta 余弦（per-band G=5）
  [4] build_one_cv 的缺省路置零是否把差异抹平
"""
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
device = "cuda" if torch.cuda.is_available() else "cpu"
NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]

gif_dir = ROOT / "data" / "gif_raw" / "noto"
fields = [encode_gif_to_field(str(gif_dir / n), TS, GRID, PATCH, WIN, IMG, D).to(device)
          for n in NAMES]
t, s = build_spacetime_index(TS, GRID, GRID, device)

cc = ROOT / "data" / "clip_cache"
T3 = np.load(cc / "text3.npy")
F3 = np.load(cc / "first3.npy")
G3 = np.load(cc / "gif3_video.npy")
print(f"[cache] text3{T3.shape} first3{F3.shape} gif3{G3.shape}")

print("\n" + "=" * 88)
print("  [1] 原始 CLIP 向量：两素材逐路余弦")
print("=" * 88)
for nm, A in (("text", T3), ("first", F3), ("gif", G3)):
    a = torch.from_numpy(A[0:1]).float().to(device).reshape(-1)
    b = torch.from_numpy(A[1:2]).float().to(device).reshape(-1)
    cs = float(a @ b / (a.norm() * b.norm()))
    print(f"  {nm:6s} dim={a.numel():4d}  cos={cs:+.6f}  "
          f"|v0|={float(a.norm()):.4f} |v1|={float(b.norm()):.4f}  "
          f"L2差={float((a-b).norm()):.4f}")

print("\n" + "=" * 88)
print("  [2] build_cond_vector3 组合后 + 模型 proj 后")
print("=" * 88)
os.environ["COND_GROUPS"] = "5"
torch.manual_seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
# ★ 可选加载训练后 ckpt（2026-10-18）：关键对照 ——
#   未训练态 proj 把素材cos 0.632 → 0.984（差异被抹掉 99%），
#   训练后是否仍如此，直接决定是「未训练的临时塌缩」还是「架构学不开」。
_CK = os.environ.get("CKPT", "")
if _CK:
    _sd = torch.load(_CK, map_location=device)
    _sd = {k: v for k, v in _sd.items() if not k.startswith("qwen.")}
    _miss, _unexp = m.load_state_dict(_sd, strict=False)
    print(f"[ckpt] 载入 {_CK}  missing={len(_miss)} unexpected={len(_unexp)}")
else:
    print("[ckpt] 未加载（未训练态）")
m.eval()
ci = m.cond_init

cvs = []
for i in (0, 1):
    cv = torch.from_numpy(build_cond_vector3(T3[i:i + 1], F3[i:i + 1],
                                              G3[i:i + 1], (1, 1, 1))).to(device)
    cvs.append(cv)
c0, c1 = cvs


def cos3(x, y):
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    return float(x @ y / (x.norm() * y.norm() + 1e-12))


print(f"  cv 逐元素cos       = {cos3(c0, c1):+.6f}   dim={c0.numel()}")
with torch.no_grad():
    h0, h1 = ci.proj(c0), ci.proj(c1)
print(f"  proj(hidden) cos   = {cos3(h0, h1):+.6f}   dim={h0.numel()}")

print("\n" + "=" * 88)
print("  [3] dtheta / dlna 跨素材余弦（per-band）")
print("=" * 88)
with torch.no_grad():
    d0, l0, _ = ci(c0, return_parts=True, n_pos=len(t))
    d1, l1, _ = ci(c1, return_parts=True, n_pos=len(t))
print(f"  dtheta shape={tuple(d0.shape)}  dlna shape={tuple(l0.shape)}")
print(f"  dtheta 全向量 cos = {cos3(d0, d1):+.6f}   |d0|={float(d0.norm()):.4f}")
print(f"  dlna   全向量 cos = {cos3(l0, l1):+.6f}   |l0|={float(l0.norm()):.4f}")
print("  逐组 dtheta 余弦：")
G = d0.shape[-1]
for g in range(G):
    a = d0[0, :, g]
    b = d1[0, :, g]
    print(f"    组{g}: cos={float(a@b/(a.norm()*b.norm()+1e-12)):+.6f}"
          f"  |差|={float((a-b).abs().mean()):.3e}"
          f"  相对={float((a-b).abs().mean()/(a.abs().mean()+1e-12)):.4f}")

print("\n" + "=" * 88)
print("  [4] 判决")
print("=" * 88)
d_cos = cos3(d0, d1)
rel = float((d0 - d1).abs().mean() / (d0.abs().mean() + 1e-12))
print(f"  条件向量 cos      = {cos3(c0, c1):+.6f}")
print(f"  dtheta 跨素材 cos = {d_cos:+.6f}")
print(f"  dtheta 相对差异   = {rel:.6f}")
if d_cos > 0.999:
    print("  → ★★ 条件通路**无法区分素材**：两素材 dtheta 几乎同一个场。")
    print("     supervise 只要求「对齐某素材的 target 方向」，而两素材 target")
    print("     主导方向相似 ⇒输出一个折中通用场即可降 loss ⇒ 压根不需要区分。")
    print("     修法= 加**对比判别损失**（只有分开素材才能降 loss）。")
elif d_cos > 0.95:
    print("  → △ 区分度不足（cos>0.95），条件信号偏弱，建议加对比判别损失。")
else:
    print(f"  → 条件通路**能区分素材**（cos={d_cos:+.4f}< 0.95），输入侧无问题。")