# -*- coding: utf-8 -*-
"""秒级诊断：条件通路的区分度在哪一层被吃掉？（第二十六步 b）

【背景】
  3000 步预热 + 对比判别损失后，跨素材 dtheta 余弦仍≈1.0（disc 顶格 0.4）。
  加惩罚压不下去 ⇒ 冲突不在监督权重，在**架构某处**。

【逐层剥离测量】
  对每一层（CLIP原始 / cond_vec / proj第1层Linear / proj第1层SiLU / proj第2层 / 输出头）
  测两素材的跨素材余弦，找余弦从低变高的**那一跳**。
  同时测「素材中心化后的余弦」—— 区分开 = 去掉共同分量后的差异。

【关键假设 H_modgap】
  CLIP 三路的**模态间隙**远大于素材间隙。proj 会先学会提取模态身份
  （文/图/GIF），素材身份只剩残差 → 余弦被推高。
  验证法：把三路分开测余弦，若三路各自余弦都高，则是模态主导。
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

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3, CLIP_EMB_DIM

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

cc = ROOT / "data" / "clip_cache"
T3 = np.load(cc / "text3.npy")
F3 = np.load(cc / "first3.npy")
G3 = np.load(cc / "gif3_video.npy")

os.environ["COND_GROUPS"] = "5"
torch.manual_seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
_CK = os.environ.get("CKPT", "")
if _CK:
    _sd = torch.load(_CK, map_location=device)
    m.load_state_dict({k: v for k, v in _sd.items() if not k.startswith("qwen.")},
                      strict=False)
    print(f"[ckpt] {_CK}")
m.eval()
ci = m.cond_init


def cos(x, y):
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    return float(x @ y / (x.norm() * y.norm() + 1e-12))


print("\n" + "=" * 90)
print("  [A] 逐层：两素材跨素材余弦（谁吃掉区分度）")
print("=" * 90)
cvs = [torch.from_numpy(build_cond_vector3(T3[i:i+1], F3[i:i+1], G3[i:i+1],
                                           (1, 1, 1))).to(device) for i in (0, 1)]
c0, c1 = cvs
with torch.no_grad():
    l0 = ci.proj[0](c0)
    l1 = ci.proj[0](c1)
    s0 = ci.proj[1](l0)
    s1 = ci.proj[1](l1)
    l2 = ci.proj[3](s0)
    l2b = ci.proj[3](s1)
    h0, h1 = ci.proj(c0), ci.proj(c1)
rows = [
    ("cond_vec (组合后)", c0, c1),
    ("proj[0] Linear1", l0, l1),
    ("proj[1] SiLU1", s0, s1),
    ("proj[3] Linear2", l2, l2b),
    ("proj[4] SiLU2  = h", h0, h1),
]
for nm, x, y in rows:
    # 中心化后余弦：剔除共同分量，看真实差异
    z = (x.reshape(-1) + y.reshape(-1)) / 2
    xc = x.reshape(-1) - z
    yc = y.reshape(-1) - z
    cc_ = cos(xc, yc)
    print(f"  {nm:22s} cos={cos(x,y):+.6f}   中心化后cos={cc_:+.6f}"
          f"   |x|={float(x.reshape(-1).norm()):8.3f}")

print("\n" + "=" * 90)
print("  [B] H_modgap 检验：三路各自过 proj[0] 的区分度")
print("=" * 90)
with torch.no_grad():
    for nm, A in (("text", T3), ("first", F3), ("gif", G3)):
        a = torch.from_numpy(A[0:1]).float().to(device)
        b = torch.from_numpy(A[1:2]).float().to(device)
        ra, rb = cos(a, b)
        pa = ci.proj[0](torch.cat([a, torch.zeros_like(a)], 1))
        pb = ci.proj[0](torch.cat([b, torch.zeros_like(a)], 1))
        print(f"  {nm:6s} 原始cos={ra:+.6f}  →  proj[0]后 cos={cos(pa,pb):+.6f}")

print("\n" + "=" * 90)
print("  [C] cond_vec 三段的方差结构（模态间隙是否占主导）")
print("=" * 90)
z = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
for nm, A in (("text", T3), ("first", F3), ("gif", G3)):
    seg = torch.from_numpy(np.concatenate([A[0:1], z, z, z], axis=-1)).to(device)
    print(f"  {nm:6s} 占位段 dim={seg.numel()}")
with torch.no_grad():
    full = torch.from_numpy(np.concatenate([T3[0:1], F3[0:1], G3[0:1]], 1)).to(device)
    ful2 = torch.from_numpy(np.concatenate([T3[1:2], F3[1:2], G3[1:2]], 1)).to(device)
    print(f"  直接三路拼接 cos = {cos(full, ful2):+.6f}")
    # 每路内 素材间差异 vs 路间差异（不同路之间）
    t = torch.from_numpy(T3[0:2]).float().to(device)
    f = torch.from_numpy(F3[0:2]).float().to(device)
    print(f"  路内(text) 两素材距离 = {float((t[0]-t[1]).norm()):.4f}")
    print(f"  路间(text0-first0) 距离 = {float((t[0]-f[0]).norm()):.4f}")
    print(f"  → 若路间 >> 路内，模态间隙主导，素材身份是残差")

print("\n" + "=" * 90)
print("  [D] 判决")
print("=" * 90)
if cos(s0, s1) > 0.95 and cos(c0, c1) < 0.9:
    print("  →区分度在 **proj[0]（第一层Linear）** 就被吃掉。")
    print("    proj 是 cond_dim→512 的大矩阵，两个差异大的输入进去后方向几乎一致，")
    print("    说明它主要提取**共同分量**（模态/全局均值），素材差异被当作噪声丢弃。")
    print("    修法方向：中心化 + 条件归一化，或给 proj 加 LayerNorm 断掉共同分量。")
else:
    print(f"  proj[0] cos={cos(s0,s1):.4f}，区分度未在此丢失，需继续下钻。")