# -*- coding: utf-8 -*-
"""诊断 A 方案相位监督的信噪比。

假设：`supervise()` 的相位项用 w_ph = a_tar² 加权，而 a_tar 逐通道均值
跨 727 倍（DC 通道独占 ~97% 能量），导致 sin/cos 的加权和被 DC 通道
独占 → 监督目标退化成"整张图的平均相位"，对 src0/src1 几乎一样
→ 模型学到的 dtheta 指向一个与素材无关的方向 → margin_B 恒负。

本脚本测三件事：
  1. cond_init 对不同条件的输出分化度（有没有在学）
  2. 相位监督目标的通道构成（DC 占多少权重）
  3. 排除 DC 后监督目标的 src0/src1 区分度
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

import numpy as np
import torch
from codec.gif_codec import encode_gif_to_field
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "台球8号"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "二次元表情包"),
]
gif_dir = ROOT / "data" / "gif_raw" / "noto"

fields, a_tars, th_tars = [], [], []
for name, _ in GIF_SPECS:
    f = encode_gif_to_field(str(gif_dir / name), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]))
a_tar = torch.stack(a_tars, 0)
th_tar = torch.stack(th_tars, 0)

ckpt = ROOT / "outputs_real" / "cond2" / "cond2_ckpt.pt"
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
sd = torch.load(ckpt, map_location=device)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[diag] ckpt 加载 missing={len(missing)} unexpected={len(unexpected)}")
model.eval()

cv = np.load(ROOT / "data" / "clip_cache" / "cv_both.npy")
print(f"[diag] 条件向量 {cv.shape}")

# ---------- 1. cond_init 的条件分化度 ----------
print("\n[1] cond_init 输出分化度（150 步后）")
with torch.inference_mode():
    dths, dlns = [], []
    for i in range(2):
        for j in range(2):
            one = torch.from_numpy(cv[i:i+1, :]).to(device)
            # 换一种条件类型再看
            one2 = torch.from_numpy(cv[j:j+1, :]).to(device)
            dt, dl, _ = model._cond_init(one2, 6144)
            dths.append(dt[0, :, 0].cpu())
            dlns.append(dl[0, :, 0].cpu())
    # 条件 A 的第 0 条 vs 第 1 条（两条不同素材）
    d0, d1 = dths[0], dths[1]
    print(f"    条件 src0: dtheta std={d0.std():.4f} mean={d0.mean():+.4f} absmax={d0.abs().max():.4f}")
    print(f"    条件 src1: dtheta std={d1.std():.4f} mean={d1.mean():+.4f} absmax={d1.abs().max():.4f}")
    diff = (d0 - d1)
    print(f"    两条件差: std={diff.std():.4f} absmax={diff.abs().max():.4f}")
    print(f"    → 相对分化度 = diff.std / d0.std = {diff.std() / max(d0.std().item(), 1e-8):.4f}")
    print(f"    （<0.1 说明两条件的 dtheta 几乎一样 = 条件通路没分化）")

# ---------- 2. 相位监督的通道权重构成 ----------
print("\n[2] 相位监督的通道权重构成（w_ph = a_tar²）")
for i, (name, _) in enumerate(GIF_SPECS):
    w = (a_tar[i].detach() ** 2).clamp(min=1e-8)     # [N,D]
    per_ch = w.mean(dim=0)                            # [D]
    tot = per_ch.sum()
    ch0 = per_ch[0] / tot
    top5 = per_ch.topk(5).values.sum() / tot
    print(f"    {name}: ch0(DC) 占 {ch0*100:.1f}%  top5 占 {top5*100:.1f}%  "
          f"ch0/ch50 = {per_ch[0]/per_ch[50].clamp(min=1e-12):.1f}x")

# ---------- 3. 监督目标的 src0/src1 区分度 ----------
print("\n[3] 相位监督目标的 src0/src1 区分度")
# a_tar / th_tar 形状 [N, D]（单素材），不是 [B,N,D]
for tag, chan_mask in [("全通道", slice(0, D)), ("排除 DC(ch0)", slice(1, D))]:
    dirs = []
    for i in range(2):
        w = (a_tar[i].detach() ** 2).clamp(min=1e-8)[:, chan_mask]   # [N, k]
        th = th_tar[i][:, chan_mask]
        s = (w * torch.sin(th)).sum(dim=1)     # [N]
        c = (w * torch.cos(th)).sum(dim=1)
        n = torch.sqrt(s ** 2 + c ** 2).clamp(min=1e-6)
        dirs.append((s / n, c / n))          # [N]
    d0, d1 = dirs[0], dirs[1]
    cos_between = (d0[0] * d1[0] + d0[1] * d1[1]).mean()
    coherence0 = d0[0].mean()                # 越高=该素材空间相位越一致
    coherence1 = d1[0].mean()
    print(f"    {tag}: src0/src1 目标方向余弦={cos_between:+.4f}  "
          f"空间一致性 src0={coherence0:+.4f} src1={coherence1:+.4f}")
    print(f"      → 目标方向余弦越接近 1，两素材的相位监督目标越难区分")
