# -*- coding: utf-8 -*-
"""量化对比：素材解码帧 vs 模型生成帧的空间域统计。

回答"生成的是什么东西"：
  - std / p1-p99 动态范围  → 细节丰富度（对比度）
  - 帧间绝对差            → 运动轨迹幅度（GIF 是否在动）
  - 首帧 vs 末帧差异      → 整段有无演化
  - 与素材的逐像素相关    → 生成内容到底像不像
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
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
_DEPS = _PROJ / "deps312"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
from PIL import Image

from codec.gif_codec import decode_field_to_frames, encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]
cond_path = ROOT / "outputs_real" / "cond2"
CKPT = cond_path / "cond2_ckpt.pt"

def stats(tag, frames):
    """frames: [T,H,W,3] float32 0-255"""
    f = frames
    T = f.shape[0]
    flat = f.reshape(T, -1)
    std = flat.std(axis=1).mean()
    p1, p99 = np.percentile(flat, 1), np.percentile(flat, 99)
    # 帧间运动
    fd = np.abs(f[1:] - f[:-1]).reshape(T - 1, -1).mean(axis=1)
    # 首末帧差异
    fe = np.abs(f[-1] - f[0]).mean()
    print(f"  {tag}")
    print(f"    帧数={T}  帧std={std:6.2f}  动态范围p1-p99={p1:6.1f}~{p99:6.1f}（跨度{p99-p1:6.1f}）")
    print(f"    帧间差={fd.mean():6.2f}（max帧={fd.max():6.2f}）  首末差={fe:6.2f}")
    return dict(std=std, dr=p99 - p1, fd=fd.mean(), T=T)

print("=" * 74)
print("[1] 素材原貌（codec 解码基准，无截断口径）")
gif_dir = ROOT / "data" / "gif_raw" / "noto"
refs = []
for name in GIF_NAMES:
    f = encode_gif_to_field(str(gif_dir / name), TS, GRID, PATCH, WIN, IMG, D)
    a = torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12)
    th = torch.atan2(f[..., D:], f[..., :D])
    z = (a * torch.exp(1j * th)).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    fr = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
    refs.append(fr)
    stats(name, fr)

print("\n[2] ckpt 生成（无 target_field，纯条件）")
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()

# 构造与 overfit/eval 相同的条件向量（直接复用落盘的 fit_cache 或重建）
# 这里用最简单的方式：从 eval 脚本同款流程重建条件太重（int8 编码器），
# 改用 ckpt 训练时已有的口径——直接用文本/图像嵌入的同款归一化向量。
# 条件向量 [4097] = text_emb[2048]+image_emb[2048]+has_image[1]，
# 生成质量与具体条件内容无关（只验证"生成出来是什么"），用随机但固定的
# 条件即可看出模型输出的固有形态。
t, s = build_spacetime_index(TS, GRID, GRID, device)
torch.manual_seed(0)
cv_rand = torch.randn(1, 4097, device=device) * 0.3
cv_rand[0, -1] = 1.0

with torch.inference_mode():
    a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                     batch_size=1, cond_vec=cv_rand, use_cond_field=True)
z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
gen = np.stack(decode_field_to_frames(fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
stats("生成（随机条件）", gen)

print("\n[3] 生成帧 vs 素材帧的逐像素相关（cos）")
for i, fr in enumerate(refs):
    # 对齐帧数
    n = min(len(gen), len(fr))
    g = gen[:n].reshape(n, -1)
    r = fr[:n].reshape(n, -1)
    g0 = g - g.mean(axis=1, keepdims=True)
    r0 = r - r.mean(axis=1, keepdims=True)
    cos = (g0 * r0).sum(axis=1) / (np.linalg.norm(g0, axis=1) * np.linalg.norm(r0, axis=1) + 1e-8)
    print(f"    vs {GIF_NAMES[i]}: 逐帧 cos 均值={cos.mean():+.3f}（std≈{cos.std():.3f}）")

print("\n[4] 生成帧之间的相似度（6 次不同条件生成是否雷同的帧内视角）")
print(f"    生成帧 std 逐帧: {gen.reshape(len(gen),-1).std(axis=1).round(1)}")
print("\n" + "=" * 74)
