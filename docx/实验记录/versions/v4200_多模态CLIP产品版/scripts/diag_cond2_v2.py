# -*- coding: utf-8 -*-
"""诊断协议：看 protos 是否学会，演化是否抹平差异，输出 wcos 等。"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ★ 依赖注入必须「向上探测 + 剔除 C 盘残缺 transformers」，否则会命中
#   C:\Python312\Lib\site-packages\transformers（缺 utils/versions.py）
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

import math
import torch
import numpy as np
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

# 素材
GIF_SPECS = [
    ("8-ball_1f3b1.gif", "一个8号台球在绿色台球桌上滚动"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "一张二次元Q版风格的动漫表情包——一个扎着红色蝴蝶结、穿深色水手服的黑色长发少女，睁着倒V型眼睛，脸颊泛红，双手举在头部两侧，前台词\"前辈前辈~\"，表情激动又撒娇"),
]
gif_dir = ROOT / "data" / "gif_raw" / "noto"
fields, a_tars, th_tars = [], [], []
for name, desc in GIF_SPECS:
    gp = gif_dir / name
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]))
a_tar = torch.stack(a_tars, 0)
th_tar = torch.stack(th_tars, 0)

t, s = build_spacetime_index(TS, GRID, GRID, device)

ckpt = ROOT / "outputs_real" / "cond2" / "cond2_ckpt.pt"
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(ckpt, map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()

# 构造条件向量（与 overfit 中的缓存一致）
ZERO = np.zeros_like(np.load(ROOT / "data" / "clip_cache" / "cv_text.npy")[0])
cv_text = np.load(ROOT / "data" / "clip_cache" / "cv_text.npy")
cv_image = np.load(ROOT / "data" / "clip_cache" / "cv_image.npy")
cv_both = np.load(ROOT / "data" / "clip_cache" / "cv_both.npy")

def parse_diag(a, th, src_i, a_tar, th_tar):
    # L1: protos 场（但这里不需要，只用演化后的输出）
    # 生成模式：a, th 是最终输出
    pe = torch.remainder(th - th_tar[src_i:src_i + 1] + math.pi, 2 * math.pi) - math.pi
    la = torch.log1p(a.clamp(min=1e-8))
    lt = torch.log1p(a_tar[src_i:src_i + 1].clamp(min=1e-8))
    w = (a_tar[src_i:src_i + 1] / a_tar[src_i:src_i + 1].max().clamp(min=1e-8)).pow(0.5)
    loss_amp = (w * (la - lt).pow(2)).mean()
    w_ph = (a_tar[src_i:src_i + 1] ** 2).clamp(min=1e-8)
    loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
    c = torch.cos(pe)
    wcos = (w_ph * c).sum().item() / w_ph.sum().item()
    # 演化后的 sin 差异（看是否抹平）
    sin_diff = torch.sin(pe).abs().mean().item()
    return dict(
        loss_amp=loss_amp.item(),
        loss_ph=loss_ph.item(),
        wcos=wcos,
        sin_diff=sin_diff,
        a=a[0].mean().item(),
        th_mean=th[0].mean().item(),
    )

with torch.inference_mode():
    for cv_name, cv in [("图生", cv_image), ("文生", cv_text), ("图文生", cv_both)]:
        print(f"\n=== {cv_name} ===")
        for src_i in range(2):
            one = torch.from_numpy(cv[src_i:src_i + 1]).to(device)
            a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=one, use_cond_field=True)
            diag = parse_diag(a, th, src_i, a_tar, th_tar)
            print(f"  src{src_i}: loss_amp={diag['loss_amp']:.4f} loss_ph={diag['loss_ph']:.4f} wcos={diag['wcos']:+.4f} sin_diff={diag['sin_diff']:.4f} a_mean={diag['a']:.5f}")

print("\n=== 分析 ===")
print("若 wcos 对 src1 仍 < 0.5，说明条件相位没学会；即使用 direct weight，还需更长时间。")
