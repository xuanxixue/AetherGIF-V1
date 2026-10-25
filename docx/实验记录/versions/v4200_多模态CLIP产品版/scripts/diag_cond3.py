# -*- coding: utf-8 -*-
"""三向量方案诊断：区分「物理核生成能力不足」vs「条件向量没起作用」。

测两件事：
  A. 重建模式（forward 传 target_field）：物理核在有答案的情况下能学多好
  B. 生成模式（target_field=None，只给条件向量）：条件能唤起多少

若 A src1 也差 → 物理核容量问题（与昨日 cond2 同源，非三向量方案问题）
若 A 好但 B 差 → 条件通路问题（训练全给/推理缺省分布不匹配）

同时记录五条件的 dtheta（初始相位锚）在两素材间的分化程度。
"""
import sys
import os
import math
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

import torch
import numpy as np
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_SPECS = [
    ("8-ball_1f3b1.gif",),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif",),
]
gif_dir = ROOT / "data" / "gif_raw" / "noto"
fields, a_tars, th_tars = [], [], []
for (name,) in GIF_SPECS:
    gp = gif_dir / name
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]))
a_tar = torch.stack(a_tars, 0)
th_tar = torch.stack(th_tars, 0)
target = torch.stack(fields, 0)
t, s = build_spacetime_index(TS, GRID, GRID, device)

# ★ ckpt 路径可用 DIAG_TAG 指定（默认 cond3）：
#     DIAG_TAG=cond3_cross150 ... 诊断短训产物
#   注意：v4200 标量加权改造后旧 ckpt（含 cond_projector、旧头 896 维）
#   无法加载（shape mismatch），只能诊断重训后的新 ckpt。
ckpt = ROOT / "outputs_real" / (os.environ.get("DIAG_TAG") or "cond3") / "cond3_ckpt.pt"
# ★ 诊断 h512 容量探针产物时必须传 FFN_HIDDEN=512，否则 shape mismatch
_FFN_H = int(os.environ.get("FFN_HIDDEN", "256"))
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=_FFN_H,
                       init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
sd = torch.load(ckpt, map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()
print(f"[LOAD] {ckpt}")


def wcos_of(a, th, src_i):
    pe = torch.remainder(th - th_tar[src_i:src_i + 1] + math.pi, 2 * math.pi) - math.pi
    w_ph = (a_tar[src_i:src_i + 1] ** 2).clamp(min=1e-8)
    c = torch.cos(pe)
    return (w_ph * c).sum().item() / w_ph.sum().item()


def amp_err_of(a, src_i):
    la = torch.log1p(a.clamp(min=1e-8))
    lt = torch.log1p(a_tar[src_i:src_i + 1].clamp(min=1e-8))
    return la.sub(lt).abs().mean().item()


CONDS = [("文生", "cv3_text.npy"), ("图生", "cv3_first.npy"), ("图文生", "cv3_tf.npy"),
         ("GIF本身", "cv3_gif.npy"), ("三路全给", "cv3_full.npy")]
cv_cache = ROOT / "data" / "clip_cache"

print("\n=== A. 重建模式（给 target_field，训练同口径）===")
with torch.inference_mode():
    for src_i in range(2):
        one_cv = torch.from_numpy(np.load(cv_cache / "cv3_full.npy")[src_i:src_i + 1]).to(device)
        a, th, diag = model(texts=None, target_field=target[src_i:src_i + 1],
                            t_coord=t, s_coord=s, batch_size=1,
                            cond_vec=one_cv, use_cond_field=True)
        print(f"  src{src_i}({GIF_SPECS[src_i][0][:12]}): wcos={wcos_of(a, th, src_i):+.4f} "
              f"amp_err={amp_err_of(a, src_i):.4f}")

print("\n=== B. 生成模式（纯条件，target_field=None）===")
dth_all = {}
with torch.inference_mode():
    for cv_name, fname in CONDS:
        cv = np.load(cv_cache / fname)
        row = []
        for src_i in range(2):
            one = torch.from_numpy(cv[src_i:src_i + 1]).to(device)
            a, th, diag = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                                batch_size=1, cond_vec=one, use_cond_field=True)
            wc = wcos_of(a, th, src_i)
            row.append(wc)
            # 记录初始相位锚的分化
            if cv_name == "三路全给":
                dth_all[f"src{src_i}"] = diag.get("dtheta")
        print(f"  {cv_name}: src0 wcos={row[0]:+.4f}  src1 wcos={row[1]:+.4f}")

print("\n=== C. 条件初始相位锚 dtheta 分化（三路全给）===")
if all(v is not None for v in dth_all.values()):
    d0 = dth_all["src0"].flatten()
    d1 = dth_all["src1"].flatten()
    # 两素材的初始锚方向余弦（若条件真区分，应该接近 -1 或至少明显偏离 +1）
    cos01 = torch.nn.functional.cosine_similarity(d0, d1, dim=0).item()
    print(f"  src0 vs src1 的 dtheta 余弦 = {cos01:+.4f}")
    print(f"  |dtheta| mean: src0={d0.abs().mean():.4f} src1={d1.abs().mean():.4f}")
    print(f"  → 接近 -1 = 两条件推出相反初始相位（条件起作用）；≈0 或 +1 = 没区分")

print("\n=== D. 结论判据 ===")
print("A.src1 差 → 物理核容量；A.src1 好但 B 差 → 条件通路；两者皆差 → 训练不足")
