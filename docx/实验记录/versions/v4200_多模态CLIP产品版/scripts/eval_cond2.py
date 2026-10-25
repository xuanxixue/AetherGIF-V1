# -*- coding: utf-8 -*-
"""独立条件生成评测 —— 不重跑训练，直接吃 overfit_cond2 存的 ckpt。

用途：训练中途（如 300/600 步）提前评 margin 趋势，方向错了立刻止损。

【运行】
cd 到 v4200 快照根目录后
    python scripts/eval_cond2.py [ckpt路径(默认 outputs_real/cond2/cond2_ckpt.pt)]

判据同 overfit_cond2：交叉 PSNR 矩阵，对角必须显著大于非对角。
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ★ 依赖注入与 overfit_cond2.py 完全一致（deps312 在项目根）
_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir():
        break
    if _PROJ.parent == _PROJ:
        break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))
_DEPS = _PROJ / "deps312"
_VENV = _PROJ / ".venv" / "Lib" / "site-packages"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
if _VENV.is_dir():
    sys.path.append(str(_VENV))
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass

try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] torch_accel_shim 加载失败：{ex}")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
from PIL import Image

from codec.gif_codec import decode_field_to_frames, encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from int8_qwen_encoder import Int8QwenEncoder
from data.clip_condition import build_cond_vector

# ---------------- 口径（与 overfit_cond2 完全一致） ----------------
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
MAX_FRAMES = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

GIF_SPECS = [
    ("8-ball_1f3b1.gif", "一个8号台球在绿色台球桌上滚动"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "一张二次元Q版风格的动漫表情包——一个扎着红色蝴蝶结、穿深色水手服的黑色长发少女，睁着倒V型眼睛，脸颊泛红，双手举在头部两侧，前台词\"前辈前辈~\"，表情激动又撒娇"),
]

CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs_real" / "cond2" / "cond2_ckpt.pt"
OUT_DIR = ROOT / "outputs_real" / "cond2"


def psnr(a_img, b_img):
    mse = ((a_img.astype(np.float32) - b_img.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


# ---------------------------------------------------------------- 素材
print("=" * 74)
print(f"[eval] ckpt = {CKPT}")
gif_dir = ROOT / "data" / "gif_raw" / "noto"
fields, a_tars, th_tars, ref_imgs = [], [], [], []
for name, desc in GIF_SPECS:
    gp = gif_dir / name
    if not gp.exists():
        gp = ROOT.parent.parent / "data" / "gif_raw" / "noto" / name
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]))
    z = (a_tars[-1] * torch.exp(1j * th_tars[-1])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    ref_imgs.append(np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32))

t, s = build_spacetime_index(TS, GRID, GRID, device)

# ---------------------------------------------------------------- 条件
print(f"[eval] int8 编码条件...")
enc = Int8QwenEncoder(max_frames=MAX_FRAMES)
desc_list = [d for _, d in GIF_SPECS]
t_emb = enc.encode_texts(desc_list, batch_size=2)
i_emb = enc.encode_gif_paths([str(gif_dir / n) for n, _ in GIF_SPECS], batch_size=1)
del enc
import gc
gc.collect()
torch.cuda.empty_cache()

ZERO = np.zeros_like(t_emb)
cv_text = build_cond_vector(t_emb, ZERO, [0, 0])
cv_image = build_cond_vector(ZERO, i_emb, [1, 1])
cv_both = build_cond_vector(t_emb, i_emb, [1, 1])

# ---------------------------------------------------------------- 模型
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(CKPT, map_location=device)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[eval] ckpt 加载: {len(sd)} 张量, missing={len(missing)}, unexpected={len(unexpected)}")
if missing:
    print(f"        missing 首几项: {missing[:5]}")
model.eval()

# ---------------------------------------------------------------- 交叉生成
print(f"\n[eval] 三种条件 × 两素材 交叉生成")
results = {}
with torch.inference_mode():
    for cv_name, cv in [("图生", cv_image), ("文生", cv_text), ("图文生", cv_both)]:
        for src_i in range(2):
            one = torch.from_numpy(cv[src_i:src_i + 1]).to(device)
            a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=one, use_cond_field=True)
            key = f"{cv_name}|src{src_i}"
            # 存 GIF（覆盖式：每次评测都产最新结果）
            z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
            fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
            frames = np.stack(decode_field_to_frames(
                fld, TS, GRID, PATCH, WIN, IMG, D, normalize=True)).astype(np.uint8)
            pil = [Image.fromarray(np.ascontiguousarray(fr)) for fr in frames]
            pil[0].save(OUT_DIR / f"{key.replace('|', '_')}.gif",
                        save_all=True, append_images=pil[1:], loop=0, duration=100)
            gen = np.stack(decode_field_to_frames(
                fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
            ps = [psnr(gen, ref_imgs[k]) for k in range(2)]
            results[key] = ps
            winner = "自己" if ps[src_i] > ps[1 - src_i] else "错了"
            print(f"    {key}: vs_A={ps[0]:6.2f} dB  vs_B={ps[1]:6.2f} dB  -> {winner}")

# ---------------------------------------------------------------- 判据
print(f"\n[eval] 判据")
all_ok = True
for cv_name in ["图生", "文生", "图文生"]:
    k0, k1 = f"{cv_name}|src0", f"{cv_name}|src1"
    if k0 not in results:
        continue
    d0 = results[k0][0] - results[k0][1]
    d1 = results[k1][1] - results[k1][0]
    ok = d0 > 0 and d1 > 0
    all_ok = all_ok and ok
    print(f"    [{'PASS' if ok else 'FAIL'}] {cv_name}: "
          f"margin_A={d0:+6.2f} dB  margin_B={d1:+6.2f} dB")

print(f"\n[eval] 总判定: {'PASS ✅ 条件可区分素材' if all_ok else 'FAIL ❌ 条件未能区分素材'}")
print("=" * 74)
