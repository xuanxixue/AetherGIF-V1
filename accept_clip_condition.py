# -*- coding: utf-8 -*-
"""验收：新 clip_condition.py 能导入 + 真实 GIF 端到端跑通。

检查项：
  1. CLIP_EMB_DIM == 2048（全局常量生效）
  2. CondProjector 形状 2048 -> 896
  3. build_cond_vector 拼出 4097 维
  4. encode_text 两条中文 -> 2048 维，L2 范数≈1
  5. encode_gif_paths 两个 GIF -> 2048 维，可区分（余弦 < 0.6）
  6. center_normalize 后区分度提升
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
VER = ROOT / "docx" / "实验记录" / "versions" / "v4200_多模态CLIP产品版"
sys.path.insert(0, str(VER / "src"))

# 必须先打 torch.accelerator 垫片（transformers 5.x 硬依赖，torch 2.5.1 没有）
sys.path.insert(0, str(ROOT))
# ★ 必须先注入 D 盘依赖：C:\Python312 的 transformers 残缺
#   （缺 utils/versions.py），好版本在 D:/AetherGIF-V1/deps312。
sys.path.insert(0, r"D:/AetherGIF-V1/deps312")
sys.path.append(r"D:/AetherGIF-V1/.venv/Lib/site-packages")
_bad = r"C:\Python312\Lib\site-packages\transformers"
import os as _os
try:
    sys.path = [p for p in sys.path
                if _os.path.normcase(_os.path.abspath(p))
                != _os.path.normcase(_os.path.abspath(_bad))]
except Exception:
    pass

from torch_accel_shim import install_full  # noqa: E402

install_full()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from data.clip_condition import (  # noqa: E402
    ClipConditionEncoder, CondProjector, CLIP_EMB_DIM, N_COND, build_cond_vector,
)

GIF_A = r"D:/AetherGIF-V1/data/gif_raw/noto/8-ball_1f3b1.gif"
GIF_B = r"D:/AetherGIF-V1/data/gif_raw/noto/Aquarius_2652.gif"
DESC_A = "一个黑色的8号台球在绿色台球桌上滚动"
DESC_B = "一个黄道十二星座的水瓶座符号在深色背景上发光"

ok = True


def check(name, cond, detail=""):
    global ok
    flag = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"  [{flag}] {name}{(' -> ' + detail) if detail else ''}")


print("=" * 72)
print("1) 常量与形状")
check("CLIP_EMB_DIM == 2048", CLIP_EMB_DIM == 2048, str(CLIP_EMB_DIM))
check("N_COND == 4096", N_COND == 4096, str(N_COND))

proj = CondProjector(out_dim=896)
x = proj(torch.randn(2, CLIP_EMB_DIM), torch.randn(2, CLIP_EMB_DIM),
         torch.tensor([1.0, 0.0]))
check("CondProjector 2048->896", tuple(x.shape) == (2, 896), str(tuple(x.shape)))

cv = build_cond_vector(np.random.randn(2, CLIP_EMB_DIM),
                       np.random.randn(2, CLIP_EMB_DIM), [1, 0])
check("build_cond_vector -> 4097", cv.shape[1] == 4097, str(cv.shape))

print("\n2) 文本条件编码")
enc = ClipConditionEncoder(device="cpu", max_frames=8)   # 8 帧：CPU 上 16 帧要 471s
t = enc.encode_text([DESC_A, DESC_B], batch_size=2)
check("shape == (2, 2048)", t.shape == (2, 2048), str(t.shape))
n = np.linalg.norm(t, axis=-1)
check("L2 范数≈1", bool(np.all(np.abs(n - 1) < 1e-3)), f"{n.round(4)}")
sim_t = float(t[0] @ t[1])
check("文本余弦 < 0.6（可区分）", sim_t < 0.6, f"{sim_t:.4f}")

print("\n3) GIF 视频条件编码（8 帧）")
g = enc.encode_gif_paths([GIF_A, GIF_B], batch_size=1)
check("shape == (2, 2048)", g.shape == (2, 2048), str(g.shape))
n = np.linalg.norm(g, axis=-1)
check("L2 范数≈1", bool(np.all(np.abs(n - 1) < 1e-3)), f"{n.round(4)}")
sim_g = float(g[0] @ g[1])
check("GIF 余弦 < 0.6（可区分）", sim_g < 0.6, f"{sim_g:.4f}")

print("\n4) 交叉模态矩阵")
mat = t @ g.T
check("行内配对最大 (A)", mat[0, 0] > mat[0, 1], f"{mat[0, 0]:.4f} > {mat[0, 1]:.4f}")
check("行内配对最大 (B)", mat[1, 1] > mat[1, 0], f"{mat[1, 1]:.4f} > {mat[1, 0]:.4f}")
margin = float(np.diag(mat).mean() - (mat.sum() - np.diag(mat).sum()) / 2)
check("margin > 0.05", margin > 0.05, f"{margin:+.4f}")

print("\n5) center_normalize 区分度提升")
# 造一个高基线集合（模拟 CLIP 各向异性）复现去均值价值
fake = np.random.randn(200, CLIP_EMB_DIM).astype(np.float32)
fake += np.tile(np.random.randn(1, CLIP_EMB_DIM).astype(np.float32) * 3.0, (200, 1))
fake = ClipConditionEncoder.l2_normalize(fake)
before = fake @ fake.T
np.fill_diagonal(before, np.nan)
after = ClipConditionEncoder.center_normalize(fake)
a2 = after @ after.T
np.fill_diagonal(a2, np.nan)
b_std = float(np.nanstd(before))
a_std = float(np.nanstd(a2))
check("去均值后 std 提升", a_std > b_std, f"{b_std:.4f} -> {a_std:.4f} ({a_std / b_std:.1f}x)")

print("\n" + "=" * 72)
print("总判定:", "PASS ✅ 新 clip_condition.py 可用" if ok else "FAIL ❌ 有检查项未过")
print("=" * 72)
sys.exit(0 if ok else 1)
