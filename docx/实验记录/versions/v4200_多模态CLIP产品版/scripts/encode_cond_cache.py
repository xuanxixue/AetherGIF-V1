# -*- coding: utf-8 -*-
"""一次性编码条件，存到 data/clip_cache/，供训练直接读取。

使用与过拟合脚本完全相同的文本描述，避免仗义不一致。
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
from int8_qwen_encoder import Int8QwenEncoder
from data.clip_condition import build_cond_vector

# 文本描述必须与 overfit_cond2.py 完全一致
GIF_SPECS = [
    ("8-ball_1f3b1.gif", "一个8号台球在绿色台球桌上滚动"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "一张二次元Q版风格的动漫表情包——一个扎着红色蝴蝶结、穿深色水手服的黑色长发少女，睁着倒V型眼睛，脸颊泛红，双手举在头部两侧，前台词\"前辈前辈~\"，表情激动又撒娇"),
]

gif_dir = ROOT / "data" / "gif_raw" / "noto"
OUT = ROOT / "data" / "clip_cache"
OUT.mkdir(parents=True, exist_ok=True)

print(f"[encode] 文本描述：")
for name, desc in GIF_SPECS:
    print(f"  {name}: {desc}")

enc = Int8QwenEncoder(max_frames=8)
desc_list = [d for _, d in GIF_SPECS]
t_emb = enc.encode_texts(desc_list, batch_size=2)
i_emb = enc.encode_gif_paths([str(gif_dir / n) for n, _ in GIF_SPECS], batch_size=1)

ZERO = np.zeros_like(t_emb)
cv_text = build_cond_vector(t_emb, ZERO, [0, 0])
cv_image = build_cond_vector(ZERO, i_emb, [1, 1])
cv_both = build_cond_vector(t_emb, i_emb, [1, 1])

# 保存
np.save(OUT / "cv_text.npy", cv_text)
np.save(OUT / "cv_image.npy", cv_image)
np.save(OUT / "cv_both.npy", cv_both)
print(f"[encode] 条件已缓存到：\n  {OUT / 'cv_text.npy'}\n  {OUT / 'cv_image.npy'}\n  {OUT / 'cv_both.npy'}")
