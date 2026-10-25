# -*- coding: utf-8 -*-
"""任务 #66 验证：正式化后的 encode_video_files() 端到端核对。

核对三件事：
  1. PyAV 后端确实由 _ensure() 自动注入（无需脚本里先 install）
  2. encode_video_files 产出与验证脚本落盘的 gif3_video.npy 一致（同管线）
  3. 两素材区分度仍在（不是 NULL 兜底）
"""
import os
import sys

sys.path.insert(0, r"D:\AetherGIF-V1")
sys.path.insert(0, r"D:\AetherGIF-V1\int8_deps")
sys.path.insert(0, r"D:\AetherGIF-V1\deps312")

import numpy as np
from int8_qwen_encoder import Int8QwenEncoder

CACHE = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache"
GIF_DIR = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto"
GIFS = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]

enc = Int8QwenEncoder()
paths = [os.path.join(GIF_DIR, g) for g in GIFS]
vecs = enc.encode_video_files(paths)          # _ensure() 自动注入 PyAV

ref = np.load(os.path.join(CACHE, "gif3_video.npy"))
old_frames = np.load(os.path.join(CACHE, "gif3.npy"))

print("形状:", vecs.shape, "范数:", np.linalg.norm(vecs, axis=-1))
print("\n=== 与验证脚本落盘的 gif3_video.npy 对比（应一致）===")
print(f"  最大绝对差 : {float(np.abs(vecs - ref).max()):.3e}")
print(f"  余弦       : {float(vecs[0] @ ref[0]):+.6f} / {float(vecs[1] @ ref[1]):+.6f}")
print("\n=== 区分度（两素材间余弦，=1.0 就是 NULL 兜底）===")
print(f"  真视频分支 : {float(vecs[0] @ vecs[1]):+.4f}")
print(f"  帧列表分支 : {float(old_frames[0] @ old_frames[1]):+.4f}")

# 顺带核对注入确实生效（脚本里没有手动 install）
from qwen_vl_utils import vision_process as vp
print("\n[DBG] get_video_reader_backend() =", vp.get_video_reader_backend())
print("[DBG] backends =", list(vp.VIDEO_READER_BACKENDS.keys()))
