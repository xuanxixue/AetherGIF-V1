# -*- coding: utf-8 -*-
"""实测：GIF 文件路径走真视频分支 vs 帧列表分支，向量差异 + 可行性。

用户指出当前 gif 向量是"帧图拼接"得到的（list 分支），
要求改用 GIF 文件本身走视频解码分支（str 路径，fps 时序重采样）。
本脚本验证：① str 分支在本机能否跑通 ② 两种向量差多少。
"""
import sys
import numpy as np
import torch

sys.path.insert(0, r"D:\AetherGIF-V1")

from int8_qwen_encoder import Int8QwenEncoder, INSTR_VIDEO

GIF_DIR = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto"
GIFS = [
    "8-ball_1f3b1.gif",
    "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif",
]
paths = [f"{GIF_DIR}\\{g}" for g in GIFS]

enc = Int8QwenEncoder()

# ---- A. 帧列表分支（当前缓存用的方式）----
frames_vecs = enc.encode_gif_paths(paths)
print(f"[A 帧列表] shape={frames_vecs.shape}")

# ---- B. GIF 文件路径直传（真视频分支）----
outs = []
for p in paths:
    emb = enc._process([{"video": p, "instruction": INSTR_VIDEO,
                         "fps": 1.0, "max_frames": 8}])
    outs.append(emb.float().cpu().numpy())
path_vecs = np.concatenate(outs, 0).astype(np.float32)
print(f"[B 视频路径] shape={path_vecs.shape}")

names = ["8-ball", "表情包"]
for i, n in enumerate(names):
    cos = float(frames_vecs[i] @ path_vecs[i])
    print(f"  {n}: 帧列表 vs 视频路径 余弦 = {cos:+.4f}")
cos01_frames = float(frames_vecs[0] @ frames_vecs[1])
cos01_path = float(path_vecs[0] @ path_vecs[1])
print(f"两素材区分度: 帧列表={cos01_frames:+.4f}  视频路径={cos01_path:+.4f}")
print(f"峰值显存 = {torch.cuda.max_memory_reserved()/1024**2:.0f} MiB")

np.save(r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache\gif3_video.npy",
        path_vecs)
print("[SAVE] gif3_video.npy")
