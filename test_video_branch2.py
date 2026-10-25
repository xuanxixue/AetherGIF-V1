# -*- coding: utf-8 -*-
"""验证两件事：
1. GIF 路径走 fetch_video 抛什么异常（证实 NULL 兜底链路）
2. GIF 转 mp4 后能否走通真视频分支、区分度是否恢复
"""
import sys
import io
import logging
import numpy as np
import torch

sys.path.insert(0, r"D:\AetherGIF-V1")
sys.path.insert(0, r"D:\AetherGIF-V1\int8_deps")
sys.path.insert(0, r"D:\AetherGIF-V1\deps312")

logging.basicConfig(level=logging.WARNING)

GIF_DIR = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto"
GIFS = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]

# ---------- 1. 显式调 fetch_video 看异常 ----------
from qwen_vl_utils.vision_process import fetch_video

for g in GIFS:
    p = f"{GIF_DIR}\\{g}"
    try:
        v = fetch_video({"video": p, "fps": 1.0, "max_frames": 8})
        print(f"[fetch_video] {g}: 成功 shape={getattr(v, 'shape', type(v))}")
    except Exception as e:
        print(f"[fetch_video] {g}: {type(e).__name__}: {e}")

# ---------- 2. GIF -> mp4 -> 真视频分支 ----------
from PIL import Image, ImageSequence
import tempfile, os

def gif_to_mp4(gif_path, mp4_path, fps=10):
    frames = []
    with Image.open(gif_path) as im:
        for fr in ImageSequence.Iterator(im):
            frames.append(np.array(fr.convert("RGB")))
    arr = np.stack(frames)  # [T,H,W,3]
    try:
        from torchvision.io import write_video
        write_video(mp4_path, torch.from_numpy(arr).permute(0, 3, 1, 2), fps=fps)
        return True
    except Exception as e:
        print(f"  [write_video 失败] {type(e).__name__}: {e}")
        return False

from int8_qwen_encoder import Int8QwenEncoder, INSTR_VIDEO

enc = Int8QwenEncoder()
tmpdir = tempfile.mkdtemp()
outs = []
for g in GIFS:
    gp = f"{GIF_DIR}\\{g}"
    mp4 = os.path.join(tmpdir, g.replace(".gif", ".mp4"))
    ok = gif_to_mp4(gp, mp4)
    if not ok:
        print(f"[跳过] {g} 转 mp4 失败")
        continue
    emb = enc._process([{"video": mp4, "instruction": INSTR_VIDEO,
                         "fps": 1.0, "max_frames": 8}])
    outs.append(emb.float().cpu().numpy())
    print(f"[mp4 真视频分支] {g}: done")

if len(outs) == 2:
    mp4_vecs = np.concatenate(outs, 0).astype(np.float32)
    cos = float(mp4_vecs[0] @ mp4_vecs[1])
    print(f"两素材区分度(mp4真视频分支) = {cos:+.4f}")
    old = np.load(r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache\gif3.npy")
    print(f"对照: 帧列表分支={float(old[0] @ old[1]):+.4f}  NULL兜底分支=+1.0030")
    np.save(r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache\gif3_mp4.npy", mp4_vecs)
    print("[SAVE] gif3_mp4.npy")
print(f"显存峰值 = {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB")
