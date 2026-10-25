# -*- coding: utf-8 -*-
"""GIF -> mp4（无损像素）-> 走真视频分支编码，验证区分度恢复。"""
import sys
import os
import tempfile
import numpy as np
import torch
from PIL import Image, ImageSequence

sys.path.insert(0, r"D:\AetherGIF-V1")

from int8_qwen_encoder import Int8QwenEncoder, INSTR_VIDEO

GIF_DIR = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto"
GIFS = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]


def gif_to_mp4(gif_path, mp4_path):
    """GIF 帧 -> mp4，用 GIF 自身帧延迟做 fps。返回 (fps, nframes)。"""
    frames = []
    with Image.open(gif_path) as im:
        for fr in ImageSequence.Iterator(im):
            frames.append(np.array(fr.convert("RGB")))
        dur = im.info.get("duration", 100)          # 每帧延迟 ms
    arr = np.stack(frames).astype(np.uint8)          # [T,H,W,3]
    fps = max(1, round(1000.0 / max(dur, 10)))
    from torchvision.io import write_video
    write_video(mp4_path, torch.from_numpy(arr), fps=fps, options={"crf": "18"})
    return fps, arr.shape[0]


enc = Int8QwenEncoder()
enc._ensure()
tmpdir = tempfile.mkdtemp()
outs, meta = [], []
for g in GIFS:
    gp = f"{GIF_DIR}\\{g}"
    mp4 = os.path.join(tmpdir, g.replace(".gif", ".mp4"))
    fps, nfr = gif_to_mp4(gp, mp4)
    meta.append((g, fps, nfr))
    emb = enc._process([{"video": mp4, "instruction": INSTR_VIDEO,
                         "fps": 1.0, "max_frames": 8}])
    outs.append(emb.float().cpu().numpy())
    print(f"[mp4 真视频] {g}: 源fps={fps} 源帧数={nfr} -> embedding 完成")

mp4_vecs = np.concatenate(outs, 0).astype(np.float32)
cos01 = float(mp4_vecs[0] @ mp4_vecs[1])
old = np.load(r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache\gif3.npy")
first3 = np.load(r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache\first3.npy")
text3 = np.load(r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache\text3.npy")
print("\n=== 三路向量区分度（两素材间余弦，越小越好）===")
print(f"  文本        : {float(text3[0] @ text3[1]):+.4f}")
print(f"  首帧静图    : {float(first3[0] @ first3[1]):+.4f}")
print(f"  GIF 帧列表  : {float(old[0] @ old[1]):+.4f}   <- 旧缓存（实际在编码帧）")
print(f"  GIF->mp4视频: {cos01:+.4f}   <- 新（真视频分支）")
np.save(r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache\gif3_video.npy", mp4_vecs)
print("[SAVE] gif3_video.npy")
