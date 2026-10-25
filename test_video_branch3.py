# -*- coding: utf-8 -*-
"""定位：GIF 路径在 process_vision_info 这一层到底发生了什么。
fetch_video 能解出帧，但最终 embedding 两素材完全相同（+1.0030），
所以问题在 process_vision_info / processor 之间。
"""
import sys
import logging
import numpy as np
import torch

sys.path.insert(0, r"D:\AetherGIF-V1")
sys.path.insert(0, r"D:\AetherGIF-V1\int8_deps")
sys.path.insert(0, r"D:\AetherGIF-V1\deps312")

logging.basicConfig(level=logging.INFO)

from int8_qwen_encoder import Int8QwenEncoder, INSTR_VIDEO

GIF_DIR = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto"
GIF_A = f"{GIF_DIR}\\8-ball_1f3b1.gif"
GIF_B = f"{GIF_DIR}\\3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"

enc = Int8QwenEncoder()
enc._ensure()
emb = enc._emb

from qwen_vl_utils.vision_process import process_vision_info

for tag, p in [("A", GIF_A), ("B", GIF_B)]:
    conv = emb.format_model_input(
        text=None, image=None, video=p,
        instruction=INSTR_VIDEO, fps=1.0, max_frames=8)
    print(f"\n=== 素材{tag}: conversation content 结构 ===")
    for c in conv[1]["content"]:
        v = c.get("video", c.get("text"))
        print(f"  type={c['type']}, val_type={type(v).__name__}, len/shape={len(v) if hasattr(v,'__len__') else v}")
    try:
        images, video_inputs, video_kwargs = process_vision_info(
            conv, image_patch_size=16,
            return_video_metadata=True, return_video_kwargs=True)
        print(f"  images={None if images is None else len(images)}")
        print(f"  video_inputs={None if video_inputs is None else [(type(v[0]).__name__, getattr(v[0],'shape',None), v[1]) for v in video_inputs]}")
        print(f"  video_kwargs={video_kwargs}")
    except Exception as e:
        import traceback
        print(f"  [process_vision_info 抛异常] {type(e).__name__}: {e}")
        traceback.print_exc()
