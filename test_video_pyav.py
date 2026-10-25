# -*- coding: utf-8 -*-
"""注入 PyAV 视频解码后端，绕开本机坏的 torchvision read_video（无 video_fps）。

torchvision 0.20.1 + 新版 PyAV 不兼容：read_video 的 info 丢失 video_fps，
导致 qwen_vl_utils fetch_video 抛 KeyError连环，官方 _preprocess_inputs
兜底成 "NULL" 文本 -> GIF 视频向量两素材完全相同（+1.0030）。

PyAV 本体可用：av.open(GIF) 能拿到 average_rate（真 fps）+ 解出帧。
"""
import os
import sys

# 必须在 import qwen_vl_utils / int8 编码器之前注入
sys.path.insert(0, r"D:\AetherGIF-V1")
sys.path.insert(0, r"D:\AetherGIF-V1\int8_deps")
sys.path.insert(0, r"D:\AetherGIF-V1\deps312")


def install_pyav_backend():
    import time
    import math
    import logging
    import numpy as np
    import torch
    import av
    from qwen_vl_utils import vision_process as vp
    from qwen_vl_utils.vision_process import (
        smart_nframes, FRAME_FACTOR, logger as vp_logger)

    def _round_by_factor(x, factor):
        return int(round(x / factor)) * factor

    def _read_video_pyav(ele):
        video_path = ele["video"]
        if video_path.startswith("file://"):
            video_path = video_path[7:]
        st = time.time()
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps_num, fps_den = stream.average_rate.numerator, stream.average_rate.denominator
            video_fps = float(fps_num) / max(fps_den, 1)
            frames = [f.to_ndarray(format="rgb24")
                      for f in container.decode(video=0)]
        if len(frames) == 0:
            raise ValueError(f"PyAV 解码出 0 帧: {video_path}")
        arr = np.stack(frames)                                  # [T,H,W,3] uint8
        total_frames = arr.shape[0]
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        video = torch.from_numpy(arr[idx]).permute(0, 3, 1, 2)  # [T,3,H,W] uint8
        vp_logger.info(
            f"pyav: {video_path=}, {total_frames=}, fps={video_fps:.2f}, "
            f"sample={nframes}, time={time.time() - st:.3f}s")
        video_metadata = dict(
            fps=video_fps,
            frames_indices=idx,
            total_num_frames=total_frames,
            video_backend="pyav",
        )
        return video, video_metadata, sample_fps

    vp.VIDEO_READER_BACKENDS["pyav"] = _read_video_pyav
    # ★ get_video_reader_backend 读的是模块加载时的 env 快照（不是实时 getenv），
    #   且套了 lru_cache —— 事后设环境变量无效。直接替换函数最彻底。
    vp.get_video_reader_backend = lambda: "pyav"
    print("[pyav 后端已注入]")


install_pyav_backend()

import numpy as np                                    # noqa: E402
from qwen_vl_utils import vision_process as _vp_dbg    # noqa: E402
print("[DBG] FORCE_QWENVL_VIDEO_READER =", os.environ.get("FORCE_QWENVL_VIDEO_READER"))
print("[DBG] get_video_reader_backend() =", _vp_dbg.get_video_reader_backend())
print("[DBG] backends =", list(_vp_dbg.VIDEO_READER_BACKENDS.keys()))

from int8_qwen_encoder import Int8QwenEncoder, INSTR_VIDEO   # noqa: E402

GIF_DIR = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto"
GIFS = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]

enc = Int8QwenEncoder()
enc._ensure()
paths = [f"{GIF_DIR}\\{g}" for g in GIFS]

# GIF 文件直传走真视频分支
outs = []
for p in paths:
    emb = enc._process([{"video": p, "instruction": INSTR_VIDEO,
                         "fps": 1.0, "max_frames": 8}])
    outs.append(emb.float().cpu().numpy())
pyav_vecs = np.concatenate(outs, 0).astype(np.float32)

old = np.load(r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache\gif3.npy")
print("\n=== 区分度（两素材间余弦，越小越好；=1.0 就是同一个向量）===")
print(f"  GIF 帧列表分支 : {float(old[0] @ old[1]):+.4f}")
print(f"  GIF 真视频分支 : {float(pyav_vecs[0] @ pyav_vecs[1]):+.4f}   <- pyav 注入后")
print(f"  两种视频向量余弦: {float(pyav_vecs[0] @ old[0]):+.4f} / {float(pyav_vecs[1] @ old[1]):+.4f}")
np.save(r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\clip_cache\gif3_video.npy",
        pyav_vecs)
print("[SAVE] gif3_video.npy")
