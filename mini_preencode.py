# -*- coding: utf-8 -*-
"""
Mini 预编码 —— 只处理两个 GIF 样本，用于验证条件生成能力。

本地路径：跳过 HuggingFace 下载，直接用 models/ 里的模型
"""
import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
v4200_root = Path('D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版')
sys.path.insert(0, str(v4200_root / 'src'))

import numpy as np
import torch
from PIL import Image

from data.clip_condition import ClipConditionEncoder, CLIP_EMB_DIM

# 模型本地路径（直接指定，跳过 HF 下载）
LOCAL_MODEL = 'D:/AetherGIF-V1/models/jina-clip-v2'

# 来源数据：v4200 的 gif_raw
GIF_ROOT = v4200_root / "data" / "gif_raw"
CACHE = ROOT / "mini_data" / "clip_cache"

# 只对这两个 GIF 进行编码
MINI_GIFS = [
    "noto/8-ball_1f3b1.gif",
    "noto/3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"
]

# 对应的文本描述（从 v4200 的 index.json 里摘的）
MINI_TEXTS = [
    "一个黑色的弹力球在白色背景下上下弹跳，球体表面有反光，弹跳幅度较大，整体是简单的几何动画，运动流畅。",
    "一个粉紫色的圆润卡通角色，身体柔软，动作缓慢，身体左右摇摆、上下起伏，类似音乐节奏或舞蹈的简单动画，没有复杂的细节，整体是柔和的2D风格。"
]

def main():
    # 设备选择
    device = "cpu"
    fp16 = False
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / 1024 ** 3
        print(f"[DEV] CUDA 可用，空闲 {free_gb:.2f} GB")
        if free_gb >= 2.5:
            device, fp16 = "cuda", True
    
    print(f"[DEV] 编码设备 {device}  fp16={fp16}")
    CACHE.mkdir(parents=True, exist_ok=True)
    
    # 直接用本地模型路径
    enc = ClipConditionEncoder(model_path=LOCAL_MODEL, device=device, fp16=fp16)
    
    # 文本编码
    t0 = time.time()
    text_emb = enc.encode_text(MINI_TEXTS, batch_size=8)
    print(f"[TEXT] 编码完成 {text_emb.shape}  耗时 {time.time()-t0:.1f}秒")
    
    # 图像编码
    images = []
    ok = []
    for path_str in MINI_GIFS:
        p = GIF_ROOT / path_str
        if not p.exists():
            ok.append(False)
            images.append(None)
        else:
            with Image.open(p) as im:
                try:
                    im.seek(0)
                except EOFError:
                    pass
                images.append(im.convert("RGB").copy())
                ok.append(True)
    
    # 验证至少有一个图像存在
    ok = np.array(ok)
    print(f"[IMG] 有效图像 {int(ok.sum())}/{len(ok)}")
    if int(ok.sum()) == 0:
        print("[FATAL] 所有图像都无法读取，请检查路径")
        return
    
    # 对有效的图像进行编码
    valid_images = [img for img, ok_ in zip(images, ok) if ok_]
    t0 = time.time()
    image_emb_valid = enc.encode_images(valid_images, batch_size=4)
    print(f"[IMG] 编码完成 {len(valid_images)} 张  耗时 {time.time()-t0:.1f}秒")
    
    # 合并到完整数组
    image_emb = np.zeros((len(ok), CLIP_EMB_DIM), dtype=np.float32)
    valid_idx = np.where(ok)[0]
    image_emb[valid_idx] = image_emb_valid
    
    # 去均值
    text_emb = ClipConditionEncoder.center_normalize(text_emb)
    image_emb[ok] = ClipConditionEncoder.center_normalize(image_emb[ok])
    
    # 保存
    np.save(CACHE / "text_emb.npy", text_emb)
    np.save(CACHE / "image_emb.npy", image_emb)
    np.save(CACHE / "image_ok.npy", ok)
    with open(CACHE / "group_list.json", "w", encoding="utf-8") as f:
        json.dump(["8ball", "3d87b1"], f)
    with open(CACHE / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "jinaai/jina-clip-v2",
            "emb_dim": CLIP_EMB_DIM,
            "n_items": len(ok),
            "n_groups": 2,
            "image_ok": int(ok.sum()),
            "n_concat": CLIP_EMB_DIM * 2 + 1,
            "centered": True,
            "text_sim": None,
            "image_sim": None,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"[SAVE] 迷你预编码完成 → {CACHE}")

if __name__ == "__main__":
    main()
