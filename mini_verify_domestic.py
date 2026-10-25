# -*- coding: utf-8 -*-
"""
Mini 条件生成验证（国内模型版）

目标：
  只用两个gif样本验证：基于CLIP特征的生成能力
  使用国内可访问的模型：BAAI/bge-m3 (文本) + clip-vit-base (视觉)

模型路径（需要你先在本地下载好，脚本用本地路径）
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, 'D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/src')

import numpy as np
import torch
from PIL import Image

# 国内模型路径（你需先手动 dl 下来）
LOCAL_TEXT_MODEL = 'D:/AetherGIF-V1/models/BAAI/bge-m3'
LOCAL_IMG_MODEL = 'D:/AetherGIF-V1/models/clip-vit-base-patch16'

# GIF 路径（两个样本）
GIFS = [
    "docx/实验记录/versions/v4200_多模态CLIP产品版/data/gif_raw/noto/8-ball_1f3b1.gif",
    "docx/实验记录/versions/v4200_多模态CLIP产品版/data/gif_raw/noto/3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"
]
TEXTS = [
    "一个黑色的弹力球在白色背景下上下弹跳，球体表面有反光，弹跳幅度较大，整体是简单的几何动画，运动流畅。",
    "一个粉紫色的圆润卡通角色，身体柔软，动作缓慢，身体左右摇摆、上下起伏，类似音乐节奏或舞蹈的简单动画，没有复杂的细节，整体是柔和的2D风格。"
]

def load_text_model(path):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, ignore_mismatched_sizes=True)
    model = AutoModel.from_pretrained(path)
    model.eval()
    return model, tok

def load_img_model(path):
    from transformers import CLIPVisionModel, CLIPImageProcessor
    proc = CLIPImageProcessor()
    model = CLIPVisionModel.from_pretrained(path)
    model.eval()
    return model, proc

def encode_text(model, tokenizer, texts, batch_size=8):
    encs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            emb = model(**tokens).last_hidden_state[:, 0, :]  # .pooler_output if needed
        encs.append(emb.float().cpu().numpy())
    return np.concatenate(encs, axis=0)

def encode_image(model, processor, images, batch_size=8):
    encs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        px = processor(images=batch, return_tensors="pt", do_resize=True, do_center_crop=True, do_normalize=True)
        px = px["pixel_values"].float()
        with torch.no_grad():
            emb = model(pixel_values=px).last_hidden_state[:, 0, :]  # pooler_output if needed
        encs.append(emb.float().cpu().numpy())
    return np.concatenate(encs, axis=0)

def load_gif_frames(gif_path):
    with Image.open(gif_path) as im:
        try:
            im.seek(0)
        except EOFError:
            pass
        return im.convert("RGB")

def main():
    print("Loading models...")
    text_model, tok = load_text_model(LOCAL_TEXT_MODEL)
    img_model, im_proc = load_img_model(LOCAL_IMG_MODEL)

    print("Encoding texts...")
    text_embs = encode_text(text_model, tok, TEXTS, batch_size=8)
    print("Text embs shape:", text_embs.shape)

    print("Encoding images...")
    img_embs = []
    for g in GIFS:
        img = load_gif_frames(g)
        img_embs.append(img)
    img_embs = encode_image(img_model, im_proc, img_embs, batch_size=8)
    print("Image embs shape:", img_embs.shape)

    # 去均值
    text_embs = text_embs - text_embs.mean(axis=0, keepdims=True)
    text_embs = text_embs / np.linalg.norm(text_embs, axis=-1, keepdims=True) + 1e-8
    img_embs = img_embs - img_embs.mean(axis=0, keepdims=True)
    img_embs = img_embs / np.linalg.norm(img_embs, axis=-1, keepdims=True) + 1e-8

    # 合并
    cond = []
    for t, i in zip(text_embs, img_embs):
        cond.append(np.concatenate([t, i, np.array([1.0], dtype=np.float32)]))
    cond = np.array(cond, dtype=np.float32)
    print("Cond shape:", cond.shape)

    # 保存到本地，用于后续验证
    import json
    os.makedirs("mini_data", exist_ok=True)
    np.save("mini_data/cond.npy", cond)
    np.save("mini_data/text_embs.npy", text_embs)
    np.save("mini_data/image_embs.npy", img_embs)
    with open("mini_data/meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_text": LOCAL_TEXT_MODEL,
            "model_img": LOCAL_IMG_MODEL,
            "n_concat": 2049,
            "centered": True
        }, f, ensure_ascii=False, indent=2)

    print("Mini condition saved to mini_data/.")

if __name__ == "__main__":
    main()
