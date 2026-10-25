# -*- coding: utf-8 -*-
"""
极简条件生成验证脚本

只使用两个gif文件：
  1. 8-ball_1f3b1.gif
  2. 3D87B14FBB731ED3A948AC6DD9EA6EBC.gif

目标：
  - 只对这两个样本进行训练
  - 测试 model 能否基于 CLIP 特征生成 GIF
  - 三种生成能力：文生、图生、图文生
"""
import sys
sys.path.insert(0, 'D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/src')

import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

# 简单结果：两个gif在原始 index.json 里的 Index
# 先直接查找原始 index.json 里的正确索引位置
index_path = 'D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/data/index.json'
gif_raw_root = 'D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/data/gif_raw'

print("Loading original index.json...")
with open(index_path, 'r', encoding='utf-8') as f:
    index_data = json.load(f)

# 找到两个gif的索引
gb1_path = 'noto/8-ball_1f3b1.gif'
gb2_path = 'noto/3D87B14FBB731ED3A948AC6DD9EA6EBC.gif'

idx1 = idx2 = None
for i, item in enumerate(index_data):
    if item.get('path') == gb1_path:
        idx1 = i
    if item.get('path') == gb2_path:
        idx2 = i
        break

print(f"Found indexes: idx1={idx1}, idx2={idx2}")

if idx1 is None or idx2 is None:
    raise RuntimeError("Not found in index.json! Should add entries.")

# 两个gif的文件路径
gif_paths = [
    Path(gif_raw_root) / gb1_path,
    Path(gif_raw_root) / gb2_path
]

# 对应的文本描述（从 index 里提取）
texts = [index_data[idx1]['text'], index_data[idx2]['text']]

# 预加载 v4200 的 clip_cache，方便后续直接读取两个样本的特征
print("Loading clip_cache features...")
text_embs = np.load('D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/data/clip_cache/text_emb.npy')
image_embs = np.load('D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/data/clip_cache/image_emb.npy')
image_ok = np.load('D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/data/clip_cache/image_ok.npy')

# 找出这两个gif在 image_embs 里的正确位置对应的索引
# 我们需要知道它们在整个 index.json 中的位置
i_e1 = image_embs[idx1]  # 因为 image_embs 的每一行对应 index.json 中的一行
i_e2 = image_embs[idx2]
print(f"i_e1 shape: {i_e1.shape} | i_e2 shape: {i_e2.shape}")

# 加载目标场（target field）
# 这里需要先调用 encode_gif_to_field 函数将 GIF 编码成 6144x250 的 target 场
from codec.gif_codec import encode_gif_to_field
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6

# 预先计算一次
target1 = encode_gif_to_field(str(gif_paths[0]), TS, GRID, PATCH, WIN, IMG, D)
target2 = encode_gif_to_field(str(gif_paths[1]), TS, GRID, PATCH, WIN, IMG, D)

print("Target fields loaded, shape:", target1.shape)

# 自定义 Dataset，只包含这两个样本
class MiniDataset(Dataset):
    def __init__(self, text_embs, image_embs, idx, target_fields):
        self.idx = idx
        self.text_embs = text_embs[idx]
        self.image_embs = image_embs[idx]
        self.target_fields = target_fields
        self.device = "cuda"
    
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, i):
        # cond_vec = [text_emb, image_emb, has_image]
        t_e = self.text_embs[i, :1024]
        i_e = self.image_embs[i, :1024]
        has_img = 1.0
        cond = np.concatenate([t_e, i_e, np.array([has_img], np.float32)])
        field = self.target_fields[i]
        return cond, field

dataset = MiniDataset(text_embs, image_embs, [idx1, idx2], torch.stack([target1, target2]))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

# 现在加载模型
print("Loading model...")
from model.aether_gif_model import AetherGIFModel
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)

# 从 v4200 的输出权重加载
ckpt_path = 'D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/outputs_real/ckpt/latest.pt'
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck["model"], strict=False)
    print(f"Loaded ckpt, step={ck.get('step','?')}")
else:
    print("No ckpt found, using newly initialized model")

# 现在训练一个 very small amount of steps (比如 1 步) 看看梯度是否正常传递
# 但关键是要看 cond_field 是否能学到特征，这需要更多步

print("Sample one batch:")
for cond, field in dataloader:
    print("cond shape:", cond.shape)
    print("field shape:", field.shape)
    break

# 测试两个模式的生成
print("\nTesting inference:")
with torch.no_grad():
    # 先测试 condition 前向 (test cond)
    for cond_vec, target in dataloader:
        target = target.to(device) if not isinstance(target, torch.Tensor) else target.to(device)
        # 简化调用，先不用 batch 而是单个样本
        # 对于 batch=2 的情况，需要确保所有数据都正确
        print("\nForward pass:")
        a, theta, diag = model(cond_vec, target_field=target,
                               t_coord=torch.load('D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/src/physics/kuramoto_sync.py').build_spacetime_index(6,32,32,device),
                               s_coord=None, batch_size=2)
        print("a, theta, diag:", a.shape, theta.shape, diag)
        break

# 接下来的任务是：运行一个简单训练，观察 loss 和 cond_field 参数是否变化

print("\nDone, this is a proof-of-concept setup. Next step: train for a few steps and test generation.")
