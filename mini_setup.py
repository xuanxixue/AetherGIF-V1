# -*- coding: utf-8 -*-
"""
超级迷你验证训练脚本

只使用两个gif样本，用于验证：基于CLIP特征的生成能力
"""
import sys
sys.path.insert(0, 'D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版/src')

import os
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# v4200 目录
V4200_ROOT = Path('D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版')

# ---- 自定义数据集，只加载两个样本 ----
class MiniDataset(Dataset):
    def __init__(self, text_embs, image_embs, groups, image_ok, device):
        self.text_embs = text_embs
        self.image_embs = image_embs
        self.groups = groups
        self.image_ok = image_ok
        self.gidx = {g: i for i, g in enumerate(groups)}
        self.device = device
    
    def __len__(self):
        return len(self.groups)
    
    def __getitem__(self, idx):
        group = self.groups[idx]
        t_row = self.gidx[group]
        t_e = np.asarray(self.text_embs[t_row], dtype=np.float32)
        # 假设 image_ok[idx] 为 True
        # image 索引对应整个数据集，这里需要找到 image_emb 索引
        # 简化处理：我们已经有 image_emb 数组，直接取对应行
        # 由于我们重新创建了一个 minidataset，需要找到对应的 image_emb 索引
        # 实际上，我们直接传入了 image_embs 对应行
        i_e = np.asarray(self.image_embs[idx], dtype=np.float32)
        has_img = 1.0
        cond = np.concatenate([t_e, i_e, np.array([has_img], np.float32)])
        
        # 目标 field 需要从 v4200 的 field_cache 加载，这里简化：假设有，实际从缓存读取
        # 为了简化，我们不在迷你脚本里跑 encode，而是直接加载
        filepath = V4200_ROOT / 'data' / 'gif_raw' / self.groups[idx]
        # 这里完全绕过 field 加载，训练脚本会负责
        return cond

# 手动构造的迷你索引
mini_index = [
    {"path": "noto/8-ball_1f3b1.gif", "group": "8ball", "text": "一个黑色的弹力球在白色背景下上下弹跳，球体表面有反光，弹跳幅度较大，整体是简单的几何动画，运动流畅。"},
    {"path": "noto/3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "group": "3d87b1", "text": "一个粉紫色的圆润卡通角色，身体柔软，动作缓慢，身体左右摇摆、上下起伏，类似音乐节奏或舞蹈的简单动画，没有复杂的细节，整体是柔和的2D风格。"}
]

# 加载 v4200 的预编码特征
text_embs = np.load(V4200_ROOT / 'data' / 'clip_cache' / 'text_emb.npy')
image_embs = np.load(V4200_ROOT / 'data' / 'clip_cache' / 'image_emb.npy')
image_ok = np.load(V4200_ROOT / 'data' / 'clip_cache' / 'image_ok.npy')
groups = ['8ball', '3d87b1']

# 找到对应 group 在 text_emb 中的行号
group_to_idx = {}
group_names = []
with open(V4200_ROOT / 'data' / 'clip_cache' / 'group_list.json', 'r', encoding='utf-8') as f:
    all_groups = json.load(f)
for i, g in enumerate(all_groups):
    if g in groups:
        group_to_idx[g] = i

print("Text embeddings for groups:", group_to_idx)
print("Image embeddings for groups:")

# 在 mini_index 中找到 image_emb 对应行的索引（需要知道 image_ok 位置）
# 简化做法：整个数据集中的前两个 index 可能不是我们的样本，因为 v4200 索引包含 14145 个
# 我们需要搜索这些 gif 在原始 index.json 中的位置，看起来我们的 gif 在 presto 目录中的 "noto" 子目录下

# 找到 image_emb 对应行的方法：搜索 index.json 中的 path
import tempfile
# 创建一个临时索引
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
    json.dump(mini_index, tmp, ensure_ascii=False)
    tmp_idx = tmp.name

# 这里无法高效找到 image_emb 索引，直接加载所有 image_emb，然后只取前两个？不行，因为前两个可能不是我们的
# 解决办法：手动指定 image_emb 索引，通过搜索 index.json 里的 path
with open(V4200_ROOT / 'data' / 'index.json', 'r', encoding='utf-8') as f:
    all_index = json.load(f)

img_idx = []
for idx, item in enumerate(all_index):
    if item['group'] in groups:
        img_idx.append(idx)

print("Image index positions:", img_idx)

# 提取对应 image_emb 行
selected_image_embs = image_embs[img_idx]
print("Selected image_embs shape:", selected_image_embs.shape)

# 提取对应 text_emb 行
selected_text_embs = text_embs[[group_to_idx[g] for g in groups]]
print("Selected text_embs shape:", selected_text_embs.shape)

# 创建 mini dataset
dataset = MiniDataset(selected_text_embs, selected_image_embs, groups, image_ok, 'cuda')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

# 现在需要加载模型并训练
print("Minidataset ready: ", len(dataset), " samples")
print("Batch sizes:", [len(b) for b in dataloader])
print("Done. This is a skeleton script to understand the data format.")
