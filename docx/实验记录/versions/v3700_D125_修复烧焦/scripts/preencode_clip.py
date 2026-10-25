# -*- coding: utf-8 -*-
"""v3700 对接 jina-clip-v2 图文双塔 —— 第一步：CLIP 离线预编码。

把 v3700 的两个素材 GIF 的首帧图像 + 文本描述，用冻结的 jina-clip-v2 编码成
[2049] 条件向量（text_emb[1024] + image_emb[1024] + has_image[1]），存 npy。

为什么要离线预编码（不在训练脚本里现编）：
  两者不能同时上卡 —— CLIP 视觉塔 512×512 fp16 前向占 1.73GB，
  物理模型 D=125 batch=1 训练占 2.55GB，4GB 卡装不下。先预编码存盘、
  释放 CLIP，再训物理模型，显存各管一摊。

本脚本退出前会显式 del 编码器 + torch.cuda.empty_cache()，把显存让给后续训练。
"""
import os
import sys
import json
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from data.clip_condition import ClipConditionEncoder, CLIP_EMB_DIM, build_cond_vector
from codec.gif_codec import load_gif_frames

CACHE = ROOT / "clip_cache"
CACHE.mkdir(exist_ok=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 两个素材 + 各自文本描述（jina-clip-v2 多语言，中英都行）
    samples = [
        {"gif": "8-ball_1f3b1.gif", "text": "a billiard eight ball, snooker ball with number 8"},
        {"gif": "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "text": "an anime character avatar"},
    ]
    gif_root = ROOT / "data" / "gif_raw" / "noto"

    enc = ClipConditionEncoder(device=device)

    text_embs, image_embs = [], []
    for s in samples:
        gp = gif_root / s["gif"]
        print(f"[CLIP] 处理 {s['gif']}", flush=True)
        # 文本（先收原始 embedding，不做 center_normalize —— 那是全集操作）
        te = enc.encode_text([s["text"]])            # [1,1024]
        # 首帧图像
        frames = load_gif_frames(str(gp), (128, 128))   # [T,128,128,3] uint8
        first = Image.fromarray(frames[0])
        ie = enc.encode_images([first])              # [1,1024]
        text_embs.append(te[0])
        image_embs.append(ie[0])

    te_all = np.stack(text_embs, 0).astype(np.float32)
    ie_all = np.stack(image_embs, 0).astype(np.float32)
    # ★ center_normalize 必须在全集上做（去各向异性均值），
    #   不能在循环里单样本做（单样本减自身均值 = 全零向量）。
    te_all = enc.center_normalize(te_all)
    ie_all = enc.center_normalize(ie_all)
    has_img = [1.0] * len(samples)
    hi_all = np.asarray(has_img, np.float32)

    np.save(CACHE / "text_emb.npy", te_all)
    np.save(CACHE / "image_emb.npy", ie_all)
    np.save(CACHE / "has_image.npy", hi_all)
    # 全 2049 拼接版本，训练脚本直接读这一个即可
    cond = build_cond_vector(te_all, ie_all, hi_all)
    np.save(CACHE / "cond.npy", cond)

    with open(CACHE / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)

    print(f"[CLIP] 完成：text_emb {te_all.shape} / image_emb {ie_all.shape} / cond {cond.shape}")
    print(f"[CLIP] 各向异性核查（去均值后）：text std={te_all.std():.4f} image std={ie_all.std():.4f}")

    # 释放 CLIP，把显存让给物理模型训练
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[CLIP] 已释放编码器，显存移交后续训练")


if __name__ == "__main__":
    main()