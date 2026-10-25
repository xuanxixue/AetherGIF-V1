# -*- coding: utf-8 -*-
"""用训练集里已有的 group 文本条件 + best.pt 权重生成 GIF。

不加载 CLIP（直接读离线 text_emb.npy），只跑物理核。
"""
import os
import sys
import json
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from model.aether_gif_model import AetherGIFModel
from physics.kuramoto_sync import build_spacetime_index
from data.clip_condition import CLIP_EMB_DIM

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
CACHE = ROOT / "data" / "clip_cache"
CKPT = ROOT / "outputs_real" / "ckpt" / "best.pt"
OUT = ROOT / "outputs_real" / "infer_text"


def main():
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_emb = np.load(CACHE / "text_emb.npy", mmap_mode="r")      # [204,1024]
    image_emb = np.load(CACHE / "image_emb.npy", mmap_mode="r")    # [14145,1024]
    image_ok = np.load(CACHE / "image_ok.npy")
    groups = json.load(open(CACHE / "group_list.json", encoding="utf-8"))
    items = json.load(open(ROOT / "data" / "index.json", encoding="utf-8"))

    model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"[CKPT] best.pt step={ck.get('step','?')} stats_best="
          f"{ck.get('stats',{}).get('best_val','?')}")

    # 训练集里真实出现过的 group（挑几个有代表性的）
    picks = [14, 18, 21, 28, 35, 36]     # Lucky-Star / Clannad / KonoSuba / Yuru-Camp / Toradora / K-On
    # 每个 group 取一条真实样本的图像条件
    first_item = {}
    for idx, it in enumerate(items):
        first_item.setdefault(it["group"], idx)

    OUT.mkdir(parents=True, exist_ok=True)
    t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)

    for gi in picks:
        gname = groups[gi]
        t_e = np.asarray(text_emb[gi], dtype=np.float32)[None, :]
        i_e = np.zeros((1, CLIP_EMB_DIM), np.float32)
        has_img = 0.0
        if gname in first_item:
            ii = first_item[gname]
            if image_ok[ii]:
                i_e = np.asarray(image_emb[ii], dtype=np.float32)[None, :]
                has_img = 1.0
        cond = np.concatenate([t_e, i_e, np.array([[has_img]], np.float32)], 1)
        tag = "text+b_img" if has_img else "text_only"

        with torch.no_grad():
            a, th = model.generate(duration=3, cond_vec=cond)
        frames = model._field_to_frames(a, th, t_coord, s_coord)
        out = OUT / f"g{gi}_{gname}_{tag}.gif"
        model._save_gif(frames, str(out))
        arr = np.stack([np.asarray(f) for f in frames])
        print(f"[SAVED] {out.name}  帧数={len(frames)} 形状={arr.shape} "
              f"std={arr.std():.1f} min={arr.min()} max={arr.max()}")

    print(f"\n输出目录: {OUT}")


if __name__ == "__main__":
    main()