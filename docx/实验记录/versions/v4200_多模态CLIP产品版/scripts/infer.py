# -*- coding: utf-8 -*-
"""
v4200 三种生成能力推理验证 —— 文生 GIF / 图生 GIF / 图文生 GIF。

用法
----
  # 文生
  C:/Python312/python.exe scripts/infer.py --text "一个弹跳的小球" --out out/txt.gif

  # 图生（需要一张参考图）
  C:/Python312/python.exe scripts/infer.py --image D:/some/ref.png --out out/img.gif

  # 图文生
  C:/Python312/python.exe scripts/infer.py --text "让它跳动" --image ref.png --out out/both.gif

  # 用训练集里某个 GIF 的首帧当图像条件（无需自备图片）
  C:/Python312/python.exe scripts/infer.py --ref-item 0 --out out/ref.gif

默认从 outputs_real/ckpt/latest.pt 读权重。
"""
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import load_gif_frames
from data.clip_condition import CLIP_EMB_DIM

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
GIF_ROOT = Path("D:/AetherGIF-V1/data/gif_raw")
CACHE = ROOT / "data" / "clip_cache"
CKPT = ROOT / "outputs_real" / "ckpt" / "latest.pt"


def fmt_dur(sec):
    sec = int(sec); h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="文本条件")
    ap.add_argument("--image", default=None, help="图像条件（png/jpg/gif 路径）")
    ap.add_argument("--ref-item", type=int, default=None,
                    help="用数据集第 N 个 GIF 的首帧当图像条件")
    ap.add_argument("--out", default="outputs_real/infer.gif")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--seconds", type=float, default=1.2, help="生成时长（秒）")
    args = ap.parse_args()

    if not (args.text or args.image or args.ref_item is not None):
        print("[ERR] 至少要给 --text / --image / --ref-item 之一")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"[CKPT] 已加载 step={ck.get('step','?')}  ({args.ckpt})")

    # ---- 组装条件向量 [1, 2049] ----
    t_e = np.zeros((1, CLIP_EMB_DIM), np.float32)
    i_e = np.zeros((1, CLIP_EMB_DIM), np.float32)
    has_img = 0.0

    if args.text:
        import sys as _s
        _s.path.insert(0, str(ROOT / "src"))
        from data.clip_condition import ClipConditionEncoder
        enc = ClipConditionEncoder(device=device)
        t_e = enc.encode_text([args.text])
        print(f"[COND] 文本已编码：{args.text[:40]}")

    if args.ref_item is not None:
        import json
        items = json.load(open(ROOT / "data" / "index.json", encoding="utf-8"))
        it = items[args.ref_item]
        gp = GIF_ROOT / it["path"]
        with Image.open(gp) as im:
            im.seek(0)
            pil = im.convert("RGB").copy()
        from data.clip_condition import ClipConditionEncoder
        enc = globals().get("enc") or ClipConditionEncoder(device=device)
        i_e = enc.encode_images([pil])
        has_img = 1.0
        print(f"[COND] 图像条件来自数据集第 {args.ref_item} 个：{it['path']}")
    elif args.image:
        pil = Image.open(args.image).convert("RGB")
        from data.clip_condition import ClipConditionEncoder
        enc = ClipConditionEncoder(device=device)
        i_e = enc.encode_images([pil])
        has_img = 1.0
        print(f"[COND] 图像条件：{args.image}")

    cond = np.concatenate([t_e, i_e, np.array([[has_img]], np.float32)], 1)

    mode = "both" if (t_e.any() and i_e.any()) else ("text" if t_e.any() else "image")
    print(f"[MODE] {mode}")

    # ---- 生成 ----
    dur = int(args.seconds)
    a, theta = model.generate(duration=dur, cond_vec=cond)
    frames = model._field_to_frames(a, theta,
                                    *_coords(dur, device))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model._save_gif(frames, args.out)
    print(f"[SAVED] {args.out}  ({len(frames)} 帧, {dur} 秒)")


def _coords(dur, device):
    from physics.kuramoto_sync import build_spacetime_index
    fps = 10
    T_step = max(2, dur * fps // 5)
    t, s = build_spacetime_index(T_step, GRID, GRID, device)
    return t, s


if __name__ == "__main__":
    main()
