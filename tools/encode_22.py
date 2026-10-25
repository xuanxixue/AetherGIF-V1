# -*- coding: utf-8 -*-
"""22 个素材 × 三路条件向量编码（INT8 Qwen3-VL-Embedding-2B）。

三路语义严格区分（不可混用）：
  文本  = encode_texts(描述)            → [22,2048]
  首帧  = encode_images(第0帧 PIL)       → [22,2048]   （静图分支）
  视频  = encode_video_files(GIF 路径)   → [22,2048]   （PyAV 真 fps + 时序重采样）
          ★ 不是 encode_gif_paths —— 那个走 PIL 解帧成 list，只是"一堆图"，
            丢动速信息。本项目 v4200 的 gif3_video.npy 就是真视频分支。

★ 防静默退化自检：若 PyAV 注入失败，视频向量会静默变成 NULL 空文本向量
  （所有素材两两余弦 ≈ +1.0）。本脚本会算素材间余弦矩阵，
  若近似退化为全 1，直接报错中止。

用法：
  C:/Python312/python.exe tools/encode_22.py
  C:/Python312/python.exe tools/encode_22.py --out 训练素材22/vectors
"""
import os
import re
import sys
import json
import time
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent

for p in [str(PROJ), str(PROJ / "deps312"), str(PROJ / "int8_deps")]:
    sys.path.insert(0, p)

import numpy as np
from PIL import Image

DATA = PROJ / "训练素材22"
DESC_MD = DATA / "描述_22条.md"


def parse_descriptions(md_path: Path):
    """解析 `## 01. 文件名（N 帧）` + 正文。返回 [(idx, filename, desc)]。"""
    txt = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n## (\d{2})\. (.+?)（\d+ 帧）\n", txt)
    out = []
    for i in range(1, len(blocks), 3):
        idx = blocks[i]
        fname = blocks[i + 1].strip()
        body = blocks[i + 2]
        # 去掉分隔线与说明行
        lines = [l.strip() for l in body.split("\n")]
        lines = [l for l in lines if l and not l.startswith("---")
                 and not l.startswith(">")]
        desc = "".join(lines)
        desc = desc.replace("（待写）", "").strip()
        out.append((idx, fname, desc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "vectors"))
    ap.add_argument("--max-frames", type=int, default=8)
    args = ap.parse_args()

    items = parse_descriptions(DESC_MD)
    print(f"[1] 解析描述：{len(items)} 条")
    assert len(items) == 22, f"期望 22 条，实际 {len(items)}"
    short = [(i, len(d)) for i, _, d in items if len(d) < 400]
    if short:
        print(f"[WARN] 字数偏短：{short}")
    for i, f, d in items:
        print(f"    {i} {f:44s} {len(d):4d} 字")

    # 文件必须都存在
    paths, firsts = [], []
    for i, fname, _ in items:
        gp = DATA / fname
        if not gp.exists():
            raise FileNotFoundError(f"{i} {gp}")
        paths.append(str(gp))
        with Image.open(gp) as im:
            im.seek(0)
            firsts.append(im.convert("RGB").copy())

    from int8_qwen_encoder import Int8QwenEncoder
    print("\n[2] 加载 INT8 编码器（首次会读 2B 权重，约 1~2 分钟）")
    t0 = time.time()
    enc = Int8QwenEncoder(max_frames=args.max_frames)
    enc._ensure()
    print(f"    就绪 {time.time()-t0:.1f}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 文本 ----
    print("\n[3/3] 文本向量")
    t0 = time.time()
    text = enc.encode_texts([d for _, _, d in items], batch_size=2)
    print(f"    {text.shape}  {time.time()-t0:.1f}s")

    # ---- 首帧 ----
    print("\n[3/3] 首帧向量（image 分支）")
    t0 = time.time()
    first = enc.encode_images(firsts, batch_size=1)
    print(f"    {first.shape}  {time.time()-t0:.1f}s")

    # ---- 视频 ----
    print("\n[3/3] 视频向量（PyAV 真 fps 分支）")
    t0 = time.time()
    video = enc.encode_video_files(paths, batch_size=1, fps=1.0)
    print(f"    {video.shape}  {time.time()-t0:.1f}s")

    # ---- 自检：余弦多样性（防 NULL 静默退化）----
    def cos_stats(name, M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        C = Mn @ Mn.T
        off = C[~np.eye(len(M), dtype=bool)]
        print(f"    {name}: 对角=1.000  非对角 min={off.min():+.4f} "
              f"max={off.max():+.4f} mean={off.mean():+.4f}")
        return float(off.max())

    print("\n[4] 余弦多样性自检（全部≈+1.0 说明向量退化成 NULL）")
    mx_t = cos_stats("文本", text)
    mx_f = cos_stats("首帧", first)
    mx_v = cos_stats("视频", video)

    bad = []
    if mx_v > 0.999:
        bad.append(f"视频向量非对角余弦 max={mx_v:.4f} ≈ 1.0 → PyAV 后端可能没注入，"
                   f"退化成 NULL 空文本向量")
    if mx_t > 0.999:
        bad.append(f"文本向量非对角余弦 max={mx_t:.4f} ≈ 1.0，异常")
    if bad:
        for b in bad:
            print(f"[FAIL] {b}")
        raise SystemExit(1)

    # 三路互检：同一素材的文本 vs 视频应正相关但不为 1
    cross = [(text[i] @ video[i]) / (np.linalg.norm(text[i]) * np.linalg.norm(video[i]) + 1e-9)
             for i in range(len(items))]
    print(f"    同素材 文本·视频 余弦: mean={np.mean(cross):+.4f} "
          f"min={np.min(cross):+.4f} max={np.max(cross):+.4f}")

    np.save(out / "text22.npy", text)
    np.save(out / "first22.npy", first)
    np.save(out / "gif22.npy", video)
    meta = dict(n=len(items), dim=int(text.shape[1]), max_frames=args.max_frames,
                items=[dict(idx=i, file=f, chars=len(d)) for i, f, d in items],
                cos_text_max=mx_t, cos_first_max=mx_f, cos_video_max=mx_v,
                cross_tv_mean=float(np.mean(cross)))
    json.dump(meta, open(out / "meta22.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n[SAVED] {out}/text22.npy first22.npy gif22.npy meta22.json")
    print("[OK] 三路向量编码完成")


if __name__ == "__main__":
    main()
