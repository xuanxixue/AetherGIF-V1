# -*- coding: utf-8 -*-
"""任务 #67：用真视频向量 gif3_video.npy 重建五个 cv3 拼接缓存。

★ 背景
  text/first 两路不受 NULL bug 影响（走 format_model_input 的独立分支），
  只有 gif 一路要换。三路原始向量已落盘，本脚本**不加载编码器**（零显存、
  秒级），只做 numpy 拼接，把 cv3_* 缓存的 gif 段从帧序列向量换成真视频向量。

★ 口径（防混用）
  - text3.npy / first3.npy / gif3.npy（旧帧序列向量）：原样保留不动
  - gif3_video.npy：真视频分支向量（PyAV 真 fps + 时序重采样）
  - cv3_*.npy：覆盖，gif 段 = gif3_video.npy
  - cv3_source.json：记录这套缓存的三路口径

★ 自检
  重建后逐段对比旧 cv3_full：只有 gif 段(4096:6144) 应该变，
  text(0:2048) / first(2048:4096) / flags(6144:6147) 必须逐位不变。

运行：
  cd 到 v4200 根目录后
  C:/Python312/python.exe -u scripts/rebuild_cv3_video.py
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "clip_cache"

from data.clip_condition import build_cond_vector3, CLIP_EMB_DIM   # noqa: E402

SEG = [
    ("text", slice(0, CLIP_EMB_DIM)),
    ("first", slice(CLIP_EMB_DIM, CLIP_EMB_DIM * 2)),
    ("gif", slice(CLIP_EMB_DIM * 2, CLIP_EMB_DIM * 3)),
    ("flags", slice(CLIP_EMB_DIM * 3, CLIP_EMB_DIM * 3 + 3)),
]


def cos(a, b):
    a = a / np.clip(np.linalg.norm(a), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b), 1e-8, None)
    return float(a @ b)


def main():
    text3 = np.load(OUT / "text3.npy")
    first3 = np.load(OUT / "first3.npy")
    gif3v = np.load(OUT / "gif3_video.npy")
    old_full = np.load(OUT / "cv3_full.npy")        # 旧缓存（gif 段=帧序列向量）

    assert text3.shape == first3.shape == gif3v.shape == (2, CLIP_EMB_DIM), \
        f"三路向量形状不对: {text3.shape} {first3.shape} {gif3v.shape}"
    n = text3.shape[0]

    print("=== 三路口径核对（src0 vs src1 余弦，越小越可分）===")
    for k, v in [("文本", text3), ("首帧", first3), ("GIF真视频", gif3v)]:
        print(f"  {k:8s}: cos={cos(v[0], v[1]):+.4f}")

    zero = np.zeros_like(text3)
    one = np.ones((n, 1), dtype=np.float32)
    zz = np.zeros((n, 1), dtype=np.float32)

    cv3_full = build_cond_vector3(text3, first3, gif3v, (one, one, one))
    cv3_text = build_cond_vector3(text3, zero, zero, (one, zz, zz))
    cv3_first = build_cond_vector3(zero, first3, zero, (zz, one, zz))
    cv3_gif = build_cond_vector3(zero, zero, gif3v, (zz, zz, one))
    cv3_tf = build_cond_vector3(text3, first3, zero, (one, one, zz))

    # ★ 自检：对比旧缓存，只有 gif 段应该变
    print("\n=== 重建自检：cv3_full 新旧逐段最大绝对差 ===")
    diff = np.abs(cv3_full - old_full)
    for name, sl in SEG:
        print(f"  {name:6s}: {diff[:, sl].max():.6f}")
    assert diff[:, 0:CLIP_EMB_DIM * 2].max() < 1e-6, "text/first 段不该变"
    assert diff[:, CLIP_EMB_DIM * 2:CLIP_EMB_DIM * 3].max() > 1e-3, "gif 段没变，重建无意义"
    assert diff[:, CLIP_EMB_DIM * 3:].max() < 1e-6, "flags 段不该变"
    print("  [PASS] 只有 gif 段变化，口径正确")

    np.save(OUT / "cv3_full.npy", cv3_full)
    np.save(OUT / "cv3_text.npy", cv3_text)
    np.save(OUT / "cv3_first.npy", cv3_first)
    np.save(OUT / "cv3_gif.npy", cv3_gif)
    np.save(OUT / "cv3_tf.npy", cv3_tf)

    with open(OUT / "cv3_source.json", "w", encoding="utf-8") as f:
        json.dump({
            "text": "text3.npy",
            "first": "first3.npy",
            "gif": "gif3_video.npy (PyAV 真视频分支，真 fps + 时序重采样)",
            "note": "gif3.npy 为旧帧序列向量，仅作历史保留，不参与本套缓存",
            "dim": cv3_full.shape[1],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 五个 cv3 缓存已重建（gif 段=真视频向量）")
    print(f"  维度 = {cv3_full.shape[1]} (= 2048*3 + 3 flags)")
    print("  覆盖：cv3_full / cv3_text / cv3_first / cv3_gif / cv3_tf")
    print("  口径记录：cv3_source.json")


if __name__ == "__main__":
    main()
