# -*- coding: utf-8 -*-
"""
离线预编码（int8 GPU 版）—— 用 Qwen3-VL-Embedding-2B 替换 jina-clip-v2。

★ 为什么要换（2026-10-15，用户定案）
------------------------------------
1. 维度升级：jina-clip-v2 = 1024 维 → Qwen3-VL-Embedding-2B = 2048 维。
   旧 `data/clip_cache/*.npy` 全是 1024 维，维度已作废，正式训练前必须重跑。
2. 时序能力：jina-clip 只能吃 GIF 首帧，时序信息全丢；Qwen3-VL 原生支持
   帧序列输入，能编码 GIF 的动感（本项目要生成的正是动图）。
3. 区分度：实测两素材条件余弦 jina-clip 原始基线 0.853（各向异性）→
   Qwen3-VL 0.277（去均值后），物理核才有素材差异可学。

★ int8 的必要性（本机硬件）
----------------------------
Qwen3-VL-Embedding-2B fp16 需 4058 MiB，本机 4GB 卡可用仅 3300 MiB，装不下。
int8 后峰值 2444 MiB。CPU fp32 单条 16 帧 GIF 要 471.7s；int8 GPU 8 帧 1.8s。
全量 14145 张 × 1.8s ≈ 7 小时（8 帧），用 4 帧可压到约 4 小时。

★ 与旧 preencode.py 的差异
--------------------------
- 编码器：ClipConditionEncoder(CPU/fp16) → Int8QwenEncoder(int8 GPU)
- 图像侧：只取首帧 → 帧序列（max_frames 可调，默认 8）
- 文本侧：去重后 204 条，int8 GPU batch=4 约 5 分钟
- 断点续跑 / 去均值 / 静默失败闸：全部保留旧逻辑

★ 运行环境（唯一能跑 int8 的组合，缺一不可）
  解释器 C:\\Python312\\python.exe（唯一有 CUDA，torch 2.5.1+cu121）
  transformers 5.19 来自 deps312；bitsandbytes 来自 int8_deps

运行：
  cd 到 v4200 根目录后
  C:/Python312/python.exe -u scripts/preencode_int8.py [MAX_FRAMES]
"""
import os
import sys
import gc
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ★ 向上探测项目根（deps312 所在），并把项目根加入 sys.path 供
#   torch_accel_shim / int8_qwen_encoder import
_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir():
        break
    if _PROJ.parent == _PROJ:
        break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))

import numpy as np
import torch

from data.dataset_index import load_index
from data.clip_condition import CLIP_EMB_DIM
from int8_qwen_encoder import Int8QwenEncoder

GIF_ROOT = ROOT.parent.parent.parent.parent / "data" / "gif_raw"
CACHE = ROOT / "data" / "clip_cache"

if not GIF_ROOT.is_dir():
    raise SystemExit(f"[FATAL] GIF_ROOT 不存在：{GIF_ROOT}")


def _fmt(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _stats(x):
    """离线条件向量的相似度统计（区分度体检）。"""
    if len(x) < 2:
        return None
    nrm = x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)
    m = nrm @ nrm.T
    off = m[~np.eye(len(nrm), dtype=bool)]
    return [float(off.mean()), float(off.std()), float(off.min()), float(off.max())]


def center_normalize(emb):
    """去均值 + 重新 L2（与 ClipConditionEncoder.center_normalize 同口径）。"""
    mu = emb.mean(axis=0, keepdims=True)
    emb = emb - mu
    n = np.linalg.norm(emb, axis=-1, keepdims=True)
    return (emb / np.clip(n, 1e-8, None)).astype(np.float32)


def main():
    max_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"[DEV] int8 GPU 编码器，GIF 帧数 max_frames={max_frames}")
    if not torch.cuda.is_available():
        raise SystemExit("[FATAL] int8 编码器必须有 CUDA，请用 C:\\Python312\\python.exe 运行")
    CACHE.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 索引 ----------
    idx_path = ROOT / "data" / "index.json"
    items = load_index(str(idx_path))
    print(f"[INDEX] {len(items)} 条 GIF")

    # ---------- 2. 文本侧：按 group 去重 ----------
    groups, gidx = [], {}
    for it in items:
        g = it["group"]
        if g not in gidx:
            gidx[g] = len(groups)
            groups.append(g)
    group_texts = [items[gidx[g]]["text"] for g in groups]
    print(f"[TEXT] {len(groups)} 个 group 去重编码")

    enc = Int8QwenEncoder(max_frames=max_frames)   # 懒加载
    t0 = time.time()
    text_emb = enc.encode_texts(group_texts, batch_size=4)
    print(f"[TEXT] 完成 {text_emb.shape}  耗时 {_fmt(time.time()-t0)}")

    # ---------- 3. 图像侧：每个 GIF 帧序列 ----------
    n = len(items)
    image_emb = np.zeros((n, CLIP_EMB_DIM), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    part_path = CACHE / "image_emb.partial.npy"
    ok_path = CACHE / "image_ok.partial.npy"
    done_before = 0
    if part_path.exists() and ok_path.exists():
        try:
            pe = np.load(part_path, mmap_mode="r")
            po = np.load(ok_path)
            if pe.shape[0] == n and po.shape[0] == n and pe.shape[1] == CLIP_EMB_DIM:
                image_emb[:] = pe
                ok[:] = po
                done_before = int(ok.sum())
                print(f"[RESUME] 已完成 {done_before}/{n}")
            else:
                print("[RESUME] 部分结果形状不匹配（旧 1024 维缓存？），丢弃重做")
        except Exception as ex:
            print(f"[RESUME] 载入失败（{type(ex).__name__}），丢弃重做")

    t0 = time.time()
    missing = 0
    last_ckpt = time.time()

    def save_partial():
        try:
            for pth, arr in ((part_path, image_emb), (ok_path, ok)):
                tmp = pth.with_name(pth.name + "_tmp.npy")
                np.save(tmp, arr)
                os.replace(tmp, pth)
        except Exception as ex:
            print(f"[WARN] 部分落盘失败：{type(ex).__name__} {str(ex)[:80]}")

    for i, it in enumerate(items):
        if ok[i]:
            continue
        p = GIF_ROOT / it["path"]
        if not p.exists():
            missing += 1
            continue
        try:
            e = enc.encode_gif_paths([str(p)], batch_size=1)
            image_emb[i] = e[0]
            ok[i] = True
        except Exception as ex:
            print(f"[WARN] 编码失败 {it['path']}：{type(ex).__name__} {str(ex)[:80]}")
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            done = int(ok.sum())
            rate = done / max(el, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            gpu = torch.cuda.memory_allocated() / 1024 ** 2
            print(f"[IMG] {done}/{n} ({done*100//n}%)  {rate:.2f} 张/s  "
                  f"已用 {_fmt(el)}  预计剩余 {_fmt(eta)}  GPU {gpu:.0f}MiB", flush=True)
        if time.time() - last_ckpt > 180:
            save_partial()
            last_ckpt = time.time()

    save_partial()
    hit = int(ok.sum())
    print(f"[IMG] 完成 {hit}/{n}  耗时 {_fmt(time.time()-t0)}")
    if missing:
        print(f"[WARN] {missing} 条路径缺失")
    if hit == 0:
        raise SystemExit("[FATAL] 图像编码成功 0 张")
    if hit / max(n, 1) < 0.5:
        raise SystemExit(f"[FATAL] 图像编码成功率过低：{hit}/{n}")

    # ---------- 3.5 去均值（全集上，区分度关键）----------
    text_emb = center_normalize(text_emb)
    image_emb[ok] = center_normalize(image_emb[ok])
    s_t = _stats(text_emb)
    s_i = _stats(image_emb[ok]) if ok.sum() >= 2 else None
    print(f"[CENTER] text  {s_t[0]:+.3f}/{s_t[1]:.3f}/{s_t[2]:+.3f}/{s_t[3]:+.3f}")
    if s_i:
        print(f"[CENTER] image {s_i[0]:+.3f}/{s_i[1]:.3f}/{s_i[2]:+.3f}/{s_i[3]:+.3f}")

    # ---------- 4. 落盘 ----------
    np.save(CACHE / "text_emb.npy", text_emb)
    np.save(CACHE / "image_emb.npy", image_emb)
    np.save(CACHE / "image_ok.npy", ok)
    with open(CACHE / "group_list.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False)
    with open(CACHE / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen/Qwen3-VL-Embedding-2B (int8)",
            "emb_dim": CLIP_EMB_DIM,
            "n_items": n, "n_groups": len(groups),
            "image_ok": hit,
            "n_concat": CLIP_EMB_DIM * 2 + 1,
            "centered": True,
            "max_frames": max_frames,
            "text_sim": s_t, "image_sim": s_i,
        }, f, ensure_ascii=False, indent=2)

    for pth in (part_path, ok_path):
        if pth.exists():
            pth.unlink()
    print(f"[DONE] 目录 {CACHE}")


if __name__ == "__main__":
    main()