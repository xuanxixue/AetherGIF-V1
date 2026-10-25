# -*- coding: utf-8 -*-
"""
离线预编码 —— 把 jina-clip-v2 的文本/图像条件提前算成 .npy，训练时直接读。

为什么离线（用户 2026-10-13 定案）
----------------------------------
实测 PMNN 每步 3.78s（batch=1, N=6144）。若每步都过 CLIP：
  - 每步额外 0.3~0.8s（文本塔 XLM-R + 视觉塔 EVA-02）
  - 显存再吃 1.6GB，4GB 卡直接 OOM
离线后：
  - 训练单步只多一次 [B,2049] 的数组读入 + 一次 Linear，**开销 <1%**
  - 显存零占用
  - 单步可望降到 1.2s 以内 → 7 天能跑的步数翻 3 倍

去重优化（关键）
----------------
14145 个 GIF 只有 ~250 个不同文本 group（anime 的角色/情感标签各异）。
文本侧按 group 去重编码：只需跑 ~250 次文本塔，而不是 14145 次。
图像侧每个 GIF 首帧都必须单独编码（这是图生 GIF 的唯一信息源），
但视觉塔是 ViT-B 级别，14145 张 224px 图在 GPU 上约十几分钟。

产出
----
  data/clip_cache/text_emb.npy      [N_groups, 1024]  L2 归一化
  data/clip_cache/image_emb.npy     [N_items, 1024]  L2 归一化（首帧）
  data/clip_cache/group_list.json   group -> 行号
  data/clip_cache/meta.json         尺寸 / 条数 / 模型 ID

运行：
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  C:/Python312/python.exe scripts/preencode.py
"""
import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image, ImageSequence

from data.dataset_index import scan_dataset, save_index, load_index
from data.clip_condition import ClipConditionEncoder, CLIP_EMB_DIM

# v4200 位于 D:/AetherGIF-V1/docx/实验记录/versions/v4200_.../
# 项目数据根在 D:/AetherGIF-V1/data/gif_raw，即要往上退 **4** 级
# （v4200 → versions → 实验记录 → docx → AetherGIF-V1）。
# ★ 曾写成 3 级，指向不存在的 docx/data/gif_raw，导致 14145 张全部
#   `p.exists()` 为 False 被静默 continue 跳过，脚本照样退出码 0、
#   打印 "[IMG] 编码完成 成功 0/14145"。静默失败比报错更难查。
GIF_ROOT = ROOT.parent.parent.parent.parent / "data" / "gif_raw"
CACHE = ROOT / "data" / "clip_cache"

# 启动即断言数据根存在：把"静默 0 张"降级成"立刻炸"。
if not GIF_ROOT.is_dir():
    raise SystemExit(
        f"[FATAL] GIF_ROOT 不存在：{GIF_ROOT}\n"
        f"        期望 D:/AetherGIF-V1/data/gif_raw（往上退 4 级）\n"
        f"        请检查本行路径层级是否改动。"
    )


def _fmt(sec):
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def load_first_frame(gif_path: Path) -> Image.Image:
    """取 GIF 首帧（PIL，必须逐帧 seek——循环里 convert 会把多帧压成单帧）。"""
    with Image.open(gif_path) as im:
        try:
            im.seek(0)
        except EOFError:
            pass
        return im.convert("RGB").copy()


def main():
    # 设备选择：GPU fp16 优先，CPU 兜底。
    # 实测数据（jina-clip-v2，865M 参数，视觉塔原生 512×512）：
    #   CPU fp32  单张约 20~30 秒 → 14145 张需 100+ 小时，完全不可行
    #   GPU fp32  直接 OOM（权重 3.5GB + 前向激活 > 4GB 卡）
    #   GPU fp16  4.8 张/s → 全量约 49 分钟  ✓
    # 另：并发跑多个 CLIP 实例会挤爆内存（实测两次 exit 139 段错误），
    #     所以本脚本必须独占运行，不要和基准测试同时开。
    device = "cpu"
    fp16 = False
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / 1024 ** 3
        print(f"[DEV] CUDA 可用，空闲 {free_gb:.2f} GB")
        if free_gb >= 2.5:
            device, fp16 = "cuda", True
        else:
            print("[DEV] 空闲显存 < 2.5 GB，CLIP fp16 放不下 → 退回 CPU（会非常慢，"
                  "建议先关掉占用显存的程序）")
    else:
        print("[DEV] 无 CUDA")
    print(f"[DEV] 编码设备 {device}  fp16={fp16}")
    CACHE.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 建立/加载索引 ----------
    idx_path = ROOT / "data" / "index.json"
    if idx_path.exists():
        items = load_index(str(idx_path))
        print(f"[INDEX] 复用已有索引：{len(items)} 条")
    else:
        items = scan_dataset(str(GIF_ROOT))
        save_index(items, str(idx_path))

    # ---------- 2. 文本侧：按 group 去重 ----------
    groups = []
    gidx = {}
    for it in items:
        g = it["group"]
        if g not in gidx:
            gidx[g] = len(groups)
            groups.append(g)
    group_texts = [items[gidx[g]]["text"] for g in groups]  # 取该 group 任一样本的 text
    print(f"[TEXT] {len(groups)} 个不同 group（去重后只需编码这么多）")

    enc = ClipConditionEncoder(device=device, fp16=fp16)
    t0 = time.time()
    # ★ fp16 下文本 batch 必须调小：bs=64 时 XLM-R LoRA 的 lora_forward
    #   一次要额外 490MB，实测在 4GB 卡上 OOM（"Tried to allocate 490 MiB,
    #   1.06 GiB is free"）。图像塔 bs=4 只占 2.1GB，两者不能共用一个大 batch。
    text_emb = enc.encode_text(group_texts, batch_size=8)
    print(f"[TEXT] 编码完成 {text_emb.shape}  耗时 {_fmt(time.time()-t0)}")

    # ---------- 3. 图像侧：每个 GIF 首帧 ----------
    n = len(items)
    image_emb = np.zeros((n, CLIP_EMB_DIM), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    # 断点续跑：已有部分结果就接着做，不重头再来。
    # 14145 张在 CPU 上要跑很久，中途被 Ctrl-C / 断电打掉不该前功尽弃。
    part_path = CACHE / "image_emb.partial.npy"
    ok_path = CACHE / "image_ok.partial.npy"
    done_before = 0
    if part_path.exists() and ok_path.exists():
        try:
            pe = np.load(part_path, mmap_mode="r")
            po = np.load(ok_path)
            if pe.shape[0] == n and po.shape[0] == n:
                image_emb[:] = pe
                ok[:] = po
                done_before = int(ok.sum())
                print(f"[RESUME] 载入部分结果，已完成 {done_before}/{n}")
            else:
                print("[RESUME] 部分结果形状不匹配，丢弃重做")
        except Exception as ex:
            print(f"[RESUME] 载入失败（{type(ex).__name__}），丢弃重做")

    batch, batch_paths = [], []
    t0 = time.time()
    processed = 0
    missing = 0
    last_ckpt = time.time()

    def flush():
        nonlocal batch, batch_paths
        if not batch:
            return
        try:
            # bs=4：512×512 视觉塔在本机 15.7GB 内存下的安全值（RSS 峰值 ~4.8GB）
            e = enc.encode_images(batch, batch_size=4)
            for k, gi in enumerate(batch_paths):
                image_emb[gi] = e[k]
                ok[gi] = True
        except Exception as ex:
            print(f"[WARN] 图像批处理失败 {len(batch)} 张：{type(ex).__name__} {str(ex)[:100]}")
        batch, batch_paths = [], []

    def save_partial():
        """原子写部分结果（先写 .tmp 再 os.replace，避免半截文件）。

        ★ np.save 会自动追加 .npy 后缀：传 "x.partial.npy.tmp" 实际写成
          "x.partial.npy.tmp.npy"，导致 os.replace 找不到文件直接抛异常。
          所以临时名用 "..._tmp.npy"，让它落在同名 .npy 上。
        """
        try:
            for pth, arr in ((part_path, image_emb), (ok_path, ok)):
                tmp = pth.with_name(pth.name + "_tmp.npy")   # np.save 会保持此名
                np.save(tmp, arr)
                os.replace(tmp, pth)
        except Exception as ex:
            print(f"[WARN] 部分结果落盘失败：{type(ex).__name__} {str(ex)[:80]}")

    for i, it in enumerate(items):
        if ok[i]:                      # 续跑：跳过已完成
            processed += 1
            continue
        p = GIF_ROOT / it["path"]
        if not p.exists():
            missing += 1
            continue
        try:
            batch.append(load_first_frame(p))
            batch_paths.append(i)
        except Exception:
            ok[i] = False
        # 攒 16 张就刷：512×512 PIL 图常驻内存会挤占 CLIP 前向所需空间。
        # 攒 64 张时 preencode 会段错误（exit 139）。
        if len(batch) >= 16:
            flush()
        processed += 1
        # 每 200 张报一次：CPU 上跑 512×512 EVA-02 很慢（约 1~2 张/s），
        # 原来的 1000 张间隔意味着要等 10+ 分钟才看到第一行反馈，
        # 无法区分"在跑"和"卡死"。同时打 RSS 便于盯内存。
        if processed % 200 == 0:
            el = time.time() - t0
            rate = processed / max(el, 1e-6)
            eta = (n - processed) / max(rate, 1e-6)
            try:
                import psutil
                rss = psutil.Process().memory_info().rss / 1024 ** 3
                mem = f" RSS {rss:.2f}GB"
            except Exception:
                mem = ""
            print(f"[IMG] {processed}/{n} ({processed*100//n}%)  {rate:.2f} 张/s  "
                  f"已用 {_fmt(el)}  预计剩余 {_fmt(eta)}{mem}", flush=True)
        # 每 3 分钟落一次部分结果：段错误/断电最多损失 3 分钟
        if time.time() - last_ckpt > 180:
            save_partial()
            last_ckpt = time.time()
    flush()
    save_partial()
    print(f"[IMG] 编码完成 成功 {int(ok.sum())}/{n}  耗时 {_fmt(time.time()-t0)}")

    # 静默失败闸：路径层级写错时，14145 张会全部 exists()==False 被跳过，
    # 但脚本依然退出码 0。这里把异常比例显式暴露出来。
    if missing:
        print(f"[WARN] 索引中 {missing} 条路径在 {GIF_ROOT} 下不存在（已跳过）")
    hit = int(ok.sum())
    if hit == 0:
        raise SystemExit(
            f"[FATAL] 图像编码成功 0 张。\n"
            f"        GIF_ROOT = {GIF_ROOT}\n"
            f"        索引条目 = {n}，路径不存在 {missing} 条\n"
            f"        多半是 GIF_ROOT 层级写错或索引与数据目录不匹配。"
        )
    if hit / max(n, 1) < 0.5:
        raise SystemExit(
            f"[FATAL] 图像编码成功率过低：{hit}/{n} = {hit/n:.1%}\n"
            f"        GIF_ROOT = {GIF_ROOT}\n"
            f"        多数 GIF 无法解码，缓存不完整，后续训练会得到退化条件。"
        )

    # ---------- 3.5 去各向异性（必须在全集上估均值，见 ClipConditionEncoder.center_normalize）----------
    # 不做这步：所有 GIF 条件向量两两余弦 0.85±0.08，物理核学不到素材差异。
    # 做了之后：std 0.077 → 0.343，区分度提升 4.5 倍。
    def _stats(x, tag):
        if len(x) < 2:
            return None
        nrm = x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)
        m = nrm @ nrm.T
        off = m[~np.eye(len(nrm), dtype=bool)]
        return float(off.mean()), float(off.std()), float(off.min()), float(off.max())

    text_emb = ClipConditionEncoder.center_normalize(text_emb)
    if ok.sum() >= 2:
        image_emb[ok] = ClipConditionEncoder.center_normalize(image_emb[ok])
    else:
        print("[WARN] 有效图像不足 2 张，跳过图像去均值")

    s_t = _stats(text_emb, "text")
    s_i = _stats(image_emb[ok], "image") if ok.sum() >= 2 else None
    print("[CENTER] 去均值后相似度统计（均值/std/最小/最大）")
    print(f"[CENTER]   text  {s_t[0]:+.3f} / {s_t[1]:.3f} / {s_t[2]:+.3f} / {s_t[3]:+.3f}")
    if s_i:
        print(f"[CENTER]   image {s_i[0]:+.3f} / {s_i[1]:.3f} / {s_i[2]:+.3f} / {s_i[3]:+.3f}")
    if s_t and s_t[1] < 0.05:
        print("[WARN] text 去均值后 std 仍 < 0.05，group 文本可能高度重复，检查 dataset_index")

    # ---------- 4. 落盘 ----------
    np.save(CACHE / "text_emb.npy", text_emb)
    np.save(CACHE / "image_emb.npy", image_emb)
    np.save(CACHE / "image_ok.npy", ok)
    with open(CACHE / "group_list.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False)
    with open(CACHE / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "jinaai/jina-clip-v2",
            "emb_dim": CLIP_EMB_DIM,
            "n_items": n, "n_groups": len(groups),
            "image_ok": int(ok.sum()),
            "n_concat": CLIP_EMB_DIM * 2 + 1,
            "centered": True,
            "text_sim": s_t, "image_sim": s_i,
        }, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] text_emb.npy  {text_emb.shape}")
    print(f"[SAVE] image_emb.npy {image_emb.shape}")
    # 全部成功后清掉断点文件，否则下次运行会误判为"已完成"而跳过真实工作
    for pth in (part_path, ok_path):
        if pth.exists():
            pth.unlink()
    print(f"[SAVE] 目录 {CACHE}")
    print(f"[DONE] 全流程耗时 {_fmt(time.time()-t0)}")


if __name__ == "__main__":
    main()
