# -*- coding: utf-8 -*-
"""GIF 批量打标签自动化（Agents-A1 多模态 → 中文 500 字描述）。

用法：
  # 小样本验证（默认跑前 20 张）
  C:/Python312/python.exe scripts/gif_captioner.py --limit 20

  # 全量（断点续跑）
  C:/Python312/python.exe scripts/gif_captioner.py

  # 指定并发
  C:/Python312/python.exe scripts/gif_captioner.py --concurrency 20

设计：
  - 扫描 data/gif_raw 全部 GIF（含 _extract，大小写扩展名）
  - 抽帧（最多 6 帧均匀采样）横排拼图 → base64 → Agents-A1
  - 强制 ≥500 字中文描述，不足自动重试（最多 2 次）
  - 结果增量写入 JSONL（断点续跑：已处理路径跳过）
  - asyncio + aiohttp 并发，接口上限 2640 req/min 远够
"""
import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import aiohttp
from PIL import Image, ImageSequence

API = "https://discovery-api.intern-ai.org.cn/v1/chat/completions"
# ★ 密钥改为环境变量读取（原硬编码 key 已于 2026-10-22 上传 GitHub 前移除）
# 本地用法：set CAPTION_KEY=sk-xxxx（PowerShell 用 $env:CAPTION_KEY=...）
KEY = os.environ.get("CAPTION_KEY", "")
MODEL = "Agents-A1"

ROOT = Path("D:/AetherGIF-V1")
GIF_ROOT = ROOT / "data" / "gif_raw"
OUT_DIR = ROOT / "data_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "gif_captions.jsonl"
DONE_LOG = OUT_DIR / "done_paths.txt"

MIN_CHARS = 500          # 严格 500 字
MAX_RETRY = 2
MAX_FRAMES = 6           # 抽帧数（动态信息足够）
FRAME_SIZE = 168         # 每帧缩到 168px，6 帧横排 ≈ 1000px
MAX_TOKENS = 8000        # reasoning 模型：思考 + 500 字正文都要装下
TIMEOUT = 180

SYSTEM = (
    "你是专业的GIF动画素材标注员。我会给你一张由多帧横排拼接的GIF抽帧图，"
    "从左到右代表时间顺序。请你描述这个GIF动画，"
    "必须严格输出不少于500个汉字的详细中文描述，逐字硬凑也要凑满500字。"
    "描述必须覆盖以下全部维度："
    "1.主体对象（是什么、形态、数量）；"
    "2.颜色与配色（主色、辅色、明暗、色调冷暖）；"
    "3.构图（布局、比例、背景、留白、前景后台空间关系）；"
    "4.动作与动态变化（逐帧发生了什么、运动轨迹、节奏快慢、循环规律）；"
    "5.风格（写实/扁平/二次元/像素/手绘等，艺术流派）；"
    "6.纹理与材质细节（表面质感、光泽、颗粒、边缘、阴影）；"
    "7.文字与符号（如有）；"
    "8.氛围与用途（情绪、使用场景）。"
    "只输出这段描述文字本身，禁止任何开场白、结尾总结、markdown标记或解释，"
    "直接从描述第一句开始写，写完即止。"
)


def scan_gifs():
    """扫描所有 GIF（含大小写扩展名），返回按绝对路径排序的列表。"""
    paths = []
    for ext in ("*.gif", "*.GIF"):
        paths.extend(GIF_ROOT.rglob(ext))
    # 去重 + 排序（稳定顺序，断点续跑靠它）
    uniq = sorted({str(p) for p in paths})
    return uniq


def gif_to_panel_b64(path: str) -> str | None:
    """抽帧横排拼图 → base64。失败返回 None。"""
    try:
        im = Image.open(path)
        frames = []
        for fr in ImageSequence.Iterator(im):
            frames.append(fr.convert("RGB"))
        if not frames:
            return None
        n = len(frames)
        if n <= 1:
            idx = [0]
        elif n <= MAX_FRAMES:
            idx = list(range(n))
        else:
            # 均匀采样 MAX_FRAMES 帧
            idx = [round(i * (n - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)]
        sel = [frames[i].resize((FRAME_SIZE, FRAME_SIZE)) for i in idx]
        gap = 4
        W = sum(f.width for f in sel) + gap * (len(sel) - 1)
        H = FRAME_SIZE
        panel = Image.new("RGB", (W, H), (255, 255, 255))
        x = 0
        for f in sel:
            panel.paste(f, (x, 0))
            x += f.width + gap
        buf = io.BytesIO()
        panel.save(buf, "PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def clean_caption(raw: str) -> str:
    """去掉思考残留/markdown，返回干净中文描述。"""
    s = raw.strip()
    # 去掉可能的 markdown 代码块围栏
    for fence in ("```", "~~~"):
        if fence in s:
            s = s.replace(fence, "")
    return s.strip()


async def caption_one(session, path: str, sem: asyncio.Semaphore) -> dict:
    """单张 GIF 打标签，返回结果 dict。"""
    rel = str(Path(path).relative_to(ROOT))
    b64 = gif_to_panel_b64(path)
    if b64 is None:
        return {"path": rel, "status": "decode_fail", "caption": "",
                "chars": 0, "sha": ""}

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "请描述这个GIF动画。"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + b64}},
            ]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + KEY}

    last_err = ""
    for attempt in range(MAX_RETRY + 1):
        try:
            async with sem:
                async with session.post(API, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
                    if r.status != 200:
                        body = await r.text()
                        last_err = f"http_{r.status}:{body[:200]}"
                        continue
                    j = await r.json()
            msg = j["choices"][0]["message"]
            content = clean_caption(msg.get("content", "") or "")
            finish = j["choices"][0].get("finish_reason", "")
            chars = len([c for c in content if "\u4e00" <= c <= "\u9fff"])

            # 成功条件：非空 且 充足字数 且 没被截断
            if content and (chars >= MIN_CHARS or finish == "stop"):
                if chars < MIN_CHARS:
                    last_err = f"too_short:{chars}"
                    continue
                return {"path": rel, "status": "ok", "caption": content,
                        "chars": chars, "finish": finish,
                        "sha": hashlib.md5(content.encode()).hexdigest()[:8]}
            last_err = f"empty_or_cut:{chars}/{finish}"
        except asyncio.TimeoutError:
            last_err = "timeout"
        except Exception as e:
            last_err = repr(e)[:200]

    return {"path": rel, "status": f"fail:{last_err}", "caption": "",
            "chars": 0, "sha": ""}


def load_done() -> set:
    if DONE_LOG.exists():
        return set(DONE_LOG.read_text(encoding="utf-8").splitlines())
    return set()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    gifs = scan_gifs()
    done = load_done()
    todo = [g for g in gifs if g not in done]
    if args.limit > 0:
        todo = todo[:args.limit]
    total = len(todo)
    print(f"[scan] 总 {len(gifs)} 张，已处理 {len(done)}，本次待处理 {total}")

    if total == 0:
        print("[done] 没有待处理项")
        return

    sem = asyncio.Semaphore(args.concurrency)
    ok = fail = short = 0
    t0 = time.time()

    def append_line(rec: dict):
        with open(OUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(DONE_LOG, "a", encoding="utf-8") as f:
            f.write(rec["path"] + "\n")

    async with aiohttp.ClientSession() as session:
        # 分批处理，避免全量任务同时挂起
        BATCH = 200
        for bi in range(0, total, BATCH):
            chunk = todo[bi:bi + BATCH]
            tasks = [asyncio.create_task(caption_one(session, p, sem)) for p in chunk]
            for t in asyncio.as_completed(tasks):
                rec = await t
                append_line(rec)
                if rec["status"] == "ok":
                    ok += 1
                elif rec["status"].startswith("fail:") and "too_short" in rec["status"]:
                    short += 1
                else:
                    fail += 1
                elapsed = time.time() - t0
                rate = (ok + fail + short) / max(elapsed, 1e-9)
                eta = (total - (ok + fail + short)) / max(rate, 1e-9)
                if (ok + fail + short) % 50 == 0 or (ok + fail + short) == total:
                    print(f"[{ok+fail+short}/{total}] ok={ok} fail={fail} short={short} "
                          f"速率={rate:.2f}/s 已耗时={elapsed:.0f}s ETA={eta:.0f}s",
                          flush=True)

    print(f"\n[sum] 完成 ok={ok} fail={fail} short={short}")
    print(f"[out] {OUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())