# -*- coding: utf-8 -*-
"""解压 7z GIF 包并按模型适配标准筛选。

筛选标准（对齐 AetherGIF 的 codec 口径）：
  - 帧数 >= 6   (TS*WIN=6*5=30 帧输入，帧数太少 resample 会重复帧)
  - 帧数 <= 120 (超过 100ms/帧的慢速 GIF 对 10fps 训练无意义)
  - 短边 >= 96  (最终 resize 到 128×128，太小的会糊成马赛克)
  - 短边 <= 800 (超大图下载/解码慢，且 resize 后无额外信息)
  - 帧间有实际变化 (排除伪动图：帧差太小 = 静态图)

输出分目录，便于按类型抽样。
"""
import os
import sys

sys.path.insert(0, r'D:\AetherGIF-V1\scripts\_libs')
import py7zr  # noqa: E402

from PIL import Image, ImageSequence  # noqa: E402
import numpy as np  # noqa: E402

PACK = 'D:/AetherGIF-V1/data/gif_raw/_packs/bulk_22000.7z'
EXTRACT = 'D:/AetherGIF-V1/data/gif_raw/_extract'
OUT = 'D:/AetherGIF-V1/data/gif_raw/'

MIN_FRAMES, MAX_FRAMES = 6, 120
# 短边阈值：抽样 1500 个实测分布 p25=40 p50=74 p75=96，
# >=96 只有 25.3%（5576 个），>=64 有 56.9%（约 12500 个）。
# 实测 64~95 档放大到 128px 后文字/边缘仍清晰可读，值得放宽。
MIN_SHORT, MAX_SHORT = 64, 800


def gif_ok(path):
    """返回 (是否合格, 帧数, 短边, 平均帧差)。异常一律 False。"""
    try:
        with Image.open(path) as im:
            n = 0
            frames = []
            short = None
            for i in range(MAX_FRAMES + 5):
                try:
                    im.seek(i)
                except EOFError:
                    break
                f = im.convert('RGB')
                if short is None:
                    short = min(f.size)
                if i < 12:                      # 只取前 12 帧判变化，够用且快
                    frames.append(np.asarray(f.resize((64, 64)), dtype=np.float32))
                n += 1
            if n < MIN_FRAMES or n > MAX_FRAMES:
                return False, n, short, 0.0
            if short is None or short < MIN_SHORT or short > MAX_SHORT:
                return False, n, short, 0.0
            if len(frames) < 3:
                return False, n, short, 0.0
            d = np.abs(np.diff(np.stack(frames), axis=0)).mean()
            if d < 1.5:                        # 帧差过小 = 伪动图/静态图
                return False, n, short, float(d)
            return True, n, short, float(d)
    except Exception:
        return False, 0, 0, 0.0


def main():
    # 已用 bsdtar 解压到 EXTRACT，直接扫描，不再走 py7zr（22000gifs.7z 用了 BCJ2，
    # py7zr 不支持；C:\Windows\System32\tar.exe = bsdtar 3.8.4 可以）
    os.makedirs(EXTRACT, exist_ok=True)
    if not os.path.isdir(EXTRACT) or not os.listdir(EXTRACT):
        print(f'[ERR] {EXTRACT} 为空，请先用 bsdtar 解压：')
        print(f'      /c/Windows/System32/tar.exe -xf {PACK} -C {EXTRACT}')
        return

    # ★ 必须大小写不敏感：包内文件是 .GIF（大写），Windows 上 find -name "*.gif" 会漏
    gifs = []
    for root, _, files in os.walk(EXTRACT):
        for f in files:
            if f.lower().endswith('.gif'):
                gifs.append(os.path.join(root, f))
    print(f'[SCAN] 找到 {len(gifs)} 个 GIF（.gif/.GIF 均计）')

    kept, stats = [], {'ok': 0, 'frames': 0, 'small': 0, 'short_edge': 0, 'static': 0, 'err': 0}
    for i, p in enumerate(gifs):
        try:
            ok, n, short, d = gif_ok(p)
        except Exception:
            stats['err'] += 1
            continue
        if ok:
            kept.append((p, n, short, d))
            stats['ok'] += 1
        elif n and (n < MIN_FRAMES or n > MAX_FRAMES):
            stats['frames'] += 1
        elif short and (short < MIN_SHORT or short > MAX_SHORT):
            stats['short_edge'] += 1
        elif n >= MIN_FRAMES:
            stats['static'] += 1
        if (i + 1) % 2000 == 0:
            print(f'  扫 {i+1}/{len(gifs)}  合格 {stats["ok"]}', flush=True)

    print(f'\n[筛选结果] 合格 {stats["ok"]} / {len(gifs)}')
    print(f'  帧数不符 {stats["frames"]} | 尺寸不符 {stats["short_edge"]} '
          f'| 静态/伪动图 {stats["static"]} | 读取失败 {stats["err"]}')

    # 存放：优先按源包的语义分类目录（200 个子类，如 SPACERB/BIRDHS/ARM/OILLAMP/MPEOPLE），
    # 没有分类信息的落到 _misc。这类标签对后续按类型抽样/分组建模很有价值。
    buckets = {'slow_6_15': [], 'mid_16_40': [], 'fast_41_120': []}
    for p, n, short, d in kept:
        k = 'slow_6_15' if n <= 15 else ('mid_16_40' if n <= 40 else 'fast_41_120')
        cat = os.path.basename(os.path.dirname(p)) or 'MISC'
        buckets[k].append((p, cat))

    for k, lst in buckets.items():
        for cat in sorted({c for _, c in lst}):
            items = [(p, c) for p, c in lst if c == cat]
            dst = os.path.join(OUT, 'bulk_' + k, cat)
            os.makedirs(dst, exist_ok=True)
            for p, _ in items:
                base = os.path.splitext(os.path.basename(p))[0].lower() + '.gif'
                tgt = os.path.join(dst, base)
                i = 1
                while os.path.exists(tgt):
                    tgt = os.path.join(dst, f'{os.path.splitext(base)[0]}_{i}.gif')
                    i += 1
                try:
                    with open(p, 'rb') as fi, open(tgt, 'wb') as fo:
                        fo.write(fi.read())
                except OSError:
                    pass
        print(f'  {k}: {len(lst)} 个 -> {OUT}bulk_{k}/')

    print(f'\n[完成] 新增 GIF {stats["ok"]} 个')


if __name__ == '__main__':
    main()
