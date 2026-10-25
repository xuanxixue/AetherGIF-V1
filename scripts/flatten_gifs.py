# -*- coding: utf-8 -*-
"""把 data/gif_raw 全部 GIF 平铺到一个目录（硬链接，不占空间、不破坏原库）。

命名：用相对路径拼接成唯一名，重名文件（如各作品的 0001.gif）自动加序号。
  data/gif_raw/anime/10165_Nichijou/0001.gif
        ->  anime__10165_Nichijou__0001.gif
txt 将同名落盘（xxx.gif -> xxx.txt），实现"GIF 与 TXT 按名字对齐"。

用法：
  C:/Python312/python.exe scripts/flatten_gifs.py
"""
import os
import json
import sys
from pathlib import Path

ROOT = Path("D:/AetherGIF-V1")
SRC = ROOT / "data" / "gif_raw"
DST = ROOT / "data_pipeline" / "gifs"
MAP = ROOT / "data_pipeline" / "flatten_map.json"


def scan_gifs(src: Path):
    """os.walk 手动判扩展名（大小写均认，绝不重复）。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(src):
        for fn in filenames:
            if fn.lower().endswith(".gif"):
                out.append(Path(dirpath) / fn)
    return out


def flatten_name(p: Path) -> str:
    """相对路径 -> 唯一平坦名。"""
    rel = p.relative_to(SRC)
    parts = list(rel.parts)
    name = "__".join(parts)                       # 用 __ 连接子目录
    name = name.replace(":", "_").replace("\\", "__").replace("/", "__")
    # 扩展名统一 .gif
    if name.lower().endswith(".gif"):
        name = name[:-4] + ".gif"
    else:
        name = name + ".gif"
    return name


def main():
    gifs = scan_gifs(SRC)
    gifs.sort(key=str)
    print(f"[scan] 共 {len(gifs)} 个 GIF")

    DST.mkdir(parents=True, exist_ok=True)

    mapping = {}
    seen = {}          # 目标名 -> 已用次数（处理重名）
    linked = copied = failed = 0

    for i, p in enumerate(gifs):
        base = flatten_name(p)
        # 处理重名
        if base in mapping:
            stem = base[:-4] if base.lower().endswith(".gif") else base
            ext = ".gif"
            badge = seen.get(base, 1)
            candidate = f"{stem}__dup{badge}{ext}"
            while candidate in mapping:
                badge += 1
                candidate = f"{stem}__dup{badge}{ext}"
            seen[base] = badge
            base = candidate
        seen[base] = seen.get(base, 1)

        dst = DST / base
        try:
            if not dst.exists():
                os.link(p, dst)                    # 同卷硬链接，不占空间
                linked += 1
            else:
                # 已存在（可能上次中断），跳过链接但保留映射
                pass
        except Exception:
            try:
                import shutil
                shutil.copy2(p, dst)               # 跨卷/失败则复制
                copied += 1
            except Exception as e:
                failed += 1
                print(f"  [失败] {p} -> {e}")
                continue
        mapping[base] = str(p.relative_to(ROOT))

        if (i + 1) % 2000 == 0 or (i + 1) == len(gifs):
            print(f"  [{i+1}/{len(gifs)}] 链接={linked} 复制={copied} 失败={failed}",
                  flush=True)

    MAP.write_text(json.dumps(mapping, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"\n[完成] 平铺到 {DST}")
    print(f"  硬链接 {linked} / 复制 {copied} / 失败 {failed}")
    print(f"  映射表 {MAP}（平坦名 -> 原相对路径）")


if __name__ == "__main__":
    main()