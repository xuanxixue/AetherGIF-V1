#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AetherGIF-V1 项目完整性体检（纯只读，不删除、不修改任何文件）

检查项：
  1. 每个 .py 文件能否编译（语法级）
  2. 每个 .py 文件的内部 import 目标是否存在
  3. 脚本中硬编码的文件路径引用是否存在
  4. 数据集索引 / 缓存文件引用的素材是否存在
  5. checkpoint 能否被 torch.load（结构级）
"""
import ast
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(r"D:\AetherGIF-V1")
V4200 = ROOT / "docx" / "实验记录" / "versions" / "v4200_多模态CLIP产品版"

SKIP_DIR_PARTS = {
    ".git", ".venv", "__pycache__", "_libs", "deps312", "int8_deps",
    "baseline_2500", "node_modules",
}
SKIP_TOP = {".git", ".venv", "__pycache__", "deps312", "int8_deps", "nul"}


def is_skipped(p: Path) -> bool:
    parts = set(p.parts)
    if parts & SKIP_TOP:
        return True
    return bool(parts & SKIP_DIR_PARTS)


def collect_py(root: Path):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        if is_skipped(dp.relative_to(ROOT)) if dp.is_relative_to(ROOT) else False:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_PARTS]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(dp / fn)
    return out


# ---------- 1. 语法编译 ----------
def check_syntax(files):
    bad = []
    for f in files:
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
            compile(src, str(f), "exec")
        except SyntaxError as e:
            bad.append((f, f"SyntaxError line {e.lineno}: {e.msg}"))
        except Exception as e:
            bad.append((f, f"{type(e).__name__}: {e}"))
    return bad


# ---------- 2. import 目标解析 ----------
INTERNAL_ROOTS = ("src", "physics", "codec", "model", "models", "data", "train",
                  "qwen_encoder", "physics_core", "codec_core")


def module_exists(mod: str, base: Path) -> bool:
    """mod 形如 physics.block / codec.gif_codec / data.clip_condition"""
    p = base / Path(*mod.split("."))
    return (p.with_suffix(".py").exists()) or (p / "__init__.py").exists()


def candidate_roots(f: Path) -> list:
    """
    为单个文件推导候选 sys.path 根目录。
    脚本通常在文件所在目录或其上级目录里 sys.path.insert('src')，
    因此向上最多 6 级，每级都尝试 '<ancestor>' 与 '<ancestor>/src'。
    """
    roots = []
    d = f.parent
    for _ in range(7):
        roots.append(d)
        roots.append(d / "src")
        if d == ROOT or d.parent == d:
            break
        d = d.parent
    # 去重保序
    seen = set()
    out = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def check_imports(files, base: Path, label: str):
    missing = []
    for f in files:
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        roots = candidate_roots(f)
        pkg_root = None
        for r in roots:
            if (r / "physics").is_dir() or (r / "codec").is_dir() or (r / "model").is_dir():
                pkg_root = r
                break
        if pkg_root is None:
            pkg_root = roots[0]

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    # 相对导入：from .xxx import / from . import xxx
                    # 规则：level=1 -> 文件同目录；level=n -> 上溯 n-1 级。
                    # namespace 包（无 __init__.py）同样适用。
                    pdir = f.parent
                    for _ in range(node.level - 1):
                        pdir = pdir.parent
                    if node.module:
                        target = pdir / Path(*node.module.split("."))
                        ok = target.with_suffix(".py").exists() or (target / "__init__.py").exists()
                        if not ok and not is_known_third_party_gap(f, target):
                            missing.append((f, node.lineno, f"相对导入 {'.'*node.level}{node.module} -> {target}"))
                    else:
                        # from . import a, b  -> 检查 a/b 是否在 pdir 下
                        for alias in node.names:
                            t = pdir / alias.name
                            if not (t.with_suffix(".py").exists() or (t / "__init__.py").exists()):
                                missing.append((f, node.lineno, f"相对导入 {'.'*node.level}import {alias.name} -> {t}"))
                else:
                    mod = node.module or ""
                    if mod.split(".")[0] not in INTERNAL_ROOTS:
                        continue
                    hit = False
                    for r in roots:
                        if module_exists(mod, r):
                            hit = True
                            break
                    if not hit:
                        missing.append((f, node.lineno, f"模块 {mod}"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod.split(".")[0] not in INTERNAL_ROOTS:
                        continue
                    hit = False
                    for r in roots:
                        if module_exists(mod, r):
                            hit = True
                            break
                    if not hit:
                        missing.append((f, node.lineno, f"模块 {mod}"))
    return missing


# 第三方仓库的已知残缺（见 docs/已知缺失清单.md），不计为项目损坏
THIRD_PARTY_KNOWN_GAPS = {
    ("models", "jina-clip-impl"): [
        "eva_model", "hf_model", "rope_embeddings", "transform",
    ],
}


def is_known_third_party_gap(f: Path, target: Path) -> bool:
    for rel_dir, names in THIRD_PARTY_KNOWN_GAPS.items():
        base = ROOT.joinpath(*rel_dir)
        if base in target.parents:
            if target.stem in names:
                return True
    return False


# ---------- 3. 硬编码路径引用 ----------
PATH_PATTERNS = [
    re.compile(r"""["']([A-Za-z]:[\\/][^"']{3,200}?)["']"""),
    re.compile(r"""["']((?:outputs_real|outputs|data|models|scripts|src|docs|Docs|logs|ckpt)[\\/][^"']{2,180}?)["']"""),
]

def check_paths(files, base: Path):
    missing = []
    outputs = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for pat in PATH_PATTERNS:
            for m in pat.finditer(text):
                raw = m.group(1)
                # 排除明显是模板/占位/URL
                if any(x in raw for x in ("{", "}", "%", "*", "<", ">", "http", "://")):
                    continue
                # 只查扩展名明确的文件引用
                if not re.search(r"\.(pt|json|npy|gif|png|jsonl|txt|mp4|pth|bin|safetensors)$", raw, re.I):
                    continue
                cand = Path(raw)
                if not cand.is_absolute():
                    cand = (base / raw)
                if cand.exists():
                    continue
                # 尝试相对于项目根
                alt = ROOT / raw
                if alt.exists():
                    continue
                # 相对 data/gif_raw 的素材引用
                if raw.startswith("data/gif_raw/") or raw.startswith("data\\gif_raw\\"):
                    sub = raw.split("gif_raw", 1)[1].lstrip("/\\")
                    if any((gr / sub).exists() for gr in find_gif_roots()):
                        continue
                # 输出产物路径：属于「尚未生成」而非「引用损坏」
                if re.match(r"^(outputs_real|outputs)[\\/]", raw):
                    outputs.append((f, raw))
                    continue
                missing.append((f, raw))
    return missing, outputs


def find_gif_roots() -> list:
    """向上查找 data/gif_raw 目录（可能存在于多处）"""
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dp = Path(dirpath)
        if is_skipped(dp.relative_to(ROOT)) if dp.is_relative_to(ROOT) else False:
            dirnames[:] = []
            continue
        if dp.name == "gif_raw" and dp.parent.name == "data":
            out.append(dp)
            dirnames[:] = []
    return out


# ---------- 4. 数据索引校验 ----------
def check_data_index(base: Path):
    report = []
    idx = base / "data" / "index.json"
    if idx.exists():
        try:
            data = json.loads(idx.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("items", [])
            gif_roots = find_gif_roots()
            ok = miss = 0
            miss_samples = []
            for it in items:
                if isinstance(it, dict):
                    rel_key = None
                    for key in ("path", "gif", "file", "src"):
                        if key in it:
                            rel_key = it[key]
                            break
                    if rel_key is None:
                        continue
                    found = False
                    for gr in gif_roots:
                        if (gr / rel_key).exists():
                            found = True
                            break
                    if found:
                        ok += 1
                    else:
                        miss += 1
                        if len(miss_samples) < 5:
                            miss_samples.append(rel_key)
            report.append(("data/index.json", f"条目 {len(items)}, 存在 {ok}, 缺失 {miss}", miss_samples))
        except Exception as e:
            report.append(("data/index.json", f"解析失败: {e}", []))
    return report


# ---------- 5. checkpoint 结构 ----------
def check_ckpt(base: Path):
    import torch
    out = []
    for ck in sorted(base.rglob("*.pt")):
        if is_skipped(ck.relative_to(ROOT)) if ck.is_relative_to(ROOT) else False:
            continue
        try:
            obj = torch.load(str(ck), map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                keys = list(obj.keys())[:6]
                desc = f"dict, keys={keys}"
            else:
                desc = type(obj).__name__
            out.append((ck.relative_to(ROOT), "OK", desc))
        except Exception as e:
            out.append((ck.relative_to(ROOT), "FAIL", f"{type(e).__name__}: {str(e)[:120]}"))
    return out


def main():
    print("=" * 78)
    print("AetherGIF-V1 项目完整性体检（只读）")
    print("=" * 78)

    for base, label in ((V4200, "v4200（最新工作副本）"), (ROOT, "项目根目录")):
        print(f"\n{'#' * 78}\n# 区域：{label}\n# 路径：{base}\n{'#' * 78}")
        if not base.exists():
            print("  [跳过] 目录不存在")
            continue

        py_files = collect_py(base)
        print(f"\n[1] Python 文件总数: {len(py_files)}")

        bad = check_syntax(py_files)
        print(f"[2] 语法编译: {'全部通过' if not bad else f'{len(bad)} 个失败'}")
        for f, e in bad:
            print(f"    ✗ {f.relative_to(ROOT) if f.is_relative_to(ROOT) else f}: {e}")

        miss = check_imports(py_files, base, label)
        print(f"[3] 内部 import 解析: {'全部命中' if not miss else f'{len(miss)} 个缺失'}")
        for f, ln, e in miss[:40]:
            rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
            print(f"    ✗ {rel}:{ln} {e}")

        pm, po = check_paths(py_files, base)
        print(f"[4] 硬编码路径引用: {len(pm)} 个损坏 / {len(po)} 个待生成产物")
        seen = set()
        for f, raw in pm:
            rel = f.relative_to(ROOT) if f.is_relative_to(ROOT) else f
            key = (str(rel), raw)
            if key in seen:
                continue
            seen.add(key)
            print(f"    ! {rel} -> {raw}")
        if po:
            print(f"    (待生成产物 {len(po)} 条，属正常，不计入损坏)")

        for name, desc, samples in check_data_index(base):
            print(f"[5] {name}: {desc}")
            for s in samples:
                print(f"    缺失样例: {s}")

    # checkpoint 检查单独跑（耗时）
    print(f"\n{'#' * 78}\n# Checkpoint 结构校验\n{'#' * 78}")
    try:
        for rel, st, desc in check_ckpt(ROOT):
            print(f"  [{st}] {rel}  ({desc})")
    except Exception as e:
        print(f"  checkpoint 检查跳过: {e}")

    print("\n" + "=" * 78)
    print("体检结束。本脚本未删除、未修改任何文件。")
    print("=" * 78)


if __name__ == "__main__":
    main()
