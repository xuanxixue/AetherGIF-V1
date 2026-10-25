# -*- coding: utf-8 -*-
"""GIF 半自动标注网页服务。

功能：
  - 网格展示 data_pipeline/gifs 里所有 GIF 动图
  - 点击卡片弹出编辑器：左侧 GIF 循环播放，右侧 textarea 填描述
  - 点「保存」→ 在 GIF 同目录生成**同名 .txt**（名称贴合对应图片）
  - 搜索 / 已标注·未标注过滤 / 进度统计 / 键盘快捷键（左右切换）

用法：
  C:/Python312/python.exe scripts/label_server.py           # 默认 127.0.0.1:8765
  然后浏览器打开 http://127.0.0.1:8765
"""
import os
import sys
from pathlib import Path
from urllib.parse import quote, unquote

from flask import Flask, request, jsonify, send_file, abort

ROOT = Path("D:/AetherGIF-V1")
GIF_DIR = ROOT / "data_pipeline" / "gifs"
STATIC_DIR = ROOT / "data_pipeline" / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# 缓存全部 gif 名（启动加载，避免每次 listdir）
_GIF_NAMES = []
_name_cache_loaded = False


def _load_names():
    global _GIF_NAMES, _name_cache_loaded
    if _name_cache_loaded:
        return
    _GIF_NAMES = sorted(
        f.name for f in GIF_DIR.iterdir() if f.name.lower().endswith(".gif")
    )
    _name_cache_loaded = True


def txt_path(gif_name: str) -> Path:
    return GIF_DIR / (gif_name[:-4] + ".txt")


def read_txt(gif_name: str) -> str:
    p = txt_path(gif_name)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return p.read_text(encoding="utf-8", errors="replace")
    return ""


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/gif/<path:name>")
def gif(name):
    name = unquote(name)
    p = (GIF_DIR / name).resolve()
    if not str(p).startswith(str(GIF_DIR.resolve())) or not p.exists():
        abort(404)
    return send_file(p, mimetype="image/gif")


@app.route("/api/items")
def items():
    _load_names()
    q = (request.args.get("q") or "").strip()
    filt = request.args.get("filter") or "all"   # all | done | undone
    offset = int(request.args.get("offset", 0))
    limit = int(request.args.get("limit", 60))

    names = _GIF_NAMES
    if q:
        ql = q.lower()
        names = [n for n in names if ql in n.lower()]
    if filt == "done":
        names = [n for n in names if txt_path(n).exists()]
    elif filt == "undone":
        names = [n for n in names if not txt_path(n).exists()]

    total = len(names)
    page = names[offset:offset + limit]
    data = []
    for n in page:
        t = read_txt(n)
        data.append({
            "name": n,
            "done": bool(t),
            "preview": t[:60],
            "chars": len(t),
        })
    return jsonify({"total": total, "items": data})


@app.route("/api/stats")
def stats():
    _load_names()
    done = sum(1 for n in _GIF_NAMES if txt_path(n).exists())
    return jsonify({"total": len(_GIF_NAMES), "done": done})


@app.route("/api/text/<path:name>")
def get_text(name):
    name = unquote(name)
    if not name.lower().endswith(".gif"):
        abort(404)
    return jsonify({"name": name, "text": read_txt(name)})


@app.route("/api/save", methods=["POST"])
def save():
    j = request.get_json(force=True)
    name = j.get("name", "")
    text = j.get("text", "")
    name = unquote(name)
    if not name.lower().endswith(".gif"):
        return jsonify({"ok": False, "err": "非法文件名"}), 400
    # 防目录穿越
    target = (GIF_DIR / name).resolve()
    if not str(target).startswith(str(GIF_DIR.resolve())):
        return jsonify({"ok": False, "err": "非法路径"}), 400
    tp = txt_path(name)
    tp.write_text(text, encoding="utf-8")
    return jsonify({"ok": True, "chars": len(text)})


@app.route("/api/goto")
def goto():
    """定位到当前未标注的第一张（带过滤条件）。"""
    return jsonify({"url": "/"})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    _load_names()
    print(f"[服务] GIF 目录: {GIF_DIR}")
    print(f"[服务] 共 {len(_GIF_NAMES)} 张，已标注 "
          f"{sum(1 for n in _GIF_NAMES if txt_path(n).exists())} 张")
    print(f"[服务] 打开浏览器: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()