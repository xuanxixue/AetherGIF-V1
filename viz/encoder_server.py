# -*- coding: utf-8 -*-
"""独立的「文本 → 特征向量」编码服务（与生成服务彻底分离）。

★ 为什么要单独一个进程
  文本编码器是 Qwen3-VL-Embedding-2B（INT8 也占 2444MB 显存），生成模型又要 1GB+。
  4GB 卡装不下两者同时常驻 —— 所以编码与生成必须分进程、分时占用显存。
  这也正好把「文本 → 向量」这一环节做成可视化的一环。

接口：
  GET /encode?text=...   -> {ok, dim, vec_b64(float32 6147维), img, norm, ms}
  GET /health            -> {ok, loaded}
  GET /unload            -> 释放显存（生成前调用，把卡让给生成模型）

启动：
  C:/Python312/python.exe viz/encoder_server.py --port 8103
"""
import os
import sys
import io
import json
import math
import time
import base64
import argparse
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
V4200 = PROJ / "docx" / "实验记录" / "versions" / "v4200_多模态CLIP产品版"

for p in [str(PROJ), str(PROJ / "deps312"), str(PROJ / "int8_deps")]:
    sys.path.insert(0, p)
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass

import numpy as np

import torch
sys.path.insert(0, str(V4200))
sys.path.insert(0, str(V4200 / "src"))

_lock = threading.Lock()
_enc = None


def get_encoder():
    global _enc
    if _enc is None:
        from int8_qwen_encoder import Int8QwenEncoder
        _enc = Int8QwenEncoder(max_frames=8)
        _enc._ensure()
    return _enc


def release():
    global _enc
    _enc = None
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass


def vec_img(vec):
    """把 6147 维向量画成一张热力图，让观众看见「文本变成了什么」。"""
    from PIL import Image
    v = np.asarray(vec).reshape(-1)
    n = v.size
    # 裁成接近 16:9 的网格
    w = 96
    h = int(math.ceil(n / w))
    pad = np.zeros(h * w, dtype=np.float32)
    pad[:n] = v
    g = pad.reshape(h, w)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    x = g[..., None]
    r = np.clip(1.5 * x - 0.35, 0, 1)
    gg = np.clip(1.5 - np.abs(x - 0.55) * 3.0, 0, 1)
    b = np.clip(1.2 - 1.8 * x, 0, 1)
    img = (np.concatenate([r, gg, b], -1) * 255).astype(np.uint8)
    im = Image.fromarray(img).resize((480, int(480 * h / w)), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8103)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    from flask import Flask, Response, request, jsonify, send_from_directory
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)

    @app.route("/health")
    def health():
        return jsonify(ok=True, loaded=_enc is not None,
                       gpu_mb=round(torch.cuda.memory_reserved() / 1024 ** 2) if torch.cuda.is_available() else 0)

    @app.route("/unload")
    def unload():
        release()
        return jsonify(ok=True, released=True)

    @app.route("/encode")
    def encode():
        text = (request.args.get("text") or "").strip()
        if not text:
            return jsonify(ok=False, msg="text 为空"), 400
        t0 = time.time()
        with _lock:
            enc = get_encoder()
            t_e = enc.encode_texts([text])          # [1,2048]
        from data.clip_condition import build_cond_vector3
        zero = np.zeros((1, 2048), dtype=np.float32)
        cv = build_cond_vector3(t_e, zero, zero, (1, 0, 0))   # [1,6147] 文生模式
        cv = np.asarray(cv, dtype=np.float32).reshape(-1)
        ms = int((time.time() - t0) * 1000)
        # ★ 编码完成后立刻释放显存，把 4GB 卡完整让给生成模型
        #   （INT8 常驻要 2444MB，与生成模型同时在场必 OOM）
        if request.args.get("keep", "0") != "1":
            release()
        return jsonify(ok=True, dim=int(cv.size),
                       vec_b64=base64.b64encode(cv.tobytes()).decode(),
                       norm=round(float(np.linalg.norm(cv)), 4),
                       text_norm=round(float(np.linalg.norm(t_e[0])), 4),
                       img=vec_img(cv), ms=ms, text=text)

    print("=" * 66)
    print(f"  文本编码服务  http://{args.host}:{args.port}/")
    print("  /encode?text=...   /unload   /health")
    print("=" * 66)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
