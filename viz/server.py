# -*- coding: utf-8 -*-
"""流式推理可视化服务（SSE 实时推送 + 每帧落盘）。

★ 为什么这么设计
  用 `register_forward_hook` 挂在 12 个 PhysicsBlock 上，每层算完立刻
  解码成图、塞进队列，主线程用 SSE 推给浏览器 —— 演示的是**真实前向过程**，
  不是事后拼的动画。同时每层结果都写进 `viz/runs/<tag>/`，可复盘、可录屏。

  之所以不用「手动拆开 PMNNCore.forward 逐层调」：model.forward 之前还有
  cond_init / fft_bypass / cond_anchor 等一大段前置逻辑，手拆极易与训练口径
  不一致（口径一偏，演示就变成骗人的）。hook 是在真实前向里插眼，最保险。

启动（两个版本必须分进程，顶层包名相同）：
  C:/Python312/python.exe viz/server.py --model v3600 --port 8101
  C:/Python312/python.exe viz/server.py --model v4200 --port 8102

浏览器打开 http://127.0.0.1:8101/  或 8102
"""
import os
import sys
import io
import json
import math
import time
import queue
import base64
import argparse
import threading
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
VERS = PROJ / "docx" / "实验记录" / "versions"
V3600 = VERS / "v3600_方案D_D125"
V4200 = VERS / "v4200_多模态CLIP产品版"

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
SDIM = PATCH * PATCH * 3
WFFT = SDIM // 2 + 1

# ---------------------------------------------------------------- 环境
def build_env(model: str):
    for p in [str(PROJ), str(PROJ / "deps312"), str(PROJ / "int8_deps")]:
        sys.path.insert(0, p)
    try:
        from torch_accel_shim import install_full
        install_full()
    except Exception:
        pass
    if model == "v4200":
        for k, v in dict(COND_INIT_PC="1", COND_PC_RANK="8", COND_GROUPS="5",
                         COND_LOWRANK="16", PHASE_ACT="arctan", RES_LIMIT="0.5",
                         W2_NORM_MAX="3.0", KAPPA_MAX="6.0", FFT_FFTW="1",
                         IDEAL_CLAMP=str(math.pi)).items():
            os.environ.setdefault(k, v)
        root = V4200
    else:
        root = V3600
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    return root


ARGV = None
MODEL_NAME = None
ROOT = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v3600", "v4200"], required=True)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    return ap.parse_args()


# ---------------------------------------------------------------- 图像编码
def png_b64(arr_u8, scale=None, fmt="PNG"):
    from PIL import Image
    im = Image.fromarray(arr_u8)
    if scale:
        im = im.resize(scale, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format=fmt, optimize=True)
    return "data:image/%s;base64,%s" % (fmt.lower(), base64.b64encode(buf.getvalue()).decode())


def field_from(a, th):
    z = a * torch.exp(1j * th)
    return torch.cat([z.real, z.imag], dim=-1).float()


def frames_from(field, normalize=True):
    """normalize=True 用于幅度未收敛的生成模型；False 用于 v3600 重建
    （与 overfit.py 出 overfit_honest.gif 的 fixed-clamp ±1.2 口径一致）。"""
    from codec.gif_codec import decode_field_to_frames
    return np.stack(decode_field_to_frames(
        field.cpu(), TS, GRID, PATCH, WIN, IMG, D, normalize=normalize)).astype(np.uint8)


def strip_of(frames, n=6):
    idx = [int(k * (len(frames) - 1) / (n - 1)) for k in range(n)]
    return np.concatenate([frames[j] for j in idx], axis=1)


def energy_map(a):
    """[N,D] 振幅(Tensor) -> [TS,H,W] 能量热力图 uint8。"""
    a = a.reshape(TS, GRID, GRID, -1).sum(-1).detach().cpu().float()
    a = a - a.min()
    a = a / (a.max() + 1e-9)
    return (a.numpy() * 255).astype(np.uint8)


def colorize(gray):
    x = gray.astype(np.float32) / 255.0
    r = np.clip(1.5 * x - 0.35, 0, 1)
    g = np.clip(1.5 - abs(x - 0.55) * 3.0, 0, 1)
    b = np.clip(1.2 - 1.8 * x, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


# ---------------------------------------------------------------- 推理
class Engine:
    """懒加载模型；每次 /stream 触发一次完整流式推理。"""

    def __init__(self, model_name: str, ckpt: str = None):
        self.model_name = model_name
        self.ckpt = ckpt
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        global torch, np
        import numpy as np
        import torch
        from model.aether_gif_model import AetherGIFModel
        from physics.kuramoto_sync import build_spacetime_index
        from codec.gif_codec import load_gif_frames, resample_frames, encode_gif_to_field

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model_name == "v4200":
            ck = Path(self.ckpt) if self.ckpt else (
                V4200 / "outputs_real" / "guid4_long" / "guided_ckpt_step1800.pt")
            m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                               ffn_hidden=256, init_mode="init", n_modules=8,
                               mod_gamma=0.15).to(device)
        else:
            ck = Path(self.ckpt) if self.ckpt else (V3600 / "outputs_real" / "overfit_ckpt.pt")
            m = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)
        sd = torch.load(str(ck), map_location=device, weights_only=True)
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
        ret = m.load_state_dict(sd, strict=False)
        m.eval()
        self.model = m
        self.ckpt_path = ck
        self.t, self.s = build_spacetime_index(TS, GRID, GRID, device)
        self.device = device
        return dict(ckpt=ck.name, missing=len(ret.missing_keys),
                    unexpected=len(ret.unexpected_keys),
                    params=int(sum(p.numel() for p in m.parameters())), device=device)

    def run_stream(self, tag: str, src: int = 0, text: str = "台球", seed: int = 1234,
                   ext_vec=None):
        """生成器：逐层产出 dict（含 dataURL 图 + 指标），并落盘。"""
        with self.lock:
            if self.model is None:
                info = self.load()
                yield {"stage": "load", **info}

            import numpy as np2
            import torch
            torch.manual_seed(seed)
            np2.random.seed(seed)

            model = self.model
            run_dir = PROJ / "viz" / "runs" / tag
            run_dir.mkdir(parents=True, exist_ok=True)

            q = queue.Queue()
            snaps = []

            def push(d):
                q.put(d)

            # ---- 准备条件 ----
            cond_meta = {}
            tf = None
            text_input = (text or "").strip()
            if self.model_name == "v4200":
                VD = PROJ / "训练素材22" / "vectors"
                text22 = np2.load(VD / "text22.npy")
                first22 = np2.load(VD / "first22.npy")
                gif22 = np2.load(VD / "gif22.npy")
                from data.clip_condition import build_cond_vector3
                i = int(src)
                if ext_vec is not None:
                    # ★ 分离架构：向量由独立的编码服务算好后 POST 过来
                    cv = np2.asarray(ext_vec, dtype=np2.float32).reshape(1, -1)
                    assert cv.shape[1] == 6147, f"外部向量维度 {cv.shape[1]} != 6147"
                    cond_meta = dict(mode="外部编码向量（文生模式）", dim=int(cv.shape[1]),
                                     item=(text_input or "(外部)"), detail={"文本": 1.0})
                    cur_text = text_input
                elif text_input:
                    # ★ 用户手输文本 → 在线编码。用 INT8 GPU 编码器（CPU fp32 版要
                    #   十几 GB 内存，本机只剩 3.66GB 会 OpenBLAS 分配失败）。
                    #   只开文本路 has=(1,0,0)，即训练过的「文生」组合。
                    cache_key = text_input
                    if cache_key in self._text_cache:
                        cv = self._text_cache[cache_key]
                    else:
                        yield {"stage": "encoding",
                               "msg": f"正在编码文本（INT8 Qwen，首次约 50s）：{text_input}"}
                        if getattr(self, "_enc", None) is None:
                            from int8_qwen_encoder import Int8QwenEncoder
                            self._enc = Int8QwenEncoder(max_frames=8)
                            self._enc._ensure()
                        t_e = self._enc.encode_texts([text_input])
                        zero = np2.zeros((1, 2048), dtype=np2.float32)
                        cv = build_cond_vector3(t_e, zero, zero, (1, 0, 0))
                        self._text_cache[cache_key] = cv
                    cond_meta = dict(mode="用户输入文本（文生模式）", dim=int(cv.shape[1]),
                                     item=text_input, detail={"文本": 1.0})
                    cur_text = text_input
                else:
                    cv = build_cond_vector3(text22[i:i + 1], first22[i:i + 1],
                                            gif22[i:i + 1], (1, 1, 1))
                    meta22 = json.loads((VD / "meta22.json").read_text(encoding="utf-8"))
                    item = meta22["items"][i]["file"]
                    cond_meta = dict(mode="三路条件向量", dim=int(cv.shape[1]), item=item,
                                     detail={"文本": round(float(np2.linalg.norm(text22[i])), 3),
                                             "首帧": round(float(np2.linalg.norm(first22[i])), 3),
                                             "视频": round(float(np2.linalg.norm(gif22[i])), 3)})
                    cur_text = ""
                    gp = PROJ / "训练素材22" / item
                    from codec.gif_codec import encode_gif_to_field
                    tf = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D
                                             ).to(self.device).unsqueeze(0)
                cv_t = torch.from_numpy(cv).to(self.device).float()
            else:
                # ★ v3600 结构模型：与 overfit.py 出 overfit_honest.gif 完全同口径 ——
                #   forward 时**给 target_field**（fft_bypass 旁路参与），这是「物理重建」
                #   而不是无中生有。不给 target 的话输出是没有意义的噪声场。
                cv_t = None
                cur_text = text_input or "台球"
                GIF_NAME = "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"
                gp = V3600 / "data" / "gif_raw" / "noto" / GIF_NAME
                if not gp.exists():
                    gp = next((V3600 / "data" / "gif_raw" / "noto").glob("*.gif"))
                from codec.gif_codec import encode_gif_to_field
                tf = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D
                                         ).to(self.device).unsqueeze(0)
                cond_meta = dict(mode="结构模型 · 给定目标场的物理重建", dim=0,
                                 item=gp.name,
                                 detail={"说明": "无文本条件，target_field 经 fft_bypass 注入"})

            # 拼一个跟"向量条数"一样宽的条形图，让观众直观看到输入
            vec_bars = []
            if self.model_name == "v4200":
                if text_input or ext_vec is not None:
                    v = np2.asarray(cv).reshape(-1)[:1536]
                else:
                    v = np2.concatenate([text22[i], first22[i], gif22[i]])[:1536]
                v = v[:1536].reshape(48, 32)          # 48×32 网格
                v = (v - v.min()) / (v.max() - v.min() + 1e-9)
                img_v = (np2.stack([v] * 3, -1) * 255).astype(np2.uint8)
                from PIL import Image as _I
                img_v = np2.array(_I.fromarray(img_v).resize((256, 96), _I.NEAREST))
                vec_bars = png_b64(img_v)
            yield {"stage": "cond", "cond": cond_meta, "vec_img": vec_bars,
                   "total_layers": 12}

            # ---- hook：抓初始态 + 每层输出 ----
            handles = []

            def pre_hook_first(mod, inp):
                a0, th0 = inp[0].detach(), inp[1].detach()
                snaps.append((a0, th0, "init"))
                _emit_layer(a0[0], th0[0], "init", -1)

            def mk_hook(idx):
                def h(mod, inp, out):
                    aa, tt = out[0].detach(), out[1].detach()
                    snaps.append((aa, tt, f"L{idx:02d}"))
                    _emit_layer(aa[0], tt[0], f"L{idx:02d}", idx)
                return h

            def _emit_layer(a0, th0, name, idx):
                fld = field_from(a0, th0)
                frames = frames_from(fld, normalize=(self.model_name != "v3600"))
                strip = strip_of(frames)
                emap = energy_map(a0)
                em_strip = np.concatenate(
                    [colorize(emap[j]) for j in
                     [int(k * (len(emap) - 1) / 5) for k in range(6)]], axis=1)
                url = png_b64(strip)
                url_e = png_b64(em_strip)
                # 落盘
                from PIL import Image as _I
                _I.fromarray(strip).save(run_dir / f"{name}_frames.png")
                _I.fromarray(em_strip).save(run_dir / f"{name}_energy.png")
                pe = None
                if tf is not None:
                    a_tar = torch.sqrt(tf[0, :, :D] ** 2 + tf[0, :, D:] ** 2 + 1e-12)
                    th_tar = torch.atan2(tf[0, :, D:], tf[0, :, :D])
                    dpe = torch.remainder(th0 - th_tar + math.pi, 2 * math.pi) - math.pi
                    w = a_tar ** 2
                    pe = float((w * torch.cos(dpe)).sum() / w.sum().clamp(min=1e-12))
                push({"stage": "layer", "name": name, "idx": idx,
                      "frames": url, "energy": url_e,
                      "metrics": {"a_mean": round(float(a0.mean()), 4),
                                  "a_max": round(float(a0.max()), 4),
                                  "sync": round(float(torch.abs(torch.exp(1j * th0).mean())), 4),
                                  "wcos": (round(pe, 4) if pe is not None else None)}})

            blocks = list(model.pmnncore.blocks)
            handles.append(blocks[0].register_forward_pre_hook(pre_hook_first))
            for i, blk in enumerate(blocks):
                handles.append(blk.register_forward_hook(mk_hook(i)))

            final_box = {}

            def worker():
                try:
                    with torch.inference_mode():
                        if self.model_name == "v4200":
                            a, th, _ = model(texts=None, target_field=None,
                                             t_coord=self.t, s_coord=self.s,
                                             batch_size=1, cond_vec=cv_t,
                                             use_cond_field=True)
                        else:
                            a, th, _ = model(texts=[cur_text], target_field=tf,
                                             t_coord=self.t, s_coord=self.s,
                                             batch_size=1)
                        final_box["a"] = a.detach()
                        final_box["th"] = th.detach()
                except Exception as e:
                    final_box["err"] = f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                finally:
                    for h in handles:
                        h.remove()
                    q.put({"stage": "__done__"})

            threading.Thread(target=worker, daemon=True).start()

            # ---- 从队列转发（顺序 = 前向顺序）----
            while True:
                try:
                    d = q.get(timeout=900)
                except queue.Empty:
                    yield {"stage": "error", "msg": "推理超时"}
                    return
                if d.get("stage") == "__done__":
                    break
                yield d

            if "err" in final_box:
                yield {"stage": "error", "msg": final_box["err"]}
                return

            # ---- 最终帧 + GIF ----
            a0, th0 = final_box["a"][0], final_box["th"][0]
            fld = field_from(a0, th0)
            frames = frames_from(fld, normalize=(self.model_name != "v3600"))
            from PIL import Image
            pil = [Image.fromarray(np2.ascontiguousarray(f)) for f in frames]
            gif_path = run_dir / "final.gif"
            pil[0].save(gif_path, save_all=True, append_images=pil[1:], loop=0, duration=100)

            # 目标是 v4200 才有
            tgt_url = ""
            if tf is not None:
                tframes = frames_from(tf[0], normalize=(self.model_name != "v3600"))
                tpil = [Image.fromarray(np2.ascontiguousarray(f)) for f in tframes]
                tpil[0].save(run_dir / "target.gif", save_all=True,
                             append_images=tpil[1:], loop=0, duration=100)
                tgt_url = png_b64(strip_of(tframes))

            summary = dict(run_dir=str(run_dir), gif=str(gif_path),
                           n_frames=len(frames),
                           a_mean=round(float(a0.mean()), 4),
                           sync=round(float(torch.abs(torch.exp(1j * th0).mean())), 4))
            (run_dir / "meta.json").write_text(
                json.dumps(dict(cond=cond_meta, summary=summary),
                           ensure_ascii=False, indent=1), encoding="utf-8")

            yield {"stage": "final", "final": png_b64(strip_of(frames)),
                   "target": tgt_url, "summary": summary,
                   "gif_url": f"/runs/{tag}/final.gif",
                   "target_gif_url": (f"/runs/{tag}/target.gif" if tf is not None else "")}


# ---------------------------------------------------------------- Flask
def make_app(model_name: str, ckpt: str):
    from flask import Flask, Response, request, jsonify, send_from_directory
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)
    engine = Engine(model_name, ckpt)
    VEC_STORE = {}

    @app.route("/set_vec", methods=["POST"])
    def set_vec():
        """分离架构：接收编码服务算好的 6147 维向量，返回一次性 token。"""
        import base64 as _b64
        import uuid as _uuid
        data = request.get_json(force=True) or {}
        try:
            raw = _b64.b64decode(data["vec_b64"])
            v = np.frombuffer(raw, dtype=np.float32).copy()
        except Exception as e:
            return jsonify(ok=False, msg=f"向量解析失败: {e}"), 400
        if v.size != 6147:
            return jsonify(ok=False, msg=f"维度 {v.size} != 6147"), 400
        token = "v" + _uuid.uuid4().hex[:10]
        VEC_STORE[token] = v
        if len(VEC_STORE) > 32:                       # 只留最近 32 个
            for k in list(VEC_STORE)[:-32]:
                VEC_STORE.pop(k, None)
        return jsonify(ok=True, token=token, dim=int(v.size),
                       norm=round(float(np.linalg.norm(v)), 4))

    @app.route("/")
    def index():
        html = (HERE / "template.html").read_text(encoding="utf-8")
        cfg = dict(model=model_name,
                   title=("v3600 · 方案D D125（结构生成）" if model_name == "v3600"
                          else "v4200 · 多模态条件生成（特征向量→GIF）"),
                   has_cond=(model_name == "v4200"))
        return html.replace("__CONFIG__", json.dumps(cfg, ensure_ascii=False))

    @app.route("/stream")
    def stream():
        src = int(request.args.get("src", 0))
        text = request.args.get("text", "")
        seed = int(request.args.get("seed", 1234))
        token = request.args.get("token", "")
        ext_vec = VEC_STORE.get(token) if token else None
        tag = f"{model_name}_{time.strftime('%m%d_%H%M%S')}"

        def gen():
            try:
                for d in engine.run_stream(tag, src=src, text=text, seed=seed,
                                           ext_vec=ext_vec):
                    yield "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
            except Exception as e:
                traceback.print_exc()
                yield "data: " + json.dumps(
                    {"stage": "error", "msg": f"{type(e).__name__}: {e}"},
                    ensure_ascii=False) + "\n\n"

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no",
                                 "Connection": "keep-alive"})

    @app.route("/runs/<path:fn>")
    def runs(fn):
        return send_from_directory(HERE / "runs", fn)

    @app.route("/health")
    def health():
        return jsonify(dict(ok=True, model=model_name,
                            loaded=engine.model is not None))

    return app


def main():
    global MODEL_NAME, ROOT, ARGV, torch, np
    args = parse_args()
    MODEL_NAME = args.model
    ROOT = build_env(args.model)
    import numpy as np
    import torch

    app = make_app(args.model, args.ckpt)
    print("=" * 66)
    print(f"  流式可视化服务  model={args.model}")
    print(f"  打开  http://{args.host}:{args.port}/")
    print(f"  权重  {args.ckpt or '(默认)'}")
    print("=" * 66)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
