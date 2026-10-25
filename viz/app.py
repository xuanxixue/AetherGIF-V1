# -*- coding: utf-8 -*-
"""AetherGIF 统一演示服务：一个页面切换权重 + 全部生成模式。

★ 显存纪律（4GB 卡）
  文本/图像编码器（INT8 Qwen，2444MB）与生成模型（1~2GB）**永不共存**：
  编码时先卸生成模型 → 编码 → 释放编码器 → 再加载目标模型。切换权重同理。
  所以每次切换有 20~60 秒加载窗口，这是硬件决定的，不是实现偷懒。

★ 三种生成模式（与 training 的 COND_COMBOS 严格对应）
  t2v  文本 → GIF        has=(1,0,0)
  i2v  图片 → GIF        has=(0,1,0)
  both 文本+图片 → GIF   has=(1,1,0)
  v3600 走「给定目标场的物理重建」，与 overfit.py 出 overfit_honest.gif 同口径。

启动：
  C:/Python312/python.exe viz/app.py --port 8100
"""
import os
import io
import sys
import json
import math
import time
import base64
import queue
import argparse
import threading
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
VERS = PROJ / "docx" / "实验记录" / "versions"
V3600 = VERS / "v3600_方案D_D125"
V4200 = VERS / "v4200_多模态CLIP产品版"
ASSETS = PROJ / "训练素材22"

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6

# ---------------------------------------------------------------- 环境
for p in [str(PROJ), str(PROJ / "deps312"), str(PROJ / "int8_deps")]:
    sys.path.insert(0, p)
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
for k, v in dict(COND_INIT_PC="1", COND_PC_RANK="8", COND_GROUPS="5",
                 COND_LOWRANK="16", PHASE_ACT="arctan", RES_LIMIT="0.5",
                 W2_NORM_MAX="3.0", KAPPA_MAX="6.0", FFT_FFTW="1",
                 IDEAL_CLAMP=str(math.pi)).items():
    os.environ.setdefault(k, v)

# ★ 关键：v3600(src 里 TEXT_DIM=896) 与 v4200(2048) 的 model 包名相同、形状不兼容，
#   同进程只能挂一套。这里**延迟到 main() 按 --kind 决定**，模块顶层不插 VERS 的 src，
#   否则 v3600 进程会 import 到 v4200 的模型类 → text_param 形状不匹配。

import numpy as np
import torch

# ---------------------------------------------------------------- 权重注册
WEIGHTS = {
    "v3600": dict(kind="v3600", label="v3600 · 结构模型 D125",
                  path=str(V3600 / "outputs_real" / "overfit_ckpt.pt"),
                  note="无文本/图像条件；演示「给定目标场的物理重建」"),
    "v4200_s1800": dict(kind="v4200", label="v4200 · 条件生成 1800 步",
                        path=str(V4200 / "outputs_real" / "guid4_long" / "guided_ckpt_step1800.pt"),
                        note="2 素材 guidance 全量训练，轮廓最清晰"),
    "v4200_600": dict(kind="v4200", label="v4200 · 22 素材 600 步（最新）",
                      path=str(V4200 / "outputs_real" / "disc22_600" / "guided_ckpt.pt"),
                      note="22 素材泛化尝试 + 判别损失，颜色仍在攻"),
}
DEFAULT_W = "v4200_s1800"

# ---------------------------------------------------------------- 预设
TEXT_PRESETS = [
    dict(id="8ball", label="8号球图标", src="01_8ball.gif",
         text='一张透明背景的台球"8号球"矢量图标：深灰黑色球体，左上方有一个白色圆形区域，'
              '白圈正中是黑色粗体无衬线数字"8"，球体左上角有高光、右下方有深色弧线阴影，'
              '整体呈扁平图标风格但有 3D 立体感'),
    dict(id="senpai", label="前辈前辈表情包", src="02_senpai.gif",
         text='一张二次元Q版动漫表情包：黑色长直发少女，齐刘海，左侧头发绑红色蝴蝶结发带，'
              '闭着倒V字型"^ ^"的眼睛，眼角有汗滴和情绪波动线，脸颊泛红带三条斜线，'
              '嘴巴微张成小U型，双手举在头部两侧握拳挥舞，身穿深色水手服领制服、白色条纹衣领、'
              '胸前红色领结、细银色十字架项链，图片顶部有手写风格粗体艺术字"前辈前辈~"，末尾波浪号拉长'),
]
IMG_PRESETS = [
    dict(id="8ball", label="8号球", file="01_8ball.gif"),
    dict(id="senpai", label="前辈前辈", file="02_senpai.gif"),
    dict(id="idol", label="偶像大师", file="03_10278_Idolmaster_0006.gif"),
    dict(id="ghibli", label="千与千寻", file="10_199_Spirited-Away_0001.gif"),
    dict(id="opm", label="一拳超人", file="15_30276_One-Punch-Man_0002.gif"),
    dict(id="pumpkin", label="南瓜小魔女", file="21_37993_Wataten_0004.gif"),
]

# ---------------------------------------------------------------- 小工具
def png_b64(arr, fmt="PNG"):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, optimize=True)
    return "data:image/%s;base64,%s" % (fmt.lower(), base64.b64encode(buf.getvalue()).decode())


def gif_first(path, size=160):
    from PIL import Image
    with Image.open(path) as im:
        im.seek(0)
        return im.convert("RGB").resize((size, size), Image.LANCZOS).copy()


def field_from(a, th):
    z = a * torch.exp(1j * th)
    return torch.cat([z.real, z.imag], dim=-1).float()


def frames_from(field, normalize=True):
    from codec.gif_codec import decode_field_to_frames
    return np.stack(decode_field_to_frames(
        field.cpu(), TS, GRID, PATCH, WIN, IMG, D, normalize=normalize)).astype(np.uint8)


def strip_of(frames, n=6):
    idx = [int(k * (len(frames) - 1) / (n - 1)) for k in range(n)]
    return np.concatenate([frames[j] for j in idx], axis=1)


def energy_map(a):
    a = a.reshape(TS, GRID, GRID, -1).sum(-1).detach().cpu().float()
    a = a - a.min()
    a = a / (a.max() + 1e-9)
    return (a.numpy() * 255).astype(np.uint8)


def colorize(g):
    x = g.astype(np.float32) / 255.0
    r = np.clip(1.5 * x - 0.35, 0, 1)
    gg = np.clip(1.5 - np.abs(x - 0.55) * 3.0, 0, 1)
    b = np.clip(1.2 - 1.8 * x, 0, 1)
    return (np.stack([r, gg, b], -1) * 255).astype(np.uint8)


def vec_img(v, w=96):
    from PIL import Image
    v = np.asarray(v).reshape(-1)
    n = v.size
    h = int(math.ceil(n / w))
    pad = np.zeros(h * w, dtype=np.float32)
    pad[:n] = v
    g = pad.reshape(h, w)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    x = g[..., None]
    r = np.clip(1.5 * x - 0.35, 0, 1)
    gg = np.clip(1.5 - np.abs(x - 0.55) * 3.0, 0, 1)
    b = np.clip(1.2 - 1.8 * x, 0, 1)
    im = Image.fromarray((np.concatenate([r, gg, b], -1) * 255).astype(np.uint8))
    im = im.resize((w * 3, max(24, int(w * 3 * h / w))), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- 运行时
class Runtime:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.cur_key = None
        self.kind = None
        self.enc = None
        self.enc_kind = None          # "text" | "image"
        self.vec_cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.t = self.s = None

    # ---- 卸载 ----
    def _drop_model(self):
        if self.model is not None:
            self.model = None
            self.cur_key = None
            self.kind = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def _drop_enc(self):
        if self.enc is not None:
            self.enc = None
            self.enc_kind = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    def unload_all(self):
        self._drop_model()
        self._drop_enc()

    # ---- 编码 ----
    def encode(self, mode, text, img_file):
        """返回 (cond_vec[1,6147], meta)。编码器用完即卸，绝不与生成模型共存。"""
        from data.clip_condition import build_cond_vector3, CLIP_EMB_DIM
        zero = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
        cache_key = (mode, text or "", img_file or "")
        if cache_key in self.vec_cache:
            cv, meta = self.vec_cache[cache_key]
            return cv, dict(meta, cached=True)

        need_text = mode in ("t2v", "both")
        need_img = mode in ("i2v", "both")
        has = (1 if need_text else 0, 1 if need_img else 0, 0)

        t_e = zero
        i_e = zero
        meta = dict(mode=mode, text="", img="", has=has)
        if need_text and not text:
            raise ValueError("文生模式需要文本")
        if need_img and not img_file:
            raise ValueError("图生模式需要选择图片")

        # 卸载生成模型，腾出整张卡
        self._drop_model()
        self._drop_enc()
        from int8_qwen_encoder import Int8QwenEncoder
        enc = Int8QwenEncoder(max_frames=8)
        enc._ensure()
        try:
            with torch.no_grad():
                if need_text:
                    t_e = enc.encode_texts([text])
                    meta["text"] = text
                    meta["text_norm"] = round(float(np.linalg.norm(t_e[0])), 4)
                if need_img:
                    pil = gif_first(str(ASSETS / img_file), 448)
                    i_e = enc.encode_images([pil])
                    meta["img"] = img_file
                    meta["img_norm"] = round(float(np.linalg.norm(i_e[0])), 4)
        finally:
            del enc
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        cv = build_cond_vector3(t_e, i_e, zero, has).astype(np.float32)
        meta["norm"] = round(float(np.linalg.norm(cv)), 4)
        self.vec_cache[cache_key] = (cv, meta)
        return cv, meta

    # ---- 加载生成模型 ----
    def load_model(self, key):
        if self.cur_key == key and self.model is not None:
            return
        self._drop_enc()
        self._drop_model()
        cfg = WEIGHTS[key]
        from model.aether_gif_model import AetherGIFModel
        from physics.kuramoto_sync import build_spacetime_index
        if cfg["kind"] == "v4200":
            m = AetherGIFModel(d_model=D, device=self.device, use_qwen=False, n_blocks=12,
                               ffn_hidden=256, init_mode="init", n_modules=8,
                               mod_gamma=0.15).to(self.device)
        else:
            m = AetherGIFModel(d_model=D, device=self.device, use_qwen=False).to(self.device)
        sd = torch.load(cfg["path"], map_location=self.device, weights_only=True)
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
        ret = m.load_state_dict(sd, strict=False)
        m.eval()
        self.model, self.cur_key, self.kind = m, key, cfg["kind"]
        self.t, self.s = build_spacetime_index(TS, GRID, GRID, self.device)
        print(f"[LOAD] {key}  missing={len(ret.missing_keys)} "
              f"unexpected={len(ret.unexpected_keys)}")

    # ---- 流式生成 ----
    def stream(self, key, mode, text, img_file, seed=1234):
        with self.lock:
            self.vec_cache = {k: v for k, v in self.vec_cache.items()}
            cfg = WEIGHTS[key]
            cond_meta = {}
            cv_t = None
            tf = None

            if cfg["kind"] == "v4200":
                if mode == "v3600_only":
                    mode = "t2v"
                yield {"stage": "status", "msg": "编码条件（文本/图像）…"}
                cv, meta = self.encode(mode, text, img_file)
                cv_t = torch.from_numpy(cv).to(self.device).float()
                cond_meta = dict(mode={"t2v": "文生（文本→GIF）",
                                       "i2v": "图生（图片→GIF）",
                                       "both": "图文生（文本+图片→GIF）"}[mode],
                                 dim=int(cv.shape[1]), item=meta.get("text") or meta.get("img") or "",
                                 detail={k: v for k, v in meta.items()
                                         if k in ("text_norm", "img_norm", "norm", "cached")})
            else:
                mode = "rebuild"
                from codec.gif_codec import encode_gif_to_field
                gp = ASSETS / "02_senpai.gif"
                tf = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D
                                         ).to(self.device).unsqueeze(0)
                cond_meta = dict(mode="结构模型 · 给定目标场的物理重建", dim=0,
                                 item=gp.name, detail={"说明": "target_field 经 fft_bypass 注入"})

            yield {"stage": "status", "msg": f"加载权重 {WEIGHTS[key]['label']}…"}
            self.load_model(key)

            tag = f"{key}_{time.strftime('%m%d_%H%M%S')}"
            run_dir = HERE / "runs" / tag
            run_dir.mkdir(parents=True, exist_ok=True)

            q = queue.Queue()

            def _emit(a0, th0, name, idx):
                fld = field_from(a0, th0)
                norm = (self.kind != "v3600")
                frames = frames_from(fld, normalize=norm)
                strip = strip_of(frames)
                em = energy_map(a0)
                ems = np.concatenate([colorize(em[j]) for j in
                                      [int(k * (len(em) - 1) / 5) for k in range(6)]], axis=1)
                from PIL import Image as _I
                _I.fromarray(strip).save(run_dir / f"{name}_frames.png")
                _I.fromarray(ems).save(run_dir / f"{name}_energy.png")
                wcos = None
                if tf is not None:
                    a_t = torch.sqrt(tf[0, :, :D] ** 2 + tf[0, :, D:] ** 2 + 1e-12)
                    th_t = torch.atan2(tf[0, :, D:], tf[0, :, :D])
                    pe = torch.remainder(th0 - th_t + math.pi, 2 * math.pi) - math.pi
                    w = a_t ** 2
                    wcos = float((w * torch.cos(pe)).sum() / w.sum().clamp(min=1e-12))
                q.put({"stage": "layer", "name": name, "idx": idx,
                       "frames": png_b64(strip), "energy": png_b64(ems),
                       "metrics": dict(a_mean=round(float(a0.mean()), 4),
                                       a_max=round(float(a0.max()), 4),
                                       sync=round(float(torch.abs(torch.exp(1j * th0).mean())), 4),
                                       wcos=(round(wcos, 4) if wcos is not None else None))})

            handles = []
            blocks = list(self.model.pmnncore.blocks)

            def pre_hook(mod, inp):
                _emit(inp[0].detach()[0], inp[1].detach()[0], "init", -1)

            def mk(i):
                def h(mod, inp, out):
                    _emit(out[0].detach()[0], out[1].detach()[0], f"L{i:02d}", i)
                return h

            handles.append(blocks[0].register_forward_pre_hook(pre_hook))
            for i, b in enumerate(blocks):
                handles.append(b.register_forward_hook(mk(i)))

            box = {}

            def worker():
                try:
                    with torch.inference_mode():
                        if self.kind == "v4200":
                            a, th, _ = self.model(texts=None, target_field=None,
                                                  t_coord=self.t, s_coord=self.s, batch_size=1,
                                                  cond_vec=cv_t, use_cond_field=True)
                        else:
                            a, th, _ = self.model(texts=["台球"], target_field=tf,
                                                  t_coord=self.t, s_coord=self.s, batch_size=1)
                        box["a"], box["th"] = a.detach(), th.detach()
                except Exception as e:
                    box["err"] = f"{type(e).__name__}: {e}"
                    traceback.print_exc()
                finally:
                    for h in handles:
                        h.remove()
                    q.put({"stage": "__done__"})

            threading.Thread(target=worker, daemon=True).start()
            while True:
                try:
                    d = q.get(timeout=600)
                except queue.Empty:
                    yield {"stage": "error", "msg": "推理超时"}
                    return
                if d.get("stage") == "__done__":
                    break
                yield d

            if "err" in box:
                yield {"stage": "error", "msg": box["err"]}
                return

            a0, th0 = box["a"][0], box["th"][0]
            norm = (self.kind != "v3600")
            frames = frames_from(field_from(a0, th0), normalize=norm)
            from PIL import Image
            pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
            pil[0].save(run_dir / "final.gif", save_all=True,
                        append_images=pil[1:], loop=0, duration=100)

            tgt_url = ""
            if tf is not None:
                tfr = frames_from(tf[0], normalize=norm)
                tp = [Image.fromarray(np.ascontiguousarray(f)) for f in tfr]
                tp[0].save(run_dir / "target.gif", save_all=True,
                           append_images=tp[1:], loop=0, duration=100)
                tgt_url = f"/runs/{tag}/target.gif"

            summary = dict(run_dir=str(run_dir), n_frames=len(frames),
                           a_mean=round(float(a0.mean()), 4),
                           sync=round(float(torch.abs(torch.exp(1j * th0).mean())), 4))
            (run_dir / "meta.json").write_text(json.dumps(
                dict(weight=key, cond=cond_meta, summary=summary),
                ensure_ascii=False, indent=1), encoding="utf-8")
            yield {"stage": "final", "final": png_b64(strip_of(frames)),
                   "gif_url": f"/runs/{tag}/final.gif", "target_gif_url": tgt_url,
                   "summary": summary, "cond": cond_meta}


RT = Runtime()


# ---------------------------------------------------------------- Flask
def make_app():
    from flask import Flask, Response, request, jsonify, send_from_directory
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        return (HERE / "app.html").read_text(encoding="utf-8")

    @app.route("/presets")
    def presets():
        imgs = []
        for p in IMG_PRESETS:
            f = ASSETS / p["file"]
            if f.exists():
                imgs.append(dict(id=p["id"], label=p["label"],
                                 thumb=png_b64(np.array(gif_first(str(f), 96)))))
        return jsonify(weights=[dict(id=k, label=v["label"], kind=v["kind"],
                                     note=v["note"]) for k, v in WEIGHTS.items()],
                       default=DEFAULT_W, texts=TEXT_PRESETS, images=imgs,
                       modes=[dict(id="t2v", label="文生（文本→GIF）"),
                              dict(id="i2v", label="图生（图片→GIF）"),
                              dict(id="both", label="图文生（文本+图片→GIF）")])

    @app.route("/stream")
    def stream():
        key = request.args.get("w", DEFAULT_W)
        mode = request.args.get("mode", "t2v")
        text = request.args.get("text", "")
        img = request.args.get("img", "")
        # 图片预设传的是 id（如 ghibli），这里映射回真实文件名
        _imap = {p["id"]: p["file"] for p in IMG_PRESETS}
        img = _imap.get(img, img)
        seed = int(request.args.get("seed", 1234))
        if key not in WEIGHTS:
            return jsonify(ok=False, msg="未知权重"), 400

        def gen():
            try:
                for d in RT.stream(key, mode, text, img, seed):
                    yield "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
            except Exception as e:
                traceback.print_exc()
                yield "data: " + json.dumps(
                    {"stage": "error", "msg": f"{type(e).__name__}: {e}"},
                    ensure_ascii=False) + "\n\n"

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/unload")
    def unload():
        RT.unload_all()
        return jsonify(ok=True)

    @app.route("/runs/<path:fn>")
    def runs(fn):
        return send_from_directory(HERE / "runs", fn)

    @app.route("/health")
    def health():
        return jsonify(ok=True, cur=RT.cur_key,
                       gpu_mb=round(torch.cuda.memory_reserved() / 1024 ** 2))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--kind", choices=["v3600", "v4200", "all"], default="all",
                    help="本进程只服务某一类权重（v3600/v4200 的模型类不兼容，"
                         "同进程无法共存：文本维度 896 vs 2048）")
    args = ap.parse_args()

    if args.kind != "all":
        global WEIGHTS, DEFAULT_W
        WEIGHTS = {k: v for k, v in WEIGHTS.items() if v["kind"] == args.kind}
        if DEFAULT_W not in WEIGHTS:
            DEFAULT_W = next(iter(WEIGHTS))
        print(f"[KIND] 本进程只加载 {args.kind} 类权重：{list(WEIGHTS)}")

    # ★ 按 kind 挂对应版本的 src（两套模型类不兼容，见文件头注释）
    root = V3600 if args.kind == "v3600" else V4200
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    print(f"[SRC] {root}")

    app = make_app()
    print("=" * 70)
    print(f"  AetherGIF 演示  kind={args.kind}  http://{args.host}:{args.port}/")
    print(f"  权重: {', '.join(WEIGHTS)}")
    print("=" * 70)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
