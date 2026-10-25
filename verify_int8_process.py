# -*- coding: utf-8 -*-
"""int8 完整 process 链路验证 —— 文本 + GIF 帧序列 都能产出 embedding。

核心：用 Qwen3VLForEmbedding + int8 加载，绕过官方 Qwen3VLEmbedder.__init__
的 .to(device) 硬编码，只复用它的 process/format/pooling 纯 Python 方法。
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, r"D:\AetherGIF-V1\deps312")
sys.path.insert(0, r"D:\AetherGIF-V1\int8_deps")

import torch

if not hasattr(torch, "accelerator"):
    class _A:
        @staticmethod
        def is_available(): return torch.cuda.is_available()
        @staticmethod
        def device_count(): return torch.cuda.device_count() if torch.cuda.is_available() else 0
        @staticmethod
        def current_accelerator(): return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        @staticmethod
        def current_device_index(): return torch.cuda.current_device() if torch.cuda.is_available() else -1
        @staticmethod
        def is_bf16_supported(device=None): return False
    torch.accelerator = _A

import importlib.util

MODEL_DIR = r"D:\AetherGIF-V1\models\qwen3vl\models\Qwen--Qwen3-VL-Embedding-2B\snapshots\master"


def load_official_impl():
    spec = importlib.util.spec_from_file_location(
        "_qwen3vl_emb",
        r"D:/AetherGIF-V1/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_qwen3vl_emb"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    impl = load_official_impl()
    Qwen3VLForEmbedding = impl.Qwen3VLForEmbedding
    Qwen3VLEmbedder = impl.Qwen3VLEmbedder

    from transformers import BitsAndBytesConfig
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    print("[1] int8 加载 Qwen3VLForEmbedding")
    t0 = time.time()
    model = Qwen3VLForEmbedding.from_pretrained(
        MODEL_DIR, trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto", low_cpu_mem_usage=True,
    )
    print(f"    加载 {time.time()-t0:.1f}s, 显存 alloc={torch.cuda.memory_allocated()/1024**2:.0f}MiB")
    model.eval()

    processor = Qwen3VLProcessor.from_pretrained(MODEL_DIR, padding_side='right')

    # 构造 Qwen3VLEmbedder 实例跳过 __init__，复用它的 process 逻辑
    emb = object.__new__(Qwen3VLEmbedder)
    emb.model = model
    emb.processor = processor
    emb.max_length = 8192
    emb.min_pixels = impl.MIN_PIXELS
    emb.max_pixels = impl.MAX_PIXELS
    emb.total_pixels = impl.MAX_TOTAL_PIXELS
    emb.fps = 1.0
    emb.max_frames = 8
    emb.default_instruction = "Represent the user's input."
    print("[2] 构造嵌入器完成")

    # int8 模型没有单一 .device（device_map='auto'），process() 里
    # `{k:v.to(self.model.device)}` 会炸。这里自己实现 process：固定 cuda:0。
    def process(items, normalize=True):
        convs = [emb.format_model_input(
            text=ele.get('text'), image=ele.get('image'),
            video=ele.get('video'), instruction=ele.get('instruction'),
            fps=ele.get('fps'), max_frames=ele.get('max_frames')) for ele in items]
        pi = emb._preprocess_inputs(convs)
        pi = {k: v.to("cuda:0") for k, v in pi.items()}
        out = emb.forward(pi)
        hidden = out['last_hidden_state']
        am = out['attention_mask']
        last = am.shape[1] - am.flip(dims=[1]).argmax(dim=1) - 1
        row = torch.arange(hidden.shape[0], device=hidden.device)
        e = hidden[row, last]
        if normalize:
            e = torch.nn.functional.normalize(e, dim=-1)
        return e

    # ---- 文本 ----
    print("\n[3] 文本 embedding")
    t0 = time.time()
    te = process([{"text": "一个黑色的8号台球在绿色台球桌上滚动",
                   "instruction": "Represent the user's input."}])
    print(f"    文本 emb: {tuple(te.shape)} 耗时 {time.time()-t0:.1f}s")

    # ---- GIF 帧序列 ----
    print("\n[4] GIF 帧序列 embedding（复刻 load_gif_frames 铁律）")
    from PIL import Image, ImageSequence
    gif = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto\8-ball_1f3b1.gif"
    frames = []
    with Image.open(gif) as im:
        for fr in ImageSequence.Iterator(im):
            frames.append(fr.convert("RGB"))
    # 采样到 8 帧
    if len(frames) > 8:
        idx = [round(i * (len(frames) - 1) / 7) for i in range(8)]
        frames = [frames[i] for i in idx]
    print(f"    解出 {len(frames)} 帧")
    t0 = time.time()
    ve = process([{"video": frames, "instruction": "Represent the animated GIF content."}])
    print(f"    视频 emb: {tuple(ve.shape)} 耗时 {time.time()-t0:.1f}s")

    # ---- 区分度 ----
    print("\n[5] 与另一素材区分度")
    gif2 = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto\3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"
    frames2 = []
    with Image.open(gif2) as im:
        for fr in ImageSequence.Iterator(im):
            frames2.append(fr.convert("RGB"))
    if len(frames2) > 8:
        idx = [round(i * (len(frames2) - 1) / 7) for i in range(8)]
        frames2 = [frames2[i] for i in idx]
    ve2 = process([{"video": frames2, "instruction": "Represent the animated GIF content."}])
    cos = float(ve @ ve2.T)
    print(f"    两素材视频余弦 = {cos:.4f}")

    print(f"\n=== 峰值显存 {torch.cuda.max_memory_reserved()/1024**2:.0f} MiB ===")
    print("PASS: int8 process 链路（文本 + GIF 帧序列）全部走通")


if __name__ == "__main__":
    main()