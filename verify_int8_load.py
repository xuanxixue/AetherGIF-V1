# -*- coding: utf-8 -*-
"""int8 加载核心验证 —— 能否把 Qwen3-VL-Embedding-2B 以 int8 塞进 4GB 卡。

验证三件事：
1. transformers 5.19 + bitsandbytes 0.50.2 能否端到端加载 Qwen3VL int8
2. 加载后显存占用是否 < 4GB（目标 ~2GB）
3. 前向传播能否产出 2048 维 embedding
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

sys.path.insert(0, r"D:\AetherGIF-V1\deps312")
sys.path.insert(0, r"D:\AetherGIF-V1\int8_deps")

import torch

# ★ torch<2.6 没有 torch.accelerator，transformers 5.x 硬依赖，必须在 import transformers 前补
if not hasattr(torch, "accelerator"):
    class _AcceleratorShim:
        @staticmethod
        def is_available():
            return torch.cuda.is_available()
        @staticmethod
        def device_count():
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        @staticmethod
        def current_accelerator():
            return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        @staticmethod
        def current_device_index():
            return torch.cuda.current_device() if torch.cuda.is_available() else -1
        @staticmethod
        def is_bf16_supported(device=None):
            if not torch.cuda.is_available():
                return False
            major, _ = torch.cuda.get_device_capability(device)
            return major >= 8
    torch.accelerator = _AcceleratorShim
    print("[shim] 已内联 patch torch.accelerator")

MODEL_DIR = r"D:\AetherGIF-V1\models\qwen3vl\models\Qwen--Qwen3-VL-Embedding-2B\snapshots\master"


def gpu_mem():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserv = torch.cuda.memory_reserved() / 1024**2
        return f"alloc={alloc:.0f}MiB reserv={reserv:.0f}MiB"
    return "no-cuda"


def main():
    print("=" * 70)
    print("[0] 环境")
    print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()} {torch.version.cuda}")
    import transformers, bitsandbytes
    print(f"    transformers {transformers.__version__}  bitsandbytes {bitsandbytes.__version__}")
    print(f"    GPU: {gpu_mem()}")

    # 1. 用 BitsAndBytesConfig 加载 int8
    print("\n[1] 加载 int8 量化模型（直接从原始 bf16 权重量化）")
    t0 = time.time()
    from transformers import BitsAndBytesConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLConfig,
    )

    # 官方 Qwen3VLForEmbedding 在 qwen3_vl_embedding.py 里，按文件加载
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_qwen3vl_emb",
        r"D:/AetherGIF-V1/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_qwen3vl_emb"] = mod
    spec.loader.exec_module(mod)
    Qwen3VLForEmbedding = mod.Qwen3VLForEmbedding

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    model = Qwen3VLForEmbedding.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    print(f"    加载完成，耗时 {time.time()-t0:.1f}s")
    print(f"    GPU: {gpu_mem()}")
    print(f"    模型 dtype 样例:", next(model.parameters()).dtype)

    # 2. 前向测试（文本 embedding）
    print("\n[2] 文本 embedding 前向")
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    processor = Qwen3VLProcessor.from_pretrained(MODEL_DIR, padding_side='right')

    model.eval()
    msg = [{"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
           {"role": "user", "content": [{"type": "text", "text": "一个黑色的8号台球在绿色台球桌上滚动"}]}]
    text = processor.apply_chat_template([msg], add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors='pt', padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
    hidden = out.last_hidden_state
    print(f"    last_hidden_state: {tuple(hidden.shape)}  dtype={hidden.dtype}")
    print(f"    GPU: {gpu_mem()}")

    # last token pooling
    am = inputs["attention_mask"]
    last = am.shape[1] - am.flip(dims=[1]).argmax(dim=1) - 1
    emb = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    emb = torch.nn.functional.normalize(emb, dim=-1)
    print(f"    embedding: {tuple(emb.shape)}  范数={emb.norm().item():.4f}")

    print("\n[3] 显存峰值评估")
    print(f"    峰值 reserved: {torch.cuda.max_memory_reserved()/1024**2:.0f} MiB")
    print("\n=== 结论 ===")
    peak = torch.cuda.max_memory_reserved() / 1024**2
    if peak < 3800:
        print(f"    PASS: 峰值 {peak:.0f}MiB < 4GB，int8 上 GPU 可行")
    else:
        print(f"    WARN: 峰值 {peak:.0f}MiB 仍超界，需 int4 或 offload")


if __name__ == "__main__":
    main()