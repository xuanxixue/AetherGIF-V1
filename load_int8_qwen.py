#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""加载魔塔社区预量化 Qwen3-VL-Embedding-2B (INT8)"""

import os
import sys
from pathlib import Path

# 设置线程数以防段错误
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

sys.path.insert(0, str(Path(__file__).parent / "deps312"))

# 补丁 torch.accelerator
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass

import torch
from data.clip_condition import _load_official_impl, default_model_path

QUANTIZED_PATH = r"D:\AetherGIF-V1\models\qwen3vl\models\Qwen--Qwen3-VL-Embedding-2B\snapshots\master_int8"

def check_quantized():
    """检查是否有预量化模型"""
    if Path(QUANTIZED_PATH).exists():
        print("✓ 找到预量化模型目录:", QUANTIZED_PATH)
        return True
    else:
        print("✗ 未找到预量化模型")
        print(f"\n预期位置: {QUANTIZED_PATH}")
        print("\n建议:")
        print("1. 从魔塔社区下载 'Qwen3-VL-Embedding-2B-INT8' 版本")
        print("2. 或使用 BitsAndBytes 官方量化加载方式")
        return False

if __name__ == "__main__":
    print("检查魔塔社区预量化 Qwen3-VL-Embedding-2B")
    print("=" * 70)
    print(f"模型路径: {QUANTIZED_PATH}")
    print()

    if not check_quantized():
        print("\n需要手动下载量化版本到上述目录")
        print("\n从魔塔社区下载命令:")
        print("  modelscope download Qwen/Qwen3-VL-Embedding-2B-INT8 --local-dir <your_dir>")
        print("\n或使用 HuggingFace:")
        print("  huggingface-cli download Qwen/Qwen3-VL-Embedding-2B --local-dir <your_dir> --revision v1.0")
        sys.exit(1)

    # 加载逻辑（稍后实现）
    print("\n✅ 环境配置完成，可以运行后续脚本")
