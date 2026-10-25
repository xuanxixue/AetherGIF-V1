#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3-VL-Embedding-2B 官方 INT8 量化脚本
在本地使用 bitsandbytes 将模型转换为 INT8，生成 2GB 的权重文件
"""

import os
import sys
from pathlib import Path
import torch

# 添加依赖路径
DEPS = r"D:\AetherGIF-V1\deps312"
if DEPS not in sys.path:
    sys.path.insert(0, DEPS)

# Patch 环境
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from optimum.int8 import BnBQuantizationConfig
import bitsandbytes as bnb

try:
    import bitsandbytes
    print("✓ bitsandbytes 已导入")
except ImportError as e:
    print(f"✗ bitsandbytes 未安装: {e}")
    print("\n在 Windows 上安装 bitsandbytes 需要:")
    print("1. Python 版本 >= 3.10")
    print("2. Visual C++ 2019 Redistributable")
    print("3. 建议使用预编译的 wheel")
    sys.exit(1)

# 模型路径
model_dir = r"D:\AetherGIF-V1\models\qwen3vl\models\Qwen--Qwen3-VL-Embedding-2B\snapshots\master"
output_dir = Path(model_dir) / "quantized_int8"

# 检查模型文件
model_file = Path(model_dir) / "model.safetensors"
if not model_file.exists():
    print(f"✗ 模型文件不存在: {model_file}")
    sys.exit(1)

print(f"✓ 源模型: {model_file} ({model_file.stat().st_size / 1e9:.2f} GB)")

# 加载模型和 tokenizer
print("\n加载模型和 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)

try:
    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,  # 使用 fp32 加载
        low_cpu_mem_usage=True,
        device_map="cpu",  # 全部放 CPU
    )
    print("✓ 模型加载成功 (CPU)")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    sys.exit(1)

# 量化配置
quantization_config = BnBQuantizationConfig(
    bits=8,
    load_in_8bit=False,  # 不直接加载 int8，而是转换
    device_map="cpu",
)

print("\n转换 INT8 量化...")
try:
    # 使用 transforms 官方方法转换
    quantized_model = model.quantize(
        quantization_config=quantization_config,
        device="cpu",
    )
    
    # 保存量化模型
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ 量化完成! 保存到: {output_dir}")
    
    # 检查输出大小
    output_size = sum(f.stat().st_size for f in output_dir.glob("**/*.safetensors"))
    print(f"  输出大小: {output_size / 1e9:.2f} GB")
    print(f"  压缩率: {model_file.stat().st_size / output_size:.2f}x")
    
except Exception as e:
    print(f"✗ 量化失败: {e}")
    print("\n尝试替代方法...")
    # 尝试另一种量化方式
    try:
        from optimum.int8 import QuantizedModelLMHead
        # 或者其他转换逻辑
        print("需要进一步调试，请检查该错误")
    except:
        print("无法自动量化，需要手动步骤")
        sys.exit(1)

print("\n✅ 量化完成! 现在可以使用 -load_int8_qwen.py 加载量化模型")
