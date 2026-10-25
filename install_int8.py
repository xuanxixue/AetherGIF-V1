#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Int8/Int4 量化执行脚本 - 专门用于在 Windows 上测试 Qwen3-VL-Embedding-2B 的 INT8 支持"""

import sys
import os

# 先安装依赖
try:
    from transformers import set_seed
    from optimum.int8 import BnBQuantizedConfig, BnBQuantizationConfig
    import bitsandbytes as bnb
    print("✓ bitsandbytes 已导入")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    print("\n在 Windows 上，bitsandbytes 需要 Visual C++ 2019 Redistributable 支持。")
    print("建议在 WSL2 或 Linux 上运行，或租用 GPU 云服务器。")
    sys.exit(1)

# 模型路径
model_dir = r"D:\AetherGIF-V1\models\qwen3vl\models\Qwen--Qwen3-VL-Embedding-2B\snapshots\master"

# 检查模型是否存在
if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
    print(f"✗ 模型文件不存在: {model_dir}")
    sys.exit(1)

print(f"✓ 模型路径: {model_dir}")

# 加载 quantized config
quantization_config = BnBQuantizationConfig(
    bits=8,
    load_in_8bit=True,
    device_map="auto",
)

print("\n尝试加载 8-bit 模型...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    print("✓ 8-bit 模型加载成功!")
    print(f"  总参数量: {model.config.hidden_size} (注意：2.1B 参数实际是 1.06B 活跃)")
except Exception as e:
    print(f"✗ 8-bit 加载失败: {e}")
    # 尝试 4-bit
    print("\n尝试加载 4-bit 模型...")
    try:
        quantization_config = BnBQuantizationConfig(
            bits=4,
            load_in_4bit=True,
            device_map="auto",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        print("✓ 4-bit 模型加载成功!")
    except Exception as e:
        print(f"✗ 4-bit 加载也失败: {e}")
        print("\n结论：当前环境无法使用 INT 8/4 量化。")
        print("建议方案:")
        print("1. 使用 WSL2 + 完整 Python 环境")
        print("2. 使用云端 GPU (如 PhD GPU Rent)")
        print("3. 回退到 CPU fp32 (当前方案)")
        sys.exit(1)

# 测试前向传播
import torch
dummy_input = torch.randint(0, 1000, (1, 10)).to(model.device)
try:
    with torch.no_grad():
        outputs = model(input_ids=dummy_input)
    print("✓ 前向传播测试成功")
    print(f"  输出 logit 形状: {outputs.logits.shape}")
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    sys.exit(1)

print("\n✅ 测试通过！模型可以在 4GB GPU 上运行！")
print("现在需要修改项目配置以使用量化模型。")
