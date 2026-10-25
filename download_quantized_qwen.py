#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从魔塔社区下载预量化 Qwen3-VL-Embedding-2B 模型"""

import os
import sys
from pathlib import Path

try:
    from modelscope import snapshot_download
    print("✓ modelscope 已安装")
except ImportError:
    print("✗ modelscope 未安装")
    sys.exit(1)

model_id = "Qwen/Qwen3-VL-Embedding-2B"
output_dir = Path(r"D:\AetherGIF-V1\models\qwen3vl\models\Qwen--Qwen3-VL-Embedding-2B" / "snapshots" / "master_int8")

print(f"\n从魔塔社区下载模型: {model_id}")
print(f"保存到: {output_dir}")

try:
    # 下载模型到指定目录
    model_path = snapshot_download(
        model_id,
        revision='v1.0',
        cache_dir=output_dir.parent.parent
    )
    print(f"✓ 模型下载成功: {model_path}")
    
    # 检查是否有量化版本
    int8_path = output_dir.with_name(f"{output_dir.name}_int8")
    if int8_path.exists():
        print(f"✓ 找到量化版本目录: {int8_path}")
        # 改为使用量化版本
        model_path = str(int8_path)
    else:
        print("✗ 未找到 quantized 子目录")
        
except Exception as e:
    print(f"✗ 下载失败: {e}")
    sys.exit(1)

print("\n模型加载路径:")
print(f"  model_path = '{model_path}'")
