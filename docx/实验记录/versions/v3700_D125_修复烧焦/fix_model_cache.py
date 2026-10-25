# -*- coding: utf-8 -*-
"""
修复 jina-clip-v2 模型缓存
清理损坏的 transforms_modules 目录并重新下载完整模型。
"""

import os
import sys
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

def main():
    model_name = "jinaai/jina-clip-v2"
    cache_dir = Path.home() / ".cache/huggingface/modules/transformers_modules/jina-embeddings-v3"

    print(f"检查缓存目录: {cache_dir}")
    if cache_dir.exists():
        print("检测 cache 存在问题，正在删除...")
        shutil.rmtree(cache_dir)
        print("缓存目录已删除。")
    else:
        print("缓存目录不存在，跳过删除。")

    print(f"从 HuggingFace Hub 重新下载模型: {model_name}")
    try:
        result = snapshot_download(
            model_name,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"],
            ignore_patterns=["onnx/*", "*openvino*", "*.msgpack", "*.h5"],
        )
        print(f"下载完成! 模型路径: {result}")
        return 0
    except Exception as e:
        print(f"下载失败: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
