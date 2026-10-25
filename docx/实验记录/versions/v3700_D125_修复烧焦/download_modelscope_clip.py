# -*- coding: utf-8 -*-
"""
从魔塔社区 (ModelScope) 下载 CLIP 模型
魔塔社区是中国版 HuggingFace，下载速度更快
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("魔塔社区模型下载 - CLIP ViT-B/16")
    print("=" * 60)

    target_dir = Path("D:/AetherGIF-V1/models/clip")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {target_dir}")

    config_exists = any(target_dir.glob("config.json"))
    if config_exists:
        print("✓ 本地模型已存在，跳过下载")
        return 0

    print("\n[1/2] 从魔塔社区下载...")
    try:
        from modelscope import snapshot_download
        model_id = "ModelScope/clip-vit-base-patch32"
        print(f"从魔塔社区获取: {model_id}")
        result = snapshot_download(
            model_id,
            local_dir=str(target_dir),
            allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "*.bin"],
            cache_dir=str(target_dir),
        )
        print(f"✓ 下载完成: {result}")
        return 0
    except Exception as e:
        print(f"✗ 魔塔社区下载失败: {e}")

    print("\n[2/2] 尝试 ModelScope API 直接下载...")
    try:
        from modelscope import snapshot_download
        result = snapshot_download(
            "modelscope/clip-vit-base-patch32",
            revision="main",
            local_dir=str(target_dir),
        )
        print(f"✓ 下载完成: {result}")
        return 0
    except Exception as e2:
        print(f"✗ ModelScope API 下载失败: {e2}")

    print(f"\n[建议] 下载失败，请尝试:")
    print("1. 确认已安装 modelscope: pip install modelscope")
    print("2. 检查网络连接或 HTTP/HTTPS 代理")
    print(f"3. 或手动从魔塔社区下载后复制到 {target_dir}")
    return 1

if __name__ == "__main__":
    exit(main())
