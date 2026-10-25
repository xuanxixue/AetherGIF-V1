# -*- coding: utf-8 -*-
"""
Windows 兼容模型下载脚本
自动下载官方 CLIP 模型到指定目录
"""

import os
import sys
from pathlib import Path

def main():
    import subprocess

    print("=" * 60)
    print("模型下载脚本 - CLIP ViT-B/16")
    print("=" * 60)

    # 目标目录
    target_dir = Path("D:/AetherGIF-V1/models/clip")
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {target_dir}")

    # 检查是否已有模型文件
    config_exists = any(target_dir.glob("config.json"))
    if config_exists:
        print("✓ 本地模型已存在，跳过下载")
        return 0

    # 方法 1: 使用 pip 下载
    print("\n[1/3] 尝试使用 pip 下载...")
    try:
        # 使用 pip download 但不需要 --out 参数
        result = subprocess.run(
            f'{sys.executable} -m pip download openai/clip-vit-base-patch32 --no-deps --quiet',
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # 找到下载的包并解压
            downloaded = list(Path().glob("*clip*"))
            if downloaded and downloaded[0].is_file():
                dest = target_dir / downloaded[0].name
                dest.write_path(downloaded[0])
                print(f"✓ 下载并保存到 {dest}")
                return 0
    except Exception as e:
        print(f"✗ pip 下载失败: {e}")

    # 方法 2: 使用 huggingface_hub
    print("\n[2/3] 尝试使用 huggingface_hub 下载...")
    try:
        from huggingface_hub import snapshot_download
        result = snapshot_download(
            "openai/clip-vit-base-patch32",
            local_dir=str(target_dir),
            allow_patterns=["*.json", "*.safetensors", "*.py"],
            max_workers=4,
            cache_dir=None,
        )
        print(f"✓ 下载完成: {result}")
        return 0
    except Exception as e:
        print(f"✗ huggingface_hub 下载失败: {e}")

    # 方法 3: 手动下载指南
    print("\n[3/3] 尝试本地手动下载:")
    print("""
需要手动下载方法:
1. 在有网络的另一台电脑上运行:
   pip install huggingface-hub

2. 使用以下命令下载模型到 U 盘/移动硬盘:
   python -c "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32', local_dir='temp_models')"

3. 将下载的外拷贝到 D:\\AetherGIF-V1\\models\\clip\\

或者:
4. 从 https://huggingface.co/openai/clip-vit-base-patch32 直接下载
   需要下载以下文件到 D:\\AetherGIF-V1\\models\\clip\\
   - config.json
   - model.safetensors (或 pytorch_model.bin)
   - preprocessor_config.json
   - tokenizer.json
   - tokenizer_config.json
   - vocab.json (如果有)
   - merges.txt (如果有)
""")
    return 1

if __name__ == "__main__":
    exit(main())
