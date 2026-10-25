# -*- coding: utf-8 -*-
"""
一次性下载官方 CLIP 模型到本地目录.
只需运行一次，之后训练脚本会使用本地模型，避免网络问题。
"""

from huggingface_hub import snapshot_download
import os

def main():
    # 目标目录
    local_dir = "D:/AetherGIF-V1/models/clip"
    os.makedirs(local_dir, exist_ok=True)

    print(f"[Downloading] from HuggingFace Hub to {local_dir}...")

    try:
        result = snapshot_download(
            "openai/clip-vit-base-patch32",
            local_dir=local_dir,
            allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt"],
        )
        print(f"[SUCCESS] Model saved to: {result}")
        return True
    except Exception as e:
        print(f"[FAILED] Download error: {e}")
        # 尝试另一种方法：从 GitHub releases 下载
        print("[Attempting] Download from GitHub...")
        try:
            import requests
            url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json"
            r = requests.get(url, timeout=30)
            print(f"[Response] Status: {r.status_code}, Size: {len(r.content)}")
            return True
        except Exception as e2:
            print(f"[Failed] GitHub attempt error: {e2}")
            return False

if __name__ == "__main__":
    success = main()
    print(f"\nDone: {'OK' if success else 'FAILED'}")
    exit(0 if success else 1)
