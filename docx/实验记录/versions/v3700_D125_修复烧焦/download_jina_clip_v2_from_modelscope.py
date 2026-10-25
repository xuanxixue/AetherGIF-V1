# -*- coding: utf-8 -*-
"""
从魔搭社区下载 jina-clip-v2 完整模型，包括 missing rotary 依赖。
"""

import os
import sys
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

def main():
    local_dir = Path("D:/AetherGIF-V1/models/jina-clip-v2")
    print(f"清理现有模型目录: {local_dir}")
    if local_dir.exists():
        shutil.rmtree(local_dir)
        print("已删除旧目录")
    
    print(f"从 HuggingFace Hub 重新下载完整模型:")
    try:
        # 从 HuggingFace Hub 下载完整模型
        result = snapshot_download(
            "jinaai/jina-clip-v2",
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.py"],
            ignore_patterns=["onnx/*", "*openvino*", "*.msgpack", "*.h5"],
        )
        print(f"下载完成！模型路径: {result}")
        
        # 验证 rotation 文件是否存在
        rotary_files = list(Path(result).glob("**/rotary.py"))
        if rotary_files:
            print(f"✓ 找到 rotation 文件: {rotary_files[0]}")
            return 0
        else:
            print("✗ 仍然未找到 rotation.py 文件")
            print("将尝试部署到本地目录...")
            
            # 将模型文件复制到指定本地目录
            source_dir = Path(result)
            for file in source_dir.iterdir():
                dest = local_dir / file.name
                if file.is_file():
                    shutil.copy2(file, dest)
                    print(f"已复制: {file.name}")
            
            print(f"目标目录: {local_dir}")
            print("执行完毕，请重新运行训练脚本。")
            return 0
    except Exception as e:
        print(f"下载失败: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
