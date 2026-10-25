# -*- coding: utf-8 -*-
"""
使用官方 HuggingFace 模型路径加载 jina-clip-v2
"""

from huggingface_hub import snapshot_download
import shutil
from pathlib import Path

def main():
    # 从 HuggingFace 下载官方模型
    model_name = "jinaai/jina-clip-v2"
    local_dir = Path("D:/AetherGIF-V1/models/jina-clip-v2")
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"从 HuggingFace Hub 下载 {model_name}...")
    try:
        result = snapshot_download(
            model_name,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.py"],
            ignore_patterns=["onnx/*", "*openvino*", "*.msgpack", "*.h5"],
        )
        print(f"下载完成！路径: {result}")
        
        # 将文件复制到新位置
        src = Path(result)
        for item in src.iterdir():
            dest = local_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest)
                print(f"  ✓ {item.name}")
            elif item.is_dir():
                print(f"  ⚠ 跳过目录: {item.name}")
        
        # 检查关键文件
        key_files = ["config.json", "modeling_clip.py", "configuration_clip.py", "tokenizer.json"]
        print("\n[检查关键文件]")
        for fname in key_files:
            if (local_dir / fname).exists():
                print(f"  ✓ {fname}")
            else:
                print(f"  ✗ {fname} 缺失!")
        
        # 搜索 rot 文件
        rotary_files = list(local_dir.rglob("*rotary*"))
        if rotary_files:
            print(f"\n[旋转嵌入相关] 找到 {len(rotary_files)} 项:")
            for f in rotary_files[:10]:
                print(f"  ✓ {f.relative_to(local_dir)}")
        else:
            print("\n[旋转嵌入相关] 未发现 rotary.py 或相关模块")
        
        return 0
    except Exception as e:
        print(f"错误: {e}", flush=True)
        return 1

if __name__ == "__main__":
    exit(main())
