# -*- coding: utf-8 -*-
"""
从魔搭社区下载 jina-clip-v2 完整模型
适用于中国境内网络环境
"""

import os
import sys
import shutil
from pathlib import Path
from modelscope import snapshot_download

def main():
    source_repo = "damo/nlp_lr-clip_jina-clip-v2"  # 魔搭社区的 jina-clip-v2
    local_dir = Path("D:/AetherGIF-V1/models/jina-clip-v2")
    
    print(f"检查现有目录: {local_dir}")
    if local_dir.exists():
        print("移除旧模型...")
        shutil.rmtree(local_dir)
        print("✓ 已清理")
    
    print(f"\n从魔搭社区下载: {source_repo}")
    print("请稍候...")
    
    try:
        # 从魔搭社区下载
        result = snapshot_download(
            source_repo,
            revision="master",
            cache_dir=str(local_dir.parent),
        )
        
        print(f"\n下载完成! 源路径: {result}")
        
        # 移动文件到目标位置
        src = Path(result)
        destination = local_dir
        if src != destination:
            destination.mkdir(parents=True, exist_ok=True)
            for item in src.iterdir():
                if item.is_file():
                    dest_file = destination / item.name
                    print(f"  复制: {item.name}")
                    shutil.copy2(item, dest_file)
            # 检查关键文件
            essential_files = ["configuration_clip.py", "modeling_clip.py", "hf_model.py", "rope_embeddings.py", "rotary.py"]
            print("\n[关键文件检查]")
            for fname in essential_files:
                if (destination / fname).exists():
                    print(f"  ✓ {fname}")
                else:
                    print(f"  ✗ {fname} 缺失!")
            
            return 0
        else:
            return 0
    except Exception as e:
        print(f"\n[错误] 下载失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
