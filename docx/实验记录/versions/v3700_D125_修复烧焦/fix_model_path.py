# -*- coding: utf-8 -*-
"""
验证模型完整性
"""

from pathlib import Path
import shutil

def main():
    local_dir = Path("D:/AetherGIF-V1/models/jina-clip-v2")
    
    if local_dir.exists():
        print(f"检查本地模型目录: {local_dir}")
        files = list(local_dir.iterdir())
        print(f"文件列表 ({len(files)} 项):")
        for f in files:
            print(f"  - {f.name}")
        
        # 查找 rotary 相关文件
        rotary_files = [f for f in files if 'rotary' in f.name.lower()]
        print(f"\n旋转嵌入相关文件 ({len(rotary_files)} 项):")
        for f in rotary_files:
            print(f"  - {f.name}")
        
        # 关键文件检查
        essential = ["configuration_clip.py", "modeling_clip.py", "hf_model.py"]
        for name in essential:
            if any(f.name == name for f in files):
                print(f"✓ {name} 存在")
            else:
                print(f"✗ {name} 缺失!")
        
        # 检查是否需要重新下载
        if not rotary_files:
            print("\n检测到缺少旋转嵌入文件！")
            print("建议从魔搭镜像重新下载模型。")
            return False
        return True
    else:
        print(f"本地模型目录不存在: {local_dir}")
        return False

if __name__ == "__main__":
    result = main()
    exit(0 if result else 1)
