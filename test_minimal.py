import torch
from pathlib import Path
import os
import sys

# 添加项目目录路径
sys.path.insert(0, 'QwenSrc')

# 验证模型文件存在
model_path = "/tmp/modelscope_cache/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master"
print(f"Checking model path: {model_path}")
print(f"Exists: {os.path.exists(model_path)}")
if not os.path.exists(model_path):
    print("ERROR: Model not found")
    sys.exit(1)

# 列出目录内容
for root, dirs, files in os.walk(model_path):
    print(f"{root}:")
    for f in files[:5]:
        print(f"  {f}")
    if len(files) > 5:
        print(f"  ... and {len(files)-5} more files")
    break

print("\nNow attempting to load...")

# 先尝试基础加载
try:
    from transformers import AutoModel
    import logging
    
    # 静默日志
    logging.basicConfig(level=logging.CRITICAL)
    
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    print(f"✅ Model loaded! Device: {next(model.parameters()).device}")
    print(f"Model type: {type(model)}")
    print(f"Model architecture: {model.__class__.__name__}")
    
    # 查看是否有 encode 方法
    if hasattr(model, 'encode'):
        print("Model has encode() method")
    elif hasattr(model, 'get_image_features'):
        print("Model has get_image_features() method")
    else:
        print("Model does not have typical encoding methods")
    
    print("Test complete!")
    
except Exception as e:
    print(f"❌ Load failed: {e}")
    import traceback
    traceback.print_exc()
