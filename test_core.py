import torch
from pathlib import Path
import os
import sys

# 添加项目目录
sys.path.insert(0, 'QwenSrc')

# 检查模型文件
model_path = "/tmp/modelscope_cache/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master"
print(f"Checking model path: {model_path}")

# 检查必要文件
required_files = ['model.safetensors', 'config.json']
for f in required_files:
    exists = (Path(model_path) / f).exists()
    print(f"  {f}: {'✅' if exists else '❌'}")

if not (Path(model_path) / 'model.safetensors').exists():
    print("ERROR: Model weights not found")
    sys.exit(1)

print("\nAttempting to load using torch.load...")

# 直接加载 safetensors 权重
try:
    # 这是一个简单测试，之后我们再优化
    from safetensors import safe_open
    with safe_open(f"{model_path}/model.safetensors", framework="pt", device="cpu") as f:
        # 获取部分权重用于测试
        keys = f.keys()
        sample_key = list(keys)[0]
        tensor = f.get_tensor(sample_key)
        print(f"✅ Loaded sample tensor: {sample_key} shape={tensor.shape}")
        print("Model loading basic test passed!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nQuick encoding test with dummy inputs...")
# 简单测试模型结构
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path)
    print(f"✅ Config loaded: {config.model_type}, hidden size={config.text_config.hidden_size}")
except Exception as e:
    print(f"❌ Config load failed: {e}")
