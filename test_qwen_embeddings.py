import torch
from modelscope import snapshot_download
import os

model_id = "Qwen/Qwen3-VL-Embedding-2B"

# 创建下载目标目录
download_dir = snapshot_download(model_id, cache_dir='/tmp/modelscope_cache')
print(f"Model downloaded to: {download_dir}")

# 检查模型文件
files = os.listdir(download_dir)
for f in ['config.json', 'pytorch_model.bin', 'model.safetensors']:
    exists = any(f in file for file in files)
    print(f"{f}: {'Found' if exists else 'Not found'}")

# 检查显存
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
