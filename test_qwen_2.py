import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image

# 模型路径 - 使用本地缓存
model_path = "/tmp/modelscope_cache/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master"

print(f"Loading model from {model_path}...")

try:
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ Model loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    
    # 检查模型配置
    config = processor.config
    print(f"\nModel Config:")
    print(f"- Embedding dim: {config.text_config.hidden_size}")
    print(f"- Layers: {config.text_config.num_hidden_layers}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    raise
