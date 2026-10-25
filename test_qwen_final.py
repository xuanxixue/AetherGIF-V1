import torch
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

# 加载模型
print("Loading Qwen3-VL-Embedding-2B...")
model = Qwen3VLEmbedder(
    model_name_or_path="/tmp/modelscope_cache/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master",
    torch_dtype=torch.float16,
    device_map="auto"
)
print(f"✅ Model loaded! -> {next(model.parameters()).device}")

# 准备数据
from PIL import Image
import numpy as np

img_path = "docx/实验记录/versions/v4200_多模态CLIP产品版/data/gif_raw/noto/8-ball_1f3b1.gif"
img = Image.open(img_path).convert("RGB")

# 测试文本描述
text_en = "a black and white billiard ball with a single stripe"
text_zh = "一个黑白相间的台球"

# 输入格式
inputs = [
    {"text": text_en},
    {"image": img},
    {"text": text_zh, "image": img}
]

print("\nGenerating embeddings...")
embeddings = model.process(inputs)
print(f"✅ Embeddings shape: {embeddings.shape}")

# 检查维度
print(f"First 10 values: {embeddings[0, :10]}")

# 保存测试
np.save("/tmp/test_qwen_embs.npy", embeddings)
print("✅ Saved to /tmp/test_qwen_embs.npy")
