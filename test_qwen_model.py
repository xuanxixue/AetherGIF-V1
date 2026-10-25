import torch
from modelscope import AutoModelForImageTextToText, AutoProcessor
import numpy as np
from PIL import Image

# 使用本地缓存的模型
model_id = "/tmp/modelscope_cache/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master"

# 加载模型和处理器
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")

print("Model loaded successfully!")

# 测试编码能力
def test_encoding():
    # 1. 测试图片编码
    img_path = "docx/实验记录/versions/v4200_多模态CLIP产品版/data/gif_raw/noto/8-ball_1f3b1.gif"
    img = Image.open(img_path).convert("RGB")
    
    # 图片编码（单模态）
    inputs_image = {"image": img}
    with torch.no_grad():
        embedding_image = model.get_feature_extractor()(inputs_image)
    print(f"Image embedding shape: {embedding_image.shape}")
    print(f"Image embedding dtype: {embedding_image.dtype}")
    print(f"Image embedding norm: {torch.norm(embedding_image).item()}")
    
    # 2. 测试文本编码（英文）
    text_en = "a black and white billiard ball with a single stripe"
    inputs_text_en = {"text": text_en}
    with torch.no_grad():
        embedding_text_en = model.get_feature_extractor()(inputs_text_en)
    print(f"Text (EN) embedding shape: {embedding_text_en.shape}")
    
    # 3. 测试文本编码（中文）
    text_zh = "一个黑白相间的台球"
    inputs_text_zh = {"text": text_zh}
    with torch.no_grad():
        embedding_text_zh = model.get_feature_extractor()(inputs_text_zh)
    print(f"Text (ZH) embedding shape: {embedding_text_zh.shape}")
    
    # 4. 测试图文联合编码
    inputs_mixed = {"text": text_en, "image": img}
    with torch.no_grad():
        embedding_mixed = model.get_feature_extractor()(inputs_mixed)
    print(f"Multimodal embedding shape: {embedding_mixed.shape}")
    
    # 保存示例
    np.save("/tmp/test_img_emb.npy", embedding_image.cpu().numpy())
    np.save("/tmp/test_text_en_emb.npy", embedding_text_en.cpu().numpy())
    print("Embeddings saved to /tmp/")

test_encoding()
print("\nAll tests passed!")
