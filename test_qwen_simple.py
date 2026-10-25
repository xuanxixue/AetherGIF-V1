import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

model_path = "/tmp/modelscope_cache/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master"

print("Loading model and processor...")
try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ Model loaded! Device: {next(model.parameters()).device}")
    print(f"Model type: {type(model)}")
    print(f"Has get_image_features: {hasattr(model, 'get_image_features')}")
    print(f"Has get_text_features: {hasattr(model, 'get_text_features')}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nLet's check model config...")
    import json
    config = json.load(open(f"{model_path}/config.json", encoding='utf-8'))
    print(json.dumps(config, indent=2)[:2000])
    raise

# Now let's see if model has a specific forward method for embedding
print("\nTesting feature extraction...")
from PIL import Image

# 准备两张不同的图片
img1 = Image.open("docx/实验记录/versions/v4200_多模态CLIP产品版/data/gif_raw/noto/8-ball_1f3b1.gif").convert("RGB")
img2 = Image.open("docx/实验记录/versions/v4200_多模态CLIP产品版/data/gif_raw/noto/3D87B14FBB731ED3A948AC6DD9EA6EBC.gif").convert("RGB")

# 准备文本描述
text1 = "A black and white billiard ball with a single stripe"
text2 = "A football with a geometric pattern and bright colors"

inputs_list = []

# 1. 纯文本
inputs_list.append({"text": text1})
inputs_list.append({"text": text2})

# 2. 纯图片
inputs_list.append({"image": img1})
inputs_list.append({"image": img2})

# 3. 图文混合
inputs_list.append({"text": text1, "image": img1})
inputs_list.append({"text": text2, "image": img2})

print(f"Total inputs: {len(inputs_list)}")

# 现在尝试提取特征
with torch.no_grad():
    for i, inp in enumerate(inputs_list):
        try:
            # 先看看怎么处理
            if "image" in inp:
                # 需要理解 Qwen3 的处理格式
                # 查看 qwen3_vl 的 pipeline
                pass
            print(f"Testing input {i}...")
        except Exception as e:
            print(f"Error at input {i}: {e}")

# 先看看 model 是否有 `model_type` 相关的属性
print(f"\nModel class: {model.__class__.__name__}")
print(f"Architecture: {model.config.architectures}")
