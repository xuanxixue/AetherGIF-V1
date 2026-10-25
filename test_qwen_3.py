import torch
from transformers import AutoModel 
from PIL import Image

model_path = '/tmp/modelscope_cache/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master'

print('Loading model...')
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

print(f"✅ Model loaded! Device: {next(model.parameters()).device}")
print(f"Hidden size: {model.config.text_config.hidden_size}")

# Test with sample image
img = Image.open('docx/实验记录/versions/v4200_多模态CLIP产品版/data/gif_raw/noto/8-ball_1f3b1.gif').convert('RGB')

# Need to use image_processor if available
print("\nTesting embeddings...")
try:
    with torch.no_grad():
        # Try standard inference
        outputs = model.generate(
            inputs=[{"image": img, "text": "test"}],
            max_new_tokens=1
        )
except Exception as e:
    print(f"Standard inference error: {e}")

print("\nTesting feature extraction...")
try:
    # Model might have get_feature_extractor or encode methods
    # Try common patterns
    if hasattr(model, 'encode'):
        result = model.encode([{"image": img}])
    elif hasattr(model, 'encode_image'):
        result = model.encode_image(img)
    elif hasattr(model, 'encode_text'):
        result = model.encode_text("test")
    else:
        # Direct forward pass
        inputs = {"images": [img], "text": ["test"]}
        outputs = model(**inputs)
        if hasattr(outputs, 'last_hidden_state'):
            result = outputs.last_hidden_state
        else:
            print(outputs.keys())
            raise ValueError("No useful output")
    
    print(f"✅ Result shape: {result.shape}")
    print(f"✅ Result dtype: {result.dtype}")
    
    # Save sample
    if torch.is_tensor(result):
        torch.save(result, '/tmp/test_emb.pt')
        print("✅ Saved test embedding to /tmp/test_emb.pt")
    
except Exception as e:
    print(f"❌ Feature extraction error: {e}")
    import traceback
    traceback.print_exc()
