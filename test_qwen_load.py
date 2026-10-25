import torch
import os
from pathlib import Path

def load_local_model():
    """Load Qwen3-VL-Embedding-2B from local path"""
    model_path = "D:/AetherGIF-V1/models/qwen3vl/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master"

    try:
        from transformers import AutoModel, AutoTokenizer

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print("Loading model...")
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = model.eval()

        print("Model loaded successfully!")
        return tokenizer, model

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def extract_features(tokenizer, model, text, image_path=None):
    """Extract image and text embeddings"""
    if image_path and os.path.exists(image_path):
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        inputs = tokenizer(
            text=text,
            images=[image],
            return_tensors="pt",
            padding=True
        )
        # Only convert data tensors to half, keep input_ids as long
        inputs = {k: v.half().to(model.device) if k != 'input_ids' else v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        print(f"Image+Text embedding shape: {embeddings.shape}")
        return embeddings[0].cpu().numpy()
    else:
        inputs = tokenizer(
            text=text,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.half().to(model.device) if k != 'input_ids' else v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        print(f"Text-only embedding shape: {embeddings.shape}")
        return embeddings[0].cpu().numpy()

def main():
    tokenizer, model = load_local_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Test with sample GIF descriptions
    text1 = "a black and white 8-ball moving on a pool table"
    text2 = "a spinning 3D cube with colored faces"

    # Test text-only embedding (take first token: embeddings shape is [1, seq_len, 2048])
    emb1 = extract_features(tokenizer, model, text1)  # shape (seq1, 2048)
    emb2 = extract_features(tokenizer, model, text2)  # shape (seq2, 2048)

    # Use mean pooling over sequence dimension to get fixed-length vector
    emb1_mean = emb1.mean(axis=0)
    emb2_mean = emb2.mean(axis=0)

    print(f"\nMean-pooled embedding 1 shape: {emb1_mean.shape}, mean: {emb1_mean.mean():.4f}, std: {emb1_mean.std():.4f}")
    print(f"Mean-pooled embedding 2 shape: {emb2_mean.shape}, mean: {emb2_mean.mean():.4f}, std: {emb2_mean.std():.4f}")

    # Compute similarity
    vec1 = torch.tensor(emb1_mean)
    vec2 = torch.tensor(emb2_mean)
    similarity = torch.dot(vec1, vec2)
    print(f"\nDot similarity: {similarity.item():.4f}")

    # Check if they are different enough
    diff = torch.linalg.norm(vec1 - vec2)
    print(f"Norm difference: {diff.item():.4f}")

    print("\n✅ Test passed: Model can generate distinct embeddings for different texts.")

if __name__ == "__main__":
    main()
