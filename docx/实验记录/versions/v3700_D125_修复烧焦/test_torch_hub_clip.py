import torch
print("Testing torch.hub.from_pretrained...")
try:
    model = torch.hub.from_pretrained("openai/clip-vit-base-patch32")
    print(f"Model loaded: {type(model)}")
    print(f"Has encode_text: {hasattr(model, 'encode_text')}")
    print(f"Has encode_images: {hasattr(model, 'encode_images')}")

    # 测试编码
    dummy_text = ["test"]
    with torch.no_grad():
        text_feat = model.encode_text(dummy_text)
        print(f"Text feat shape: {text_feat.shape}")

    from PIL import Image
    dummy_img = Image.new("RGB", (224, 224))
    with torch.no_grad():
        img_feat = model.encode_images([dummy_img])
        print(f"Img feat shape: {img_feat.shape}")

    print("\n[test_torch_hub_clip.py] SUCCESS!")
except Exception as e:
    print(f"\n[test_torch_hub_clip.py] FAILED: {e}")
    import traceback
    traceback.print_exc()
