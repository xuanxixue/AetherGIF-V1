# -*- coding: utf-8 -*-
"""加载 checkpoint 重新生成 GIF（验证归一化解码）。"""
import os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import torch
from model.aether_gif_model import AetherGIFModel

ck = sys.argv[1]
out = sys.argv[2]
txt = sys.argv[3] if len(sys.argv) > 3 else "简约线条动画"

model = AetherGIFModel(device="cuda")
model.load_checkpoint(ck, map_location="cuda")
model.to("cuda").eval()
with torch.inference_mode():
    model.generate(txt, duration=3, save_path=out)
print("done:", out)
