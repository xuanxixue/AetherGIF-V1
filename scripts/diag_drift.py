# -*- coding: utf-8 -*-
"""漂移场通路检验：零训练单次前向 + 单样本过拟合，看相位对齐能否建立。"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import torch
from pathlib import Path
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
device = "cuda"
gp = sorted(Path("data/gif_raw/noto").glob("*.gif"))[0]
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re**2 + im**2 + 1e-12)
th_tar = torch.atan2(im, re)

def probe(model, tag):
    torch.manual_seed(0)
    with torch.no_grad():
        a, th, _ = model(texts=["测"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
    pe = (torch.remainder(th - th_tar + math.pi, 2*math.pi) - math.pi)
    cos = torch.cos(pe).mean().item()
    corr_a = torch.corrcoef(torch.stack([a.flatten(), a_tar.flatten()]))[0,1].item()
    print(f"[{tag}] cos(θ-θtar)={cos:+.4f}  |err|中位={pe.abs().median().item():.3f}rad  "
          f"corr(a,atar)={corr_a:+.4f}  loss_amp={(a-a_tar).pow(2).mean().item():.4f}")

model = AetherGIFModel(device=device).to(device)
probe(model, "零训练 单次前向")

# 单样本过拟合 20 步：通路通的化相位应对齐到 target
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
for step in range(1, 21):
    torch.manual_seed(step)
    opt.zero_grad()
    a, th, _ = model(texts=["测"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
    pe = torch.remainder(th - th_tar + math.pi, 2*math.pi) - math.pi
    loss = (a - a_tar).pow(2).mean() + pe.pow(2).mean()
    loss.backward()
    opt.step()
    if step % 5 == 0:
        with torch.no_grad():
            cos = torch.cos(pe).mean().item()
        print(f"  step {step:2d}: loss={loss.item():.4f}  cos={cos:+.4f}")
probe(model, "过拟合 20 步后")
