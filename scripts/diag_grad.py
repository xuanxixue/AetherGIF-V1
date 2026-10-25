# -*- coding: utf-8 -*-
"""相位梯度诊断：phase loss 对各参数组的梯度 norm。

铁律：调通路顺序前先算清梯度。wcos 训练 200 步振荡不提升，
需确认梯度是否到达 κ / drift / time_field，以及量级是否被 amp 项淹没。
"""
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

gp = next(Path("data/gif_raw/noto").glob("8-ball_1f3b1.gif"))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
th_tar = torch.atan2(im, re)

model = AetherGIFModel(device=device).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train()
torch.manual_seed(1234)

# ---- 组别 ----
GROUPS = {
    "qwen(frozen)": [p for n, p in model.named_parameters() if "qwen" in n],
    "omega_bias_head": [model.omega_bias_head.weight, model.omega_bias_head.bias],
    "drift_target_head": [model.drift_target_head.weight, model.drift_target_head.bias],
    "kappa_scale_head": [model.kappa_scale_head.weight, model.kappa_scale_head.bias],
    "T_scale_head": [model.T_scale_head.weight, model.T_scale_head.bias],
    "time_field": [p for n, p in model.named_parameters() if "time_field" in n],
    "drift_field": [p for n, p in model.named_parameters() if "blocks.0.drift" in n],
    "kuramoto": [p for n, p in model.named_parameters() if "blocks.0.kuramoto" in n],
    "ffn(block0)": [p for n, p in model.named_parameters() if "blocks.0.ffn" in n],
    "ffn(all)": [p for n, p in model.named_parameters() if ".ffn" in n],
}


def step():
    opt.zero_grad()
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
    pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
    loss_amp = (torch.log1p(a) - torch.log1p(a_tar)).pow(2).mean()
    w_ph = (a_tar ** 2).clamp(min=1e-8)
    loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
    loss = loss_amp + loss_ph
    loss.backward()
    with torch.no_grad():
        c = torch.cos(pe)
        wcos = (w_ph * c).sum().item() / w_ph.sum().item()
    return loss.item(), loss_amp.item(), loss_ph.item(), wcos


for trial in range(1, 6):
    loss, la, lp, wcos = step()
    print(f"\n=== step {trial}  loss={loss:.4f} amp={la:.4f} phase={lp:.4f} wcos={wcos:+.4f}")
    for name, params in GROUPS.items():
        gn = torch.nn.utils.clip_grad_norm_(params, 1e9)
        print(f"  {name:18s} grad_norm={gn.item():.6e}")
    opt.step()
