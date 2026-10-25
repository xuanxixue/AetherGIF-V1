# -*- coding: utf-8 -*-
"""分解 loss 各项 + 测模型输出与 target 的对齐度，定位 loss 不动的原因。"""
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

# 取一个真实样本
gp = sorted(Path("data/gif_raw/noto").glob("*.gif"))[0]
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)  # [1,N,2D]

re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re**2 + im**2 + 1e-12)
th_tar = torch.atan2(im, re)

def breakdown(model, tag):
    torch.manual_seed(0)
    with torch.no_grad():
        a, th, diag = model(texts=["测试"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
    amp = (a - a_tar).pow(2).mean().item()
    pe = (torch.remainder(th - th_tar + math.pi, 2*math.pi) - math.pi).pow(2).mean().item()
    # 相位对齐
    cos_align = torch.cos(th - th_tar).mean().item()
    # 相位误差绝对值中位数
    pe_abs = (torch.remainder(th - th_tar + math.pi, 2*math.pi) - math.pi).abs()
    print(f"\n[{tag}]")
    print(f"  loss_amp   = {amp:.4f}")
    print(f"  loss_phase = {pe:.4f}   <- 地板(随机) = {math.pi**2/3:.4f}")
    print(f"  cos(th-th_tar) = {cos_align:+.4f}   <- 0=随机, 1=完全对齐")
    print(f"  |phase_err| 中位数 = {pe_abs.median().item():.3f} rad (π={math.pi:.3f})")
    print(f"  a 输出量级 mean={a.mean().item():.4f}  target a 量级 mean={a_tar.mean().item():.4f}")
    # 只看振幅对齐：模型 a 与 target a 的相关系数
    av, tb = a.flatten(), a_tar.flatten()
    corr = ((av-av.mean())*(tb-tb.mean())).mean() / (av.std()*tb.std() + 1e-12)
    print(f"  corr(a, a_tar) = {corr.item():+.4f}   <- 振幅结构是否学到")

m0 = AetherGIFModel(device=device).to(device)
breakdown(m0, "随机初始化")
del m0; torch.cuda.empty_cache()

m1 = AetherGIFModel(device=device).to(device)
m1.load_checkpoint("outputs_real/checkpoint_epoch6.pt", map_location="cuda")
breakdown(m1, "训练 6 轮后")
