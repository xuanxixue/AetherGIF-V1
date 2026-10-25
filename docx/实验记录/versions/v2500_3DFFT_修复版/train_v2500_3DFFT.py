# -*- coding: utf-8 -*-
"""
AetherGIF 训练脚本 (严格修复版 - 3D FFT + V=V⁺-λV⁻)
"""
import sys
import os

# 添加项目根目录到 sys.path
# 当前文件位于：.../v2500_3DFFT_修复版/train_v2500_3DFFT.py
# 项目根目录在：.../v2500_3DFFT_修复版/
# 所以我们需要添加当前目录，这样 src 才能被找到
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from pathlib import Path
from src.model.aether_gif_model import AetherGIFModel
from src.codec.gif_codec import encode_gif_to_field


def train():
    # 参数
    duration = 3  # seconds
    fps = 10
    time_step = duration * fps // 5  # =6
    H, W = 128, 128
    patch_size = (4, 4)
    time_step_win = 5
    d_model = 125
    ffn_hidden = 512
    n_blocks = 12

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 构建模型
    model = AetherGIFModel(
        d_model=d_model,
        ffn_hidden=ffn_hidden,
        n_blocks=n_blocks,
        device=device,
        use_qwen=False,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 数据
    gif_path = "data/gif_raw/noto/8-ball_1f3b1.gif"
    # 编码为复数场 [N, 125]
    target_field = encode_gif_to_field(gif_path, time_step=time_step, d_model=d_model).to(device)
    print(f"Target field shape: {target_field.shape}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 训练
    n_steps = 2500
    for step in range(n_steps):
        model.train()
        optimizer.zero_grad()

        # 前向：模型产生实值振幅和相位
        a_gen, theta_gen, _ = model(target_field, time_step=time_step, return_diag=False)

        # 目标振幅和相位
        # target_field 是复数 [N, 64]
        z_tar = target_field
        a_tar = torch.abs(z_tar)
        theta_tar = torch.angle(z_tar)

        # Loss (与 v2500 方案 D 一致)
        # 振幅 loss (log1p 域，等权) - 但在修复版中我们用 log 域
        la = torch.log(a_gen.clamp(min=1e-8))
        lt = torch.log(a_tar.clamp(min=1e-8))
        loss_amp = (la - lt).pow(2).mean()

        # 相位 loss (功率加权)
        # wrap to [-pi, pi]
        phase_diff = torch.abs(torch.remainder(theta_gen - theta_tar + np.pi, 2*np.pi) - np.pi)
        # 功率加权
        w_ph = (a_tar ** 2).clamp(min=1e-8)
        loss_phase = (w_ph * phase_diff.pow(2)).sum() / w_ph.sum()

        loss = loss_amp + loss_phase

        # 反向
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}, loss_amp={loss_amp.item():.4f}, loss_phase={loss_phase.item():.4f}, a_avg={a_gen.mean().item():.4f}")

        # 保存检查点
        if step % 500 == 0 and step > 0:
            ckpt_path = f"outputs_real/overfit_{step}pt.pt"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # 保存最终权重
    ckpt_path = "outputs_real/overfit_2500.pt"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': n_steps,
    }, ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")

    # 生成 GIF (简化：只检查输出)
    with torch.no_grad():
        a_gen, theta_gen, _ = model(target_field, time_step=time_step, return_diag=False)
        # 这里可以添加 decode 为 GIF 的代码

if __name__ == "__main__":
    train()
