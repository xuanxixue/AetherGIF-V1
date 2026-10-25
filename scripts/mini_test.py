# -*- coding: utf-8 -*-
"""
迷你测试：只测试模型 forward，不处理 GIF 生成。
验证物理核是否正常工作，显存占用多少。
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.aether_gif_model import AetherGIFModel
from src.physics.kuramoto_sync import build_spacetime_index

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 测试 3 秒和 10 秒
    for duration in [3, 10]:
        print(f"\n{'='*60}")
        print(f"测试 {duration} 秒配置")
        print(f"{'='*60}")

        # 模型初始化
        model = AetherGIFModel(
            d_model=64,
            ffn_hidden=256,
            n_blocks=12,
            lambda_t=0.5,
            lambda_s=3.0,
            temporal_window=3,
            device=device,
        )

        # 计算时间步
        fps = 10
        time_step = duration * fps // 5
        H, W = 128, 128
        patch_h, patch_w = 4, 4
        n_h = H // patch_h
        n_w = W // patch_w
        t_coord, s_coord = build_spacetime_index(time_step, n_h, n_w, device)

        print(f"time_step={time_step}, N={len(t_coord)}")

        # 在 GPU 上启用锚点，计算显存占用
        print(" allocating model...")
        model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"参数量 = {n_params:,} ({n_params * 4 / 1e6:.2f} MB 权重)")

        torch.cuda.empty_cache()
        with torch.no_grad():
            print(" Running forward...")
            a, theta, diag = model(
                texts=["简约线条动画"],
                target_field=None,
                t_coord=t_coord,
                s_coord=s_coord,
                batch_size=1,
            )
            print(f" a shape = {a.shape}")
            print(f" theta shape = {theta.shape}")
            print(f" diag r = {diag.get('r', 'N/A')}")

            # 显存统计
            mem_allocated = torch.cuda.memory_allocated(device)
            mem_reserved = torch.cuda.memory_reserved(device)
            print(f" 分配显存: {mem_allocated/1024**2:.0f} MB")
            print(f" 保留显存: {mem_reserved/1024**2:.0f} MB")

            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f" 峰值显存: {peak:.0f} MB")

        print(f"测试通过")

if __name__ == "__main__":
    main()
