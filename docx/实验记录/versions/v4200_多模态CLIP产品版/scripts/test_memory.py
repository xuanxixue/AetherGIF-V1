#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""内存测试脚本 - 快速适配不同硬件配置"""

import os
import sys
from pathlib import Path
import torch

proj_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(proj_root))
sys.path.insert(0, str(proj_root / "src"))

from model.aether_gif_model import AetherGIFModel  # noqa: E402

# 限制线程
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

def test_memory():
    """测试不同参数下的内存占用"""

    print("===== 内存测试 =====")

    # 测试不同配置
    configs = [
        # (d_model, n_blocks, time_step, grid, patch, win, img)
        # 保持 grid * patch = img
        (32, 4, 3, 32, 4, 5, 128),  # N = 3072
        (32, 4, 2, 32, 4, 5, 128),  # N = 2048
        (32, 2, 2, 32, 4, 5, 128),  # N = 2048
        (16, 2, 2, 32, 4, 5, 128),  # N = 2048
    ]

    for d_model, n_blocks, time_step, grid, patch, win, img in configs:
        print(f"\n测试配置：d_model={d_model}, n_blocks={n_blocks}, time_step={time_step}, grid={grid}, img={img}")

        # 检查基本对齐
        if grid * patch != img:
            print(f"  Warning: grid*patch={grid*patch} != img={img}")
            continue

        # 计算参数量
        try:
            # from model.aether_gif_model import AetherGIFModel
            model = AetherGIFModel(
                d_model=d_model,
                n_blocks=n_blocks,
                time_step=time_step,
                grid=grid,
                patch=patch,
                win=win,
                img=img,
                device="cpu"
            )
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  参数量: {total_params:,}")
            print(f"  显存(理论最小): {total_params * 4 / 1024**2:.2f} MB")
            print(f"  N = {time_step * grid * grid}")

        except Exception as e:
            print(f"  错误: {e}")

        print()

if __name__ == "__main__":
    test_memory()
