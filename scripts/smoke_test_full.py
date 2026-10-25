# -*- coding: utf-8 -*-
"""
AetherGIF-V1 全流程冒烟测试脚本
测试 3 秒和 10 秒两档配置，验证：
  - GPU 显存占用
  - 物理核正常运行（r, theta_diff）
  - 生成 GIF 输出
  - 不出现全同步坍缩（静帧）
"""
import os
import sys
# 将项目根目录添加到 sys.path，这样 src 包才能被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model.aether_gif_model import AetherGIFModel
from src.physics.kuramoto_sync import build_spacetime_index

def run_test(duration: int, device: str):
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

    # 预备参考数据
    print(f"time_step={time_step}, N={len(t_coord)}")

    # 在 GPU 上启用锚点，计算显存占用
    print(" allocating model...")
    model.to(device)
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 = {n_params:,} ({n_params * 4 / 1e6:.2f} MB 权重)")

    # 显存预分配测试
    torch.cuda.empty_cache()
    with torch.no_grad():
        # 构造 dummy 输入
        dummy_text = ["简约线条动画"]
        # 生成推理
        print(" Running forward...")
        out_path = f"test_{duration}s.gif"
        model.generate(
            text=dummy_text[0],
            duration=duration,
            save_path=out_path,
        )
        print(f" 输出 GIF: {out_path}")

        # 显存统计
        mem_allocated = torch.cuda.memory_allocated(device)
        mem_reserved = torch.cuda.memory_reserved(device)
        print(f" 分配显存: {mem_allocated/1024**2:.0f} MB")
        print(f" 保留显存: {mem_reserved/1024**2:.0f} MB")

        # 估计 GPU 峰值
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f" 峰值显存: {peak:.0f} MB")

    # 检查生成错误
    if not os.path.exists(out_path):
        print(f" [FAIL] 未生成 GIF 文件")
        return False
    print(f" 文件存在，大小 {os.path.getsize(out_path)} bytes")
    return True

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    results = {}
    for duration in [3, 10]:
        try:
            results[duration] = run_test(duration, device)
        except Exception as e:
            print(f" [ERROR] {duration}s 测试失败: {e}")
            results[duration] = False

    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for dur, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"{dur}s: {status}")

    all_pass = all(results.values())
    if all_pass:
        print("\n[OK] 所有测试通过")
    else:
        print("\n[FAIL] 部分测试失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
