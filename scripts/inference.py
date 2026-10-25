# -*- coding: utf-8 -*-
"""
AetherGIF-V1 推理脚本
- 从任意文本条件生成 GIF（3–10 秒）
- 自适应时间步长 N
- 使用已训练好的模型权重
"""
import os
import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, 'src')
from model.aether_gif_model import AetherGIFModel
from physics.kuramoto_sync import build_spacetime_index


def main():
    parser = argparse.ArgumentParser(description="AetherGIF-V1 推理脚本")
    parser.add_argument(
        "text",
        type=str,
        help="文本条件（例如：'简约线条动画'）"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型检查点路径（默认自动找 outputs_real/ 下兼容当前架构的权重）"
    )
    parser.add_argument(
        "--duration",
        type=int,
        choices=[3, 4, 5, 6, 7, 8, 9, 10],
        default=5,
        help="生成 GIF 时长（秒）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/inferenced",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备"
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    # 注意：outputs_real/*.pt 是 use_qwen=False 旁路验证模式产物（text_param 替代 Qwen 编码），
    # 与当前代码唯一完全兼容。这里默认 use_qwen=False 与权重口径对齐；
    # 若未来训练出带 Qwen 编码器的新 ckpt，再改回 use_qwen=True。
    model = AetherGIFModel(
        d_model=64,
        ffn_hidden=256,
        n_blocks=12,
        lambda_t=0.5,
        lambda_s=3.0,
        temporal_window=3,
        device=args.device,
        use_qwen=False,
    )

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if ckpt_path is None or not ckpt_path.exists():
        # ★ 自动选择兼容当前架构（d_model=64）的权重。
        #   注意：outputs/checkpoint_epoch*.pt 与 outputs_real/checkpoint_epoch*.pt
        #   是旧架构（ffn.w2 形状 [64,256]，当前为 [128,256]），不可用，
        #   不要选它们，否则会 shape mismatch。
        _preferred = [
            Path("outputs_real/overfit_ckpt.pt"),
            Path("outputs_real/ckpt_schemeG_1000.pt"),
            Path("outputs_real/ckpt_s150_nocut.pt"),
        ]
        for _p in _preferred:
            if _p.exists():
                ckpt_path = _p
                print(f"[AUTO] 自动选择兼容权重: {ckpt_path}")
                break
        if ckpt_path is None:
            ckpt_path = Path(args.checkpoint) if args.checkpoint else Path("outputs_real/overfit_ckpt.pt")
    if ckpt_path.exists():
        print(f"[LOAD] 从 {ckpt_path} 加载权重")
        model.load_checkpoint(str(ckpt_path))
    else:
        print(f"[ERROR] 权重文件不存在: {ckpt_path}")
        print("请先运行训练脚本。")
        return

    model.eval()

    # 推理
    print(f"[GEN] 生成文本: '{args.text}', 时长: {args.duration}s")
    output_dir = Path(args.output_dir) / f"gen_{args.text.replace(' ', '_')}_{args.duration}s.gif"
    with torch.no_grad():
        model.generate(
            text=args.text,
            duration=args.duration,
            save_path=str(output_dir),
        )
    print(f"[DONE] 生成到: {output_dir}")


if __name__ == "__main__":
    main()
