# -*- coding: utf-8 -*-
"""
AetherGIF-V1 训练脚本
- 基于 PMNN 物理矩阵神经网络生成 GIF
- 支持 3–10 秒任意时长（自适应 N）
- 包含 AetherSentry 三条护栏 + 全同步坍缩防护
- 遵循物理公式，零简化
"""
import os
import sys
import argparse
import math
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index


def main():
    parser = argparse.ArgumentParser(description="AetherGIF-V1 训练脚本")
    parser.add_argument("--data_dir", type=str, default="data/gif_filtered", help="训练数据目录")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小")
    parser.add_argument("--duration", type=int, choices=[3, 4, 5, 6, 7, 8, 9, 10], default=5, help="目标 GIF 时长（秒）")
    parser.add_argument("--t_steps", type=int, default=None, help="时间步数（自动计算 if None）")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--n_blocks", type=int, default=12, help="物理块数量")
    parser.add_argument("--d_model", type=int, default=64, help="模型维度")
    parser.add_argument("--ffn_hidden", type=int, default=256, help="FFN 隐藏层维度")
    parser.add_argument("--lambda_t", type=float, default=0.5, help="时间衰减率")
    parser.add_argument("--lambda_s", type=float, default=3.0, help="空间衰减率")
    parser.add_argument("--temporal_window", type=int, default=3, help="时间局部窗")
    parser.add_argument("--temperature", type=float, default=1.0, help="退火温度系数")
    parser.add_argument("--steps_per_epoch", type=int, default=8, help="每轮步数（dummy 数据用）")
    parser.add_argument("--max_gifs", type=int, default=8, help="真实数据集最多编码多少个 GIF")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="检查点间隔")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 学习率 warmup
    lr = args.learning_rate

    # 实例化模型
    model = AetherGIFModel(
        d_model=args.d_model,
        ffn_hidden=args.ffn_hidden,
        n_blocks=args.n_blocks,
        lambda_t=args.lambda_t,
        lambda_s=args.lambda_s,
        temporal_window=args.temporal_window,
        device=device,
    )

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # 可选：scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optim, args.num_epochs, eta_min=1e-6)

    # 数据加载
    fps = 10
    time_step = int(args.t_steps) if args.t_steps else (args.duration * fps // 5)
    GRID, PATCH, WIN, IMG = 32, 4, 5, 128

    data_dir = Path(args.data_dir)
    use_real_data = False
    if data_dir.exists():
        gif_files = sorted(data_dir.glob("**/*.gif"))[: args.max_gifs]
        if gif_files:
            use_real_data = True
            print(f"[INFO] 找到 {len(gif_files)} 个 GIF，编码中（最多 {args.max_gifs}）...")

    class GifFieldDataset(torch.utils.data.Dataset):
        """真实 GIF：预编码成 [N, 2D] 振子场。"""

        def __init__(self, gif_paths):
            self.items = []
            for gp in gif_paths:
                try:
                    field = encode_gif_to_field(str(gp), time_step, GRID, PATCH, WIN, IMG, args.d_model)
                    self.items.append({"target": field.cpu(), "name": gp.stem})
                except Exception as e:
                    print(f"[WARN] 跳过 {gp.name}: {e}")
            self.t_coord, self.s_coord = build_spacetime_index(time_step, GRID, GRID, "cpu")
            print(f"[INFO] 成功编码 {len(self.items)}/{len(gif_paths)} 个 GIF")

        def __getitem__(self, idx):
            it = self.items[idx]
            return {"target": it["target"], "t_coord": self.t_coord,
                    "s_coord": self.s_coord, "name": it["name"]}

        def __len__(self):
            return len(self.items)

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_items):
            self.num_items = num_items
            self.t_coord, self.s_coord = build_spacetime_index(time_step, GRID, GRID, "cpu")

        def __getitem__(self, idx):
            N = len(self.t_coord)
            target = torch.randn(N, 2 * args.d_model)
            return {"target": target, "t_coord": self.t_coord,
                    "s_coord": self.s_coord, "name": "dummy_random"}

        def __len__(self):
            return self.num_items

    def gif_collate(batch):
        """坐标对所有样本相同；names 保持 list，绕开默认 collate 拆字符串。"""
        return {
            "target": torch.stack([b["target"] for b in batch]),
            "t_coord": batch[0]["t_coord"],
            "s_coord": batch[0]["s_coord"],
            "names": [b["name"] for b in batch],
        }

    if use_real_data:
        dataset = GifFieldDataset(gif_files)
        num_samples = max(len(dataset), 1)
    else:
        print(f"[WARN] 未找到可用 GIF（{args.data_dir}），退回随机噪声。")
        dataset = DummyDataset(args.batch_size * args.steps_per_epoch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=gif_collate)

    # 训练
    _losses = defaultdict(list)
    _diag = defaultdict(list)

    print(f"[START] 开始训练，device={device}, batch_size={args.batch_size}, epochs={args.num_epochs}")

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        steps = 0
        for batch in dataloader:

            # 坐标对所有样本相同，collate 只存一份 [N]；直接上 GPU
            t_coord = batch["t_coord"].to(device)
            s_coord = batch["s_coord"].to(device)
            target_field = batch["target"].to(device)
            # texts 不走 DataLoader collate，避免 str 被拆成单字符序列
            texts = ["简约线条动画"] * target_field.shape[0]

            # 优化器清零
            optimizer.zero_grad()

            # 前向：AetherGIFModel
            a, theta, diag = model(
                texts=texts,
                target_field=target_field,
                t_coord=t_coord,
                s_coord=s_coord,
                batch_size=args.batch_size,
            )

            # 损失：目标复值场 [B,N,2D] → (a_tar, theta_tar)，对齐振幅与相位
            re = target_field[..., :args.d_model]
            im = target_field[..., args.d_model:]
            a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
            th_tar = torch.atan2(im, re)
            # 相位走周期距离，避免 ±π 附近被惩罚成跨半周
            phase_err = torch.remainder(theta - th_tar + math.pi, 2 * math.pi) - math.pi
            loss = (a - a_tar).pow(2).mean() + phase_err.pow(2).mean()
            # 额外时间正则：帧间差（单边 hinge）
            eps = 0.05
            frame_diff = (theta[:, 1:] - theta[:, :-1]).norm(dim=-1)
            time_loss = (eps - frame_diff).relu().mean()  # AetherSentry 护栏：只罚不动
            loss += 0.1 * time_loss  # λ_time

            # 损失分解监控：定位哪一项主导
            if epoch == 0 and steps <= 2:
                print(f"[DIAG] amp={ (a - a_tar).pow(2).mean().item():.4f} "
                      f"phase={phase_err.pow(2).mean().item():.4f} "
                      f"time={time_loss.item():.4f} "
                      f"a_out={a.mean().item():.4f} a_tar={a_tar.mean().item():.4f}",
                      flush=True)

            # 总 loss = L = L_phase + L_amp + λ_time * L_time
            # 反向
            loss.backward()

            # 优化
            optimizer.step()

            # 记录
            epoch_loss += loss.item()
            steps += 1
            _losses["train"].append(loss.item())
            _diag["r"].append(diag["r"])
            _diag["theta_diff"].append(diag["theta_diff"])

        #  epoch 结束统计
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / steps
        print(f"[Epoch {epoch+1}/{args.num_epochs}] Loss={avg_loss:.4f}, Time={epoch_time:.1f}s, r_mean={sum(_diag['r'])/len(_diag['r']):.4f}")

        # 保存检查点
        if (epoch + 1) % args.checkpoint_interval == 0:
            ckpt_path = Path(args.output_dir) / f"checkpoint_epoch{epoch+1}.pt"
            model.save_checkpoint(str(ckpt_path), epoch=epoch)

        # 可选：在 epoch 后做短推理（3 秒）
        if (epoch + 1) % args.checkpoint_interval == 0:
            gen_text = "简约线条动画"
            gen_path = Path(args.output_dir) / f"gen_epoch{epoch+1}_3s.gif"
            with torch.no_grad():
                model.generate(gen_text, duration=3, save_path=str(gen_path))
            print(f"[GEN] Generated {gen_path}")

    print("[FINISH] 训练完成")


if __name__ == "__main__":
    main()
