# -*- coding: utf-8 -*-
"""
条件生成模型 v4：纯 CLIP 感知损失（无重建）

核心：物理核 + 条件调制器，仅通过 CLIP 特征相似度驱动训练。
不使用原 GIF 作为监督信号，实现真正的条件生成。

损失公式：L = -mean(cos_sim(clip(image_features), condition_features))
"""

import os
import sys
import math
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torch.nn as nn

# 环境设置
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", " NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from physics.block import PhysicsBlock
from physics.time_field import TimeField
from data.clip_condition import ClipConditionEncoder

# 编解码口径
D, GRID, PATCH, WIN, IMG, TS = 64, 16, 4, 2, 64, 3  # 大幅减小尺寸以降低显存

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
GIF = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4

OUT = "outputs_real/cond_gen_v4.gif"
CACHE = ROOT / "clip_cache"
device = "cuda"

print(f"[准备] 开始条件生成 v4 训练  lr={LR}  GIF={GIF}  STEPS={STEPS}")


class CondGenModelV4(torch.nn.Module):
    """条件生成模型 v4：条件直接驱动 DriftField，基于 CLIP 感知损失"""

    def __init__(self, cond_dim=2049, d_model=D, n_blocks=12):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks

        # 条件调制器：cond → 原型块
        from physics.condition_mod import ConditionModulator
        self.cond_mod = ConditionModulator(cond_dim=cond_dim, d_model=d_model, n_prototypes=16, hidden_dim=256)

        # 物理核心组件
        self.time_field = TimeField(n_blocks=n_blocks)
        self.blocks = nn.ModuleList([
            PhysicsBlock(
                d_model=d_model,
                ffn_hidden=256,
                lambda_t=0.5,
                lambda_s=3.0,
                temporal_window=3,
                dt=0.1,
                n_prototypes=16,
                T_ref=20.0,
                X_ref=32.0,
                Y_ref=32.0,
                a_mu=None,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, cond, t_coord, s_coord):
        B = cond.shape[0]
        proto = self.cond_mod(cond)  # [B, 16, 2D]

        N = t_coord.shape[0]
        a0 = torch.exp(torch.randn(B, N, self.d_model, device=cond.device) * 0.8 - 2.5)
        theta0 = torch.rand(B, N, self.d_model, device=cond.device) * 2 * math.pi - math.pi

        omega_base = self._init_omega(t_coord, s_coord)

        a, theta = a0, theta0
        for i, blk in enumerate(self.blocks):
            tf = self.time_field.forward(i, cond.device)
            a, theta, d = blk(
                a, theta, omega_base, t_coord, s_coord,
                kappa=tf["kappa"],
                T=tf["T"],
                omega_d=tf["omega_d"],
                targets=None,
                proto=proto,
            )

        return a, theta, None

    def _init_omega(self, t_coord, s_coord):
        d_idx = torch.arange(self.d_model, device=t_coord.device).view(1, -1)
        base = 0.4 * torch.sin(
            s_coord[..., 0:1] * 12.9898
            + s_coord[..., 1:2] * 78.233
            + t_coord.unsqueeze(-1) * 37.719
            + d_idx * 4.1
        )
        return base.unsqueeze(0)


def _decodable(a_, th_):
    z = torch.complex(a_ * torch.cos(th_), a_ * torch.sin(th_))
    N = z.shape[0]
    z = z * 10.0
    W_FFT = (PATCH * PATCH * 3) // 2 + 1
    # 需要确保输入形状正确
    reshaped = z.reshape(N, -1, PATCH * PATCH * 3)
    flat = torch.fft.irfft2(reshaped, s=(WIN, PATCH * PATCH * 3), dim=(-2, -1)).real
    flat = flat.reshape(N, WIN * PATCH * PATCH * 3)
    blocks = flat.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)
    return xt.reshape(TS * WIN, IMG, IMG, 3)


def _clip_loss(gif_frames, clip_encoder, condition_features):
    """
    gif_frames: numpy array [T, H, W, C] uint8
    step=1: 编码关键帧 [0, T//2, T-1]
    """
    key_indices = [0, len(gif_frames) // 2, len(gif_frames) - 1]
    key_frames = [Image.fromarray(gif_frames[i]) for i in key_indices]

    with torch.no_grad():
        img_features = clip_encoder.encode_images(key_frames, batch_size=1)
        img_features = torch.from_numpy(img_features).float().to(condition_features.device)

    cos_sim = torch.nn.functional.cosine_similarity(
        img_features, condition_features.expand_as(img_features), dim=-1
    )
    loss = -cos_sim.mean()
    return loss, cos_sim.item()


def main():
    # 读取 CLIP 条件
    cond_np = np.load(CACHE / "cond.npy")
    sample_idx = 0 if GIF.startswith("8-ball") else 1
    condition = torch.from_numpy(cond_np[sample_idx:sample_idx + 1]).float().to(device)
    print(f"[CLIP] 条件特征 shape: {condition.shape}")

    # 时空坐标
    t, s = build_spacetime_index(TS, GRID, GRID, device)

    # 模型
    model = CondGenModelV4(cond_dim=2049, d_model=D, n_blocks=12)
    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] 总参数量：{total_params:,}")

    # CLIP 编码器（冻结）
    clip_encoder = ClipConditionEncoder(
        device=device,
        image_size=512,
        fp16=False
    )
    clip_encoder.eval()
    clip_encoder = clip_encoder.to(device)

    # 优化显存使用
    torch.cuda.empty_cache()

    torch.manual_seed(1234)
    print(f"[RUN] {STEPS} 步纯条件生成 v4 训练  lr={LR}  GIF={GIF}")
    print(f"[注意] 使用 CLIP 感知损失，无需真实 GIF 监督")

    for step in range(1, STEPS + 1):
        opt.zero_grad()

        # 条件 → 物理演化
        a, theta, _ = model(condition, t, s)

        # 解码为 GIF 帧
        frames = _decodable(a[0], theta[0])
        frames_np = frames.detach().cpu().numpy().astype(np.uint8)
        frames_gif = frames_np.reshape(TS, WIN, IMG, IMG, 3)
        gif_frames = [f[0] for f in frames_gif]

        # 计算 CLIP 感知损失
        loss, sim = _clip_loss(gif_frames, clip_encoder, condition)
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == 1 or step == STEPS:
            print(f"[{step:4d}] loss={loss.item():.4f} sim={sim:.4f}")

    # 保存生成 GIF
    model.eval()
    with torch.inference_mode():
        a, theta, _ = model(condition, t, s)
        frames = _decodable(a[0], theta[0])
        frames_np = frames.detach().cpu().numpy().astype(np.uint8)
        gif_frames = [Image.fromarray(f) for f in (frames_np.reshape(TS, WIN, IMG, IMG, 3)[:, 0])]
        gif_frames[0].save(OUT, save_all=True, append_images=gif_frames[1:], loop=0, duration=100)

    print(f"[输出] {OUT}")


if __name__ == "__main__":
    main()
