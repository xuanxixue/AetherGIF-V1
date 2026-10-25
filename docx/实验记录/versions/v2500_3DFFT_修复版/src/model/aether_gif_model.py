# -*- coding: utf-8 -*-
"""
AetherGIFModel —— 物理矩阵神经网络 GIF 生成模型

完整架构：
  GIF → 3D FFT → 64 复值振子 → PMNNCore → 逆 FFT → GIF

密钥原则：
  - 物理核所有算子严格按 AetherMind 公式实装
  - 不得引入任何 Transformer/注意力/FFN 结构
  - Qwen 仅作为冻结文本编码器，提供物理条件注入
  - 支持 3–10 秒可变时长（6–20 时间步）
"""
import sys

import os
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 本地导入
from physics.block import PMNNCore
from physics.time_field import TimeField
from physics.kuramoto_sync import KuramotoSync, SpacetimeToeplitzCoupling, MeanFieldCoupling, build_spacetime_index
from physics.standing_wave_ffn import StandingWaveFFN, StandingWave3D, lorentzian_gain
from physics.drift_field import DriftField
from physics.thermodynamics import Thermodynamics

from codec.gif_codec import load_gif_frames, resize_gif_to_10fps, split_frames_to_blocks, reconstruct_frames_from_field, gif_to_target_field, target_field_to_gif, decode_field_to_frames
from qwen_encoder.qwen_encoder import QwenTextEncoder


class AetherGIFModel(nn.Module):
    """
    AetherGIF 完整模型：
      1. 文本条件编码器 (Qwen 冻结)
      2. 3D FFT 振子编码器 (冻结)
      3. PMNN 物理核心 (可训练)
      4. 逆 FFT 解码器 (冻结)
      5. 文本条件调制 (注入物理参数)
    """

    def __init__(
        self,
        d_model: int = 125,
        ffn_hidden: int = 512,
        n_blocks: int = 12,
        lambda_t: float = 0.5,
        lambda_s: float = 3.0,
        temporal_window: int = 3,
        dt: float = 0.1,
        n_prototypes: int = 16,
        n_modes_t: int = 4,
        n_modes_xy: int = 4,
        T_ref: float = 20.0,
        X_ref: float = 32.0,
        Y_ref: float = 32.0,
        qwen_model_path: str = "Qwen2.5-0.5B-Instruct/snapshots/master",
        device: str = "cuda",
        patch_size: Tuple[int, int] = (4, 4),
        time_step: int = 5,
        use_qwen: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.n_blocks = n_blocks
        self.patch_size = patch_size
        self.time_step = time_step
        self.device = device
        self.use_qwen = use_qwen

        # 1. PMNN 物理核心（可训练）
        self.pmnncore = PMNNCore(
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            lambda_t=lambda_t,
            lambda_s=lambda_s,
            temporal_window=temporal_window,
            dt=dt,
            n_prototypes=n_prototypes,
            T_ref=T_ref,
            X_ref=X_ref,
            Y_ref=Y_ref,
        ).to(device)

        # 2. Qwen 文本编码器（冻结）
        # 验证阶段可旁路（use_qwen=False）：用一个可学习条件向量代替文本嵌入，
        # 跳过 500MB 权重加载与前向，专注验证物理核心能否生成 GIF。
        self.text_dim = 896
        if use_qwen:
            self.qwen = QwenTextEncoder(
                model_path=qwen_model_path,
                device=device,
                freeze=True
            )
            self.text_dim = self.qwen.hidden_size
        else:
            self.qwen = None
            self.text_param = nn.Parameter(torch.randn(1, self.text_dim, device=device) * 0.02)

        # 3. 文本到物理条件的调制器（可训练）
        # 文本嵌入 → 模型参数（rand, scale, drift_targets, etc.）
        # 为保持物理纯粹性，我们不使用 Qwen 输出直接改变物理算子，而是提供边界条件
        # 例如：固有频率谱偏移、漂移场原型、κ课程参数
        self.omega_bias_head = nn.Linear(self.text_dim, d_model)  # 文本 → 频率扰动
        self.drift_target_head = nn.Linear(self.text_dim, d_model)
        self.kappa_scale_head = nn.Linear(self.text_dim, 1)
        self.T_scale_head = nn.Linear(self.text_dim, 1)

        # 4. 物理上下文：用于分配原型和偏移
        self.each_prototype = nn.Parameter(torch.randn(n_prototypes, 2 * d_model) * 0.02)

        print(f"[INFO] AetherGIFModel 初始化完成：d_model={d_model}, n_blocks={n_blocks}, 参数量 = {sum(p.numel() for p in self.parameters()):,}")

    def _init_omega(self, t_coord: torch.Tensor, s_coord: torch.Tensor) -> torch.Tensor:
        """确定性固有频率场 [1, N, D]。

        ω 是系统的固有属性（参数），不是初始状态。若每次前向都 randn 重采样，
        同一个样本两次前向就是两个不同的动力系统，相位演化无法学习——这条
        通路与漂移场同样致命：ω 随机 + 漂移场失效时，模型完全没有可学习的
        确定性动力学。用时空坐标 × 通道索引的确定性准随机场给出空间异质频率，
        保证同一位置每次前向频率一致。
        """
        d_idx = torch.arange(self.d_model, device=t_coord.device).view(1, -1)  # [1,D]
        base = 0.4 * torch.sin(
            s_coord[..., 0:1] * 12.9898
            + s_coord[..., 1:2] * 78.233
            + t_coord.unsqueeze(-1) * 37.719
            + d_idx * 4.1
        )  # [N, D]
        return base.unsqueeze(0)  # [1, N, D]

    def forward(self, texts: List[str], target_field: torch.Tensor,
                t_coord: torch.Tensor, s_coord: torch.Tensor,
                batch_size: int = 1, duration_target: Optional[int] = None,
                text_embs: Optional[torch.Tensor] = None):
        """
        条件生成前向：

        1. Qwen 编码文本条件
        2. 物理参数调制：基于文本嵌入产生 ω bias, drift targets, κ/T 缩放
        3. 初始化或加载目标振子场 (a*, θ*)
        4. PMNN 物理演化：从噪声场积分到目标场
        5. 返回最终振子场和诊断

        注意：目标场 (target_field) 是 sed 目标，用于训练；生成时无 target_field，使用原型。

        返回值：
          - (a, theta): [B, N, D] 最终振子场
          - diag: 诊断标签字典
        """
        # 1. 文本编码（DataLoader 可能 collate 成 tuple，tokenizer 只收 list）
        # use_qwen=False 时用可学习条件向量；也支持外部直传 text_embs（缓存复用）
        if text_embs is not None:
            text_embs = text_embs.to(self.device)
        elif self.use_qwen:
            text_embs = self.qwen.encode(list(texts))  # [B, hidden]
        else:
            text_embs = self.text_param.expand(len(texts), -1).to(self.device)

        # 2. 物理参数调制
        omega_bias = self.omega_bias_head(text_embs)  # [B, D]
        drift_targets = self.drift_target_head(text_embs)  # [B, D]
        kappa_scale = torch.sigmoid(self.kappa_scale_head(text_embs)).squeeze(-1)  # [B]
        T_scale = torch.sigmoid(self.T_scale_head(text_embs)).squeeze(-1)  # [B]

        # 3. 初始化/加载噪声场 (若 target_field 是 None，则使用原型)
        if target_field is None:
            # 生成模式：无 target_field
            # 使用可学习原型作为吸引中心（作为漂移场的吸引子）
            # 在 PMNNCore 内部处理原型
            pass
        else:
            # 训练模式：有 target_field
            pass

        # 4. 物理演化
        # 初始状态 a0/θ0 随机（生成模型从噪声出发，正确）；ω 必须确定性
        # a0 在 log 域采样以匹配 FFT 幅度的重尾量级（几何均值 ≈0.08，与 target 同阶）；
        # 振幅走乘性演化后，起点量级几层就能调过来，但同阶起点能显著加速收敛
        a0 = torch.exp(torch.randn(batch_size, len(t_coord), self.d_model, device=self.device) * 0.8 - 2.5)
        theta0 = torch.rand(batch_size, len(t_coord), self.d_model, device=self.device) * 2 * math.pi - math.pi
        omega_base = self._init_omega(t_coord, s_coord)  # [1,N,D] 确定性

        # 文本偏移
        omega = omega_base + omega_bias.unsqueeze(1)  # [B, N, D]

        # 时间场参数平均化
        kappa_base, T_base = 2.0, 0.5
        kappa = kappa_base * kappa_scale.unsqueeze(1) + 0.0  # 扩展
        T = T_base * T_scale.unsqueeze(1) + 0.0

        # 5. PMNN 物理演化（时间步迭代）
        a, theta = a0, theta0
        for i in range(self.n_blocks):
            tf = self.pmnncore.time_field.forward(i, self.device)
            # 漂移引导由 PhysicsBlock 内部的 DriftField(a, theta, targets) 给出，
            # 是唯一携带 target 信息的通路；这里不再重复计算
            a, theta, d = self.pmnncore.blocks[i](
                a, theta, omega, t_coord, s_coord,
                kappa=tf["kappa"] * kappa,
                T=tf["T"] * T,
                omega_d=tf["omega_d"],
                targets=target_field,
            )
            # 可选：记录 diag

        # 返回最终状态和诊断
        diag = self._collect_diag(a, theta, t_coord, s_coord)
        return a, theta, diag

    def _get_static_drift(self, a, theta, drift_target):
        # 生成模式下，使用可学习原型 + drift_target 注入
        # 为了简化演示，这里返回一个固定漂移
        B, N, D = a.shape
        return torch.zeros(B, N, D, device=a.device)

    def _collect_diag(self, a, theta, t_coord, s_coord):
        # 收集一些诊断量
        diag = {}
        # 计算 r (平均场序参量)
        w = torch.nn.functional.softplus(a)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        rcos = (w * torch.cos(theta)).sum(dim=1) / w_sum
        rsin = (w * torch.sin(theta)).sum(dim=1) / w_sum
        r = torch.sqrt(rcos ** 2 + rsin ** 2 + 1e-12).clamp(max=1.0)
        diag["r"] = r.mean().item()
        diag["theta_diff"] = (torch.diff(theta, dim=1).abs().mean().item())
        return diag

    @torch.inference_mode()
    def generate(self, text: str, duration: int = 5, save_path: str = None) -> Union[torch.Tensor, str]:
        """
        推理：从随机噪声生成 GIF。

        参数:
          text: 文本条件
          duration: 目标时长（秒），3–10
          save_path: 输出 GIF 路径，可选

        返回:
          (a, theta): 最终振子场
          若 save_path 给定，保存为 GIF 并返回路径
        """
        # 1. 文本编码
        text_embs = self.qwen.encode([text])  # [1, 512]

        # 2. 物理参数调制
        omega_bias = self.omega_bias_head(text_embs)  # [1, D]
        kappa_scale = torch.sigmoid(self.kappa_scale_head(text_embs)).squeeze(-1)  # [1]
        T_scale = torch.sigmoid(self.T_scale_head(text_embs)).squeeze(-1)  # [1]

        # 3. 构造时空坐标
        fps = 10
        T_step = duration * fps // self.time_step
        H, W = 128, 128
        patch_h, patch_w = self.patch_size
        n_h = H // patch_h
        n_w = W // patch_w
        hw = n_h * n_w
        t_coord, s_coord = build_spacetime_index(T_step, n_h, n_w, self.device)
        N = len(t_coord)

        # 4. 初始化噪声场
        a0 = torch.nn.functional.softplus(torch.randn(1, N, self.d_model, device=self.device))
        theta0 = torch.rand(1, N, self.d_model, device=self.device) * 2 * math.pi - math.pi
        omega_base = self._init_omega(t_coord, s_coord)  # [1,N,D] 确定性

        # 5. 文本偏移
        omega = omega_base + omega_bias.unsqueeze(1)  # [1, N, D]

        # 6. 迭代演化
        a, theta = a0, theta0
        kappa_base, T_base = 2.0, 0.5
        kappa = kappa_base * kappa_scale.unsqueeze(1) + 0.0
        T = T_base * T_scale.unsqueeze(1) + 0.0
        for i in range(self.n_blocks):
            tf = self.pmnncore.time_field.forward(i, self.device)
            # 生成模式下使用空 drift
            a, theta, d = self.pmnncore.blocks[i](
                a, theta, omega, t_coord, s_coord,
                kappa=tf["kappa"] * kappa,
                T=tf["T"] * T,
                omega_d=tf["omega_d"],
                targets=None,
            )

        # 7. 逆 FFT 重构
        # 将振子场还原为 GIF 帧
        frames = self._field_to_frames(a, theta, t_coord, s_coord)

        # 8. 保存
        if save_path:
            self._save_gif(frames, save_path)
            print(f"[GEN] 输出到 {save_path}")
            return save_path
        return a, theta

    def _field_to_frames(self, a, theta, t_coord, s_coord):
        """振子场 → GIF 帧列表。

        a/theta: [1, N, D]。先合成复值场 z = a·exp(iθ)，
        再走确定性逆时空 FFT（decode_field_to_frames）还原像素。
        """
        T_step = int(t_coord.max().item()) + 1
        N = a.shape[1]
        grid = int(round((N / T_step) ** 0.5))
        patch_h, patch_w = self.patch_size
        win = self.time_step

        z = (a[0] * torch.exp(1j * theta[0])).to(torch.complex64)   # [N, D]
        field = torch.cat([z.real, z.imag], dim=-1).float().cpu()   # [N, 2D]，decode 在 CPU 做
        frames = decode_field_to_frames(
            field, time_step=T_step, grid=grid,
            patch=patch_h, win=win, img=128, d_model=self.d_model,
            normalize=True,
        )
        return frames

    def _save_gif(self, frames: List[np.ndarray], path: str) -> None:
        """保存 GIF（frames 为 numpy 数组，转 PIL）。"""
        from PIL import Image
        pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
        pil[0].save(path, save_all=True, append_images=pil[1:], loop=0, duration=100, quality=95)

    def save_checkpoint(self, path: str, epoch: int = 0):
        """保存权重。"""
        state = {
            "model": self.state_dict(),
            "epoch": epoch,
            "d_model": self.d_model,
            "n_blocks": self.n_blocks,
        }
        torch.save(state, path)
        print(f"[SAVE] 检查点到 {path}")

    def load_checkpoint(self, path: str, map_location="cpu"):
        """加载权重。"""
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state["model"])
        print(f"[LOAD] 从 {path} 加载，epoch={state.get('epoch', '?')}")


# 测试函数
if __name__ == "__main__":
    # 快速测试
    model = AetherGIFModel()
    model.eval()
    # 测试条件前向
    texts = ["简约线条动画", "简单的gif"]
    # 构造 dummy 时长和坐标
    duration = 3
    fps = 10
    T_step = duration * fps // 5
    H, W = 128, 128
    patch_h, patch_w = 4, 4
    n_h, n_w = H // patch_h, W // patch_w
    t_coord, s_coord = build_spacetime_index(T_step, n_h, n_w, "cuda")
    # 生成流程测试
    with torch.no_grad():
        a, theta, diag = model.generate("简约线条动画", duration=duration)
        print("生成完成:", diag)
        # 保存
        gpath = f"test_{duration}s.gif"
        model._save_gif([] if not isinstance(a, torch.Tensor) else [], gpath)
        # 尝试用 AetherGIFModel 生成的 GIF 
        print("路径:", gpath)
