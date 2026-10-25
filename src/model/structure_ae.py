# -*- coding: utf-8 -*-
"""StructureAutoEncoder —— 物理结构自编码器（Step 1，用户两阶段方案第一阶段）

【目标】训练一个**有泛化能力**的物理结构生成模型，把 GIF 结构压缩成一个
结构向量 z = [z_scene ‖ z_motion]，两个因子正交解耦。之后 z 再与文本/图像
向量结合（第二阶段）。

【验证设计（用户指定，20 张图）】
  A 组 10 张：动作相同、画面内容不同 → 要求 z_scene 分开、z_motion 靠近
  B 组 10 张：画面内容相同、动作不同 → 要求 z_scene 靠近、z_motion 分开
  （靠这个判据证明结构模型「把动作与画面解耦」，从而有组合泛化能力）

【架构铁律（与物理核零冲突）】
  - Encoder：驻波基（无参数三角函数）+ 通道压缩 Linear + 低秩 Linear。
    任何 Attention / 归一化 / 循环 FFN 都不出现。
  - Decoder：z_scene 只进空间维、z_motion 只进时间/频率维（正交），
    设定初始相位/幅度调制 → 走 **PMNNCore（零改动）** 物理演化回场。
  - 物理核（kuramoto/standing_wave_ffn/drift_field/thermodynamics/time_field/
    block.py）一行未动，只做「调用」。
"""

import math
import torch
import torch.nn as nn

from physics.block import PMNNCore
from physics.kuramoto_sync import build_spacetime_index


# ---------------------------------------------------------------------------
# 物理基底（无参数三角函数）
# ---------------------------------------------------------------------------
def sw_temporal_basis(TS: int, Kt: int, device, dtype=torch.float32):
    """1D 时间驻波基（含 DC）：SW_t[t,k] = cos(π·k·t/(TS-1))，k∈{0..Kt-1}。

    返回 [TS, Kt]。
    """
    t = torch.arange(TS, device=device, dtype=dtype) / max(TS - 1, 1)
    k = torch.arange(Kt, device=device, dtype=dtype)
    return torch.cos(math.pi * k.unsqueeze(0) * t.unsqueeze(1))


def sw_spatial_basis(GRID: int, Kxy: int, device, dtype=torch.float32):
    """2D 空间驻波基（含 DC）：SW_s[(gy,gx), kx*Kxy+ky] = cos(π kx x) cos(π ky y)。

    返回 [GRID*GRID, Kxy*Kxy]。
    """
    yy, xx = torch.meshgrid(
        torch.arange(GRID, device=device, dtype=dtype),
        torch.arange(GRID, device=device, dtype=dtype), indexing="ij")
    x = xx.reshape(-1) / max(GRID - 1, 1)      # [P]
    y = yy.reshape(-1) / max(GRID - 1, 1)
    ks = torch.arange(Kxy, device=device, dtype=dtype)
    bx = torch.cos(math.pi * ks.unsqueeze(0) * x.unsqueeze(-1))   # [P, Kxy]
    by = torch.cos(math.pi * ks.unsqueeze(0) * y.unsqueeze(-1))   # [P, Kxy]
    # 外积 -> [P, Kxy*Kxy]
    return (bx.unsqueeze(-1) * by.unsqueeze(1)).reshape(x.shape[0], -1)


def _init_omega(t_coord, s_coord, d_model):
    """确定性固有频率场 [1,N,D]（与 AetherGIFModel._init_omega 同口径，
    独立实现，避免依赖 model 模块）。ω 必须确定性：每次前向重采样 = 换动力系统。
    """
    d_idx = torch.arange(d_model, device=t_coord.device).view(1, -1)
    base = 0.4 * torch.sin(
        s_coord[..., 0:1] * 12.9898
        + s_coord[..., 1:2] * 78.233
        + t_coord.unsqueeze(-1) * 37.719
        + d_idx * 4.1
    )
    return base.unsqueeze(0)   # [1,N,D]


# ---------------------------------------------------------------------------
# Encoder：场 -> z_scene (+) z_motion
# ---------------------------------------------------------------------------
class StructureEncoder(nn.Module):
    """把 [B, N, 2D] 复值场压缩成解耦结构向量 z_scene / z_motion。

    流程（全物理基底 + 共享 Linear，无 attention）：
      1. field 按 (t, gy, gx) 拆成 [B, TS, P, 2D]
      2. scene = 时间维平均（静态画面）；motion = 场 − scene（帧间变化）
      3. 通道压缩 Linear(2D -> C)
      4. 空间驻波投影（P -> Kxy²）；motion 额外先过时间驻波
      5. 低秩 Linear 得到 z_scene [B, dim_z]、z_motion [B, dim_z]
    """

    def __init__(self, d_model: int = 125, dim_z: int = 256,
                 chan: int = 12, kxy: int = 8, kt: int = 4):
        super().__init__()
        self.d_model = d_model
        self.dim_z = dim_z
        self.chan = chan
        self.kxy = kxy
        self.kt = kt
        self.chan_proj = nn.Linear(2 * d_model, chan)          # 通道压缩（共享）
        self.k_sp = kxy * kxy
        # scene: [B, k_sp*chan] -> z_scene
        self.to_scene = nn.Linear(self.k_sp * chan, dim_z)
        # motion: [B, kt*chan] -> z_motion（空间维已全局池化，只剩时间×通道）
        self.to_motion = nn.Linear(self.kt * chan, dim_z)

    def forward(self, field: torch.Tensor, TS: int, GRID: int):
        """
        field: [B, N, 2D] 实虚拼接目标场（N = TS*GRID*GRID）
        返回 (z_scene, z_motion, diag)
        """
        B = field.shape[0]
        P = GRID * GRID
        dev, dt = field.device, field.dtype
        f = field.reshape(B, TS, P, 2 * self.d_model)          # [B,TS,P,2D]

        scene = f.mean(dim=1)                                   # [B,P,2D]
        motion = f - scene.unsqueeze(1)                         # [B,TS,P,2D]

        # 通道压缩
        s_c = self.chan_proj(scene)                             # [B,P,C]
        m_c = self.chan_proj(motion)                            # [B,TS,P,C]

        # 空间驻波基底（无参数）
        sw_s = sw_spatial_basis(GRID, self.kxy, dev, dt)        # [P, K]
        z_s = torch.einsum("pk,bpc->bkc", sw_s, s_c)
        z_s = z_s.reshape(B, -1)                                # [B, K*C]
        z_scene = self.to_scene(z_s)                            # [B, dim_z]

        # 时间驻波 + 空间全局池化
        # ★ motion 因子必须与画面解耦：空间维做全局池化，把「画面内容」
        #   平均掉，只保留「每时间步的动作统计特征」。之前对 motion 也做
        #   空间驻波投影，会把画面信息重新混进动作因子（冒烟实测解耦度 −0.164）。
        sw_t = sw_temporal_basis(TS, self.kt, dev, dt)          # [TS, Kt]
        m_t = torch.einsum("tk,btpc->bkpc", sw_t, m_c)          # [B,Kt,P,C]
        m_pool = m_t.mean(dim=2)                                 # [B,Kt,C]  空间平均
        z_m = m_pool.reshape(B, -1)                              # [B, Kt*C]
        z_motion = self.to_motion(z_m)                           # [B, dim_z]

        diag = {"z_scene_norm": z_scene.norm(dim=-1).mean().item(),
                "z_motion_norm": z_motion.norm(dim=-1).mean().item()}
        return z_scene, z_motion, diag


# ---------------------------------------------------------------------------
# Decoder：z -> 初始调制 -> PMNNCore -> 场
# ---------------------------------------------------------------------------
class StructureDecoder(nn.Module):
    """z_scene(空间/画面) ⊕ z_motion(时间/动作) -> 物理演化 -> 复值场。

    正交设计：z_scene 只进空间维、z_motion 只进时间/频率维，
    保证「换画面不动动作、换动作不动画面」。
    物理演化复用 PMNNCore（零改动），targets 恒 None（纯物理自洽，不抄答案）。
    """

    def __init__(self, d_model: int = 125, dim_z: int = 256,
                 n_blocks: int = 12, ffn_hidden: int = 256,
                 n_modules: int = 8, mod_gamma: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.dim_z = dim_z
        # z_scene -> 空间初始形状（幅度增益 + 相位偏移）
        self.scene_dlna = nn.Linear(dim_z, 1024)
        self.scene_dtheta = nn.Linear(dim_z, 1024)
        # z_motion -> 时间初始相位偏移（每时间步一个基偏移）
        self.motion_dtheta = nn.Linear(dim_z, 6)
        # z -> 频率调制（固有频率偏置，物理上"动作=振荡频率/相位演化"）
        self.omega_bias = nn.Linear(dim_z, d_model)
        # 零初始化：起步纯物理自洽，调制量平滑过渡（与 cond_init 同款reasoning）
        for m in [self.scene_dlna, self.scene_dtheta, self.motion_dtheta, self.omega_bias]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        # 物理核（零改动）
        self.pmnncore = PMNNCore(d_model=d_model, n_blocks=n_blocks,
                                 ffn_hidden=ffn_hidden,
                                 n_modules=n_modules, mod_gamma=mod_gamma)

    def forward(self, z_scene: torch.Tensor, z_motion: torch.Tensor,
                t_coord: torch.Tensor, s_coord: torch.Tensor,
                TS: int = 6, GRID: int = 32):
        """
        返回 (a, theta)：[B,N,D] 最终振子场。
        """
        B = z_scene.shape[0]
        N = TS * GRID * GRID
        dev = z_scene.device

        # ---- 初始调制（场景进空间维、动作进时间维）----
        dlna_s = torch.tanh(self.scene_dlna(z_scene)) * 1.5       # [B, P=1024]
        dth_s = torch.tanh(self.scene_dtheta(z_scene)) * math.pi  # [B, P]
        dth_t = torch.tanh(self.motion_dtheta(z_motion)) * math.pi  # [B, TS]

        # 广播：scene 沿时间复制、motion 沿空间复制
        dlna = dlna_s.reshape(B, 1, GRID, GRID).expand(B, TS, GRID, GRID).reshape(B, N, 1)
        dth = (dth_s.reshape(B, 1, GRID, GRID) + dth_t.reshape(B, TS, 1, 1)).reshape(B, N, 1)

        # ---- 初始状态 ----
        a0 = torch.exp(torch.randn(B, N, self.d_model, device=dev) * 0.8 - 2.5)
        theta0 = torch.rand(B, N, self.d_model, device=dev) * 2 * math.pi - math.pi
        a0 = a0 * torch.exp(dlna)
        theta0 = theta0 + dth

        # ---- 频率调制（动作进频率维）----
        omega = _init_omega(t_coord, s_coord, self.d_model)          # [1,N,D]
        omega = omega + self.omega_bias(z_motion).reshape(B, 1, self.d_model)

        # ---- 物理演化（targets=None：纯物理自洽，不抄答案）----
        a, theta = self.pmnncore(a0, theta0, omega, t_coord, s_coord,
                                 targets=None, protos=None,
                                 mod_id=None, cond_anchor=None)
        return a, theta


# ---------------------------------------------------------------------------
# 完整结构自编码器
# ---------------------------------------------------------------------------
class StructureAutoEncoder(nn.Module):
    """field -> z -> field 的端到端物理结构模型。"""

    def __init__(self, d_model: int = 125, dim_z: int = 256,
                 n_blocks: int = 12, n_modules: int = 8, mod_gamma: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.encoder = StructureEncoder(d_model=d_model, dim_z=dim_z)
        self.decoder = StructureDecoder(d_model=d_model, dim_z=dim_z,
                                        n_blocks=n_blocks,
                                        n_modules=n_modules, mod_gamma=mod_gamma)

    def forward(self, field: torch.Tensor, t_coord: torch.Tensor,
                s_coord: torch.Tensor, TS: int = 6, GRID: int = 32):
        """
        field: [B, N, 2D] target 场
        返回 (a, theta, z_scene, z_motion, diag)
        """
        z_scene, z_motion, diag = self.encoder(field, TS, GRID)
        a, theta = self.decoder(z_scene, z_motion, t_coord, s_coord, TS, GRID)
        diag["z_scene"] = z_scene.detach()
        diag["z_motion"] = z_motion.detach()
        return a, theta, z_scene, z_motion, diag