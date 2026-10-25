# 07 · AetherGIFModel 模型源码逐行分析

源文件：`src/model/aether_gif_model.py`（376行）
v4200 版本：`docx/实验记录/versions/v4200_多模态CLIP产品版/src/model/aether_gif_model.py`（已改造）

---

## 1. 文件头声明（第1-13行）

```python
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
```

---

## 2. 导入与路径处理（第14-37行）

```python
import sys
sys.path.insert(0, '.' if __name__ == "__main__" else '..')  # for imports

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
from src.physics.block import PMNNCore
from src.physics.time_field import TimeField
from src.physics.kuramoto_sync import KuramotoSync, SpacetimeToeplitzCoupling, MeanFieldCoupling, build_spacetime_index
from src.physics.standing_wave_ffn import StandingWaveFFN, StandingWave3D, lorentzian_gain
from src.physics.drift_field import DriftField
from src.physics.thermodynamics import Thermodynamics

from src.codec.gif_codec import load_gif_frames, resize_gif_to_10fps, split_frames_to_blocks, reconstruct_frames_from_field, gif_to_target_field, target_field_to_gif, decode_field_to_frames
from src.qwen_encoder.qwen_encoder import QwenTextEncoder
```

注意：`from transformers import AutoModelForCausalLM, AutoTokenizer` 在文件顶层导入。这是 v4200 之前的问题：即使 use_qwen=False，导入 transformers 也会触发版本兼容问题。v4200 改为懒加载。

---

## 3. `AetherGIFModel` 类（第40-349行）

### 3.1 `__init__`（第50-121行）

```python
def __init__(
    self,
    d_model: int = 64,
    ffn_hidden: int = 256,
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
```

#### 模块1：PMNNCore（第80-92行）

```python
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
```

#### 模块2：Qwen 文本编码器（第94-107行）

```python
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
```

- use_qwen=True：加载冻结 Qwen2.5-0.5B，text_dim = qwen.hidden_size
- use_qwen=False：用一个可学习条件向量 `text_param` 代替，shape [1, 896]

QwenTextEncoder.hidden_size 来自 `self.model.config.hidden_size`（qwen_encoder.py:36）。Qwen2.5-0.5B 的 hidden_size 实际是 896。

#### 模块3：文本到物理条件的调制头（第109-116行）

```python
self.omega_bias_head = nn.Linear(self.text_dim, d_model)  # 文本 → 频率扰动
self.drift_target_head = nn.Linear(self.text_dim, d_model)
self.kappa_scale_head = nn.Linear(self.text_dim, 1)
self.T_scale_head = nn.Linear(self.text_dim, 1)
```

四个调制头：
- `omega_bias_head`: text_dim → d_model，文本 → ω 偏移
- `drift_target_head`: text_dim → d_model，文本 → 漂移目标（定义了但 forward 未使用）
- `kappa_scale_head`: text_dim → 1，文本 → κ 缩放
- `T_scale_head`: text_dim → 1，文本 → T 缩放

#### 模块4：原型参数（第118-119行）

```python
self.each_prototype = nn.Parameter(torch.randn(n_prototypes, 2 * d_model) * 0.02)
```

定义了但 forward 和 generate 中均未使用。

#### 参数量打印（第121行）

```python
print(f"[INFO] AetherGIFModel 初始化完成：d_model={d_model}, n_blocks={n_blocks}, 参数量 = {sum(p.numel() for p in self.parameters()):,}")
```

### 3.2 `_init_omega` 方法（第123-139行）

```python
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
```

- 确定性频率场：用时空坐标和通道索引的 sin 组合
- 系数 12.9898, 78.233, 37.719, 4.1 是常数
- 幅度 0.4，所以 ω ∈ [−0.4, 0.4]
- 注释：「ω 随机 + 漂移场失效时，模型完全没有可学习的确定性动力学」

### 3.3 `forward` 方法（第141-218行）

```python
def forward(self, texts: List[str], target_field: torch.Tensor,
            t_coord: torch.Tensor, s_coord: torch.Tensor,
            batch_size: int = 1, duration_target: Optional[int] = None,
            text_embs: Optional[torch.Tensor] = None):
```

参数：
- texts: 文本列表
- target_field: 目标振子场（训练时给）
- t_coord, s_coord: 时空坐标
- batch_size
- duration_target: 未使用
- text_embs: 外部直传的文本嵌入（缓存复用）

注意：主目录版本没有 `cond_vec` 和 `use_cond_field` 参数。这两个是 v4200 版本新增的。

#### 步骤1：文本编码（第160-167行）

```python
if text_embs is not None:
    text_embs = text_embs.to(self.device)
elif self.use_qwen:
    text_embs = self.qwen.encode(list(texts))  # [B, hidden]
else:
    text_embs = self.text_param.expand(len(texts), -1).to(self.device)
```

#### 步骤2：物理参数调制（第169-173行）

```python
omega_bias = self.omega_bias_head(text_embs)  # [B, D]
drift_targets = self.drift_target_head(text_embs)  # [B, D]
kappa_scale = torch.sigmoid(self.kappa_scale_head(text_embs)).squeeze(-1)  # [B]
T_scale = torch.sigmoid(self.T_scale_head(text_embs)).squeeze(-1)  # [B]
```

- `drift_targets` 计算了但后续未使用（第194行只用 omega_bias）
- kappa_scale, T_scale 经 sigmoid，范围 (0,1)

#### 步骤3：初始化噪声场（第186-194行）

```python
# 初始状态 a0/θ0 随机（生成模型从噪声出发，正确）；ω 必须确定性
# a0 在 log 域采样以匹配 FFT 幅度的重尾量级（几何均值 ≈0.08，与 target 同阶）；
# 振幅走乘性演化后，起点量级几层就能调过来，但同阶起点能显著加速收敛
a0 = torch.exp(torch.randn(batch_size, len(t_coord), self.d_model, device=self.device) * 0.8 - 2.5)
theta0 = torch.rand(batch_size, len(t_coord), self.d_model, device=self.device) * 2 * math.pi - math.pi
omega_base = self._init_omega(t_coord, s_coord)  # [1,N,D] 确定性

# 文本偏移
omega = omega_base + omega_bias.unsqueeze(1)  # [B, N, D]
```

- a0 = exp(randn × 0.8 − 2.5)：对数正态采样，几何均值 ≈ exp(−2.5) ≈ 0.082
- theta0 = U(−π, π)
- omega = 确定性基础频率 + 文本偏移

#### 步骤4：kappa/T 缩放（第196-199行）

```python
kappa_base, T_base = 2.0, 0.5
kappa = kappa_base * kappa_scale.unsqueeze(1) + 0.0  # 扩展
T = T_base * T_scale.unsqueeze(1) + 0.0
```

- kappa_base=2.0, T_base=0.5
- kappa ∈ (0, 2.0)，T ∈ (0, 0.5)
- 这些会乘到 TimeField 的 κ(t) 和 T(t) 上

#### 步骤5：12轮物理演化（第201-213行）

```python
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
```

注意：直接调用 `self.pmnncore.blocks[i]`，不走 `PMNNCore.forward`。kappa 和 T 被文本缩放因子进一步缩放。

实际 κ 范围：TimeField κ(t) ∈ [1.907, 15.851] × kappa_scale ∈ (0,1) × kappa_base=2.0 → (0, 31.7)
实际 T 范围：TimeField T(t) ∈ [0.05, 1.05] × T_scale ∈ (0,1) × T_base=0.5 → (0, 0.525)

#### 返回值（第217-218行）

```python
diag = self._collect_diag(a, theta, t_coord, s_coord)
return a, theta, diag
```

### 3.4 `_collect_diag` 方法（第226-237行）

```python
def _collect_diag(self, a, theta, t_coord, s_coord):
    diag = {}
    w = torch.nn.functional.softplus(a)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    rcos = (w * torch.cos(theta)).sum(dim=1) / w_sum
    rsin = (w * torch.sin(theta)).sum(dim=1) / w_sum
    r = torch.sqrt(rcos ** 2 + rsin ** 2 + 1e-12).clamp(max=1.0)
    diag["r"] = r.mean().item()
    diag["theta_diff"] = (torch.diff(theta, dim=1).abs().mean().item())
    return diag
```

只返回两个标量：r（序参量均值）和 theta_diff（相邻振子相位差均值）。

### 3.5 `generate` 方法（第239-305行）

```python
@torch.inference_mode()
def generate(self, text: str, duration: int = 5, save_path: str = None) -> Union[torch.Tensor, str]:
```

生成流程：
1. 文本编码（第254行）
2. 物理参数调制（第257-259行）
3. 构造时空坐标（第261-270行）：
   ```python
   fps = 10
   T_step = duration * fps // self.time_step
   H, W = 128, 128
   patch_h, patch_w = self.patch_size
   n_h = H // patch_h
   n_w = W // patch_w
   hw = n_h * n_w
   t_coord, s_coord = build_spacetime_index(T_step, n_h, n_w, self.device)
   N = len(t_coord)
   ```
   duration=5 → T_step = 50//5 = 10 → N = 10×32×32 = 10,240
4. 初始化噪声场（第272-274行）：
   ```python
   a0 = torch.nn.functional.softplus(torch.randn(1, N, self.d_model, device=self.device))
   theta0 = torch.rand(1, N, self.d_model, device=self.device) * 2 * math.pi - math.pi
   ```
   注意：generate 用 `softplus(randn)`，而 forward 用 `exp(randn*0.8−2.5)`。两种初始化不同。
5. 12 轮演化（第285-294行），`targets=None`（生成模式）
6. 逆 FFT 重构（第296-298行）
7. 保存 GIF（第300-304行）

### 3.6 `_field_to_frames` 方法（第307-326行）

```python
def _field_to_frames(self, a, theta, t_coord, s_coord):
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
```

- `normalize=True`：per-clip min-max 拉伸（decode_field_to_frames 第312-314行）
- grid 从 N/T_step 反推

### 3.7 `_save_gif` 方法（第328-332行）

```python
def _save_gif(self, frames: List[np.ndarray], path: str) -> None:
    from PIL import Image
    pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:], loop=0, duration=100, quality=95)
```

- duration=100ms → 10fps
- loop=0 → 无限循环

### 3.8 检查点方法（第334-349行）

```python
def save_checkpoint(self, path: str, epoch: int = 0):
    state = {
        "model": self.state_dict(),
        "epoch": epoch,
        "d_model": self.d_model,
        "n_blocks": self.n_blocks,
    }
    torch.save(state, path)

def load_checkpoint(self, path: str, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    self.load_state_dict(state["model"])
```

---

## 4. v4200 版本的改造

根据交接文档，v4200 的 `aether_gif_model.py` 做了以下改造：

1. `forward` 增加 `cond_vec` 参数（CLIP 条件向量）和 `use_cond_field` 开关
2. `use_cond_field=True` 时用 `cond_field(cond)` 当吸引中心，`drift_target=None`
3. diag 里回传 `pred_field` 供训练侧直接监督
4. `generate()` 已支持 `cond_vec` 直传（读离线特征，不加载 CLIP）
5. Qwen 加载改为懒加载，`use_qwen=False` 路径不触发 transformers 依赖

训练脚本调用（train_multimodal.py:947-949）：
```python
a, th, diag = model(texts=None, target_field=field, t_coord=t_coord,
                    s_coord=s_coord, batch_size=B, cond_vec=cond,
                    use_cond_field=is_gen)
```

---

## 5. 两条演化路径的对比

| 路径 | kappa | T | targets | 用途 |
|---|---|---|---|---|
| `forward`（训练） | tf["kappa"] × kappa_scale × 2.0 | tf["T"] × T_scale × 0.5 | target_field（重建）或 cond_field 预测（生成） | 训练 |
| `generate`（推理） | tf["kappa"] × kappa_scale × 2.0 | tf["T"] × T_scale × 0.5 | None（用 prototypes） | 推理 |
| `PMNNCore.forward` | tf["kappa"] | tf["T"] | targets | 未使用 |

---

## 6. 初始化方式的差异

| 方法 | a0 初始化 | 代码位置 |
|---|---|---|
| `forward` | `exp(randn × 0.8 − 2.5)`，几何均值 ≈0.082 | aether_gif_model.py:189 |
| `generate` | `softplus(randn)`，softplus(0)≈0.694 | aether_gif_model.py:273 |

softplus(randn) 的均值：randn 均值 0，softplus(0) = ln(2) ≈ 0.694。所以 generate 的 a0 均值约 0.7，远大于 forward 的 0.082。

注释（第186-188行）说明 forward 用 log 域采样是为了「匹配 FFT 幅度的重尾量级（几何均值 ≈0.08，与 target 同阶）」。generate 用 softplus 没有这个考虑，两种初始化不一致。

---

## 7. 未验证项

1. `drift_target_head` 和 `each_prototype` 定义了但未使用，无实验数据。
2. `duration_target` 参数未使用。
3. `PMNNCore.forward` 路径未被调用，无实验数据。
4. generate 与 forward 的 a0 初始化不一致，对生成质量的影响无对比实验。
5. `omega_bias` 的实际范围（text_embs × Linear 权重）无实测数据。
6. `kappa_scale` / `T_scale` 经 sigmoid 后的典型值无训练日志记录。
7. v4200 的 `cond_vec` 和 `use_cond_field` 参数的完整实现需读 v4200 版本文档，主目录版本不存在。

---

## 8. 代码事实汇总

| 项 | 值 | 来源 |
|---|---|---|
| d_model | 64（默认） | aether_gif_model.py:52 |
| n_blocks | 12 | aether_gif_model.py:54 |
| text_dim | 896（use_qwen=False） | aether_gif_model.py:97 |
| kappa_base / T_base | 2.0 / 0.5 | aether_gif_model.py:197 |
| a0（forward） | exp(randn×0.8−2.5) | aether_gif_model.py:189 |
| a0（generate） | softplus(randn) | aether_gif_model.py:273 |
| theta0 | U(−π,π) | aether_gif_model.py:190, 274 |
| omega 幅度 | 0.4×sin(...) ∈ [−0.4,0.4] | aether_gif_model.py:133 |
| 4个调制头 | omega_bias/drift_target/kappa_scale/T_scale | aether_gif_model.py:113-116 |
| 未使用参数 | drift_target_head, each_prototype, duration_target | 代码分析 |
| generate duration | 3-10 秒，T_step=duration×10//5 | aether_gif_model.py:263 |
| GIF duration | 100ms = 10fps | aether_gif_model.py:332 |
