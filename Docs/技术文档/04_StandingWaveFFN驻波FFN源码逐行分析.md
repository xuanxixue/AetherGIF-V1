# 04 · StandingWaveFFN 驻波FFN源码逐行分析

源文件：`src/physics/standing_wave_ffn.py`（163行）
调用点：`block.py:117`（`ffn_out = self.ffn(a, theta_new, omega_d, t_coord, s_coord)`）

---

## 1. 文件头声明（第1-18行）

```python
"""
§4.3 StandingWaveFFN —— 共振-驻波前馈（替代 FFN）

  h = σ( W₂·(W₁·z + b₁) ⊙ A(ω_d, ω_n) ⊙ SW(p) + b₂ )

(a) 洛伦兹共振增益（频率选择性）
    A(ω_d, ω_n) = 1 / sqrt((ω_d² - ω_n²)² + (2γ ω_d)²)
    驱动频率 ω_d(t) 接近某通道固有频率 ω_n 时该通道被放大（阻尼 γ=0.3）。

(b) 3D 驻波调制（位置感知，物理版 RoPE）
    SW_{m}(t,x,y) = cos(2π m_t·t / T_ref) · cos(2π m_x·x / X_ref) · cos(2π m_y·y / Y_ref)
    前向波 + 后向波叠加成驻波，振子时空坐标充当空间坐标。
    N_ref 是参考长度，与运行时序列长度解耦（P1-1 修复口径）。

关键：这不是普通 FFN。线性层 W₁/W₂ 只是「共振选择后的场」的载体，
真正的非线性来自洛伦兹共振的频率选择性 + 驻波的位置调制。
"""
```

---

## 2. 导入（第19-22行）

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## 3. `lorentzian_gain` 函数（第25-36行）

```python
def lorentzian_gain(omega_d: torch.Tensor, omega_n: torch.Tensor, gamma: float = 0.3):
    """A(ω_d, ω_n) = 1 / sqrt((ω_d²-ω_n²)² + (2γ ω_d)²)

    omega_d: [B] 或标量，驱动频率（时间场给出）
    omega_n: [hidden]，各通道固有频率（可学习）
    返回 [B, hidden]
    """
    od = omega_d.unsqueeze(-1) if omega_d.dim() > 0 else omega_d.reshape(1, 1)
    # device 对齐：omega_n 可能是 Parameter 在 CPU，omega_d 可能在 CUDA
    omega_n = omega_n.to(od.device)
    denom = torch.sqrt((od ** 2 - omega_n.unsqueeze(0) ** 2) ** 2 + (2 * gamma * od) ** 2 + 1e-12)
    return 1.0 / denom
```

- gamma=0.3：阻尼系数
- `omega_n` 是 `nn.Parameter(torch.linspace(0.5, 4.0, ffn_hidden))`（第124行），ffn_hidden=256
- 当 omega_d = omega_n 时，gain = 1/(2γ·omega_d)，共振峰
- device 对齐处理：omega_n 可能在 CPU，omega_d 在 CUDA

omega_n 初始化：linspace(0.5, 4.0, 256)，即 256 个从 0.5 到 4.0 均匀分布的固有频率。

omega_d 范围（来自 TimeField.drive_freq）：初始 0.75~2.75。所以 omega_d 只会命中 omega_n 的一部分通道（0.5~4.0 中的 0.75~2.75 区间）。

---

## 4. `StandingWave3D` 类（第39-83行）

### 4.1 `__init__`（第47-55行）

```python
def __init__(self, n_modes_t=4, n_modes_xy=4, T_ref=20.0, X_ref=32.0, Y_ref=32.0):
    super().__init__()
    self.n_modes_t = n_modes_t
    self.n_modes_xy = n_modes_xy
    self.T_ref = T_ref
    self.X_ref = X_ref
    self.Y_ref = Y_ref
    # 总模式数
    self.n_modes = n_modes_t * n_modes_xy * n_modes_xy
```

- n_modes_t=4, n_modes_xy=4 → n_modes = 4×4×4 = 64
- 参考长度固定：T_ref=20.0, X_ref=32.0, Y_ref=32.0（P1-1 口径，与运行时长度解耦）

### 4.2 `forward` 方法（第57-83行）

```python
def forward(self, t_coord: torch.Tensor, s_coord: torch.Tensor):
    """
    t_coord: [N] 数据时间坐标（整数步 0..T_step-1）
    s_coord: [N,2] 归一化空间坐标 ∈[0,1]
    返回 [N, n_modes]
    """
    N = t_coord.shape[0]
    device, dtype = t_coord.device, t_coord.dtype

    mt = torch.arange(1, self.n_modes_t + 1, device=device, dtype=dtype)  # [Mt]
    mx = torch.arange(1, self.n_modes_xy + 1, device=device, dtype=dtype)
    my = torch.arange(1, self.n_modes_xy + 1, device=device, dtype=dtype)

    # 时间驻波 [N, Mt]
    # t 坐标归一化到 [0,1] 再乘 T_ref 尺度
    t_norm = t_coord / max(t_coord.max().item(), 1.0)
    sw_t = torch.cos(2 * math.pi * mt.unsqueeze(0) * t_norm.unsqueeze(1) / 1.0)  # [N, Mt]

    # 空间驻波 [N, Mxy] ×2
    x = s_coord[:, 0:1]  # [N,1]
    y = s_coord[:, 1:2]
    sw_x = torch.cos(2 * math.pi * mx.unsqueeze(0) * x / 1.0)  # [N, Mxy]
    sw_y = torch.cos(2 * math.pi * my.unsqueeze(0) * y / 1.0)

    # 三维外积 [N, Mt, Mxy, Mxy]
    sw = sw_t.unsqueeze(-1).unsqueeze(-1) * sw_x.unsqueeze(1).unsqueeze(-1) * sw_y.unsqueeze(1).unsqueeze(2)
    return sw.reshape(N, -1)  # [N, n_modes]
```

计算细节：
- `mt = [1,2,3,4]`, `mx = [1,2,3,4]`, `my = [1,2,3,4]`
- `t_norm = t_coord / max(t_coord)`：t_coord ∈ [0, T_step−1]，t_norm ∈ [0,1]
- `sw_t = cos(2π · mt · t_norm)`：[N, 4]
- `sw_x = cos(2π · mx · x)`：[N, 4]，x 已归一化到 [0,1]
- `sw_y = cos(2π · my · y)`：[N, 4]
- 三维外积 → [N, 4, 4, 4] → reshape → [N, 64]

**注意**：文件头公式写 `cos(2π m_t·t / T_ref)`，但代码实际是 `cos(2π · mt · t_norm / 1.0)`，其中 t_norm = t_coord/max(t_coord)。即代码没有用 T_ref，而是用运行时 t_coord 的最大值归一化。T_ref/X_ref/Y_ref 在 `__init__` 中存储但 `forward` 未使用。这是文件头与代码的不一致。

同理，空间驻波 `cos(2π · mx · x / 1.0)`，x 已是 s_coord[:,0]（归一化到 [0,1]），也没有用 X_ref/Y_ref。

这意味着：如果运行时 T_step 变化（如从 6 变到 20），t_norm 的归一化基准变化，同一个 mt 对应的绝对波数会变化。这与文件头声明的「N_ref 是参考长度，与运行时序列长度解耦（P1-1 修复口径）」矛盾。

---

## 5. `StandingWaveFFN` 类（第86-163行）

### 5.1 `__init__`（第97-129行）

```python
def __init__(
    self,
    d_model: int = 64,
    ffn_hidden: int = 256,
    gamma: float = 0.3,
    n_modes_t: int = 4,
    n_modes_xy: int = 4,
    T_ref: float = 20.0,
    X_ref: float = 32.0,
    Y_ref: float = 32.0,
):
    super().__init__()
    self.d_model = d_model
    self.ffn_hidden = ffn_hidden
    self.gamma = gamma

    # 输入是复值场实虚部拼接：2D -> hidden
    self.w1 = nn.Linear(2 * d_model, ffn_hidden, bias=True)
    # 输出两路：[log 振幅残差, 相位残差]
    self.w2 = nn.Linear(ffn_hidden, 2 * d_model, bias=True)
    # 残差零初始化：初始 log_res=ph_res=0，振幅与相位完全由物理演化
    # （Kuramoto/初值）决定，FFN 从恒等映射出发逐步学到修正 —— 标准残差
    # 网络做法，避免随机初始化的相位残差一开始就把动画打散
    nn.init.zeros_(self.w2.weight)
    nn.init.zeros_(self.w2.bias)

    # 可学习固有频率谱（洛伦兹共振用）
    self.natural_freq = nn.Parameter(torch.linspace(0.5, 4.0, ffn_hidden))

    # 3D 驻波
    self.sw3d = StandingWave3D(n_modes_t, n_modes_xy, T_ref, X_ref, Y_ref)
    # 驻波 [N, n_modes] -> hidden 维投影（可学习）
    self.sw_proj = nn.Linear(self.sw3d.n_modes, ffn_hidden, bias=False)
```

参数：
- `w1`: Linear(2D=128, 256)，输入复值场实虚部
- `w2`: Linear(256, 2D=128)，输出 log 振幅残差 + 相位残差
- `w2` 零初始化：残差从恒等映射出发
- `natural_freq`: linspace(0.5, 4.0, 256)，洛伦兹共振用
- `sw_proj`: Linear(64, 256, bias=False)，驻波基投影

参数量计算（d_model=64, ffn_hidden=256）：
- w1: 128×256 + 256 = 33,280
- w2: 256×128 + 128 = 32,896
- natural_freq: 256
- sw_proj: 64×256 = 16,384
- 单块 StandingWaveFFN 总计: 82,816

12 块共享？不，每个 PhysicsBlock 各自实例化一个 StandingWaveFFN（block.py:53-54），所以 12×82,816 = 993,792。但 a_mu 是共享的（block.py:209）。

### 5.2 `forward` 方法（第131-162行）

```python
def forward(self, a: torch.Tensor, theta: torch.Tensor, omega_d: torch.Tensor,
            t_coord: torch.Tensor, s_coord: torch.Tensor):
    """
    a:       [B,N,D] 振幅
    theta:   [B,N,D] 相位
    omega_d: 标量/张量 驱动频率
    t_coord: [N]
    s_coord: [N,2]
    返回 [B,N,2D]：前 D = log 振幅残差，后 D = 相位残差
    """
    B, N, D = a.shape

    # 复值场实虚部
    z_re = a * torch.cos(theta)  # [B,N,D]
    z_im = a * torch.sin(theta)
    z = torch.cat([z_re, z_im], dim=-1)  # [B,N,2D]

    # W₁ 线性
    h = self.w1(z)  # [B,N,H]

    # (a) 洛伦兹共振增益 [.,H] -> 广播到 [B,N,H]
    gain = lorentzian_gain(omega_d.to(h.dtype), self.natural_freq, self.gamma)
    h = h * gain.unsqueeze(1)

    # (b) 3D 驻波调制 [N,H]
    sw = self.sw3d(t_coord, s_coord)  # [N, n_modes]
    sw_h = self.sw_proj(sw)  # [N, H]
    h = h * sw_h.unsqueeze(0)

    # σ + W₂
    out = self.w2(F.gelu(h))  # [B,N,D]
    return out
```

执行流程：
1. `z = cat([a·cosθ, a·sinθ])` → [B,N,2D]
2. `h = w1(z)` → [B,N,256]
3. `gain = lorentzian_gain(omega_d, natural_freq)` → [1,256] 或 [B,256]
4. `h = h * gain` → 频率选择性放大
5. `sw = sw3d(t_coord, s_coord)` → [N,64]
6. `sw_h = sw_proj(sw)` → [N,256]
7. `h = h * sw_h` → 位置调制
8. `out = w2(gelu(h))` → [B,N,2D]

注意：第161行 `out = self.w2(F.gelu(h))`，注释说返回 [B,N,D] 但实际是 [B,N,2D]（w2 输出 2*d_model）。注释笔误。

---

## 6. 输出如何被 PhysicsBlock 使用

`block.py:117-118`：
```python
ffn_out = self.ffn(a, theta_new, omega_d, t_coord, s_coord)  # [B,N,2D]
log_res, ph_res = ffn_out.chunk(2, dim=-1)
```

`block.py:149`：
```python
res = 2.0 * torch.tanh(log_res / 2.0)
```

`block.py:179`：
```python
theta_new = theta_new + ph_res.clamp(-1.5, 1.5)
```

- 前 D=125 维是 log 振幅残差，经 tanh 软饱和（res ∈ [−2,2]）
- 后 D=125 维是相位残差，clamp 到 ±1.5

---

## 7. 与 `cond_field.py` 中 StandingWave3D 的复用

`cond_field.py:45`：
```python
from .standing_wave_ffn import StandingWave3D
```

`cond_field.py:73`：
```python
self.sw3d = StandingWave3D(n_modes_t, n_modes_xy, 20.0, 32.0, 32.0)
```
其中 n_modes_t=3, n_modes_xy=3（cond_field.py:62-63），所以 n_basis = 3×3×3 = 27。

对比主物理核的 n_modes_t=4, n_modes_xy=4 → 64 个基。cond_field 降到 27 个基，注释（cond_field.py:69-72）：
「生成通路的 GPU 峰值实测 3.41GB/4GB（贴边），瓶颈是驻波基与场的中间张量，不是 drift chunk。降到 27 基后峰值回落到 2.7GB 区间」。

---

## 8. 文件头与代码的不一致

| 项 | 文件头声明 | 代码实际 |
|---|---|---|
| 驻波公式 | `cos(2π m_t·t / T_ref)` | `cos(2π · mt · t_norm / 1.0)`，t_norm = t_coord/max(t_coord)（第72-73行） |
| T_ref/X_ref/Y_ref | 「参考长度固定，与运行时长度解耦」 | `__init__` 存储但 `forward` 未使用（第51-53行 vs 第57-83行） |
| 输出形状注释 | `返回 [B,N,D]`（第139行） | 实际 [B,N,2D]（w2 输出 2*d_model） |
| 激活函数 | `σ(...)`（文件头） | `F.gelu(h)`（第161行） |

---

## 9. 驻波计算的数值行为

以 TS=6, GRID=32 为例（t_coord ∈ [0,5], s_coord ∈ [0,1]）：

**时间驻波**（t_norm = t_coord/5）：
- t_coord=0: t_norm=0, sw_t = cos(0) = [1,1,1,1]
- t_coord=1: t_norm=0.2, sw_t = cos(2π×[1,2,3,4]×0.2) = cos([0.4π, 0.8π, 1.2π, 1.6π]) = [0.309, −0.809, −0.809, 0.309]
- t_coord=2: t_norm=0.4, sw_t = cos([0.8π, 1.6π, 2.4π, 3.2π]) = [−0.809, 0.309, 0.309, −0.809]
- t_coord=3: t_norm=0.6, sw_t = cos([1.2π, 2.4π, 3.6π, 4.8π]) = [−0.809, 0.309, 0.309, −0.809]
- t_coord=4: t_norm=0.8, sw_t = cos([1.6π, 3.2π, 4.8π, 6.4π]) = [0.309, −0.809, −0.809, 0.309]
- t_coord=5: t_norm=1.0, sw_t = cos([2π, 4π, 6π, 8π]) = [1,1,1,1]

**空间驻波**（x = s_coord[:,0]，32 个格点归一化到 [0,1]，步长 1/31）：
- x=0: sw_x = [1,1,1,1]
- x=1/31≈0.0323: sw_x = cos(2π×[1,2,3,4]×0.0323) = cos([0.203, 0.406, 0.608, 0.811]) ≈ [0.980, 0.919, 0.821, 0.689]
- x=0.5: sw_x = cos(π×[1,2,3,4]) = [−1, 1, −1, 1]
- x=1: sw_x = cos(2π×[1,2,3,4]) = [1,1,1,1]

驻波值域 [−1,1]，sw_proj 是无 bias 的 Linear(64, 256)。

---

## 10. 洛伦兹共振的数值行为

omega_d 范围（TimeField.drive_freq 初始）：0.75~2.75
natural_freq：linspace(0.5, 4.0, 256)，步长 (4.0−0.5)/255 ≈ 0.0137

gain = 1/sqrt((ω_d²−ω_n²)² + (2γ ω_d)²)，γ=0.3

以 ω_d=1.0 为例：
- ω_n=1.0 附近：gain ≈ 1/(2×0.3×1.0) = 1/0.6 = 1.667（共振峰）
- ω_n=0.5：denom = sqrt((1−0.25)² + 0.36) = sqrt(0.5625+0.36) = sqrt(0.9225) = 0.960, gain = 1.042
- ω_n=2.0：denom = sqrt((1−4)² + 0.36) = sqrt(9+0.36) = 3.059, gain = 0.327
- ω_n=4.0：denom = sqrt((1−16)² + 0.36) = sqrt(225+0.36) = 15.012, gain = 0.0666

即 ω_d 只放大与之接近的 natural_freq 通道，远处的被抑制。omega_d 随时间从 0.75 扫到 2.75，所以不同物理时间会激活不同的频率通道组合。

---

## 11. 未验证项

1. T_ref/X_ref/Y_ref 未在 forward 中使用，变长（T_step 从 6 到 20）时驻波基的绝对波数会变化，与文件头声明的 P1-1 修复口径矛盾。无跨时长迁移实验验证这一点的实际影响。
2. `natural_freq` 作为可学习参数，训练中是否有效改变无实验记录。
3. w2 零初始化的实际效果（对比非零初始化）无消融实验。
4. 洛伦兹共振增益的具体激活模式无诊断数据。
5. n_modes_t=4, n_modes_xy=4（64 基）与 cond_field 的 3×3×3（27 基）的表达力差异无定量对比。

---

## 12. 代码事实汇总

| 项 | 值 | 来源 |
|---|---|---|
| d_model | 64（默认） | standing_wave_ffn.py:99 |
| ffn_hidden | 256 | standing_wave_ffn.py:100 |
| gamma | 0.3 | standing_wave_ffn.py:101 |
| n_modes_t / n_modes_xy | 4 / 4 → 64 基 | standing_wave_ffn.py:102-103 |
| T_ref / X_ref / Y_ref | 20.0 / 32.0 / 32.0（存储未用） | standing_wave_ffn.py:104-106 |
| w1 | Linear(128, 256) | standing_wave_ffn.py:114 |
| w2 | Linear(256, 128)，零初始化 | standing_wave_ffn.py:116, 120-121 |
| natural_freq | linspace(0.5, 4.0, 256) | standing_wave_ffn.py:124 |
| sw_proj | Linear(64, 256, bias=False) | standing_wave_ffn.py:129 |
| 激活函数 | F.gelu | standing_wave_ffn.py:161 |
| t_norm | t_coord / max(t_coord) | standing_wave_ffn.py:72 |
| 空间坐标 | s_coord 已归一化 [0,1] | standing_wave_ffn.py:76-79 |
| 单块参数量 | 82,816 | 计算值 |
| 12块总计 | 993,792 | 计算值 |
