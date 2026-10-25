# 02 · DriftField 漂移场源码逐行分析

源文件：`src/physics/drift_field.py`（257行）
调用点：`block.py:103`（`g = self.drift(a, theta, targets)`）

---

## 1. 文件头声明（第1-26行）

```python
"""
§4.4 DriftField —— 漂移场

  V = V⁺ − λ·V⁻
  V⁺ = Σ_k softmax_τ(−‖z_i − z_k^tar‖²)·(z_k^tar − z_i)    （吸引：向目标靠拢）
  V⁻ = Σ_j softmax_τ(−‖z_i − z_j‖²)·(z_j − z_i)            （排斥：防坍缩）

- 多温度 τ 求和取多尺度
- λ = 0.5（排斥权重）
- 训练时目标是标签/条件嵌入；推理时用可学习原型作吸引中心
- 距离在实+虚拼接的 2D 空间上定义

显存工程（不改动任何物理公式）：
  全 pairwise 距离矩阵是 O(N·M) 的；训练时吸引中心 targets 也是 N 个振子
  （M = N 可达 10240+），单个 [B,N,N] fp32 就 400MB，连续物化 + 反向必爆
  4GB 显存。采用「分块两遍 + 梯度检查点」：
    Pass 1 分块求每行 i 的全局 logsumexp（softmax 归一化常数 logZ_i）
    Pass 2 分块累加 Σ_j w_ij·y_j 与 Σ_j w_ij，w_ij = exp(l_ij − logZ_i)
  数学与全物化严格等价（标准 logsumexp 分块合并，见 test_drift_chunk.py），
  峰值显存从 O(N·M) 降到 O(N·chunk)。反向由 checkpoint 整体重算，
  不保存任何 [N,M] 中间矩阵。

数值要求（必须 fp32）：z=ae^{iθ} 是 2D 维、元素 ±a（max≈8），‖z‖² 求和可达 ~10⁵，
超 fp16 上限 65504 直接溢出；且三数相减有灾难性抵消。距离全程升 fp32。
"""
```

事实陈述：
- 漂移场 = 吸引分量 V⁺ − 排斥分量 V⁻，λ=0.5
- 训练时 targets 是目标场（M=N），推理时 targets=None，用可学习 prototypes（16个）
- 必须 fp32：‖z‖² 可达 ~10⁵，fp16 上限 65504 会溢出
- N=10240 时 [B,N,N] fp32 = 400MB，4GB 卡放不下，所以分块

---

## 2. 导入（第27-30行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
```

---

## 3. `_pairwise_logits` 函数（第33-36行）

```python
def _pairwise_logits(x_sq, y_sq, x, y_chunk, tau):
    """logits_ij = −(‖x_i‖² + ‖y_j‖² − 2 x_i·y_j)/τ，fp32。"""
    cross = torch.einsum("bnc,bmc->bnm", x, y_chunk)
    return -(x_sq.unsqueeze(2) + y_sq.unsqueeze(1) - 2.0 * cross) / tau
```

- `x_sq`: [B,N]，`y_sq`: [B,M_chunk]
- `cross`: [B,N,M_chunk]，x_i·y_j 内积
- 输出 logits_ij: [B,N,M_chunk]
- 这是标准欧氏距离展开式：‖x−y‖² = ‖x‖² + ‖y‖² − 2x·y

---

## 4. `_chunk_lse` 函数（第39-42行）

```python
def _chunk_lse(l):
    """块内 log Σ_j exp(l_ij) = m + log(Σ exp(l−m))。"""
    m = l.amax(-1)
    return m + torch.log((l - m.unsqueeze(-1)).exp().sum(-1) + 1e-30)
```

- 数值稳定的 logsumexp：先减最大值 m，再 exp、求和、log
- `+1e-30` 防 log(0)

---

## 5. `_merge_lse` 函数（第45-48行）

```python
def _merge_lse(z1, z2):
    """合并两个 logsumexp：log(e^z1 + e^z2)。"""
    m = torch.maximum(z1, z2)
    return m + torch.log((z1 - m).exp() + (z2 - m).exp() + 1e-30)
```

- 分块合并的核心：两个块的 logsumexp 如何合并成一个全局 logsumexp
- 这是 `test_drift_chunk.py` 要验证的数学等价性

---

## 6. `softmax_kernel_field` 函数（第51-95行）

```python
def softmax_kernel_field(x, y, tau, self_mask=False, chunk=2048):
    """V_i = Σ_j softmax_j(−‖x_i−y_j‖²/τ)·(y_j − x_i)

    分块两遍，与全物化严格等价：
      V_i = Σ_j w_ij·y_j − x_i·Σ_j w_ij，  w_ij = softmax_j(−d²_ij/τ)

    x: [B,N,C]，y: [B,M,C]
    self_mask: x 与 y 同源时屏蔽 i==j（排斥项用，对角 logits 填 −1e4）
    返回 [B,N,C]，fp32
    """
    B, N, C = x.shape
    M = y.shape[1]
    xf, yf = x.float(), y.float()
    x_sq = (xf ** 2).sum(-1)   # [B,N]
    y_sq = (yf ** 2).sum(-1)   # [B,M]

    gi = torch.arange(N, device=x.device) if self_mask else None

    # ---- Pass 1：每行 i 的全局 logZ_i = log Σ_j exp(l_ij) ----
    logZ = None
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xf, yf[:, cs:ce], tau)
        if self_mask:
            cj = torch.arange(cs, ce, device=x.device)
            l = l.masked_fill((gi.unsqueeze(1) == cj.unsqueeze(0)).unsqueeze(0), -1e4)
        z_blk = _chunk_lse(l)
        logZ = z_blk if logZ is None else _merge_lse(logZ, z_blk)
        del l, z_blk

    # ---- Pass 2：归一化权重加权和 ----
    acc = torch.zeros(B, N, C, device=x.device, dtype=torch.float32)
    wsum = torch.zeros(B, N, device=x.device, dtype=torch.float32)
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xf, yf[:, cs:ce], tau)
        if self_mask:
            cj = torch.arange(cs, ce, device=x.device)
            l = l.masked_fill((gi.unsqueeze(1) == cj.unsqueeze(0)).unsqueeze(0), -1e4)
        w = (l - logZ.unsqueeze(-1)).exp()  # [B,N,chunk]
        acc = acc + torch.einsum("bnm,bmc->bnc", w, yf[:, cs:ce])
        wsum = wsum + w.sum(-1)
        del l, w

    return acc - xf * wsum.unsqueeze(-1)
```

数学等价性推导：
- softmax_j(−d²_ij/τ) = exp(l_ij) / Σ_j exp(l_ij) = exp(l_ij − logZ_i)
- V_i = Σ_j w_ij·(y_j − x_i) = Σ_j w_ij·y_j − x_i·Σ_j w_ij = acc_i − x_i·wsum_i
- Pass 1 算 logZ_i，Pass 2 算 acc_i 和 wsum_i

`self_mask` 机制：
- 排斥项（x 和 y 同源）时，i==j 的对角线要屏蔽
- 将对角 logits 填 −1e4，exp(−1e4−logZ) ≈ 0

chunk 默认 2048，v4200 中降到 512（`train_multimodal.py:59`：「单块 logits 张量从 201MB 降到 48MB」）。

---

## 7. `softmax_kernel_phase` 函数（第98-153行）

这是修复「旧实现死穴」的关键函数。

```python
def softmax_kernel_phase(x, y, theta, theta_tar, tau, chunk=2048):
    """相位空间漂移吸引：

      g_i = Σ_j softmax_j(−‖ẑ_i − ȳ_j‖²/τ) · sin(θ_j^tar − θ_i)

    x: [B,N,C] 当前复值场，y: [B,M,C] 目标复值场
    theta: [B,N,D] 当前相位，theta_tar: [B,M,D] 目标相位，C = 2D

    两点关键设计（修掉旧实现的死穴）：
    (1) 距离在 L2 归一化后的复值场上算。‖z‖ = a·√D，模型 a 与 target a
        相差可达 90 倍，未归一化时 logits 被 ‖z_i‖² 主导、y_j 之间差异被
        淹没，softmax 退化为均匀分布，吸引场退化成 −z_i（拉向原点），
        完全丢失 target 相位信息。归一化后 ẑ·ȳ = mean_d cos(θ_i−θ_j)，
        距离自动成为相位余弦距离，与振幅量级无关。
    (2) 力用 sin(θ_j^tar − θ_i)：周期、有界 ∈[−1,1]、天然是相位流形上的
        梯度方向（θ_i 追上 θ_j^tar 时归零，稳定）。用复值场差 (y_j − x_i)
        会被振幅量级失配污染成"压振幅"而非"对齐相位"。

    sin(θ_j − θ_i) = sinθ_j cosθ_i − cosθ_j sinθ_i，分块累加复用 logsumexp 框架。
    返回 [B,N,D]，fp32。
    """
    B, N, C = x.shape
    M = y.shape[1]
    D = theta.shape[-1]
    # L2 归一化：消除振幅量级失配
    xn = F.normalize(x.float(), dim=-1)   # [B,N,C]
    yn = F.normalize(y.float(), dim=-1)   # [B,M,C]
    x_sq = (xn ** 2).sum(-1)              # ≈1
    y_sq = (yn ** 2).sum(-1)              # ≈1

    sin_tar = torch.sin(theta_tar).float()  # [B,M,D]
    cos_tar = torch.cos(theta_tar).float()
    sin_i = torch.sin(theta).float()        # [B,N,D]
    cos_i = torch.cos(theta).float()

    # ---- Pass 1：每行 i 的全局 logZ_i ----
    logZ = None
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xn, yn[:, cs:ce], tau)
        z_blk = _chunk_lse(l)
        logZ = z_blk if logZ is None else _merge_lse(logZ, z_blk)
        del l, z_blk

    # ---- Pass 2：g_i = cosθ_i·(Σ_j w_ij sinθ_j) − sinθ_i·(Σ_j w_ij cosθ_j) ----
    acc_sin = torch.zeros(B, N, D, device=x.device, dtype=torch.float32)
    acc_cos = torch.zeros(B, N, D, device=x.device, dtype=torch.float32)
    for cs in range(0, M, chunk):
        ce = min(cs + chunk, M)
        l = _pairwise_logits(x_sq, y_sq[:, cs:ce], xn, yn[:, cs:ce], tau)
        w = (l - logZ.unsqueeze(-1)).exp()  # [B,N,chunk]
        acc_sin = acc_sin + torch.einsum("bnm,bmd->bnd", w, sin_tar[:, cs:ce])
        acc_cos = acc_cos + torch.einsum("bnm,bmd->bnd", w, cos_tar[:, cs:ce])
        del l, w

    return cos_i * acc_sin - sin_i * acc_cos
```

两个关键设计的实测依据（代码注释原文）：

**设计(1)：L2 归一化**
- 问题：‖z‖ = a·√D，模型 a 与 target a 相差可达 90 倍（注释原文）
- 未归一化时 logits 被 ‖z_i‖² 主导，softmax 退化为均匀分布
- 归一化后 ẑ·ȳ = mean_d cos(θ_i−θ_j)，距离成为相位余弦距离
- `encode_gif_to_field` 中 NORM=10.0 把 target 均值对齐到 ~0.3，模型 softplus 输出 ~0.3-0.7，仍差 90 倍（`gif_codec.py:266-270`）

**设计(2)：力用 sin(θ_j^tar − θ_i)**
- 周期、有界 ∈[−1,1]
- θ_i 追上 θ_j^tar 时归零，稳定
- 用复值场差 (y_j − x_i) 会被振幅量级失配污染成「压振幅」

三角恒等式应用：
- sin(θ_j − θ_i) = sinθ_j·cosθ_i − cosθ_j·sinθ_i
- g_i = Σ_j w_ij·sin(θ_j^tar − θ_i)
     = cosθ_i·(Σ_j w_ij·sinθ_j^tar) − sinθ_i·(Σ_j w_ij·cosθ_j^tar)
     = cos_i·acc_sin − sin_i·acc_cos

---

## 8. `pairwise_sq_dist` 函数（第156-172行）

```python
def pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor):
    """dist²_nm = ‖x_n‖² + ‖y_m‖² − 2 x_n y_mᵀ

    x: [B, N, C]
    y: [B, M, C] 或 [M, C]
    返回 [B, N, M]
    全程 fp32（数值护栏）。
    """
    x = x.float()
    if y.dim() == 2:
        y = y.unsqueeze(0).expand(x.shape[0], -1, -1).float()
    else:
        y = y.float()
    x_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [B,N,1]
    y_sq = (y ** 2).sum(dim=-1, keepdim=True)  # [B,M,1]
    cross = torch.einsum("bnc,bmc->bnm", x, y)
    return x_sq + y_sq.transpose(1, 2) - 2.0 * cross
```

支持 y 是 [M,C]（无 batch 维）的情况，自动 expand。

---

## 9. `DriftField` 类定义（第175-256行）

### 9.1 `__init__`（第178-191行）

```python
def __init__(self, d_model: int = 64, n_prototypes: int = 16, lam: float = 0.5,
             n_scales: int = 2, tau: float = 1.0,
             chunk: int = 2048, use_checkpoint: bool = True):
    super().__init__()
    self.d_model = d_model
    self.lam = lam
    self.n_scales = n_scales
    self.tau = tau
    self.chunk = chunk
    self.use_checkpoint = use_checkpoint
    # 推理时用可学习原型作吸引中心
    self.prototypes = nn.Parameter(torch.randn(n_prototypes, 2 * d_model) * 0.02)
    # 直接目标吸引分量的权重（可学习）
    self.direct_weight = nn.Parameter(torch.tensor(1.0))
```

参数：
- `n_prototypes=16`：推理时的可学习原型数
- `lam=0.5`：排斥权重（注意：forward 中实际未使用 lam，见下文）
- `n_scales=2`：多温度尺度数，tau_s = tau × 2^s
- `tau=1.0`：基础温度
- `chunk=2048`：分块大小（v4200 中 DriftField chunk 用 512）
- `prototypes`：`randn(16, 2*d_model) * 0.02`，初始 std=0.02
- `direct_weight`：初始化为 1.0 的可学习标量

**注意**：`lam` 在 `__init__` 中存储但 `forward` 中未使用。文件头说 `λ=0.5（排斥权重）`，但 forward 只返回 g（相位漂移引导），排斥项 `_repel` 定义了却未在 forward 中调用。代码事实：排斥分量未实际执行。

### 9.2 `_to_complex_field` 静态方法（第193-196行）

```python
@staticmethod
def _to_complex_field(a: torch.Tensor, theta: torch.Tensor):
    """复值场 -> 实虚拼接 [B, N, 2D]。"""
    return torch.cat([a * torch.cos(theta), a * torch.sin(theta)], dim=-1)
```

z = a·cosθ + i·a·sinθ，实虚拼接成 [B,N,2D]。

### 9.3 三个私有方法（第198-206行）

```python
def _attract(self, z, tar, tau_s):
    return softmax_kernel_field(z, tar, tau_s, self_mask=False, chunk=self.chunk)

def _attract_phase(self, z, tar, theta, theta_tar, tau_s):
    """相位空间吸引：g_i = Σ_j softmax_j(−相位距离/τ)·sin(θ_j^tar−θ_i)。"""
    return softmax_kernel_phase(z, tar, theta, theta_tar, tau_s, chunk=self.chunk)

def _repel(self, z, tau_s):
    return softmax_kernel_field(z, z, tau_s, self_mask=True, chunk=self.chunk)
```

- `_attract`：复值场空间的吸引（未在 forward 中使用）
- `_attract_phase`：相位空间的吸引（forward 实际使用）
- `_repel`：排斥（定义了但未在 forward 中调用）

### 9.4 `forward` 方法（第208-256行）

```python
def forward(self, a: torch.Tensor, theta: torch.Tensor,
            targets: torch.Tensor = None):
    """
    a:     [B,N,D] 振幅
    theta: [B,N,D] 相位
    targets: [B,M,2D] 或 None。
            训练时=目标复值场（M=N）；推理时=None，用可学习 prototypes。

    返回 g: [B,N,D] 相位漂移引导，进入 Kuramoto 的 κ·g 通路。

    漂移场是模型获取 target 信息的唯一通路（初始 a0/θ0 是噪声、ω 确定性）：
    相位引导必须在相位空间上定义，否则振幅量级失配会把吸引场污染成
    "压振幅"，target 相位信息进不了相位演化，loss 贴随机地板。
    """
    z = self._to_complex_field(a, theta)  # [B,N,2D]

    # 吸引中心 + 目标相位
    if targets is None:
        tar = self.prototypes.unsqueeze(0).expand(z.shape[0], -1, -1).contiguous()
    else:
        tar = targets
        if tar.dim() == 2:
            tar = tar.unsqueeze(0).expand(z.shape[0], -1, -1).contiguous()
    re_t, im_t = tar[..., :self.d_model], tar[..., self.d_model:]
    theta_tar = torch.atan2(im_t, re_t)  # [B,M,D]

    # 训练时 M=N，[B,N,N] 爆显存 → 分块 + 梯度检查点（推理/no_grad 时无需检查点）
    use_ckpt = self.use_checkpoint and self.training and z.requires_grad

    g = torch.zeros_like(theta)
    for s in range(self.n_scales):
        tau_s = self.tau * (2 ** s)
        if use_ckpt:
            g = g + checkpoint(self._attract_phase, z, tar, theta, theta_tar,
                               tau_s, use_reentrant=False)
        else:
            g = g + self._attract_phase(z, tar, theta, theta_tar, tau_s)

    # 直接目标吸引分量：sin(θ_i^tar − θ_i)，逐点精确注入 target 相位。
    # 空间近邻平均的对齐上限只有 0.42（实测 κ 响应饱和曲线）——FFT 系数相位
    # 在数据空间上不平滑，近邻平均把相邻块的不同相位平均掉，引入不可消除的
    # 结构误差。直接分量把零训练 cos 提到 0.998。物理上外场既可直接作用在
    # 每个振子上，也可通过振子间耦合传播，两分量叠加。
    # 仅训练时加（targets 逐点已知且 M=N）；生成时 targets=None 走原型通路。
    if targets is not None and tar.shape[1] == theta.shape[1]:
        g = g + self.direct_weight * torch.sin(theta_tar - theta)

    # 排斥项仍在复值场空间（防振幅坍缩），不进相位通路
    return g.to(a.dtype)
```

执行流程：
1. `z = _to_complex_field(a, theta)` → [B,N,2D]
2. 确定 `tar`（吸引中心）：
   - targets=None → 用 prototypes（expand 到 batch）
   - targets 给定 → 直接用；若 targets 是 [M,2D] 无 batch 维，则 expand
3. 从 tar 提取 theta_tar = atan2(im_t, re_t)
4. `use_ckpt` 条件：`use_checkpoint AND self.training AND z.requires_grad`
5. 多尺度吸引（n_scales=2，tau_s = 1.0 和 2.0）：
   - 每个尺度调用 `_attract_phase`（即 `softmax_kernel_phase`）
   - 若 use_ckpt，用 `checkpoint()` 包裹
6. 直接目标吸引分量（仅训练时 targets 非 None 且 M==N）：
   - `g += direct_weight * sin(theta_tar - theta)`
7. 返回 g（相位漂移引导），`g.to(a.dtype)`

**关键实测数据（代码注释第246-250行）**：
- 「空间近邻平均的对齐上限只有 0.42（实测 κ 响应饱和曲线）」
- 「FFT 系数相位在数据空间上不平滑，近邻平均把相邻块的不同相位平均掉」
- 「直接分量把零训练 cos 提到 0.998」

即：不加 direct_weight 分量时，softmax 近邻平均的相位对齐上限是 cos=0.42；加上逐点直接分量后，零训练（未训练）时 cos 就能到 0.998。

---

## 10. DriftField 的输出如何进入 Kuramoto

`block.py:103`：
```python
g = self.drift(a, theta, targets)
```

`block.py:106-108`：
```python
theta_new, sync_diag = self.kuramoto(
    theta, a, omega, g, kappa, T, t_coord, s_coord
)
```

`kuramoto_sync.py:237`：
```python
dtheta = omega + coupling_struct + coupling_mf + kappa * g
```

即 g 以 κ(t)·g 的形式进入相位演化方程。κ(t) 由 TimeField 给出，范围 1.907~15.851（见文档 01）。

---

## 11. v4200 中 DriftField 的改动

v4200 的 `drift_field.py` 与主目录的差异需实际比对。根据交接文档，v4200 物理核「一行未改」，DriftField chunk 参数在训练脚本中设为 512（`train_multimodal.py:59`）。

代码注释第14-22行提到的显存工程数据：
- 全 pairwise [B,N,N] fp32 = 400MB（N=10240 时）
- 分块后峰值 O(N·chunk)
- chunk=2048 时单块 logits 201MB
- chunk=512 时单块 logits 48MB

---

## 12. 未验证项与代码事实

### 12.1 代码事实

| 项 | 值 | 来源 |
|---|---|---|
| n_prototypes | 16 | drift_field.py:179 |
| lam | 0.5（存储但 forward 未用） | drift_field.py:179, 208-256 |
| n_scales | 2 | drift_field.py:180 |
| tau | 1.0（多尺度：1.0, 2.0） | drift_field.py:180, 239 |
| chunk | 2048（v4200 训练用 512） | drift_field.py:180 |
| prototypes 初始化 | randn(16, 2D) × 0.02 | drift_field.py:189 |
| direct_weight 初始化 | 1.0 | drift_field.py:191 |
| prototypes 实测 std | 0.01963（best.pt，几乎未动） | cond_field.py:13 |
| 零训练 cos | 0.998（加 direct 分量后） | drift_field.py:249 |
| 空间近邻平均对齐上限 | 0.42 | drift_field.py:247 |
| fp32 必须性 | ‖z‖² 可达 ~10⁵，fp16 上限 65504 | drift_field.py:24 |

### 12.2 未验证项

1. `lam=0.5` 排斥权重：`__init__` 接收并存储，但 `forward` 未调用 `_repel`，排斥分量实际未执行。文件头声称 `V = V⁺ − λ·V⁻`，与代码实现不符。
2. `prototypes` 梯度恒为 0 的问题（`cond_field.py:10-13` 实测）：训练时 targets 永远非 None，`if targets is None` 分支一次都没执行，prototypes std=0.01963 ≈ 初始值 0.02。这是 v4200 新增 CondFieldPredictor 的直接原因。
3. `direct_weight` 作为可学习参数，其实际训练后的值无实验记录。
4. 多尺度 tau（1.0 和 2.0）两个尺度各自的贡献无单独实测数据。

---

## 13. 与 KuramotoSync 的接口

DriftField.forward 返回 g: [B,N,D]，进入 `kuramoto_sync.py:237` 的 `kappa * g` 项。

g 的物理意义：相位空间的漂移引导，即把振子相位拉向目标相位的外力。其量级：
- softmax_kernel_phase 输出：Σ_j w_ij·sin(θ_j^tar−θ_i)，w_ij 是 softmax 权重（和≈1），sin ∈[−1,1]，所以输出 ∈[−1,1]
- direct_weight·sin(theta_tar−theta)：direct_weight=1.0，sin ∈[−1,1]，所以 ∈[−1,1]
- 两个尺度相加，g 的典型量级在 [−2,2]

再乘 κ(t)（1.907~15.851），kappa·g 的量级在 [−32, 32]，随后进入 `omega_max·tanh(dtheta/omega_max)` 软饱和（omega_max=8.0）。
