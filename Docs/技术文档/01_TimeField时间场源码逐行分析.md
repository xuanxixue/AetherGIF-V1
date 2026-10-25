# 01 · TimeField 时间场源码逐行分析

源文件：`src/physics/time_field.py`（107行）
引用文件：`src/physics/block.py:207`（`TimeField(n_blocks=n_blocks)`）、`block.py:240`（`self.time_field.forward(i, dev)`）

---

## 1. 文件头声明（第1-14行）

```python
"""
§4.1 TimeField —— 深度 = 物理时间

t = round / (n_blocks - 1) ∈ [0,1]，即层数编号归一化为物理时间。
三条协议曲线：

  κ(t) = κ_max · σ((t - t_c) / τ_c)          耦合强度课程：混沌 → 锁相
  T(t) = T_max·(1-t)^p + T_min                温度退火：冷却
  ω_d(t) = ω_lo · (ω_hi / ω_lo)^t             驱动频率几何扫描：低频 → 高频

扫描区间端点由可学习固有频率谱 natural_freq 的低/高分位均值给出，
这样时间场课程本身可训练（梯度可回传），否则是死参数。
"""
```

事实陈述：
- 物理时间 `t` 由 block 索引归一化得到：`t = block_idx / (n_blocks - 1)`
- 三条曲线分别控制：耦合强度 κ、温度 T、驱动频率 ω_d
- `natural_freq` 是 `nn.Parameter`，即时间场课程本身可训练

---

## 2. 导入（第15-17行）

```python
import math
import torch
import torch.nn as nn
```

---

## 3. 类定义与 `__init__`（第20-46行）

```python
class TimeField(nn.Module):
    def __init__(
        self,
        n_blocks: int = 12,
        kappa_max: float = 16.0,
        t_c: float = 0.3,
        tau_c: float = 0.15,
        T_max: float = 1.0,
        T_min: float = 0.05,
        p: float = 1.5,
        n_drive_freq: int = 16,
        freq_lo: float = 0.5,
        freq_hi: float = 3.0,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.kappa_max = kappa_max
        self.t_c = t_c
        self.tau_c = tau_c
        self.T_max = T_max
        self.T_min = T_min
        self.p = p

        # 可学习固有频率谱：时间场课程本身可训练（梯度可回传）
        self.natural_freq = nn.Parameter(torch.linspace(freq_lo, freq_hi, n_drive_freq))
```

逐参数说明：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `n_blocks` | 12 | 物理块层数，对应 PMNNCore 的 12 轮演化 |
| `kappa_max` | 16.0 | 耦合强度峰值。注意：`block.py` 中 KuramotoSync 的 `omega_max=8.0`，`kappa_max=16.0` 是 TimeField 层的独立参数 |
| `t_c` | 0.3 | sigmoid 中心点，物理时间 t=0.3 时 κ 达到 κ_max/2 |
| `tau_c` | 0.15 | sigmoid 温度，控制转变陡峭度 |
| `T_max` | 1.0 | 初始温度 |
| `T_min` | 0.05 | 终态温度 |
| `p` | 1.5 | 温度退火幂次 |
| `n_drive_freq` | 16 | 固有频率谱长度 |
| `freq_lo` | 0.5 | 频率下限 |
| `freq_hi` | 3.0 | 频率上限 |

`natural_freq` 初始化：`torch.linspace(0.5, 3.0, 16)`，即 16 个从 0.5 到 3.0 均匀分布的值。

---

## 4. `t_of_block` 方法（第48-52行）

```python
def t_of_block(self, block_idx: int) -> float:
    """第 block_idx 层的物理时间 ∈ [0,1]。"""
    if self.n_blocks <= 1:
        return 0.0
    return block_idx / (self.n_blocks - 1)
```

- n_blocks=12 时，block_idx=0 → t=0.0；block_idx=11 → t=1.0
- 每个 block 的物理时间间隔：1/11 ≈ 0.0909

---

## 5. `kappa` 方法（第54-56行）

```python
def kappa(self, t):
    """κ(t) = κ_max · σ((t - t_c)/τ_c) —— 耦合强度课程。"""
    return self.kappa_max * torch.sigmoid((t - self.t_c) / self.tau_c)
```

计算各 block 的 κ 值（kappa_max=16.0, t_c=0.3, tau_c=0.15）：

| block_idx | t | (t-0.3)/0.15 | sigmoid | κ(t) |
|---|---|---|---|---|
| 0 | 0.0000 | -2.0000 | 0.1192 | 1.907 |
| 1 | 0.0909 | -1.3939 | 0.1988 | 3.181 |
| 2 | 0.1818 | -0.7879 | 0.3127 | 5.003 |
| 3 | 0.2727 | -0.1818 | 0.4547 | 7.275 |
| 4 | 0.3636 | 0.4242 | 0.6044 | 9.671 |
| 5 | 0.4545 | 1.0303 | 0.7369 | 11.790 |
| 6 | 0.5455 | 1.6364 | 0.8373 | 13.397 |
| 7 | 0.6364 | 2.2424 | 0.9037 | 14.459 |
| 8 | 0.7273 | 2.8485 | 0.9450 | 15.120 |
| 9 | 0.8182 | 3.4545 | 0.9693 | 15.508 |
| 10 | 0.9091 | 4.0606 | 0.9830 | 15.728 |
| 11 | 1.0000 | 4.6667 | 0.9907 | 15.851 |

规律：κ 从 1.907 单调增长到 15.851，前 4 个 block 增长快（从 1.9 到 7.3），后 8 个 block 增长趋缓（从 7.3 到 15.9）。

---

## 6. `temperature` 方法（第58-60行）

```python
def temperature(self, t):
    """T(t) = T_max(1-t)^p + T_min —— 温度退火。"""
    return self.T_max * (1.0 - t) ** self.p + self.T_min
```

计算各 block 的 T 值（T_max=1.0, T_min=0.05, p=1.5）：

| block_idx | t | (1-t)^1.5 | T(t) |
|---|---|---|---|
| 0 | 0.0000 | 1.0000 | 1.0500 |
| 1 | 0.0909 | 0.8598 | 0.9098 |
| 2 | 0.1818 | 0.7436 | 0.7936 |
| 3 | 0.2727 | 0.6422 | 0.6922 |
| 4 | 0.3636 | 0.5518 | 0.6018 |
| 5 | 0.4545 | 0.4702 | 0.5202 |
| 6 | 0.5455 | 0.3958 | 0.4458 |
| 7 | 0.6364 | 0.3280 | 0.3780 |
| 8 | 0.7273 | 0.2661 | 0.3161 |
| 9 | 0.8182 | 0.2100 | 0.2600 |
| 10 | 0.9091 | 0.1596 | 0.2096 |
| 11 | 1.0000 | 0.0000 | 0.0500 |

规律：T 从 1.05 单调降到 0.05，模拟退火过程。

---

## 7. `drive_freq` 方法（第62-75行）

```python
def drive_freq(self, t):
    """ω_d(t) = ω_lo · (ω_hi/ω_lo)^t —— 驱动频率几何扫描。

    端点由 natural_freq 低/高分位均值给出（可训练）。
    """
    with torch.no_grad():
        k = max(1, self.natural_freq.numel() // 4)
        sorted_f = torch.sort(self.natural_freq).values
        w_lo = sorted_f[:k].mean()
        w_hi = sorted_f[-k:].mean()
    # 保证 lo < hi，避免几何扫描方向翻转
    w_lo = torch.clamp(w_lo, min=1e-3)
    w_hi = torch.clamp(w_hi, min=w_lo + 1e-3)
    return w_lo * (w_hi / w_lo) ** t
```

关键点：
- `k = 16 // 4 = 4`，取排序后最低 4 个和最高 4 个频率的均值
- 初始状态下 `natural_freq = linspace(0.5, 3.0, 16)`，排序后最低 4 个是 [0.5, 0.667, 0.833, 1.0]，均值 = 0.75；最高 4 个是 [2.5, 2.667, 2.833, 3.0]，均值 = 2.75
- 即初始 `w_lo=0.75, w_hi=2.75`
- `w_lo * (w_hi/w_lo)^t` 是指数插值：t=0 时 ω_d=w_lo=0.75，t=1 时 ω_d=w_hi=2.75

计算各 block 的 ω_d（初始 natural_freq 下）：

| block_idx | t | ω_d = 0.75 × (2.75/0.75)^t |
|---|---|---|
| 0 | 0.0000 | 0.7500 |
| 1 | 0.0909 | 0.8557 |
| 2 | 0.1818 | 0.9761 |
| 3 | 0.2727 | 1.1136 |
| 4 | 0.3636 | 1.2704 |
| 5 | 0.4545 | 1.4498 |
| 6 | 0.5455 | 1.6546 |
| 7 | 0.6364 | 1.8878 |
| 8 | 0.7273 | 2.1539 |
| 9 | 0.8182 | 2.4576 |
| 10 | 0.9091 | 2.8042 |
| 11 | 1.0000 | 2.7500*（注：w_hi=2.75）→ 实际 3.2000 |

注：最后一行应为 w_lo × (w_hi/w_lo)^1 = w_hi = 2.75。上表 block 11 处计算有误，正确值为 2.75。

修正后：

| block_idx | t | ω_d |
|---|---|---|
| 11 | 1.0000 | 2.7500 |

`drive_freq` 的三个保护：
1. `w_lo = clamp(w_lo, min=1e-3)`：防止 lo 为 0 导致除零
2. `w_hi = clamp(w_hi, min=w_lo + 1e-3)`：保证 hi > lo，防止几何扫描方向翻转
3. `with torch.no_grad()`：端点计算不回传梯度，但 `natural_freq` 本身仍是 Parameter，可通过其他路径获得梯度

---

## 8. `forward` 方法（第77-90行）

```python
def forward(self, block_idx: int, device=None):
    """返回第 block_idx 层的三条曲线值 (kappa, T, omega_d)。

    device: 若给定，把标量张量搬到该 device（避免后续运算 device 不匹配）。
    """
    t = torch.tensor(float(self.t_of_block(block_idx)), dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return {
        "t": t,
        "kappa": self.kappa(t),
        "T": self.temperature(t),
        "omega_d": self.drive_freq(t),
    }
```

返回值是 dict，包含 `t`、`kappa`、`T`、`omega_d` 四个标量张量。

调用点：`block.py:240`
```python
tf = self.time_field.forward(i, dev)
```
然后 `block.py:243-244`：
```python
kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
```

---

## 9. `all_curves` 方法（第92-106行）

```python
def all_curves(self):
    """返回全部 n_blocks 层的曲线，用于诊断与日志。"""
    rows = []
    for i in range(self.n_blocks):
        c = self.forward(i)
        rows.append(
            {
                "layer": i,
                "t": c["t"].item(),
                "kappa": c["kappa"].item(),
                "T": c["T"].item(),
                "omega_d": c["omega_d"].item(),
            }
        )
    return rows
```

用于诊断和日志输出，返回 list of dict。

---

## 10. TimeField 在模型中的实际调用链

```
PMNNCore.forward (block.py:229)
  └─ for i, blk in enumerate(self.blocks):  (block.py:239)
       └─ tf = self.time_field.forward(i, dev)  (block.py:240)
            └─ TimeField.forward 返回 {t, kappa, T, omega_d}
       └─ blk(a, theta, omega, t_coord, s_coord,
              kappa=tf["kappa"], T=tf["T"], omega_d=tf["omega_d"],
              targets=targets)  (block.py:241-245)
```

另外在 `aether_gif_model.py:204` 也直接调用：
```python
tf = self.pmnncore.time_field.forward(i, self.device)
```
并在 `aether_gif_model.py:209-211` 对 kappa 和 T 做了文本条件的缩放：
```python
kappa=tf["kappa"] * kappa,
T=tf["T"] * T,
omega_d=tf["omega_d"],
```
其中 `kappa = kappa_base * kappa_scale`，`T = T_base * T_scale`，`kappa_base=2.0, T_base=0.5`（`aether_gif_model.py:197`），`kappa_scale` 和 `T_scale` 来自文本条件经 sigmoid 后的值（`aether_gif_model.py:172-173`）。

即：主目录模型中，TimeField 给出的 κ(t) 还会被文本条件进一步缩放，缩放因子 ∈ (0,1)（sigmoid 输出）。因此实际 κ 范围是 (0, 16.0) × (0, 1.0) = (0, 16.0)，但典型值远小于 16.0。

---

## 11. 与 AetherMind 技术文档的对应关系

`time_field.py` 文件头声明「§4.1 TimeField」，对应 AetherMind-V5-300M-Instruct 技术文档 §4.1。三条曲线公式逐字照搬，仅参数值针对 GIF 场景调整。

对比 AetherMind 蓝图（`docx/AetherGIF-V1_蓝图_4GB显存振子直生成GIF.md` 第216-220行）：
```
κ(t) = κ_max · σ((t − t_c)/τ_c)     # κ_max=8.0, t_c=0.5, τ_c=0.15
T(t) = T_max(1−t)^p + T_min          # T_max=1.0, T_min=0.05, p=1.5
ω_d(t) = ω_lo · (ω_hi/ω_lo)^t        # 由 natural_freq 低/高分位均值给出
```

蓝图建议 κ_max=8.0, t_c=0.5，但代码实际值是 κ_max=16.0, t_c=0.3。代码与蓝图不一致，以代码为准。

---

## 12. 未验证项

1. `natural_freq` 作为可训练参数，其梯度是否真正回传并有效学习——代码注释声称「时间场课程本身可训练」，但无实验数据证明 natural_freq 在训练中发生了有效变化。
2. κ_max=16.0 与 KuramotoSync 中 omega_max=8.0 的关系：κ(t)·g 项中 κ 最大 15.851，g 是 DriftField 输出（有界 [-1,1] 量级的 sin），乘积最大约 15.851，随后进入 `dtheta = omega_max * tanh(dtheta / omega_max)` 软饱和（`kuramoto_sync.py:247`）。这一链条的数值行为无专门实测记录。
3. 三条曲线的具体数值（κ/T/ω_d 表）无对应实验日志记录，本文表格为基于代码公式的手工计算。

---

## 13. 代码事实汇总

| 项 | 值 | 来源 |
|---|---|---|
| n_blocks | 12 | time_field.py:26 默认值 |
| kappa_max | 16.0 | time_field.py:27 |
| t_c | 0.3 | time_field.py:28 |
| tau_c | 0.15 | time_field.py:29 |
| T_max | 1.0 | time_field.py:30 |
| T_min | 0.05 | time_field.py:31 |
| p | 1.5 | time_field.py:32 |
| n_drive_freq | 16 | time_field.py:33 |
| freq_lo / freq_hi | 0.5 / 3.0 | time_field.py:34-35 |
| natural_freq 初始化 | linspace(0.5, 3.0, 16) | time_field.py:46 |
| k 分位数 | numel // 4 = 4 | time_field.py:68 |
| t 计算 | block_idx / (n_blocks - 1) | time_field.py:52 |
