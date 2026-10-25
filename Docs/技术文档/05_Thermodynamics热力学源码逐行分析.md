# 05 · Thermodynamics 热力学源码逐行分析

源文件：`src/physics/thermodynamics.py`（110行）
调用点：`block.py:255-257`（末层诊断）、`thermodynamics.py:89`（forward）

---

## 1. 文件头声明（第1-19行）

```python
"""
§4.5 Thermodynamics —— 热力学判据层

  U(Z) = −Σ_ij W_ij·cos(θ_j − θ_i)     （XY 能量：锁相 = 能量最小化）
  S(Z) = −Σ_i p_i·log p_i, p_i = softmax(a_i)   （振幅分布熵）
  F = U − T·S                          （自由能）

核心恒等式（AetherMind §4.5 / AetherSentry §2.3）：
  ∂U/∂θ_i = Σ_j W_ij·sin(θ_j − θ_i)

即 Kuramoto 的 sin 耦合就是 XY 能量梯度。同步层不是「模拟一个物理现象」，
而是在做能量最小化。这个同源性必须保留，不能简化掉。

AetherSentry §8.3 的教训（通道正交）：
  U 依赖时间维序参量，S 依赖通道维振幅分布。
  若要用 F = U − T·S 修时间通道的问题，S 必须显式计算在时间维度上。
  本实现提供两种 S：通道维熵 S_chan 与时间维熵 S_time，禁止混用。
"""
```

核心恒等式：∂U/∂θ_i = Σ_j W_ij·sin(θ_j−θ_i)，即 KuramotoSync 计算的 sin 耦合就是 XY 能量 U 对 θ_i 的负梯度。这意味着相位演化 = 能量下降方向。

---

## 2. 导入（第20-22行）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## 3. `Thermodynamics` 类定义（第25-109行）

### 3.1 `__init__`（第28-29行）

```python
def __init__(self):
    super().__init__()
```

无可学习参数。

### 3.2 `order_parameter` 静态方法（第31-44行）

```python
@staticmethod
def order_parameter(a: torch.Tensor, theta: torch.Tensor):
    """序参量 r 与平均相位 ψ（振幅加权 + 归一化口径修正）。

    r cosψ = Σw_i cosθ_i / Σw_i,  w_i = softplus(a_i)
    返回 r: [B,D], psi: [B,D]
    """
    w = F.softplus(a)
    w_sum = w.sum(dim=1).clamp(min=1e-8)  # [B,D]
    rcos = (w * torch.cos(theta)).sum(dim=1) / w_sum
    rsin = (w * torch.sin(theta)).sum(dim=1) / w_sum
    r = torch.sqrt(rcos ** 2 + rsin ** 2 + 1e-12).clamp(max=1.0)
    psi = torch.atan2(rsin, rcos)
    return r, psi
```

- 沿 dim=1（N 维）求和
- r ∈ [0,1]，clamp(max=1.0) 防数值溢出
- 与 `MeanFieldCoupling.forward`（kuramoto_sync.py:166-182）公式完全一致

### 3.3 `xy_energy` 静态方法（第46-62行）

```python
@staticmethod
def xy_energy(theta: torch.Tensor, W: torch.Tensor = None):
    """U = −Σ_ij W_ij cos(θ_j − θ_i)

    若不给 W（时空耦合是隐式的、分块计算的），用平均场近似：
    U ≈ −Σ_i w_i·r·cos(ψ − θ_i) / Σw_i
    （AetherMind §4.5 的平均场近似能量）
    """
    if W is not None:
        cos_diff = torch.cos(theta.unsqueeze(2) - theta.unsqueeze(1))  # [B,N,N,D]
        return -(W.unsqueeze(-1) * cos_diff).sum(dim=(1, 2)).mean()
    # 平均场近似
    w = F.softplus(theta.abs())  # 退化情形：无 a 时用 |θ|
    r, psi = Thermodynamics.order_parameter(w, theta)
    U = -(w * r.unsqueeze(1) * torch.cos(psi.unsqueeze(1) - theta)).sum(dim=1)
    U = U / w.sum(dim=1).clamp(min=1e-8)
    return U.mean()
```

两条路径：
1. W 给定：精确计算 [B,N,N,D] 的 cos 差矩阵（显存爆炸，实际不用）
2. W 为 None（实际调用路径）：平均场近似
   - `w = F.softplus(theta.abs())`：注意这里用 |θ| 而不是 a，因为方法签名只收 theta
   - U = −Σ_i w_i·r·cos(ψ−θ_i) / Σw_i
   - 返回 U.mean()（对所有 batch 和通道取均值）

调用点 `thermodynamics.py:94`：`U = self.xy_energy(theta)`，W 为 None，走平均场近似。

### 3.4 `amplitude_entropy_channel` 静态方法（第64-73行）

```python
@staticmethod
def amplitude_entropy_channel(a: torch.Tensor):
    """通道维振幅分布熵 S_chan = −Σ p log p, p = softmax(a_i)

    a: [B,N,D]，沿 N 维 softmax（每个通道一个分布）
    返回 [B,D] 归一化熵 ∈[0,1]
    """
    p = F.softmax(a, dim=1)  # 沿 N
    S = -(p * torch.log(p + 1e-12)).sum(dim=1)
    return S / torch.log(torch.tensor(float(a.shape[1]), device=a.device))
```

- 沿 N 维 softmax：每个通道 c 有一个跨 N 个位置的概率分布 p_i
- S = −Σ p·log(p)，归一化到 [0,1]（除以 log(N)）
- 这是**通道维**熵：衡量每个通道的振幅在空间上分布有多均匀

### 3.5 `amplitude_entropy_time` 静态方法（第75-87行）

```python
@staticmethod
def amplitude_entropy_time(a: torch.Tensor, t_steps: int, hw: int):
    """时间维振幅分布熵 S_time（AetherSentry §8.3 教训：修时间通道必须用时间维的 S）

    把 [B, N, D] 重排为 [B, t_steps, hw, D]，
    沿时间维 softmax 计算熵。用于时间通道的诊断与正则。
    """
    B, N, D = a.shape
    a_view = a.reshape(B, t_steps, hw, D)
    # 沿时间维 softmax（固定空间位置与通道）
    p = F.softmax(a_view, dim=1)  # [B,T,hw,D]
    S = -(p * torch.log(p + 1e-12)).sum(dim=1)  # [B,hw,D]
    return S.mean(dim=(1, 2)) / torch.log(torch.tensor(float(t_steps), device=a.device))
```

- 把 [B,N,D] reshape 成 [B,t_steps,hw,D]
- 沿时间维（dim=1）softmax：每个空间位置和通道有一个跨时间的概率分布
- S.mean(dim=(1,2))：对空间位置和通道取均值
- 归一化除以 log(t_steps)
- 这是**时间维**熵：衡量每个空间位置的振幅在时间上分布有多均匀

AetherSentry §8.3 教训：「若要用 F = U − T·S 修时间通道的问题，S 必须显式计算在时间维度上」。本实现因此提供两种 S，禁止混用。

### 3.6 `forward` 方法（第89-109行）

```python
def forward(self, a: torch.Tensor, theta: torch.Tensor, T: torch.Tensor,
            t_steps: int = None, hw: int = None):
    """返回 {U, S_chan, S_time, F, r, psi}，全部 detach 只做诊断。"""
    with torch.no_grad():
        r, psi = self.order_parameter(a, theta)
        U = self.xy_energy(theta)
        S_chan = self.amplitude_entropy_channel(a)
        S_time = (
            self.amplitude_entropy_time(a, t_steps, hw)
            if t_steps is not None and hw is not None
            else torch.zeros_like(S_chan)
        )
        F_val = U - T * S_chan
        return {
            "U": U.detach(),
            "S_chan": S_chan.detach(),
            "S_time": S_time.detach(),
            "F": F_val.detach(),
            "r": r.detach(),
            "psi": psi.detach(),
        }
```

- 全部 `@torch.no_grad()` + `.detach()`：只做诊断，不参与梯度
- `F = U − T·S_chan`：注意用的是 S_chan 不是 S_time。文件头说「修时间通道必须用 S_time」，但 F 的计算用的是 S_chan。这是文件头与代码的不一致。
- 调用点 `block.py:255-257`：
  ```python
  thermo_diag = self.blocks[-1].thermo(a, theta,
                                       self.time_field.forward(self.n_blocks - 1, dev)["T"],
                                       t_steps, hw)
  ```
  传入最后一个 block 的 T 值（t=1.0 时 T=0.05）

---

## 4. 诊断量的物理意义

| 量 | 公式 | 范围 | 物理意义 |
|---|---|---|---|
| r | \|Σw e^{iθ}/Σw\| | [0,1] | 全局相位有序度。1=完全锁相，0=完全无序 |
| psi | atan2(rsin, rcos) | [−π,π] | 平均相位方向 |
| U | −Σ w·r·cos(ψ−θ)/Σw | 负值 | XY 能量。越负越有序 |
| S_chan | −Σp log p / log N | [0,1] | 通道维振幅熵。1=均匀分布，0=集中在少数位置 |
| S_time | 同 S_chan 但沿时间维 | [0,1] | 时间维振幅熵 |
| F | U − T·S_chan | — | 自由能 |

---

## 5. Thermodynamics 在 PMNNCore 中的调用

`block.py:252-257`：
```python
# 末层热力学诊断
t_steps = int(t_coord.max().item()) + 1
hw = a.shape[1] // t_steps
thermo_diag = self.blocks[-1].thermo(a, theta,
                                     self.time_field.forward(self.n_blocks - 1, dev)["T"],
                                     t_steps, hw)
```

只在 `return_diag=True` 时返回（`block.py:258-259`）：
```python
if return_diag:
    return a, theta, {"per_block": diags, "thermo": thermo_diag}
```

`return_diag` 默认 False（`block.py:229` 签名），所以 Thermodynamics 默认不执行。

---

## 6. 与 KuramotoSync 的能量同源性

核心恒等式：∂U/∂θ_i = Σ_j W_ij·sin(θ_j−θ_i)

`KuramotoSync.forward`（kuramoto_sync.py:231）计算的 `coupling_struct = self.toeplitz.apply_coupling(theta, ...)` 正是 Σ_j W_ij·sin(θ_j−θ_i)。

`MeanFieldCoupling`（kuramoto_sync.py:181）计算的 `coupling = rsin·cosθ − rcos·sinθ = r·sin(ψ−θ_i)`，这是平均场近似下的同一梯度。

所以相位演化 dθ/dt 中的耦合项 = −∂U/∂θ 方向，即相位自发向能量降低方向演化。

---

## 7. 未验证项

1. `F = U − T·S_chan` 用 S_chan 而非 S_time，与文件头「修时间通道必须用 S_time」的声明不一致。无实验数据说明这个选择的影响。
2. Thermodynamics 全部输出为 detach，不参与任何 loss 或正则化。文件头称「热力学判据层」，但实际只是诊断，无「判据」动作（无基于 U/S/F 的早停或正则）。「判据」功能未实现。
3. `xy_energy` 的 W 路径（精确计算 [B,N,N,D]）从未被调用，因为调用点不传 W。
4. r/psi/U/S_chan/S_time 的历史数值无训练日志记录。
5. `w = F.softplus(theta.abs())` 在 xy_energy 平均场近似中用 |θ| 代替 a，这是「退化情形」，无实验验证其诊断有效性。

---

## 8. 代码事实汇总

| 项 | 值 | 来源 |
|---|---|---|
| 可学习参数 | 无 | thermodynamics.py:28-29 |
| order_parameter | 沿 dim=1（N维）求和 | thermodynamics.py:39 |
| xy_energy W=None 路径 | 平均场近似，w=softplus(\|θ\|) | thermodynamics.py:58-62 |
| xy_energy W 给定路径 | [B,N,N,D] 精确，未调用 | thermodynamics.py:55-56 |
| S_chan | 沿 N 维 softmax，归一化 /log(N) | thermodynamics.py:71-73 |
| S_time | reshape [B,T,hw,D]，沿 T softmax，/log(T) | thermodynamics.py:83-87 |
| F | U − T·S_chan（用 chan 不是 time） | thermodynamics.py:101 |
| no_grad | 全部诊断 | thermodynamics.py:92 |
| 调用 T 值 | 最后 block 的 T（t=1.0 → T=0.05） | block.py:256 |
| return_diag 默认 | False | block.py:229 |
| per_block diag 内容 | layer, kappa, T + sync_diag | block.py:247-249 |
| sync_diag 内容 | r, psi, dtheta_norm | kuramoto_sync.py:182, 268 |
