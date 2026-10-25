# AetherWorld：基于 PMNN 底座的物理世界模型架构

> 定位：PMNN 通用底座的**第四个实例化**（前三个：AetherMind 语言 / AetherSentry 安全 / AetherGIF 视觉）。
> 本文只讲架构设计：状态、方程、算子、训练协议。不含商业判断与规划。

---

## 1. PMNN 通用底座回顾

### 1.1 统一状态方程（所有变体的共同内核）

$$\frac{dZ}{dt} = \underbrace{M_\kappa(t)\cdot Z}_{\text{结构算子}} + \underbrace{\nabla_Z[-U(Z) + T(t)\,S(Z)]}_{\text{自由能梯度}} + \underbrace{v_\varphi(Z,t)}_{\text{漂移场}} + \underbrace{\eta(t)}_{\text{涨落}}$$

状态为复值振子场：

$$z_i = a_i\,e^{i\theta_i},\qquad a_i \in \mathbb{R}_{\ge 0},\ \theta_i \in [-\pi,\pi]$$

工程上以实张量 `(a, theta)` 各 `[B,N,D]` 存放，复运算通过 `(a·cosθ, a·sinθ)` 拆分。

### 1.2 四项的物理含义

| 项 | 作用 | 关键性质 |
|---|---|---|
| $M_\kappa(t)\cdot Z$ | Kuramoto 相位耦合（替代注意力） | 含耦合强度 $\kappa(t)$、频率选择、Toeplitz 或内容寻址结构 |
| $\nabla_Z[-U+TS]$ | 锁相（降能量）vs 熵压（升熵）的竞争 | $\partial U/\partial\theta_i = \sum_j W_{ij}\sin(\theta_j-\theta_i)$ —— **Kuramoto 耦合就是 XY 能量梯度** |
| $v_\varphi(Z,t)$ | 学习残差（流匹配速度场） | 承载"数据分布偏离物理势"的那部分 |
| $\eta(t)$ | 朗之万噪声 | 标准差必须 $\sqrt{2T\Delta t}$（涨落-耗散） |

### 1.3 五类通用算子

`TimeField`（时间场）· `KuramotoSync`（相位同步）· `StandingWaveFFN`（共振-驻波前馈）· `DriftField`（漂移场）· `Thermodynamics`（热力学判据）

---

## 2. 三个已有变体：同一底座，不同实例化

| 维度 | **AetherMind V5**（语言） | **AetherSentry Omni**（安全） | **AetherGIF**（视觉） |
|---|---|---|---|
| 状态 $Z$ 的语义 | token 位置上的振子 | flow 序列上的振子 | 时空频域块上的振子 |
| $N$ 的含义 | 序列长度（≤128） | 窗口内 flow 数 | 时空块数（6×32×32=6144） |
| $D$ 的含义 | 1024 通道 | 80 特征维 | 125 频域系数 |
| 输入编码 | 词嵌入 → 振子化 | 特征 → 振子化 | 3D-RFFT 块 → 复系数 |
| 输出解码 | `lm_head` + 流匹配双分支 | 异常分 + 逐维归因 | 逆 FFT → 帧 |
| 时间轴 $t$ | **层深**（16 层） | **层深** | **层深**（12 层） |
| 特有机制 | ELF 双分支、$L_{sync}$ 同步课程 | 三层锁定（空间/特征/模式）、$d_{sync}$ 判据 | `CondInitPC` 三路条件注入、w=16 颜色通道 |
| 规模 | 208M | — | 23~109M |

### 2.1 三个变体的共同约定（不能破）

1. **相位不做 LayerNorm**（相位是周期流形，归一化破坏几何结构），只在振幅上归一
2. **距离计算必须 fp32**（$\|z\|^2$ 可达 $10^5$，超 fp16 上限 65504）
3. **$\kappa(t)$ 必须 clamp 有效乘积**（不是 clamp 单侧）
4. **漂移场是唯一把 target 送入相位演化的通路**
5. **振幅演化必须 log 域乘性**（`a·exp(clamped x)`），不能线性叠加

### 2.2 三个变体的关键差异（可变的）

- **时间轴的含义**：层深 vs 真实时间 ← **这是世界模型必须改的那一条**
- **$N$/$D$ 的语义**：由任务决定
- **输出解码方式**：由监督信号决定
- **是否引入记忆**：语言靠 KV 隐式记忆、安全靠窗口、视觉无记忆

---

## 3. ★ 世界模型的核心改造：时间轴从「层深」改为「真实时间」

### 3.1 问题所在

现有三个变体的共同结构是：

```
输入 → [16 或 12 层物理块串联，t = i/(n_blocks-1)] → 输出
         ↑ 一次前向走完全部演化，t 只用于调度 κ/T/ω_d
```

**层深既承担"子步"又承担"时间进程"**。AetherMind 的文档写明：层 5 出现临界塌缩（$r$ 降到 0.041）、$\kappa$ 从 0.28 升到 7.72——这是**在深度轴上完成一次退火**。

对世界模型而言，这个结构有一个根本局限：**无法持续步进**。跑完 12 层就结束，没有"下一步"的概念；而且每步都从头退火，无法保持稳定状态。

### 3.2 改造：双层时间结构

```
外层（真实时间）   t = 0, 1, 2, ..., H          ← rollout 循环
    │
    ├─ 内层（物理子步） k = 0..K-1               ← 原 12 层降级为"一步内的子步"
    │      s_{t,k+1} = s_{t,k} + Δτ · dZ/dτ |_{s_{t,k}, u_t}
    │
    └─ 状态继承：s_{t+1,0} = s_{t,K}            ← 不再是"从条件重新生成"
```

**这是本质区别**：

| | 现有变体 | AetherWorld |
|---|---|---|
| 演化单位 | 层（12/16 次） | **真实时间步 × K 子步** |
| 状态来源 | 每层从前一层来 | **跨时间步继承** |
| $\kappa/T$ 调度 | 沿层深退火一次 | **每步保持工作温度区间**（不退火） |
| 可否持续 | 否（走完即止） | **是（可 rollout 任意步）** |

### 3.3 $\kappa/T$ 的新调度协议

原协议（AetherMind）：

$$\kappa(t)=\kappa_{max}\sigma\!\left(\frac{t-t_c}{\tau_c}\right),\quad T(t)=T_{max}(1-t)^p+T_{min},\quad t=\frac{i}{n_{blocks}-1}$$

**世界模型的问题**：若沿用此式，每个时间步都要重新从 $\kappa\approx0$（混沌）退火到 $\kappa_{max}$（锁相），意味着**每步都会先崩解再重建**，状态无法连续。

**改造**：把"层深退火"改为"**子步内微调 + 跨步保持**"：

$$\kappa_{t,k} = \kappa_{work} + \Delta\kappa\cdot\sigma\!\left(\frac{k/K - t_c}{\tau_c}\right)\cdot\epsilon$$

- $\kappa_{work}$：工作耦合强度（跨步保持，**不每步归零**）
- $\Delta\kappa\cdot\epsilon$：子步内的小幅调制（$\epsilon = 0.1$），只用于提供局部梯度
- $T_{t,k} \approx T_{work}$：保持在工作温度（如 0.3），**不退火到 $T_{min}$**

理由：退火是"求解"过程（一次前向求一个解）；世界模型需要的是**持续维持在解附近**（每一步都在解附近小幅更新）。

---

## 4. AetherWorld 完整架构

### 4.1 状态定义

$$\mathbf{s}_t = (a_t,\ \theta_t,\ m_t)$$

| 分量 | 形状 | 语义 |
|---|---|---|
| $a_t$ | `[B, N, D]` | 振幅场（"强度/能量"） |
| $\theta_t$ | `[B, N, D]` | 相位场（"结构/关系"） |
| $m_t$ | `[B, N, D]` | **记忆场**（慢变量，新增） |

$N$ = 空间位置数（由观测分辨率决定），$D$ = 每位置通道数。

### 4.2 ★ 新增：记忆场（世界模型必需，PMNN 原设计缺）

**为什么需要**：Kuramoto 耦合是**当前场内的耦合**，没有跨时间的信息保持。世界模型要求"物体在离开视野后仍然存在"（object permanence），这需要跨步记忆。

**设计**：慢变量场，指数滑动平均 + 与当前场的一致性约束

$$m_{t+1} = (1-\alpha)\,m_t + \alpha\,\phi(z_t),\qquad \phi = \text{可学习投影}$$

- $\alpha \in (0,1)$：记忆更新速率（$\alpha$ 小 = 记忆长，$\alpha$ 大 = 响应快）
- 初始 $\alpha = 0.1$（时间常数约 10 步）
- 记忆场通过 $v_\varphi$ 项回注到演化：

$$v_\varphi^{\text{new}} = v_\varphi(Z, t) + \gamma_m\cdot\phi^{-1}(m_t - z_t)$$

**物理类比**：这是**惯性/滞后**——系统不会瞬时响应，而是带着上一状态的"惯性"演化。对应到物理系统就是阻尼记忆。

**与 DriftField 原型的区别**：`prototypes` 是全局可学习吸引中心（32 个，与样本无关）；记忆场是**逐位置、逐样本**的状态，随时间演化。两者不冲突，可共存。

### 4.3 ★ 动作条件：作为状态方程的第五项

**不改动前四项，只加一项**：

$$\frac{dZ}{d\tau} = M_\kappa\cdot Z + \nabla_Z[-U+TS] + v_\varphi(Z,t) + \underbrace{B\,u_t}_{\text{动作}} + \eta$$

$$\boxed{B\,u_t:\quad u_t \in \mathbb{R}^{k}\ \xrightarrow{\ \text{Linear}(k,64)\ }\ \xrightarrow{\ \text{Linear}(64,D)\ }\ [B,1,D]\ \text{广播到 } N}$$

| 符号 | 含义 | 说明 |
|---|---|---|
| $u_t$ | 动作向量 | $k=8\sim16$ 维（隐动作或物理控制量） |
| $B$ | 控制矩阵 | 低秩分解，`Linear(k,64)→Linear(64,D)`，参数量约 8K |
| 广播 | `[B,1,D] → [B,N,D]` | 同一动作对全场施加同向力 |

**为什么动作是"力"而不是"条件向量"**：

- 条件向量（如 AetherGIF 的 `cond_vec`）作用于**初始态**（`CondInitPC`），决定"从哪出发"
- 动作作用于**演化过程**（每步的 `dZ/dτ`），决定"往哪走"

这与物理直觉一致：初态是位置，动作是力。世界模型两者都需要。

**动作来源两条路**：

| 路线 | 来源 | 适用 |
|---|---|---|
| **A. 无监督隐动作（推荐起步）** | 从帧对 `(s_t, s_{t+1})` 学编码器，输出 $u_t$ | 无动作标签数据 |
| B. 有监督物理动作 | 关节角速度/方向盘/推力等 | 有标注数据 |

路线 A 的参考：Genie 用 3 万小时无标注视频，隐动作空间压到 **8 个离散值**；本设计保留连续 $u\in\mathbb{R}^8$（物理力是连续的，不需要离散化）。

### 4.4 转移算子完整结构

```python
def step(a, theta, m, u_t, K=4):
    """一个真实时间步：K 个物理子步"""
    for k in range(K):
        # ① 记忆回注（惯性）
        v_mem = gamma_m * (inverse_proj(m) - z(a, theta))

        # ② 漂移场残差
        v_drift = drift_field(a, theta)          # 学习残差

        # ③ 动作力（新）
        force = B_ctrl(u_t).unsqueeze(1)         # [B,1,D] → 广播

        # ④ Kuramoto 相位演化 + 自由能梯度
        dtheta = kuramoto_sync(theta, kappa_work, T_work) + force_phase

        # ⑤ 振幅演化（log 域乘性）
        a = a * exp(clamp(res_amp + force_amp, -6, 3))

        # ⑥ 相位积分（不折叠，loss 处才 wrap）
        theta = theta + dt * (dtheta + v_mem + v_drift)

    # ⑦ 记忆更新
    m = (1 - alpha) * m + alpha * project(z(a, theta))
    return a, theta, m
```

**关键约束（沿用 PMNN 约定）**：
- 相位不折叠、不做 LayerNorm
- 振幅用 `softplus(LayerNorm(...))` 保证非负与尺度
- 距离相关计算 fp32
- $\kappa$ clamp 有效乘积

### 4.5 长期稳定性：三重物理约束

主流世界模型靠 loss 硬压或 EMA；本架构用**可测量的物理量**：

| 约束 | 公式 | 判据 |
|---|---|---|
| **能量守恒** | $U_t = -\sum_{ij}W_{ij}\cos(\theta_j-\theta_i)$；无动作时 $\|U_{t+1}-U_t\| < \epsilon$ | 能量漂移率 |
| **序参量平滑** | $r_t = \left\|\frac{\sum_i w_ie^{i\theta_i}}{\sum_i w_i}\right\|$；$\|r_{t+1}-r_t\| < \delta$ | 同步度跳变 |
| **熵地板** | $S_t = -\sum p_i\log p_i$；$S_t > S_{min}$ | 振幅分布不塌缩 |

**为什么这是优势**：Dreamer 4 的复现报告指出"稳定性占开发时间最大份额，且多数问题发生在 loss 正常下降时"——**loss 不能反映稳定性**。而 $U/r/S$ 是**无参数、可直接观测**的量，能在 loss 还正常时就发现发散。

### 4.6 输出解码

| 输出 | 方式 | 用途 |
|---|---|---|
| **下一状态** $s_{t+1}$ | 直接输出 `(a, θ, m)` | rollout 的核心 |
| **可视化** | 逆 FFT（若状态是频域）或轻量 decoder | 人看/demo |
| **预测奖励/终止**（可选） | 小 head | 若做 RL 训练环境 |

**与 AetherGIF 的关系**：若 $Z$ 的语义沿用 AetherGIF 的时空频域（$D=125$），则解码可直接复用 `decode_field_to_frames`；若用于更通用的世界模型，$Z$ 可以是任意空间的振子场（如 3D 体素），此时需要对应的 decoder。

---

## 5. 训练协议

### 5.1 总损失

$$L = \underbrace{L_{pred}}_{\text{状态预测}} + \lambda_1\underbrace{L_{inv}}_{\text{逆动作}} + \lambda_2\underbrace{L_{energy}}_{\text{守恒}} + \lambda_3\underbrace{L_{ctrl}}_{\text{可控性}} + \lambda_4\underbrace{L_{sync}}_{\text{同步课程}}$$

| 项 | 定义 | 作用 |
|---|---|---|
| $L_{pred}$ | `log1p` 振幅加权 MSE + 功率加权相位误差（沿用 AetherGIF 口径） | 基础预测 |
| $L_{inv}$ | 从 $(s_t,s_{t+1})$ 反推 $u_t$ 的 MSE/CE | **防捷径**：Dreamer 4 验证——模型会学"下一帧=当前帧"，逆模型强制 latent 保留动作信息 |
| $L_{energy}$ | 无动作时 $\|U_{t+1}-U_t\|$；有动作时 $\|U_{t+1}-U_t-W_u\|$ | 物理守恒 |
| $L_{ctrl}$ | 同一 $s_t$、两个不同 $u$ → 轨迹差应 > 阈值 | **动作可控性**（判据也是损失） |
| $L_{sync}$ | 序参量跟踪课程（AetherMind 已有） | 防全同步坍缩 |

### 5.2 关键训练细节（沿用已验证的坑）

1. **多步展开（BPTT）**：rollout 2~4 步反传，不要 1 步（学不到长期依赖），不要 >8 步（梯度爆）
2. **`loss_gen` 式短路监督**：动作通路梯度穿 K 子步会衰减，需对 $B u_t$ 加直接监督
3. **EMA 必须**（Dreamer 4 教训：扩散/流推理必须用 EMA 权重）
4. **精度边界**：参数 fp32，BF16 仅用于 matmul；归一化与输出头 fp32
5. **`W2_NORM_MAX` 投影**：每步后投影 FFN 输出权重范数，防数值爆炸
6. **逆模型不能 detach**：逆模型与主模型共享状态表示，必须联合训练

### 5.3 参数量估算

| 配置 | $N$ | $D$ | 层数 | 估计参数 |
|---|---|---|---|---|
| **AetherWorld-Tiny**（验证用） | 1024 | 125 | 12 | ~25M |
| **AetherWorld-Small** | 4096 | 128 | 12 | ~45M |
| **AetherWorld-Base** | 4096 | 256 | 16 | ~110M |
| **AetherWorld-Large** | 16384 | 256 | 24 | ~450M |

**4GB 显存可训的上限约 110M**（需梯度检查点 + 8bit 优化器 + batch=1）。

---

## 6. 与三个既有变体的对比总表

| 维度 | AetherMind V5 | AetherSentry | AetherGIF | **AetherWorld** |
|---|---|---|---|---|
| 任务 | 语言建模 | 安全检测 | GIF 生成 | **世界模拟** |
| 状态方程 | 同一式 | 同一式 | 同一式 | **同一式 + 动作项 + 记忆项** |
| 时间轴 | 层深（16） | 层深 | 层深（12） | **真实时间 × K 子步** |
| 动作条件 | ❌ | ❌ | ❌（条件向量作用于初态） | ✅ **$B u_t$ 作用于演化** |
| 跨步记忆 | KV（隐式） | 窗口 | ❌ | ✅ **显式记忆场 $m_t$** |
| 长期稳定 | $L_{sync}$ 课程 | $d_{sync}$ 判据 | 无 | **能量 + 序参量 + 熵三重** |
| 输出 | logits + 流匹配 | 异常分 + 归因 | 帧序列 | **下一状态（可 rollout）** |
| 复用 AetherGIF | — | — | — | 可复用 codec 与 `decode_field_to_frames` |

**一句话**：AetherWorld 不是新架构，是**同一状态方程在"持续时间"这个新场景下的实例化**——新增的只有两项：动作力 $B u_t$ 与记忆场 $m_t$。

---

## 7. 最小验证实验（4GB 可做）

**目标**：验证"动作可控性"——这是世界模型的第一张入场券。

```
【数据】3 个有明确运动的序列（可用现有 GIF：8ball 滚动 / senpai 挥手 / idolmaster 摆动）
【步骤】
  1. 训 LAM：帧对 (s_t, s_{t+1}) → u_t ∈ R^8（重建 s_{t+1} 作为监督）
  2. 冻结 LAM，用 u_t 训 B_ctrl 注入（$L_{pred} + L_{ctrl}$）
  3. 测试：固定 s_0，分别给 u_a、u_b（差异最大的两个隐动作）
     各 rollout 5 步（K=4 子步/步）

【判据】
  · 轨迹相位场 L2 距离 > 阈值（初始态相同时）
  · 同一 u 多次采样方差 < 阈值（稳定性）
  · 5 步末能量漂移 |ΔU| < 10%

【判失败】
  若两条轨迹无显著差异 → 动作不可控 → 需检查：
    ① B_ctrl 是否学到非零权重
    ② u_t 是否真的编码了运动差异（看 LAM 的隐动作聚类）
    ③ 动作力是否被 Kuramoto 同步抹平（参考 AetherGIF 的 μ 共享教训）
```

**预估**：4GB 卡、现有数据、约 1 周。

---

## 8. 诚实的风险标注

| 风险 | 依据 | 影响 |
|---|---|---|
| **架构 scale 性未验证** | AetherGIF 在 23M→109M 时 22 素材 sat 极差从 5.32 降到 0.68（负向证据） | 扩到 450M 前必须先验证 scale 曲线单调 |
| **无动作标签数据** | 现有 GIF 无控制信号 | 走 LAM 无监督路线，但 LAM 在极小数据上可能学不出有意义的动作 |
| **记忆场可能干扰主演化** | 新增项与原有四项耦合关系未验证 | 先设 $\gamma_m$ 很小（0.1），逐步加大 |
| **4GB 显存做不了长 rollout** | 长序列激活爆显存 | 短窗口（2~4 步 BPTT）+ 梯度检查点 |
| **物理准确性未必优于数据驱动** | 全领域 Physics-IQ 最高分仅 24.1% | 显式物理不是万能药，需实测对比 |

---

## 9. 一页纸公式速查（AetherWorld 版）

| 概念 | 公式 |
|---|---|
| 状态 | $\mathbf{s}_t=(a_t,\theta_t,m_t)$，$z=a e^{i\theta}$ |
| **动作方程**（新增第五项） | $\dfrac{dZ}{d\tau}=M_\kappa Z+\nabla_Z[-U+TS]+v_\varphi+B u_t+\eta$ |
| 转移（单步） | $s_{t+1}=\text{Integrate}_K(s_t,u_t)$，K 个子步 |
| **记忆场**（新增） | $m_{t+1}=(1-\alpha)m_t+\alpha\phi(z_t)$，$\alpha=0.1$ |
| 记忆回注 | $v_\varphi^{\text{new}}=v_\varphi+\gamma_m(\phi^{-1}(m_t)-z_t)$ |
| 控制矩阵 | $B:\ \mathbb{R}^k\to\mathbb{R}^D$，低秩 `Linear(k,64)→Linear(64,D)` |
| 能量守恒判据 | $\left\|U_{t+1}-U_t\right\|<\epsilon$，$U=-\sum W_{ij}\cos(\theta_j-\theta_i)$ |
| 序参量平滑 | $\left\|r_{t+1}-r_t\right\|<\delta$，$r=\left\|\sum w_ie^{i\theta_i}/\sum w_i\right\|$ |
| $\kappa$ 调度（改造后） | $\kappa_{t,k}=\kappa_{work}+\Delta\kappa\cdot\sigma((k/K-t_c)/\tau_c)\cdot\epsilon$ |
| 总损失 | $L=L_{pred}+\lambda_1L_{inv}+\lambda_2L_{energy}+\lambda_3L_{ctrl}+\lambda_4L_{sync}$ |
