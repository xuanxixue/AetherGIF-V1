# AetherMind-V5-300M-Instruct（PMNN）技术文档

> 本文只讲技术：架构、物理模型、数学公式、实现与训练。
> 不讲目录结构、不讲文件清单、不讲未来规划。

---

## 1. 一句话定位

一个**非 Transformer** 的自回归语言模型。每一层不含 QKV 注意力、不含前馈网络、不含位置编码，而是把 token 表示成**复值振子**，用 **Kuramoto 相位锁相**在 token 之间传递信息，用**洛伦兹共振 × 驻波调制**代替 FFN，并把网络深度解释为**物理时间**。

| | |
|---|---|
| 参数量 | 208.2M |
| 物理核层数 | 16 |
| 词表 | 8192（自带 tokenizer） |
| 上下文（训练） | 128 |
| 预训练数据 | 约 2.13 亿 token（中文为主） |
| 精度 | fp32（CPU 可直接推理） |

---

## 2. 统一状态表示

所有中间状态不是实值向量，而是**复值振子场**：

$$z_i = a_i \cdot e^{i\theta_i}$$

- $a_i \in \mathbb{R}_{\ge 0}$：振幅（每个 token、每个通道一个）
- $\theta_i \in [-\pi, \pi]$：相位

工程上用**实数分离表示**存放两个实张量 `a`、`theta`，形状均为 `[B, N, D]`（D = 1024 通道）。复值运算通过 `(a·cosθ, a·sinθ)` 拆分实现。

**振子化编码**（token → 复值场）：

$$a = \text{softplus}(W_a \cdot \text{LN}(E[x])), \quad \theta = (W_\theta \cdot \text{LN}(E[x])) \bmod 2\pi$$

振幅过 softplus 保证非负，相位折叠到 $[-\pi, \pi]$。

---

## 3. 与 Transformer 的结构对照

| 维度 | Transformer | PMNN |
|---|---|---|
| 信息混合 | QKV 注意力（学出来的点积相似度） | Kuramoto 锁相 $d\theta_i/dt = \omega_i + \sum_j W_{ij}\sin(\theta_j - \theta_i)$ |
| 状态 | 实值向量 | 复值振子 $z = a e^{i\theta}$ |
| 前馈 | FFN（线性层堆叠 + 激活） | 洛伦兹共振增益 × 驻波调制 |
| 位置信息 | 位置编码（RoPE / 绝对） | **无位置编码**，靠因果距离 $i-j$ 直接耦合 |
| 深度语义 | 堆叠层数 | **物理时间** $t$：$\kappa(t)$ 与 $T(t)$ 沿深度退火 |
| 可观测中间量 | 注意力矩阵 | 序参量 $r$、平均相位 $\psi$、$\kappa$、$T$、振幅熵 $S$、自由能 $F$ |

关键点：$r / \psi / \kappa / T / S / F$ 在 Transformer 里**根本不存在**。这不是"换个说法画注意力"，而是另一个物理系统在演化。

---

## 4. 物理核：五类算子

### 4.1 时间场 TimeField —— 深度 = 物理时间

$t = \text{round} / (n_{blocks}-1) \in [0,1]$，即**层数编号归一化为物理时间**。时间场给出三条协议曲线：

$$\kappa(t) = \kappa_{max} \cdot \sigma\!\left(\frac{t - t_c}{\tau_c}\right) \quad \text{（耦合强度课程：混沌} \to \text{锁相）}$$

$$T(t) = T_{max}(1-t)^{p} + T_{min} \quad \text{（温度退火：冷却）}$$

$$\omega_d(t) = \omega_{lo} \cdot \left(\frac{\omega_{hi}}{\omega_{lo}}\right)^{t} \quad \text{（驱动频率几何扫描：低频} \to \text{高频）}$$

300M 配置：$\kappa_{max}=8.0$、$t_c=0.5$、$\tau_c=0.15$、$T_{max}=1.0$、$T_{min}=0.05$、$p=1.5$。

扫描区间端点 $\omega_{lo}/\omega_{hi}$ 由**可学习的固有频率谱** `natural_freq`（`nn.Parameter`，`linspace(0.5, 3.0, n_drive_freq)`）的低/高分位均值给出。这样时间场课程本身可训练（梯度可回传），否则是死参数。

### 4.2 KuramotoSync —— 相位同步（替代注意力）

状态方程：

$$\frac{d\theta_i}{dt} = \omega_i + \sum_j W_{ij}\sin(\theta_j - \theta_i) + \kappa(t)\, g_i + \eta(t)$$

四条耦合通路：

**(1) 平均场序参量耦合（全局，秩-1，无参数）**

$$\sum_j \sin(\theta_j - \theta_i) \;\to\; r\sin(\psi - \theta_i) = (r\sin\psi)\cos\theta_i - (r\cos\psi)\sin\theta_i$$

其中 $r$ 是序参量、$\psi$ 是平均相位。用振幅加权 + mask 加权：

$$r\cos\psi = \frac{\sum_i w_i \cos\theta_i}{\sum_i w_i}, \quad r\sin\psi = \frac{\sum_i w_i \sin\theta_i}{\sum_i w_i}, \quad w_i = \text{softplus}(a_i)$$

复杂度 $O(N \cdot d)$，秩-1，**无需任何参数**。

> **归一化口径修正（重要）**：分母必须用 $w$ 求和，不能是位置数 $N$。旧实现分子 $w$ 加权、分母用 $N$，当 $w = \text{softplus}(a) > 1$ 时 $r$ 虚高甚至 $>1$（理论 $r \in [0,1]$）。300m 实测初始 $r=0.73$、训练后飙到 **1.03**，熵归零 = 全同步坍缩。修正后 $r$ 严格 $\in[0,1]$。

**(2) 低秩固定衰减耦合（mean_field 模式）**

$W = U V^\top$，秩 $r_{rank} = 96$。因果记忆用指数衰减前缀和：

$$A_i = \sum_{j \le i} e^{-\lambda(i-j)} v_j = e^{-\lambda i} \odot \text{cumsum}\!\left(e^{\lambda j} \odot v_j\right)$$

$\lambda = \ln 2 / \text{halflife}$。用 cumsum 并行计算，复杂度 $O(N \cdot r)$，低于 self-attention 的 $O(N^2 D)$，且**天然因果**。

> **已知局限**：聚合权重 $e^{-\lambda(i-j)}$ 只取决于**距离**，与内容无关，只能给出上下文的 rank-1 摘要。A/B 对照（causal ON/OFF，同 seed 400 步）CE 5.1596 vs 5.1622，差异 0.05%，证实此局限。

**(2b) 内容寻址因果耦合（content 模式，物理版注意力）**

这是 mean_field 的替代，300M 默认启用（`coupling="content"`）。耦合强度由振子状态本身决定 —— 「锁相即强耦合」：

$$W_{ij} \propto \exp\!\left(\frac{q_i \cdot k_j}{\sqrt{r}} - \lambda (i-j)\right), \quad i \ge j$$

- $q_i, k_j$ 由复值场 $[a\cos\theta,\; a\sin\theta]$ 经线性投影得到（查询/键各一个 `Linear(2D, r)`）
- 输出 = $W @ \sin(\theta)$，再经值投影 `cv`
- 距离项 $-\lambda(i-j)$ 同时给出**光锥因果性**与**长程衰减**
- pad 位置不能作 key；整行被屏蔽时用有限负数 $-10^4$ 而非 $-\infty$，避免 softmax 出 NaN

**(3) 漂移引导项**：$\kappa(t) \cdot g_i$，$g_i$ 由漂移场（§4.4）给出。

**(4) 朗之万噪声**（默认关）：$\eta \sim \mathcal{N}(0, 2T\Delta t)$，由涨落-耗散定理给出。标准差必须是 $\sqrt{2T\Delta t}$，漏掉 $\Delta t$ 会让噪声量级随步长错误缩放。只在训练期注入；$T(t)$ 退火到 $T_{min}$ 后噪声自然趋 0。

**数值积分**：Euler，步长 $dt = 0.1$：

$$\theta_{next} = \theta + dt \cdot d\theta$$

积分后折叠回 $[-\pi, \pi]$：`remainder(θ+π, 2π) − π`。pad 位置相位保持不变。

> **相位不能做 LayerNorm**：相位是周期流形，LayerNorm 会破坏其几何结构。模型只在振幅上做归一（`norm_a`）。

### 4.3 StandingWaveFFN —— 共振-驻波前馈（替代 FFN）

$$h = \sigma\!\left(W_2 \cdot \big(W_1 z + b_1\big) \odot A(\omega_d, \omega_n) \odot SW(p) + b_2\right)$$

**(a) 洛伦兹共振增益（频率选择性）**

$$A(\omega_d, \omega_n) = \frac{1}{\sqrt{(\omega_d^2 - \omega_n^2)^2 + (2\gamma \omega_d)^2}}$$

驱动频率 $\omega_d(t)$ 接近某通道固有频率 $\omega_n$ 时该通道被放大（阻尼 $\gamma = 0.3$）。$\omega_n$ 是 hidden 维可学习参数（`linspace(0.5, 4.0, ffn_hidden)`）。

**(b) 驻波调制（位置感知，物理版 RoPE）**

$$SW_m(p) = \cos(k_m p), \quad k_m = \frac{2\pi(m+1)}{N_{ref}}$$

前向波 + 后向波叠加成驻波，振子序号 $p$ 充当空间坐标。$N_{ref}$ 是**参考长度**（`wave_ref_len=512`），与运行时序列长度解耦：

> **P1-1 修复**：旧实现 $k_m = 2\pi(m+1)/N$ 用当前序列长度，导致长度外推时驻波模式错位（同一个 $m$ 在 $N=64$ / $N=128$ 下对应不同绝对波数）。改为固定 $N_{ref}$ 后外推一致。`wave_ref_len=0` 回退旧行为（仅复现旧权重）。

驻波 `[N, modes]` 经可学习投影 `sw_proj` 映射到 hidden 维。300M 有 16 个驻波模式。

**MoE 版（v2.1，300M 未启用）**：把 FFN 换成 8 专家 TopK=2 稀疏路由，门控输入是复值场 $[a\cos\theta, a\sin\theta]$（相位感知路由）。**只拆 FFN，不拆 KuramotoSync** —— 耦合矩阵 $W$ 是 PMNN 的物理本体（相位耦合 = XY 能量梯度的离散化），拆掉就不是物理架构了。

### 4.4 DriftField —— 漂移场

$$V = V^+ - \lambda V^-$$

$$V^+ = \sum_k \text{softmax}_\tau\!\left(-\|z_i - z^{tar}_k\|^2\right)(z^{tar}_k - z_i) \quad \text{（吸引：向目标靠拢）}$$

$$V^- = \sum_j \text{softmax}_\tau\!\left(-\|z_i - z_j\|^2\right)(z_j - z_i) \quad \text{（排斥：防坍缩）}$$

- 多温度 $\tau \in \{\tau, 2\tau, \dots\}$ 求和取多尺度（300M 用 2 个尺度，$\tau=1.0$）
- $\lambda = 0.5$（排斥权重）
- 训练时目标是标签嵌入；推理时用 32 个**可学习原型** `prototypes` 作吸引中心
- 距离在实+虚拼接的 $2D$ 空间上定义

**高效实现**（避免物化 4D 差异张量）：

$$\text{dist}^2_{nm} = \|x_n\|^2 + \|y_m\|^2 - 2 x_n y_m^\top, \quad \text{pull} = \text{einsum}(w, Y) - x \cdot \text{rowsum}(w)$$

> **数值要求（必须 fp32）**：$z = ae^{i\theta}$ 是 $2D$ 维、元素 $\pm a (\max \approx 8)$，$\|z\|^2$ 求和可达 $\sim 10^5$，**超过 fp16 上限 65504 直接溢出**；且 bf16/fp16 下「$\|x\|^2+\|y\|^2-2xy$ 三数相减」有灾难性抵消。距离相关计算全程升 fp32，softmax 权重算完再 cast 回输入 dtype。

### 4.5 Thermodynamics —— 热力学判据层

$$U(Z) = -\sum_{i,j} W_{ij}\cos(\theta_j - \theta_i) \quad \text{（XY 能量：锁相} = \text{能量最小化）}$$

$$S(Z) = -\sum_i p_i \log p_i, \quad p_i = \text{softmax}(a_i) \quad \text{（振幅分布熵）}$$

$$F = U - T S \quad \text{（自由能）}$$

**核心恒等式**：

$$\frac{\partial U}{\partial \theta_i} = \sum_j W_{ij}\sin(\theta_j - \theta_i)$$

即 **Kuramoto 的 sin 耦合就是 XY 能量梯度**。同步层不是"模拟一个物理现象"，而是在做能量最小化。

平均场近似下的能量：

$$U = -\frac{\sum_i w_i \cdot r\cos(\psi - \theta_i)}{\sum_i w_i}$$

**自由能振幅梯度**（状态方程的振幅分量，默认关）：

$$\nabla_a[-U + TS]$$

状态方程 $dZ/dt = MZ + \nabla_Z[-U+TS] + v_\phi + \eta$ 有两个自由度：相位分量 $\nabla_\theta[-U]$ 由 KuramotoSync 承担（且 $S$ 只依赖 $a$，$\nabla_\theta[T S] \equiv 0$）；**振幅分量此前完全缺失**，振幅只靠 FNN 残差更新。用 autograd 精确求导补齐这一半，`create_graph=False`（梯度作物理引导信号，不回传二阶）。

---

## 5. 单轮物理块的数据流

```
(a, θ) ──→ DriftField ──→ 漂移引导 g ──┐
                                        ↓
       ──→ KuramotoSync(κ, g, T) ──→ θ_new     （相位演化，Euler 积分）
                                        ↓
       ──→ StandingWaveFFN(ω_d) ──→ ffn_out    （振幅更新）
                                        ↓
       a_new = softplus(LayerNorm(a + ffn_out))（振幅非负 + 尺度稳定）
                                        ↓
       Thermodynamics(κ, T) ──→ {U, S, F, r}    （判据/诊断）
```

- **相位不做 LayerNorm**（保持周期流形），只在振幅上做
- pad 位置 $a$、$\theta$ 都保持原值，不参与平均场与聚合
- 16 轮物理块串联，每轮的 $t = i/(n_{blocks}-1)$ 驱动时间场

---

## 6. ELF 输出融合

### 6.1 双分支输出

模型输出不是单一路径，而是**解码分支 + 去噪分支**共享物理核心权重：

**(a) 解码分支（语言建模）**

$$\text{logits} = W_{out} \cdot [a\cos\theta,\; a\sin\theta]$$

末步场直接反嵌入到词表，`lm_head: Linear(2D, V)`。

**(b) 去噪分支（ELF 流匹配）**

速度场预测：

$$v_\theta(x_t, t, c) \approx x_1 - x_0$$

数据流：

$$x_t \to \text{共享物理核（1 round）} \to \text{场}[2d] \to \textbf{128 维瓶颈 + LayerNorm} \to \text{升维} \to v$$

**瓶颈是 ELF 的核心技巧**：强制轨迹走低维信息瓶颈，同时让去噪分支与解码分支通过共享的物理核心权重耦合。

流匹配目标构造：

$$x_t = t_r x_1 + (1-t_r) x_0, \quad t_r \sim U[0,1], \quad x_0 \sim \mathcal{N}(0, I), \quad x_1 = z_{\text{干净场}}$$

### 6.2 mode 门控（默认关）

$$g = \sigma\!\left(W_g \cdot [t,\; r(t),\; \kappa(t)]\right)$$

按热力学状态软切换「解码 vs 去噪」：$g$ 大 → $L_{ce}$ 主导；$g$ 小 → $L_{fm}$ 主导。初始偏置 2.0（$g \approx 0.88$），贴近「物理时间末期走解码分支」的设计先验。开启时 $w_{ce}, w_{fm} = g, 1-g$，并 clamp 到 $[0.05, 0.95]$ 防止把另一分支彻底掐死。

---

## 7. 训练目标

总损失（双通道）：

$$L = w_{ce} L_{ce} + w_{fm} L_{fm} + w_{sync} L_{sync} + w_{drift} L_{drift} + w_{free} L_F$$

| 项 | 权重 | 定义 |
|---|---|---|
| $L_{ce}$ | 1.0（或 $g$） | 交叉熵语言建模，`ignore_index=-100` |
| $L_{fm}$ | 0.1（或 $1-g$） | ELF 流匹配速度场 MSE：$\|v_\theta(x_t) - (x_1-x_0)\|^2$ |
| $L_{sync}$ | 0.1 | 序参量课程跟踪：$\text{MSE}(r, r_{target}(t))$ |
| $L_{drift}$ | 0.05 | 漂移场一致性：$\text{MSE}(\phi(x), \text{stopgrad}(\phi(x)+V))$ |
| $L_F$ | 0.05 | 自由能正则：$F + w_{ent} \cdot \text{relu}(S_{target} - S)$ |

**$L_{sync}$ 的课程**（`sync_curriculum` 开启时）：

$$r_{target}(t) = \sigma\!\left(\frac{t - t_c}{\tau_c}\right), \quad t = \frac{i}{n_{blocks}-1}$$

逐层跟踪：第 0 层 $r \approx 0.034$（混沌探索）→ 末层 $r \approx 0.966$（同步整合）。只约束末层会把整条课程曲线压成一个常数 0.9656，等于丢掉 K 值导航。

**$L_F$ 的单边 hinge**：只惩罚 $S < S_{target}$，不能写成 $-w(S - target)$ —— 那会奖励熵无限增长，把振幅分布推成均匀分布 = 结构性坍缩。

所有 mask 加权项都在非 pad 位置统计。

**MoE aux loss**（启用时）：Switch Transformer 式负载均衡 $L = E \cdot \sum_e f_e P_e$，完全均衡时 $=1$。必须把**带梯度的张量**加进总 loss —— detach 成 float 等于把 aux loss 变成常数，路由不学均衡 → 专家坍缩，且训练不报任何错（典型静默失败）。

---

## 8. 关键工程优化

### 8.1 漂移场 einsum 重写

反向传播从 **10.5s → 0.46s**，峰值显存 **0.39GB → 0.11GB**。

### 8.2 动态补齐 + 长度分桶（v2.8）

语料平均 188.7 token、16% 样本正好 512，固定补齐到 512 ⇒ 每步 **63% 算力打在 pad 上**。开启 `dynamic_pad`（按 batch 动态补齐）+ `bucket_batches`（按长度分桶组批）。

> 分桶是动态补齐真正生效的前提：随机组批时 32 条里出现 512 长样本的概率 99.6%，动态补齐等于没开。这不改数据分布（只改组批顺序），loss 只在非 pad 位统计，逐样本逐 token 的值不变。

### 8.3 数值精度护栏

| 位置 | 问题 | 解法 |
|---|---|---|
| DriftField 距离 | $\|z\|^2 \approx 10^5$ 超 fp16 上限 65504 | 全程升 fp32 |
| softmax 分数 | bf16 下 $[B,N,N]$ 点积失精 | score 在 fp32 计算 |
| mask dtype | 硬编码 `.float()` 在 bf16 推理时把场提升成 fp32，下一层 dtype 不匹配 | mask 跟权重/场同 dtype |
| 时间场输出 | 默认 fp32 的 $\kappa$ 在 `dtheta` 处把相位链提升成 fp32 | dtype 跟场状态 $a$ 一致 |
| 驻波 | fp32 的 `sw @ bf16 sw_proj` 不匹配 | 引用 `natural_freq`（稠密/MoE 两版都有）做 dtype 锚 |

### 8.4 防呆与反静默失败

- `--resume` 路径不存在 → **报错退出**，绝不静默从零重训
- checkpoints 目录已有存档但没传 `--resume` → 中止，需显式 `--fresh-ok`
- 同一存档目录被多进程写入 → 单实例锁硬拒绝
- wall-clock 软退出：`max_hours` 到点前自动存盘退出（云端平台到点直接强杀，最后一个存档间隔全丢）

### 8.5 生成端防绕圈

- **不做左 padding**：左 padding 会让内容寻址/驻波位置编码被 pad token 污染，模型对末尾位置的 `<eom>` 概率从 0.0000 飙到 0.9998 → 第一个 token 就输出结束符、生成塌缩。右 padding 与「不 pad 直接变长前向」都正常。采用「不 pad 变长前向」：物理核心所有跨位置运算都按当前序列长度 $N$ 动态计算，变长天然可跑（训练时固定 128 只是 batch 需要）。
- **重复惩罚**（CTRL 口径）：已出现过的 token 正 logit 除以 $p$、负 logit 乘以 $p$。未充分训练的 checkpoint 极易卡进「喜欢喜欢喜欢的喜欢」这类循环 —— 不用重训就能让输出从「复读机」变成可读句子。
- **n-gram 禁词**：禁止重复出现过的 n-gram（0/1 = 关闭）。
- **NaN 兜底**：极端情况下（惩罚把候选全禁了）退回未加惩罚的分布，不产出 NaN。

### 8.6 序参量护栏

训练时把本步序参量 $r$ 喂给自研优化器 PMNNAdafactor，作 $r \to 1$ 全同步坍缩的临界保护（见 §10）。

---

## 9. 实测：训练出来的 Kuramoto 相变

> 这不是设计出来的调度，是**训练之后权重里自然出现的**行为。

用 `pmnn_300m_step13000.pt` 的真实权重做一次前向，逐层读取物理量：

```
layer :      0      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15
r     :  0.317  0.284  0.222  0.129  0.054  0.041  0.157  0.258  0.372  0.424  0.427  0.229  0.156  0.069  0.141  0.462
psi   :  3.115  3.104  3.083  3.103  3.062 -0.044 -0.011 -0.010 -0.013 -0.023 -0.018 -0.005  0.018 -0.166 -0.166 -0.044
kappa :  0.276  0.422  0.639  0.954  1.394  1.981  2.714  3.557  4.443  5.286  6.019  6.606  7.046  7.361  7.578  7.724
temp  :  1.050  0.952  0.857  0.766  0.678  0.594  0.515  0.439  0.369  0.303  0.242  0.188  0.139  0.099  0.067  0.050
```

三个现象：

1. **临界塌缩**：序参量 $r$ 从 0.317 一路降到第 5 层的 **0.041** —— 振子几乎完全失同步，这是 Kuramoto 模型里的临界点；
2. **相位翻转**：同一处平均相位 $\psi$ 从 ≈3.11 跳到 ≈0（跨度约 3.12 弧度，≈**180°**），第 5 层正是翻转面；
3. **退火**：$\kappa$ 从 0.28 单调升到 7.72，$T$ 从 1.05 单调降到 0.05 —— 耦合越来越强、温度越来越低，是典型的**模拟退火**走向。

即模型自己学会了「**先解耦探索、再低温锁相收敛**」的分层策略。Transformer 没有可以画出这张图的中间量。

---

## 10. 优化器

自研 **PMNNAdafactor**（4GB 显存训练 300M 的首选），另支持 AdamW / Adafactor。

- 优化器状态可驻 CPU（`opt_offload`），是 4GB 显存训练 300M 的关键开关
- 接收序参量 $r$ 作护栏输入：$r \to 1$ 时触发全同步坍缩临界保护
- checkpoint 默认携带优化器动量（续训必需；不存的话 AdamW 动量从零重建，续训初期 loss 反弹）

---

## 11. 训练概况

- 规模预设 `300m`：$d_{model}=1024$ / $d_{field}=1024$ / $n_{blocks}=16$ / $ffn_{hidden}=4096$ / $r_{rank}=96$
- 数据：中文清洗语料约 2.13 亿 token
- 训练：自回归语言建模（next-token），深度轴上的 $\kappa/T$ 退火调度
- 硬件：单机 RTX 3050 Laptop (4GB) 起训，云端续训
- 关键工程优化：漂移场用 einsum 重写，反向从 10.5s → 0.46s，峰值显存 0.39GB → 0.11GB

**Chinchilla 对照**：

| | 本模型 | Chinchilla 最优 |
|---|---|---|
| token / 参数 | ≈ **1** | ≈ 20 |
| 差距 | — | 只喂了约 **1/20** |

数据量差两个数量级。这组权重**只能证明架构能训练、能生成语言**，不能证明它能用。

---

## 12. 已知缺陷（技术层面）

1. **会复读** —— 训练不充分的典型症状。已用重复惩罚 + n-gram 禁词缓解，无法根除。
2. **答非所问** —— 它学的是「续写」不是「回答」：**无 SFT、无指令对齐、无 RLHF**。
3. **事实错误率高** —— 没有知识注入训练，事实性完全不可信。
4. **收尾不稳** —— 有时停不下来，有时过早输出 `<eom>`。
5. **中文生硬** —— 语料规模差得远，能成句但不自然。
6. **词表偏小** —— 8192 词表，中文约 0.7~1.0 字 / token；扩充到 32K 可把压缩比改善到 1.228×。
7. **上下文短** —— 训练 128，超过会截断。

根因是数据量差两个数量级，不是架构问题。

---

## 13. 真实输出示例

`step13000`，`--max-new 24`：

```
输入：你好，请介绍一下你自己。
输出： 你好，我想想到个隐藏自己的朋友出
```

```
输入：南京是
输出： 东省，是南美洲。
```

**能起句、能吐中文词，但逻辑和事实完全不可信。** 这是预期现象。

---

## 14. 一页纸公式速查

| 概念 | 公式 |
|---|---|
| 复值振子 | $z_i = a_i e^{i\theta_i}$ |
| Kuramoto 状态方程 | $\dot\theta_i = \omega_i + \sum_j W_{ij}\sin(\theta_j-\theta_i) + \kappa g_i + \eta$ |
| 序参量 | $r = \big\|\frac{\sum_i w_i e^{i\theta_i}}{\sum_i w_i}\big\|$, $w_i = \text{softplus}(a_i)$ |
| 平均场耦合 | $\sum_j \sin(\theta_j-\theta_i) \to r\sin(\psi-\theta_i)$ |
| XY 能量 | $U = -\sum_{ij} W_{ij}\cos(\theta_j-\theta_i)$ |
| 能量-耦合恒等式 | $\partial U/\partial\theta_i = \sum_j W_{ij}\sin(\theta_j-\theta_i)$ |
| 振幅熵 | $S = -\sum_i \text{softmax}(a_i)\log\text{softmax}(a_i)$ |
| 自由能 | $F = U - TS$ |
| 洛伦兹共振 | $A(\omega_d,\omega_n) = [(\omega_d^2-\omega_n^2)^2+(2\gamma\omega_d)^2]^{-1/2}$ |
| 驻波 | $SW_m(p) = \cos(2\pi(m+1)p/N_{ref})$ |
| 耦合强度课程 | $\kappa(t) = \kappa_{max}\sigma((t-t_c)/\tau_c)$ |
| 温度退火 | $T(t) = T_{max}(1-t)^p + T_{min}$ |
| 内容寻址耦合 | $W_{ij} \propto \exp(q_ik_j/\sqrt r - \lambda(i-j))$, $i\ge j$ |
| 因果衰减前缀和 | $A_i = e^{-\lambda i}\odot\text{cumsum}(e^{\lambda j}\odot v_j)$ |
| 流匹配目标 | $v = x_1 - x_0$, $x_t = t_rx_1+(1-t_r)x_0$ |
| 总损失 | $L = w_{ce}L_{ce} + w_{fm}L_{fm} + w_{sync}L_{sync} + w_{drift}L_{drift} + w_{free}L_F$ |
