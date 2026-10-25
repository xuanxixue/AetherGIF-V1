# -*- coding: utf-8 -*-
"""
三场分层物理注意力（Three-Field Physical Attention）—— 零频率先验版。

病根回顾
--------
模型没有"内容性选择"：Kuramoto 耦合只看空间距离（Toeplitz 衰减）和全局相位
（平均场），不看两个频点承载的内容是否相关，导致噪声和细节被无差别耦合/学习。

三场分层（用户定义，与华为 WEWA 命名无关，概念独立成立）
-------------------------------------------------------
把每个频点按"内容重要性"分入三个场：

  核心场 core  —— 该学的（结构、必须保真的内容）
  边缘场 edge  —— 需泛化的（可软化拟合、但要在空间上连贯）
  外层场 outer —— 绝不能学的（纯噪声，学了=过拟合=烧焦感）

关键设计约束（本条是用户本人纠正，务必守住）
--------------------------------------------
**场归属与频率位置无关。** 低频完全可能是大面积闪烁噪声（应进外层场），
高频完全可能是锐利边缘细节（应进核心场）。因此：

  ✗ 禁止用「低频→核心 / 中频→边缘 / 高频→外层」的单调 sigmoid 切点
    （w1 < w2 沿频率轴切分 = 拍脑袋锁死，正是我们犯过的错）
  ✓ 每个频点 c 持有自己独立的三个 logits，softmax 归一为 (g_core, g_edge, g_outer)
    —— 频点之间无共享、无平滑、无频率序诱导，归属完全由数据自己分化

公式
----
  g_core(c), g_edge(c), g_outer(c) = softmax(logits_c)        （和 = 1，互斥分配）

  生效点 1（耦合隔离）：couple_g(c) = g_core(c) + g_edge(c)  = 1 − g_outer(c)
    乘进 Kuramoto 耦合矩阵的「源端」，外层场通道不再向其它振子传播噪声相位。

  生效点 2（学习强度）：learn_g(c) = g_core(c) + edge_w · g_edge(c)
    edge_w = sigmoid(edge_soften) ∈ (0,1)，边缘场在 loss 上被软化（只泛化不硬拟合）。
    learn_g 乘进振幅 loss，外层场通道的振幅惩罚归零 = 停止学习噪声系数。

参数量
------
  field_logits: D×3（D=125 → 375）
  edge_soften:  1
  合计 376 个参数，相对基线 1,135,658 仅 +0.03%——这正是「零参数物理注意力」
  的意义：不加任何可学习的内容相似度打分器，只加一段"该学谁"的归属表。

初始化
------
  field_logits 第一列（core）预置 +5.0，其余 0：
    softmax([5,0,0]) ≈ [0.928, 0.036, 0.036] → couple_g≈0.964, learn_g≈0.946
  即初始所有频点几乎都是核心场，行为与 v3700 基线近乎一致（从基线起步），
  随后由数据把噪声频点逐批「推出」到 edge / outer。edge_soften 初始 sigmoid(0)=0.5。

  这与「频率 sigmoid 切点」的本质区别：切点是**人规定**的频率单调函数，
  这里是**逐频点自由参数**，两者的梯度自由度完全不同。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldAttention(nn.Module):
    """三场分层物理注意力：逐频点可学习场归属，无频率顺序先验。"""

    def __init__(self, d_model: int, core_init: float = 5.0):
        super().__init__()
        self.d_model = d_model
        # 逐频点独立的三场 logits，行和经 softmax 归一为 (core, edge, outer)
        # 初始 core 列预置 core_init，让模型从「全是核心场（=基线）」起步再分化
        logits = torch.zeros(d_model, 3)
        logits[:, 0] = core_init
        self.field_logits = nn.Parameter(logits)
        # 边缘场在 loss 上的软化系数：sigmoid → (0,1)，初始 0.5（半拟合）
        self.edge_soften = nn.Parameter(torch.tensor(0.0))

    def field_probs(self):
        """三场互斥归属 [D, 3]，行和 = 1。"""
        return F.softmax(self.field_logits, dim=-1)

    def forward(self):
        """返回 (couple_g, learn_g)，均为 [D]。

        couple_g: 耦合参与度，1 − g_outer（外层场被隔离，不传播噪声相位）
        learn_g:  学习强度，g_core + edge_w·g_edge（外层场不拟合、边缘场软化）
        """
        g = self.field_probs()                       # [D, 3]
        g_core, g_edge, g_outer = g[:, 0], g[:, 1], g[:, 2]
        edge_w = torch.sigmoid(self.edge_soften)
        couple_g = g_core + g_edge                    # = 1 − g_outer
        learn_g = g_core + edge_w * g_edge
        return couple_g, learn_g