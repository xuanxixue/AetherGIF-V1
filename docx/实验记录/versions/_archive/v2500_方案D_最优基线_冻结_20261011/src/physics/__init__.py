# -*- coding: utf-8 -*-
"""
PMNN 物理矩阵神经网络 —— 通用底层架构（视频/GIF 生成专用）。

严格按 AetherMind-V5 技术文档实现，五大算子一个不少：
  §4.1 TimeField       时间场：深度 = 物理时间，三条协议曲线
  §4.2 KuramotoSync    相位同步（替代注意力）：四条耦合通路
  §4.3 StandingWaveFFN 共振-驻波前馈（替代 FFN）
  §4.4 DriftField      漂移场：吸引-排斥双分量
  §4.5 Thermodynamics  热力学判据层：U / S / F

零简化声明：
  - 不含任何 QKV 注意力
  - 不含任何标准 FFN（只有共振-驻波形式的频率选择性放大）
  - 不含任何位置编码（只有驻波，物理版 RoPE）
  - Kuramoto sin 耦合 = XY 能量梯度，物理同源性保留
  - 相位不做 LayerNorm（周期流形），只归一振幅
"""
