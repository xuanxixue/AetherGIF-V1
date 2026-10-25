# -*- coding: utf-8 -*-
"""cards_b.py — 卡片 06-11：第二幕「五块物理」"""
import math
from deck import *
from deck import CUR

TOTAL = 18


# ============================================================ 06 DriftField
def card_06():
    c = Card(6, TOTAL, "PHYSICS 01 · DRIFT FIELD", "唯一的目标信息通道",
             "初始 a₀/θ₀ 是噪声，ω 是确定性的。所以目标信息想进入相位演化，只剩这一条路。",
             accent=AM, uid="c06")
    c.addcss("""
@keyframes driftIn{0%{opacity:0;transform:scale(.3)}100%{opacity:1;transform:scale(1)}}
.drift-p{animation:driftIn 3.2s cubic-bezier(.2,.8,.2,1) both}
@keyframes pullLine{0%{opacity:0;transform:scaleX(.1)}40%{opacity:.9}100%{opacity:.25;transform:scaleX(1)}}
.pull{animation:pullLine 3.2s ease-out both;transform-box:fill-box;transform-origin:left center}
""")
    # 左：公式 + 两个设计决策
    c.add(code(150, 350, 830, 190, [
        "g_i = Σ_j softmax_j(−‖ẑ_i − ẑ_j^tar‖² / τ)",
        "        · sin(θ_j^tar − θ_i)",
    ], accent=AM, size=22, lead=40, title="DRIFT FIELD · 相位空间吸引子"))
    c.add(f'<g class="anim-rise" style="animation-delay:.2s">')
    c.add(panel(150, 570, 830, 160, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=16))
    c.add(badge_num(184, 618, "1", AM))
    c.add(T(220, 626, "距离必须 L2 归一化", 23, C["ink"], "800"))
    c.add(Twrap(220, 660, "模型与目标振幅可差 90×。不归一化，softmax logits 被 ‖zᵢ‖² 主导，"
                          "吸引退化成「拉向原点」，目标相位信息全部丢失。", 730, 18, C["ink2"], "500", 26))
    c.add(Twrap(220, 712, "归一化后 ẑ·ẑ = mean_d cos(θᵢ−θⱼ)：距离变成了相位余弦距离。",
                730, 18, AM, "600", 26))
    c.add("</g>")
    c.add(f'<g class="anim-rise" style="animation-delay:.34s">')
    c.add(panel(150, 750, 830, 160, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=16))
    c.add(badge_num(184, 798, "2", AM))
    c.add(T(220, 806, "力必须是 sin(θⱼᵗᵃʳ − θᵢ)", 23, C["ink"], "800"))
    c.add(Twrap(220, 840, "周期有界 [−1,1]，恰好是相位流形上的梯度方向；θᵢ 追上目标时力自动归零。",
                730, 18, C["ink2"], "500", 26))
    c.add(Twrap(220, 892, "若直接用复数差 (yⱼ − xᵢ)，幅值不匹配会污染相位更新，"
                          "模型会去「压幅度」而不是「对齐相位」。", 730, 18, AM, "600", 26))
    c.add("</g>")
    # 右：吸引子收敛动画
    rx = 1030
    c.add(panel(rx, 350, 740, 560, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=20))
    c.add(T(rx + 32, 396, "吸引子如何工作", 23, AM, "800"))
    c.add(T(rx + 32, 428, "40 个振子被各自的相位吸引子拉向目标相位", 17, C["ink3"], "500"))
    # 目标点
    import random as _r
    rnd = _r.Random(21)
    ox, oy = rx + 370, 630
    tgt = []
    for i in range(10):
        a = i * 0.628
        rr = 105 + (i % 3) * 46
        tgt.append((ox + math.cos(a) * rr, oy + math.sin(a) * rr * 0.72, a))
    for (tx, ty, a) in tgt:
        c.add(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="6" fill="{AM}" opacity="0.5" '
              f'filter="url(#c06_glows)"/>')
        c.add(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="13" fill="none" stroke="{AM}" '
              f'stroke-width="1" opacity="0.28"/>')
    # 拉线 + 移动的振子
    for i in range(40):
        t = tgt[i % 10]
        sx = ox + rnd.uniform(-250, 250)
        sy = oy + rnd.uniform(-160, 160)
        dx, dy = t[0] - sx, t[1] - sy
        dist = math.hypot(dx, dy) or 1
        c.add(f'<line x1="{sx:.0f}" y1="{sy:.0f}" x2="{t[0]:.0f}" y2="{t[1]:.0f}" '
              f'stroke="{AM}" stroke-width="1" opacity="0.20" class="pull" '
              f'style="animation-delay:{(i%10)*0.14:.2f}s"/>')
        k = 0.34 + 0.05 * (i % 6)
        c.add(f'<circle cx="{sx:.0f}" cy="{sy:.0f}" r="4" fill="{CY if i%3 else MG}" '
              f'opacity="0.9" class="drift-p" '
              f'style="animation-delay:{(i%10)*0.14:.2f}s;'
              f'transform-origin:{sx:.0f}px {sy:.0f}px;'
              f'--tx:{dx*k:.0f}px;--ty:{dy*k:.0f}px;'
              f'transform:translate({dx*k:.0f}px,{dy*k:.0f}px);'
              f'animation:converge 3.2s cubic-bezier(.2,.8,.2,1) both"/>')
    c.add(f'<circle cx="{ox}" cy="{oy}" r="3" fill="{C["ink3"]}" opacity="0.5"/>')
    # 底部：多尺度 τ
    c.add(f'<g class="anim-rise" style="animation-delay:.6s">')
    c.add(panel(rx + 32, 800, 676, 86, fill="#0C1424", stroke=CY, op=0.7, rx=14))
    c.add(T(rx + 56, 838, "多尺度 τ（两个八度）", 19, CY, "700"))
    c.add(T(rx + 56, 868, "外加逐振子直连项 sin(θᵗᵃʳ−θ) 与可学习原型（推理时向原型松弛）",
            17, C["ink3"], "500"))
    c.add("</g>")
    return c.render()


# ============================================================ 07 Kuramoto
def card_07():
    c = Card(7, TOTAL, "PHYSICS 02 · KURAMOTO SYNCHRONIZATION", "用同步替掉注意力",
             "注意力问的是「谁该关注谁」。Kuramoto 问的是「谁的相位该被谁拉动」。耦合权重由物理给出，不需学习。",
             accent=CY, uid="c07")
    c.addcss("""
@keyframes lockIn{0%{transform:rotate(0deg)}100%{transform:rotate(var(--to,0deg))}}
.lock{animation:lockIn 6s cubic-bezier(.3,.7,.3,1) both;transform-box:fill-box;transform-origin:center}
@keyframes rGrow{0%{transform:scaleX(.06)}100%{transform:scaleX(1)}}
.rbar{animation:rGrow 6s cubic-bezier(.2,.8,.2,1) both;transform-origin:left center}
@keyframes orderVec{0%{opacity:.2;transform:scale(.1)}100%{opacity:1;transform:scale(1)}}
.ovec{animation:orderVec 6s cubic-bezier(.2,.8,.2,1) both;transform-box:fill-box;transform-origin:center}
""")
    # 左：相位圆
    cx, cy, R = 470, 620, 205
    c.add(panel(150, 350, 700, 570, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=20))
    c.add(T(186, 396, "18 个振子，序参量 r 从 0.06 涨到 0.94", 21, CY, "800"))
    c.add(T(186, 426, "r·e^(iψ) 是所有相位的复平均——r 就是「大家有多同步」", 16, C["ink3"], "500"))
    c.add(f'<g transform="translate({cx},{cy})">')
    c.add(f'<circle r="{R}" fill="none" stroke="{C["line"]}" stroke-width="1.4"/>')
    c.add(f'<circle r="{R*0.66}" fill="none" stroke="{C["line"]}" stroke-width="1" opacity="0.5"/>')
    for i in range(24):
        a = i * math.pi / 12
        c.add(f'<line x1="{math.cos(a)*(R-10):.1f}" y1="{math.sin(a)*(R-10):.1f}" '
              f'x2="{math.cos(a)*R:.1f}" y2="{math.sin(a)*R:.1f}" stroke="{C["ink3"]}" '
              f'stroke-width="1" opacity="0.6"/>')
    # 18 个振子，各自不同的初始角和锁定期
    import random as _r
    rnd = _r.Random(33)
    for i in range(18):
        a0 = rnd.uniform(0, 2 * math.pi)
        a1 = 0.30 + rnd.uniform(-0.16, 0.16)
        px, py = math.cos(a1) * R, math.sin(a1) * R
        col = CY if i % 3 else MG
        c.add(f'<g class="lock" style="--to:{(a1-a0)*57.2958:.1f}deg;'
              f'animation-delay:{.2+i*0.12:.2f}s;transform-box:view-box;'
              f'transform-origin:{cx}px {cy}px">'
              f'<circle cx="{math.cos(a0)*R:.1f}" cy="{math.sin(a0)*R:.1f}" r="7" '
              f'fill="{col}" opacity="0.92" filter="url(#c07_glows)"/></g>')
    # 序参量矢量
    c.add(f'<g class="ovec" style="animation-delay:1.6s">')
    c.add(f'<line x1="0" y1="0" x2="{math.cos(0.30)*R*0.94:.1f}" '
          f'y2="{math.sin(0.30)*R*0.94:.1f}" stroke="{AM}" stroke-width="4" '
          f'filter="url(#c07_glows)"/>')
    c.add(f'<circle cx="{math.cos(0.30)*R*0.94:.1f}" cy="{math.sin(0.30)*R*0.94:.1f}" '
          f'r="8" fill="{AM}" filter="url(#c07_glow)"/>')
    c.add("</g>")
    c.add("</g>")
    c.add(T(cx, cy + R + 34, "0", 16, C["ink3"], "600", anchor="middle"))
    c.add(T(cx + R + 26, cy + 6, "π/2", 16, C["ink3"], "600"))
    # r 进度条
    c.add(f'<g class="anim-rise" style="animation-delay:.3s">')
    c.add(T(196, 878, "序参量 r", 17, C["ink3"], "600"))
    c.add(f'<rect x="266" y="866" width="450" height="12" rx="6" fill="#101A2C"/>')
    c.add(f'<rect class="rbar" x="266" y="866" width="450" height="12" rx="6" fill="{AM}"/>')
    c.add(T(790, 878, "0.06 → 0.94", 18, AM, "700", mono=True))
    c.add("</g>")
    # 右：四条耦合通路
    rx = 900
    c.add(T(rx, 372, "四条耦合通路", 24, C["ink"], "800"))
    paths = [
        ("①", "均场序参量耦合", "Σ sin(θⱼ−θᵢ) → r·sin(ψ−θᵢ)", "秩-1、零参数；归一化必须用 Σwᵢ 而非 N，否则 r 溢出 [0,1]", CY),
        ("②", "时空 Toeplitz 耦合", "W_ij = exp(−λ_t|tᵢ−tⱼ| − λ_s‖pᵢ−pⱼ‖²)", "时间衰减给光锥因果性，空间高斯衰减给局部性", MG),
        ("③", "漂移引导", "κ(t) · g_i", "来自 DriftField 的目标吸引子，随课程逐步增强", AM),
        ("④", "朗之万噪声", "η ~ N(0, 2T·Δt)", "涨落—耗散联系；Δt 因子必须带上才能正确缩放", GR),
    ]
    y = 404
    for i, (no, name, formula, desc, col) in enumerate(paths):
        c.add(f'<g class="anim-riseL" style="animation-delay:{.15+i*0.12}s">')
        c.add(panel(rx, y, 870, 112, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=14))
        c.add(f'<rect x="{rx}" y="{y}" width="4" height="112" rx="2" fill="{col}"/>')
        c.add(T(rx + 26, y + 34, no, 21, col, "800"))
        c.add(T(rx + 62, y + 34, name, 21, C["ink"], "800"))
        c.add(f'<rect x="{rx+26}" y="{y+44}" width="818" height="30" rx="8" fill="#060A14" opacity="0.8"/>')
        c.add(T(rx + 40, y + 65, formula, 17, col, "600", mono=True))
        c.add(Twrap(rx + 26, y + 94, desc, 810, 15, C["ink3"], "500", lead=20, maxlines=1))
        c.add("</g>")
        y += 122
    # 数值细节
    c.add(f'<g class="anim-rise" style="animation-delay:.7s">')
    c.add(panel(rx, 902, 870, 44, fill="#0C1424", stroke=CY, op=0.7, rx=12))
    c.add(T(rx + 22, 930, "Euler dt=0.1 · Ω_max=8 的 tanh 软饱和 · 相位链永不折叠（只在 loss 里 wrap）",
            16, CY, "600"))
    c.add("</g>")
    return c.render()


# ============================================================ 08 StandingWaveFFN
def card_08():
    c = Card(8, TOTAL, "PHYSICS 03 · STANDING WAVE FFN", "用共振替掉 FFN",
             "普通 FFN 靠矩阵乘做通道混合。这里通道混合改由一条洛伦兹共振曲线决定——扫频穿过谁，谁就被放大。",
             accent=MG, uid="c08")
    c.addcss("""
@keyframes sweep{0%{transform:translateX(0)}100%{transform:translateX(430px)}}
.sw{animation:sweep 5.5s cubic-bezier(.45,0,.55,1) infinite alternate}
@keyframes peakPulse{0%,100%{opacity:.35;r:16}50%{opacity:1;r:26}}
.pk{animation:peakPulse 2.6s ease-in-out infinite}
@keyframes wavePhase{0%{transform:translateX(0)}100%{transform:translateX(60px)}}
.wv{animation:wavePhase 4s linear infinite}
""")
    # 左：洛伦兹共振曲线
    c.add(panel(150, 350, 700, 560, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=20))
    c.add(T(186, 396, "洛伦兹共振增益", 22, MG, "800"))
    c.add(T(186, 426, "A(ω_d, ω_n) = 1/√((ω_d²−ω_n²)² + (2γω_d)²)", 16, C["ink3"], "500", mono=True))
    # 画曲线
    gx, gy, gw, gh = 240, 480, 540, 260
    pts = []
    for i in range(121):
        t = i / 120
        wd = 0.25 + t * 2.4
        wn = 1.0
        gamma = 0.3
        A = 1.0 / math.sqrt((wd * wd - wn * wn) ** 2 + (2 * gamma * wd) ** 2)
        pts.append((gx + gw * t, gy + gh - gh * min(A / 1.9, 1.0)))
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    c.add(f'<path d="{d}" fill="none" stroke="{MG}" stroke-width="3.2" stroke-linecap="round" '
          f'filter="url(#c08_glows)"/>')
    # 峰值
    pk = min(pts, key=lambda p: p[1])
    c.add(f'<circle class="pk" cx="{pk[0]:.1f}" cy="{pk[1]:.1f}" fill="none" stroke="{AM}" stroke-width="2.4"/>')
    c.add(f'<circle cx="{pk[0]:.1f}" cy="{pk[1]:.1f}" r="6" fill="{AM}"/>')
    # 扫频标记
    c.add(f'<g class="sw"><line x1="{gx}" y1="{gy-24}" x2="{gx}" y2="{gy+gh}" stroke="{CY}" '
          f'stroke-width="2" opacity="0.75" filter="url(#c08_glows)"/></g>')
    c.add(f'<text x="{gx}" y="{gy+gh+38}" font-size="15" fill="{C["ink3"]}" font-weight="600">ω_d 低</text>')
    c.add(f'<text x="{gx+gw}" y="{gy+gh+38}" font-size="15" fill="{C["ink3"]}" font-weight="600" '
          f'text-anchor="end">ω_d 高</text>')
    c.add(f'<text x="{gx+gw*0.45}" y="{gy-30}" font-size="16" fill="{AM}" font-weight="700">ω_d 扫过 ω_n → 该通道被放大</text>')
    c.add(f'<line x1="{gx+gw*0.45}" y1="{gy-22}" x2="{gx+gw*0.45}" y2="{gy}" stroke="{AM}" '
          f'stroke-width="1" stroke-dasharray="4 4" opacity="0.6"/>')
    c.add(f'<g class="anim-rise" style="animation-delay:.4s">')
    c.add(panel(186, 792, 628, 96, fill="#0C1424", stroke=MG, op=0.7, rx=14))
    c.add(T(210, 828, "频率选择性 = 物理版的「通道混合」", 19, MG, "700"))
    c.add(T(210, 858, "γ=0.3 决定共振峰的锐度；ω_n 是每个通道可学习的本征频率", 16, C["ink3"], "500"))
    c.add("</g>")
    # 右：驻波 + 双残差头
    rx = 900
    c.add(code(rx, 350, 870, 168, [
        "h    = W₂( GELU(W₁·z) ⊙ A(ω_d,ω_n) ⊙ SW(p) )",
        "out  = h → (log-amplitude 残差, phase 残差)",
    ], accent=MG, size=20, lead=40, title="STANDING WAVE FFN"))
    # 3D 驻波
    c.add(panel(rx, 542, 870, 200, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=18))
    c.add(T(rx + 28, 580, "3D 驻波调制 = 物理版位置编码", 20, CY, "800"))
    c.add(T(rx + 28, 608, "SW_m(t,x,y) = cos(2πm_t t/T)·cos(2πm_x x/X)·cos(2πm_y y/Y)", 15, C["ink3"], "500", mono=True))
    # 波形
    for k in range(4):
        yy = 648 + k * 22
        pth = []
        for i in range(81):
            t = i / 80
            v = math.cos(2 * math.pi * (2 + k) * t)
            pth.append(f"{rx+40+t*790:.1f} {yy+v*9:.1f}")
        c.add(f'<path d="M{" L".join(pth)}" fill="none" stroke="{[CY,MG,AM,GR][k]}" '
              f'stroke-width="1.8" opacity="{0.75-k*0.13:.2f}" class="wv" '
              f'style="animation-delay:{k*0.3:.1f}s"/>')
    # 底部两条设计要点
    c.add(f'<g class="anim-rise" style="animation-delay:.3s">')
    c.add(panel(rx, 762, 425, 148, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=16))
    c.add(T(rx + 24, 800, "为什么必须两个残差头？", 19, AM, "800"))
    for i, ln in enumerate(wrap("受迫振子的传递函数是 |H(ω)|·e^(iφ(ω))，同时改幅度和相位。"
                                "少了相位头，相位只剩 ω 偏置和 κ 两个弱自由度，"
                                "加权相位余弦只会震荡、不会收敛。", 16, 377)[:4]):
        c.add(T(rx + 24, 830 + i * 24, ln, 16, C["ink2"], "500"))
    c.add("</g>")
    c.add(f'<g class="anim-rise" style="animation-delay:.44s">')
    c.add(panel(rx + 445, 762, 425, 148, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=16))
    c.add(T(rx + 469, 800, "为什么 W₂ 零初始化？", 19, GR, "800"))
    for i, ln in enumerate(wrap("模型起点是恒等映射，最初几步完全由物理（Kuramoto + 初始条件）"
                                "决定，层只学后续修正。驻波参考长度固定，"
                                "因此跨时长外推行为一致。", 16, 377)[:4]):
        c.add(T(rx + 469, 830 + i * 24, ln, 16, C["ink2"], "500"))
    c.add("</g>")
    return c.render()


# ============================================================ 09 OU 振幅
def card_09():
    c = Card(9, TOTAL, "PHYSICS 04 · OU RESTORING FORCE", "振幅在 log 域上乘性演化",
             "FFT 振幅横跨几个数量级，加性更新会锁死动态范围、把重尾频谱压平。log 域 + 恢复力才是正解。",
             accent=GR, uid="c09")
    c.addcss("""
@keyframes growUp{from{transform:scaleY(0)}to{transform:scaleY(1)}}
.gy{animation:growUp 1.8s cubic-bezier(.2,.8,.2,1) both;transform-box:fill-box;transform-origin:bottom}
@keyframes oscPulse{0%,100%{opacity:.5}50%{opacity:1}}
.op{animation:oscPulse 2.2s ease-in-out infinite}
""")
    # 左：公式
    c.add(code(150, 350, 800, 260, [
        "res       = 2·tanh(log_res / 2)        # 每层残差有界",
        "log a_new = log a + res − γ·(log a − μ)",
        "log a_new = clamp(log a_new, −20, 5)",
        "a_new     = exp(log a_new)",
    ], accent=GR, size=20, lead=40, title="PLAN D 振幅递推（log 域）"))
    c.add(f'<g class="anim-rise" style="animation-delay:.2s">')
    c.add(panel(150, 640, 800, 130, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=16))
    c.add(T(178, 682, "为什么必须乘性？", 21, GR, "800"))
    c.add(T(178, 716, "纯累加下振幅 12 层会爆到目标的 100×。而 −γ(log a − μ) 是布朗运动加恢复力，",
            18, C["ink2"], "500"))
    c.add(T(178, 746, "其 Fokker–Planck 平稳分布有界——这是经典的稳定化机制。", 18, C["ink2"], "500"))
    c.add("</g>")
    c.add(f'<g class="anim-rise" style="animation-delay:.34s">')
    c.add(panel(150, 790, 800, 120, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=16))
    c.add(T(178, 830, "为什么用 log1p 之外的 log 域？", 20, AM, "800"))
    c.add(T(178, 864, "log 域乘性更新让梯度在对数尺度上均匀，各通道都能被有效监督。",
            17, C["ink2"], "500"))
    c.add("</g>")
    # 右：直方图 + 方案D弱点
    rx = 1000
    c.add(panel(rx, 350, 770, 300, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=20))
    c.add(T(rx + 30, 394, "通道振幅的量纲差异有多大？", 21, AM, "800"))
    # 直方图
    bx, by, bw, bh = rx + 40, 470, 690, 130
    vals = [1.0, 0.42, 0.20, 0.115, 0.075, 0.05, 0.036, 0.028, 0.021, 0.016, 0.012, 0.009]
    for i, v in enumerate(vals):
        h = bh * (v / vals[0]) ** 0.42
        w = bw / len(vals) - 9
        x = bx + i * (bw / len(vals))
        col = AM if i == 0 else (RD if i < 3 else GR)
        c.add(f'<rect class="gy" x="{x:.1f}" y="{by+bh-h:.1f}" width="{w:.1f}" height="{h:.1f}" '
              f'rx="4" fill="{col}" opacity="{0.9-i*0.045:.2f}" '
              f'style="animation-delay:{.1+i*0.07:.2f}s"/>')
    c.add(f'<line x1="{bx}" y1="{by+bh}" x2="{bx+bw}" y2="{by+bh}" stroke="{C["line"]}" stroke-width="1.4"/>')
    c.add(T(bx, by + bh + 30, "0.0247（中频）", 15, C["ink3"], "600", mono=True))
    c.add(T(bx + bw, by + bh + 30, "7.84（DC）", 15, AM, "700", anchor="end", mono=True))
    c.add(T(rx + 30, 632, "各通道均值最大差 727× · std/mean ≈ 5.75 · 重尾到 p99≈5.4 / 中位数 0.0015",
            16, C["ink3"], "500"))
    # 方案 D 弱点
    c.add(f'<g class="anim-rise" style="animation-delay:.4s">')
    c.add(panel(rx, 672, 770, 238, fill="#120C10", stroke=RD, op=0.72, rx=20))
    c.add(f'<rect x="{rx}" y="672" width="770" height="3" rx="1.5" fill="{RD}" opacity="0.9"/>')
    c.add(T(rx + 30, 716, "方案 D 的已知弱点（文档如实记录）", 20, RD, "800"))
    c.add(T(rx + 30, 754, "一个 γ 同时管两件事：水平收敛到 μ，以及形状衰减。", 17, C["ink2"], "500"))
    c.add(T(rx + 30, 786, "当 γ 自训到 ≈0.04，初始水平的 61% 撑过 12 层，μ 压不住它。",
            17, C["ink2"], "500"))
    c.add(T(rx + 30, 822, "另外：架构在 log 域递推，损失却在 log1p 域聚合，两者从第一步就分岔",
            17, AM, "600"))
    c.add(T(rx + 30, 852, "（实测差 ≈3.05 个单位）——这就是 μ 一直振荡而不收敛的原因。", 17, AM, "600"))
    c.add(T(rx + 30, 890, "v3600 保留原状以维持基线保真度；后续版本把 γ 拆成 γ_level / γ_shape。",
            16, C["ink3"], "500"))
    c.add("</g>")
    return c.render()


# ============================================================ 10 TimeField
def card_10():
    c = Card(10, TOTAL, "PHYSICS 05 · TIME FIELD", "深度就是物理时间",
             "12 层物理块不是「堆得更深」，而是一条从混乱到锁定的热力学时间线。",
             accent=CY, uid="c10")
    c.addcss("""
@keyframes playhead{0%{transform:translateX(0)}100%{transform:translateX(1080px)}}
.ph{animation:playhead 7s linear infinite}
@keyframes sigFill{0%{opacity:.25}50%{opacity:1}100%{opacity:.25}}
.sg{animation:sigFill 7s ease-in-out infinite}
""")
    # 三条课程曲线
    gx, gy, gw, gh = 250, 420, 1080, 230
    c.add(T(150, 380, "三条课程曲线，全部由 t = block/(R−1) ∈ [0,1] 驱动", 21, C["ink"], "800"))
    # 网格
    for i in range(11):
        x = gx + gw * i / 10
        c.add(f'<line x1="{x:.0f}" y1="{gy}" x2="{x:.0f}" y2="{gy+gh}" stroke="{C["line2"]}" stroke-width="1"/>')
    c.add(f'<line x1="{gx}" y1="{gy+gh}" x2="{gx+gw}" y2="{gy+gh}" stroke="{C["line"]}" stroke-width="1.6"/>')
    c.add(f'<line x1="{gx}" y1="{gy}" x2="{gx}" y2="{gy+gh}" stroke="{C["line"]}" stroke-width="1.6"/>')
    # κ 课程
    def sig(x): return 1 / (1 + math.exp(-(x - 0.52) * 9))
    p1 = [(gx + gw * i / 100, gy + gh - gh * sig(i / 100)) for i in range(101)]
    c.add(f'<path d="M{" L".join(f"{x:.1f} {y:.1f}" for x,y in p1)}" fill="none" stroke="{AM}" '
          f'stroke-width="3.4" stroke-linecap="round" filter="url(#c10_glows)"/>')
    # T 退火
    p2 = [(gx + gw * i / 100, gy + gh - gh * (0.12 + 0.85 * (1 - i / 100) ** 2.1)) for i in range(101)]
    c.add(f'<path d="M{" L".join(f"{x:.1f} {y:.1f}" for x,y in p2)}" fill="none" stroke="{RD}" '
          f'stroke-width="3.4" stroke-linecap="round"/>')
    # ω_d 几何扫频
    p3 = [(gx + gw * i / 100, gy + gh - gh * (0.15 + 0.8 * (i / 100) ** 1.25)) for i in range(101)]
    c.add(f'<path d="M{" L".join(f"{x:.1f} {y:.1f}" for x,y in p3)}" fill="none" stroke="{CY}" '
          f'stroke-width="3.4" stroke-linecap="round"/>')
    # playhead
    c.add(f'<g class="ph"><line x1="{gx}" y1="{gy-20}" x2="{gx}" y2="{gy+gh+16}" stroke="{MG}" '
          f'stroke-width="2.4" opacity="0.85" filter="url(#c10_glows)"/></g>')
    # 标签
    labs = [(AM, "κ(t) 耦合强度", "混乱 → 锁定", gy - 26),
            (RD, "T(t) 温度", "热探索 → 冷锁定", gy + gh + 40),
            (CY, "ω_d(t) 驱动频率", "低 → 高 几何扫频", gy + gh + 40)]
    c.add(T(gx + 8, gy - 26, "κ(t) 耦合强度课程", 17, AM, "800"))
    c.add(T(gx + 8, gy + gh + 40, "T(t) 温度退火（热探索 → 冷锁定）", 17, RD, "800"))
    c.add(T(gx + gw - 8, gy + gh + 40, "ω_d(t) 几何扫频（低 → 高）", 17, CY, "800", anchor="end"))
    # t 轴
    for i in range(13):
        x = gx + gw * i / 12
        c.add(f'<line x1="{x:.0f}" y1="{gy+gh}" x2="{x:.0f}" y2="{gy+gh+9}" stroke="{C["line"]}" stroke-width="1.4"/>')
        c.add(T(x, gy + gh + 32, str(i), 15, C["ink3"], "600", anchor="middle", mono=True))
    c.add(T(gx, gy + gh + 62, "物理块序号 i  →  t = i/(R−1)  从 0 到 1  （R=12）", 17, C["ink3"], "600"))
    # 底部三张说明卡
    cards = [
        ("κ(t) = κ_max·σ((t−t_c)/τ_c)", "耦合强度课程", "前期弱耦合让相位自由游走，后期强耦合拉进锁定。", AM),
        ("T(t) = T_max·(1−t)^p + T_min", "温度退火", " Langevin 噪声随温度衰减，涨落逐渐让位于秩序。", RD),
        ("ω_d(t) = ω_lo·(ω_hi/ω_lo)^t", "驱动频率扫掠", "端点由可学习的本征频率谱导出——课程本身可训练。", CY),
    ]
    for i, (f, name, desc, col) in enumerate(cards):
        x = 150 + i * 545
        c.add(f'<g class="anim-rise" style="animation-delay:{.2+i*0.12}s">')
        c.add(panel(x, 792, 505, 130, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=16))
        c.add(f'<rect x="{x}" y="792" width="4" height="130" rx="2" fill="{col}"/>')
        c.add(T(x + 26, 832, name, 19, col, "800"))
        c.add(T(x + 26, 866, f, 17, C["ink"], "600", mono=True))
        c.add(T(x + 26, 898, desc, 16, C["ink3"], "500"))
        c.add("</g>")
    return c.render()


# ============================================================ 11 Codec
def card_11():
    c = Card(11, TOTAL, "THE CODEC", "零参数的双向变换",
             "编解码器是固定线性变换，一个可学习参数都没有。它把 GIF 变成复数振子场，再原路变回。",
             accent=AM, uid="c11")
    # 左侧：编码步骤
    c.add(T(150, 372, "编码：GIF → 振荡器场", 23, CY, "800"))
    steps = [
        ("01", "逐帧 seek 后再转 RGB", "先转 RGB 会把多帧动画塌成单帧"),
        ("02", "缩放到 128×128，时间重采样到 30 帧", "TS=6 × WIN=5，10 fps，共 3 秒"),
        ("03", "像素映射到 [−1,1]", "x/127.5 − 1"),
        ("04", "切成时空块 [TS,WIN,GRID,PATCH,…]", "N = 6×32² = 6,144 块，每块 S=48 実数"),
        ("05", "rfft2 → 5×25 = 125 个复系数", "全部保留，不再截断；再乘固定常数 NORM=10.0"),
    ]
    y = 410
    for i, (no, name, desc) in enumerate(steps):
        c.add(f'<g class="anim-riseL" style="animation-delay:{.1+i*0.11}s">')
        c.add(f'<circle cx="{172}" cy="{y+16}" r="16" fill="{CY}" opacity="0.16" stroke="{CY}" stroke-width="1.2"/>')
        c.add(T(172, y + 22, no, 15, CY, "800", anchor="middle", mono=True))
        c.add(T(206, y + 12, name, 20, C["ink"], "700"))
        c.add(T(206, y + 40, desc, 16, C["ink3"], "500"))
        c.add("</g>")
        y += 78
    # 右侧：频谱 → 场
    rx = 900
    c.add(panel(rx, 350, 870, 330, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=20))
    c.add(T(rx + 30, 392, "5 × 25 = 125 个复系数，全部保留", 21, AM, "800"))
    # 5x25 频谱网格
    import random as _r
    rnd = _r.Random(77)
    gxx, gyy, cw, ch = rx + 40, 428, 31, 30
    for j in range(5):
        for i in range(25):
            v = rnd.uniform(0.05, 1.0)
            col = AM if v > 0.72 else (CY if v > 0.42 else (MG if v > 0.22 else C["ink3"]))
            c.add(f'<rect x="{gxx+i*cw:.1f}" y="{gyy+j*ch:.1f}" width="{cw-3}" height="{ch-3}" '
                  f'rx="3" fill="{col}" opacity="{0.22+v*0.72:.2f}"/>')
    c.add(f'<text x="{gxx}" y="{gyy-10}" font-size="14" fill="{C["ink3"]}" font-weight="600">h=0 (DC)</text>')
    c.add(f'<text x="{gxx}" y="{gyy+5*ch+26}" font-size="14" fill="{C["ink3"]}" font-weight="600">h=4</text>')
    c.add(f'<text x="{gxx+25*cw}" y="{gyy-10}" font-size="14" fill="{C["ink3"]}" font-weight="600" '
          f'text-anchor="end">w=24</text>')
    c.add(T(rx + 30, 652, "实部 + 虚部拼起来 → 振荡器场 [N, 2D] = [6144, 250]", 18, C["ink2"], "600"))
    # 解码 + 无损
    c.add(f'<g class="anim-rise" style="animation-delay:.3s">')
    c.add(panel(rx, 700, 425, 210, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=18))
    c.add(T(rx + 26, 742, "解码：场 → GIF", 20, CY, "800"))
    for i, ln in enumerate(wrap("逆变换回 z → 乘回 10.0 → irfft2 → 还原 [30,128,128,3] → "
                                "clamp 到 [−1.2,1.2]（诚实口径）或逐片 min–max（可视化口径）→ uint8。",
                                16, 373)[:5]):
        c.add(T(rx + 26, 776 + i * 25, ln, 16, C["ink2"], "500"))
    c.add("</g>")
    c.add(f'<g class="anim-rise" style="animation-delay:.42s">')
    c.add(panel(rx + 445, 700, 425, 210, fill="#0C1424", stroke=GR, op=0.75, rx=18))
    c.add(T(rx + 471, 742, "为什么 NORM = 10？", 20, GR, "800"))
    for i, ln in enumerate(wrap("rfft2 系数是极重尾的（均值≈2.95、中位数≈0.0015、p99≈5.4）。"
                                "除以原始像素数会把目标振幅压到 ~0.008，比模型自然输出尺度低两个数量级，"
                                "迫使振幅通路用 12 层去学 90× 压缩。NORM=10 让两侧尺度对齐。",
                                15, 373)[:6]):
        c.add(T(rx + 471, 774 + i * 23, ln, 15, C["ink2"], "500"))
    c.add("</g>")
    # 无损事实条
    c.add(f'<g class="anim-rise" style="animation-delay:.55s">')
    c.add(panel(150, 828, 700, 82, fill="#0C1424", stroke=GR, op=0.75, rx=14))
    c.add(T(178, 878, "往返无损：rfft2 → irfft2 误差 ≈ 6e-7", 20, GR, "700", mono=True))
    c.add("</g>")
    c.add(T(150, 945, "FFT 本身无噪声——这正是 v3600 消融实验能成立的前提。",
            18, C["ink3"], "500"))
    return c.render()
