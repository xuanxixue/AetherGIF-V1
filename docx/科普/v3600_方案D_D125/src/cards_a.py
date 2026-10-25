# -*- coding: utf-8 -*-
"""cards_a.py — 卡片 01-05：第一幕「为什么」+ 第二幕「物理是什么」"""
import math
from deck import *
from deck import CUR

TOTAL = 18


# ============================================================ 01 封面
def card_01():
    c = Card(1, TOTAL, "AETHERGIF v3600 · 方案 D · D=125",
             "用物理造 GIF", "", accent=CY, uid="c01", variant="hero", title_size=118,
             use_header=False)
    c.addcss("""
@keyframes titleIn{from{opacity:0;transform:translateY(50px) scale(.96);filter:blur(14px)}
 to{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}}
.anim-title{animation:titleIn 1.5s cubic-bezier(.16,.84,.24,1) both}
@keyframes ringPulse{0%{r:120;opacity:.5}100%{r:520;opacity:0}}
.ring-p{animation:ringPulse 3.6s ease-out infinite}
""")
    c.add(
        # 透视地网格
        persp_grid("c01", horizon=700, color=CY, op=0.22),
        # 中心同心环（Kuramoto 序参量意象）
        f'<g transform="translate(1420,540)">'
        f'<circle r="150" fill="none" stroke="{CY}" stroke-width="1.2" opacity="0.35"/>'
        f'<circle r="150" fill="none" stroke="{MG}" stroke-width="2.4" opacity="0.5" '
        f'stroke-dasharray="26 18" class="anim-spin"/>'
        f'<circle r="248" fill="none" stroke="{CY}" stroke-width="1" opacity="0.22" '
        f'stroke-dasharray="6 14"/>'
        f'<circle r="248" fill="none" stroke="{MG}" stroke-width="1.6" opacity="0.42" '
        f'stroke-dasharray="40 60" class="anim-spinr"/>'
        f'<circle r="346" fill="none" stroke="{CY}" stroke-width="0.9" opacity="0.16"/>'
        f'<circle class="ring-p" r="120" fill="none" stroke="{CY}" stroke-width="2"/>'
        f'<circle class="ring-p" r="120" fill="none" stroke="{MG}" stroke-width="2" '
        f'style="animation-delay:1.8s"/>'
        f'<circle r="14" fill="{C["ink"]}" filter="url(#c01_glow)"/>'
        f'<circle r="26" fill="none" stroke="{C["ink"]}" stroke-width="1" opacity="0.35"/>'
        # 内外两圈振子
        + "".join(
            f'<circle cx="{math.cos(a)*248:.1f}" cy="{math.sin(a)*248:.1f}" r="{5+3*math.sin(a*3):.1f}" '
            f'fill="{MG if i%2 else CY}" opacity="0.9" filter="url(#c01_glows)"/>'
            for i, a in enumerate([i * 0.42 for i in range(15)]))
        + "".join(
            f'<circle cx="{math.cos(a)*150:.1f}" cy="{math.sin(a)*150:.1f}" r="3.4" '
            f'fill="{CY}" opacity="0.6"/>'
            for i, a in enumerate([i * 0.62 for i in range(10)]))
        + "".join(
            f'<circle cx="{math.cos(a)*346:.1f}" cy="{math.sin(a)*346:.1f}" r="2.2" '
            f'fill="{MG}" opacity="0.42"/>'
            for i, a in enumerate([i * 0.78 for i in range(8)]))
        + "</g>",
        particles("c01", n=54, seed=11, color=CY),
        # 标题
        f'<g class="anim-title" style="animation-delay:.15s">',
        T(150, 268, "AETHERGIF v3600 · 方案 D · D=125", 25, CY, "700", ls=9),
        f'<rect x="150" y="296" width="120" height="4" rx="2" fill="{MG}"/>',
        T(150, 430, "用物理造 GIF", 122, C["ink"], "800", ls="-3",
          extra='filter="url(#c01_titleglow)"'),
        f'<rect x="150" y="462" width="600" height="2" rx="1" fill="url(#c01_accent)" opacity="0.55"/>',
        T(150, 500, "不用 self-attention、不用 Transformer block——", 30, C["ink2"], "500"),
        T(150, 542, "用 6,144 个耦合振子，从噪声一路演化到像素。", 30, C["ink2"], "500"),
        "</g>",
        # 三张指标条
        f'<g class="anim-rise" style="animation-delay:.75s">',
        panel(150, 606, 300, 148, fill="#080D1A", stroke=C["line"], op=0.72, rx=18),
        T(178, 656, "功率加权相位余弦", 17, C["ink3"], "600"),
        T(178, 728, "0.9991", 62, CY, "800", mono=True),
        "</g>",
        f'<g class="anim-rise" style="animation-delay:.92s">',
        panel(478, 606, 300, 148, fill="#080D1A", stroke=C["line"], op=0.72, rx=18),
        T(506, 656, "对编解码天花板 PSNR", 17, C["ink3"], "600"),
        T(506, 728, "19.84dB", 62, MG, "800", mono=True),
        "</g>",
        f'<g class="anim-rise" style="animation-delay:1.09s">',
        panel(806, 606, 300, 148, fill="#080D1A", stroke=C["line"], op=0.72, rx=18),
        T(834, 656, "可训练参数", 17, C["ink3"], "600"),
        T(834, 728, "2,022,232", 50, AM, "800", mono=True),
        "</g>",
        # 底部 tagline
        f'<g class="anim-rise" style="animation-delay:1.3s">',
        T(150, 856, "一份 2,500 步的单样本过拟合实验，", 25, C["ink3"], "500"),
        T(150, 896, "用来验证「噪声 → 振子 → 像素」这条物理通路到底走不走得通。", 25, C["ink3"], "500"),
        "</g>",
        f'<g class="anim-rise" style="animation-delay:1.5s">',
        label_row(150, 940, [("Kuramoto 同步", CY), ("洛伦兹共振", MG), ("OU 恢复力", AM),
                             ("退火课程", GR), ("全谱 codec D=125", RD)]),
        "</g>",
    )
    return c.render()


# ============================================================ 02 研究问题
def card_02():
    c = Card(2, TOTAL, "THE RESEARCH QUESTION", "能不能不用注意力？",
             "主流视频模型靠 self-attention 让远距离像素互相「看见」。AetherGIF 问的是："
             "如果只留下一堆会互相拉扯的振子，这条路还通不通？", accent=MG, uid="c02")
    c.addcss("""
@keyframes vsIn{from{opacity:0;transform:translateX(-50px)}to{opacity:1;transform:translateX(0)}}
.vs-l{animation:vsIn 1s cubic-bezier(.2,.8,.2,1) both}
.vs-r{animation:vsIn 1s cubic-bezier(.2,.8,.2,1) both;animation-delay:.2s}
""")
    # 左：Transformer 路线
    c.add(f'<g class="vs-l">')
    c.add(panel(150, 360, 700, 470, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=22))
    c.add(T(186, 414, "常规 Transformer 路线", 27, C["ink3"], "800"))
    c.add(f'<rect x="186" y="432" width="120" height="3" rx="1.5" fill="{C["line"]}"/>')
    rows = [
        ("Self-Attention", "Q·K·V 两两算相似度，O(N²) 显存"),
        ("LayerNorm + FFN", "逐位置归一化 + 两层 MLP 混合通道"),
        ("位置编码", "外挂的正弦 / 可学习位置向量"),
        ("残差 + 堆叠", "靠深度堆叠换取表达力"),
    ]
    yy = 500
    for i, (a, b) in enumerate(rows):
        c.add(f'<g class="anim-rise" style="animation-delay:{.3+i*0.12}s">')
        c.add(f'<rect x="186" y="{yy-34}" width="628" height="72" rx="12" fill="#111A2C" opacity="0.72"/>')
        c.add(f'<circle cx="212" cy="{yy+2}" r="5" fill="{C["ink3"]}"/>')
        c.add(T(234, yy + 9, a, 23, C["ink2"], "700"))
        c.add(T(470, yy + 9, b, 19, C["ink3"], "500"))
        c.add("</g>")
        yy += 88
    c.add("</g>")
    # 中间 VS
    c.add(f'<g class="anim-pop" style="animation-delay:.45s">')
    c.add(f'<circle cx="960" cy="595" r="56" fill="#0B1220" stroke="{MG}" stroke-width="1.6" opacity="0.9"/>')
    c.add(T(960, 606, "VS", 34, MG, "800", anchor="middle", ls=2))
    c.add("</g>")
    # 右：物理路线
    c.add(f'<g class="vs-r">')
    c.add(panel(1070, 360, 700, 470, fill="#0C1120", stroke=MG, op=0.78, rx=22))
    c.add(f'<rect x="1070" y="360" width="700" height="3" rx="1.5" fill="{MG}" opacity="0.9"/>')
    c.add(T(1106, 414, "AetherGIF 物理路线", 27, MG, "800"))
    c.add(f'<rect x="1106" y="432" width="120" height="3" rx="1.5" fill="{MG}"/>')
    rows2 = [
        ("Kuramoto 同步", "sin(θⱼ−θᵢ) 耦合，序参量 r 描述全局对齐"),
        ("洛伦兹共振", "扫频穿过通道本征频率，放大该通道"),
        ("驻波调制", "cos 三维乘积，替代位置编码"),
        ("OU 恢复力", "log 域乘性更新 + 回归力，压住振幅"),
    ]
    yy = 500
    for i, (a, b) in enumerate(rows2):
        c.add(f'<g class="anim-rise" style="animation-delay:{.5+i*0.12}s">')
        c.add(f'<rect x="1106" y="{yy-34}" width="628" height="72" rx="12" fill="#141B33" opacity="0.8"/>')
        c.add(f'<circle cx="1132" cy="{yy+2}" r="5" fill="{MG}" filter="url(#c02_glows)"/>')
        c.add(T(1154, yy + 9, a, 23, C["ink"], "700"))
        c.add(T(1400, yy + 9, b, 18, C["ink2"], "500"))
        c.add("</g>")
        yy += 88
    c.add("</g>")
    # 底部结论
    c.add(f'<g class="anim-rise" style="animation-delay:1.1s">')
    c.add(panel(150, 862, 1620, 96, fill="#0A1220", stroke=CY, op=0.7, rx=16))
    c.add(f'<rect x="150" y="862" width="4" height="96" rx="2" fill="{CY}"/>')
    c.add(T(186, 906, "结论", 18, CY, "700", ls=2))
    c.add(T(186, 940, "这 202 万个参数里没有一个 self-attention 矩阵——所有结构性与非线性都来自物理。",
            24, C["ink"], "600"))
    c.add("</g>")
    return c.render()


# ============================================================ 03 全景架构
def card_03():
    c = Card(3, TOTAL, "ARCHITECTURE", "全景：一帧 GIF 的完整旅程",
             "冻结的编解码器把像素变成复数场，12 层物理块在场上演化，再原路变回像素。",
             accent=CY, uid="c03")
    # 主流程条
    y = 470
    boxes = [
        (150, 250, "GIF 输入", "128×128 · 30 帧", MG),
        (470, 300, "Codec 编码", "rfft2 · D=125 全谱", CY),
        (840, 300, "振荡器场", "6,144 × 125 复通道", CY),
        (1210, 300, "12 × PhysicsBlock", "噪声 → 目标场", AM),
        (1580, 250, "GIF 输出", "irfft2 · 解码", MG),
    ]
    for i, (x, w, t, s, col) in enumerate(boxes):
        c.add(f'<g class="anim-rise" style="animation-delay:{.15+i*0.13}s">')
        c.add(panel(x, y, w, 168, fill="#0A1020", stroke=col, op=0.85, rx=18))
        c.add(f'<rect x="{x}" y="{y}" width="{w}" height="3" rx="1.5" fill="{col}" opacity="0.9"/>')
        c.add(T(x + w / 2, y + 62, t, 25, C["ink"], "800", anchor="middle"))
        c.add(T(x + w / 2, y + 100, s, 18, C["ink3"], "500", anchor="middle"))
        c.add("</g>")
    for i in range(4):
        x1 = boxes[i][0] + boxes[i][1]
        x2 = boxes[i + 1][0]
        c.add(f'<g class="anim-rise" style="animation-delay:{.5+i*0.13}s">')
        c.add(arrow(x1 + 10, y + 84, x2 - 12, y + 84, CY, 2.6))
        c.add("</g>")
    # 场可视化：网格点阵
    c.add(f'<g opacity="0.9">')
    c.add(lattice(950, 660, 26, 8, 8.4, r=1.9, color=CY, dim=CY, op=0.55))
    c.add("</g>")
    c.add(T(1160, 700, "a₀ , θ₀  ← 噪声初始化", 18, C["ink3"], "600", mono=True))
    c.add(T(1160, 732, "a , θ  ← 演化后", 18, C["ink3"], "600", mono=True))
    # 上方：text 条件支路
    c.add(f'<g class="anim-rise" style="animation-delay:.9s">')
    c.add(panel(560, 300, 760, 92, fill="#0C0F1E", stroke=MG, op=0.72, rx=16))
    c.add(T(590, 342, "Qwen2.5-0.5B（冻结，可选）", 21, MG, "700"))
    c.add(T(590, 374, "→ 4 个 Linear 头 → ω 偏置 / κ / T / drift 目标（只改边界条件，不改物理算子）",
            18, C["ink3"], "500"))
    c.add(arrow(940, 392, 940, 462, MG, 2.2))
    c.add("</g>")
    # 深度即时间
    c.add(f'<g class="anim-rise" style="animation-delay:1.05s">')
    c.add(panel(150, 780, 1620, 130, fill="#0A0E1B", stroke=C["line"], op=0.68, rx=18))
    c.add(T(186, 826, "关键设计：深度就是物理时间", 24, CY, "800"))
    c.add(T(186, 868, "第 i 个物理块对应 t = i/(R−1) ∈ [0,1]。12 层不是「堆得更深」，"
                      "而是一条从混乱到锁定的热力学时间线。", 21, C["ink2"], "500"))
    c.add("</g>")
    return c.render()


# ============================================================ 04 状态空间
def card_04():
    c = Card(4, TOTAL, "STATE SPACE", "振子的两个数字：幅度与相位",
             "每个振子携带 125 个复通道，通道 d 各自有一对 (a, θ)。深度即时间，振幅与相位在物理中演化。",
             accent=AM, uid="c04")
    c.addcss("""
@keyframes spinC{to{transform:rotate(360deg)}}
.rot{animation:spinC 40s linear infinite;transform-box:fill-box;transform-origin:center}
@keyframes breathe2{0%,100%{transform:scale(1);opacity:.9}50%{transform:scale(1.07);opacity:1}}
.br{animation:breathe2 4s ease-in-out infinite;transform-box:fill-box;transform-origin:center}
""")
    # 左：极坐标示意
    c.add(panel(150, 360, 700, 560, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=22))
    cx, cy, R = 500, 640, 215
    c.add(f'<g transform="translate({cx},{cy})">')
    c.add(f'<circle r="{R}" fill="none" stroke="{C["line"]}" stroke-width="1.4"/>')
    c.add(f'<circle r="{R*0.72}" fill="none" stroke="{C["line"]}" stroke-width="1" opacity="0.6"/>')
    c.add(f'<circle r="{R*0.44}" fill="none" stroke="{C["line"]}" stroke-width="1" opacity="0.6"/>')
    c.add(f'<line x1="{-R-30}" y1="0" x2="{R+30}" y2="0" stroke="{C["line"]}" stroke-width="1.2"/>')
    c.add(f'<line x1="0" y1="{-R-30}" x2="0" y2="{R+30}" stroke="{C["line"]}" stroke-width="1.2"/>')
    for i in range(24):
        a = i * math.pi / 12
        c.add(f'<line x1="{math.cos(a)*(R-8):.1f}" y1="{math.sin(a)*(R-8):.1f}" '
              f'x2="{math.cos(a)*R:.1f}" y2="{math.sin(a)*R:.1f}" stroke="{C["ink3"]}" '
              f'stroke-width="1" opacity="0.7"/>')
    # 若干振子
    import random as _r
    rnd = _r.Random(4)
    for i in range(16):
        a = rnd.uniform(0, 2 * math.pi)
        rr = rnd.uniform(R * 0.25, R * 0.95)
        col = AM if i % 3 == 0 else (MG if i % 3 == 1 else CY)
        c.add(f'<circle cx="{math.cos(a)*rr:.1f}" cy="{math.sin(a)*rr:.1f}" r="{rnd.uniform(4,7):.1f}" '
              f'fill="{col}" opacity="0.9" filter="url(#c04_glows)"/>')
        c.add(f'<line x1="0" y1="0" x2="{math.cos(a)*rr:.1f}" y2="{math.sin(a)*rr:.1f}" '
              f'stroke="{col}" stroke-width="1.2" opacity="0.4"/>')
    # 一个高亮振子：a 与 θ
    a0 = -0.62
    rr = R * 0.78
    px, py = math.cos(a0) * rr, math.sin(a0) * rr
    c.add(f'<g class="br">')
    c.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="9" fill="{AM}" filter="url(#c04_glow)"/>')
    c.add("</g>")
    c.add(f'<line x1="0" y1="0" x2="{px:.1f}" y2="{py:.1f}" stroke="{AM}" stroke-width="2.6"/>')
    c.add(f'<path d="M {math.cos(a0)*rr*0.55:.1f} {math.sin(a0)*rr*0.55:.1f} '
          f'A {rr*0.55:.1f} {rr*0.55:.1f} 0 0 1 {math.cos(a0)*rr*0.82:.1f} {math.sin(a0)*rr*0.82:.1f}" '
          f'fill="none" stroke="{AM}" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.8"/>')
    c.add(f'<text x="{math.cos(a0)*rr*0.66:.1f}" y="{math.sin(a0)*rr*0.66-14:.1f}" '
          f'font-size="20" fill="{AM}" font-weight="700" class="mono">θ</text>')
    c.add(f'<text x="{px+22:.1f}" y="{py+8:.1f}" font-size="22" fill="{AM}" '
          f'font-weight="800" class="mono">a</text>')
    c.add("</g>")
    c.add(T(186, 410, "复数场：z = a·e^(iθ)", 24, AM, "800", mono=True))
    c.add(T(186, 470, "6,144 个振子 · 125 个复通道 · 相位随机铺开", 19, C["ink3"], "500"))
    # 右：数据形状
    rx = 900
    c.add(panel(rx, 360, 870, 560, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=22))
    c.add(T(rx + 36, 410, "场的形状", 24, CY, "800"))
    shapes = [
        ("N", "6,144", "振子总数 = TS × GRID² = 6 × 32²", CY),
        ("S", "48", "每帧块实数 = PATCH² × 3 = 4² × 3", MG),
        ("D", "125", "每块复系数 = 5 × 25（rfft2 全谱）", AM),
        ("field", "[6144, 250]", "实部 + 虚部拼成的最终张量", GR),
    ]
    yy = 462
    for k, (tag, val, desc, col) in enumerate(shapes):
        c.add(f'<g class="anim-rise" style="animation-delay:{.15+k*0.12}s">')
        c.add(f'<rect x="{rx+36}" y="{yy-38}" width="798" height="88" rx="14" fill="#111A2C" opacity="0.62"/>')
        c.add(f'<rect x="{rx+36}" y="{yy-38}" width="5" height="88" rx="2.5" fill="{col}"/>')
        c.add(T(rx + 62, yy - 8, tag, 20, col, "700", mono=True))
        c.add(T(rx + 62, yy + 32, val, 38, C["ink"], "800", mono=True))
        c.add(T(rx + 330, yy + 20, desc, 19, C["ink2"], "500"))
        c.add("</g>")
        yy += 100
    c.add(f'<g class="anim-rise" style="animation-delay:.7s">')
    c.add(panel(rx + 36, 852, 798, 52, fill="#0C1424", stroke=GR, op=0.7, rx=12))
    c.add(T(rx + 60, 886, "总自由度 1,536,000 个实数，可训练参数只有 2,022,232", 19, GR, "600"))
    c.add("</g>")
    return c.render()


# ============================================================ 05 五个物理零件
def card_05():
    c = Card(5, TOTAL, "THE FIVE PHYSICS PARTS", "五块物理，替掉整条 Transformer 栈",
             "整个模型里唯一的稠密投影，是驻波层的 W₁/W₂ 和四个文本条件头。剩下的全是物理。",
             accent=MG, uid="c05")
    parts = [
        ("01", "KuramotoSync", "相位同步", "dθ/dt = ω + ΣW·sin(θⱼ−θᵢ) + κ·g + η",
         "替掉 self-attention：谁跟谁耦合，由物理权重决定", CY),
        ("02", "StandingWaveFFN", "共振混合", "h = W₂(GELU(W₁z) ⊙ A(ω_d,ω_n) ⊙ SW(p))",
         "替掉 FFN：通道混合由洛伦兹共振曲线决定", MG),
        ("03", "DriftField", "目标吸引", "gᵢ = Σ softmax(−‖ẑᵢ−ẑⱼ‖²/τ)·sin(θⱼᵗᵃʳ−θᵢ)",
         "唯一的目标信息通道：让相位朝目标相位收敛", AM),
        ("04", "OU 振幅", "恢复力", "log a ← log a + res − γ(log a − μ)",
         "替掉 LayerNorm：log 域乘性更新 + 布朗恢复力", GR),
        ("05", "TimeField", "时间课程", "κ(t) 课程 · T(t) 退火 · ω_d(t) 扫频",
         "替掉堆层深度：12 层 = 一条从混乱到锁定的物理时间线", RD),
    ]
    y = 350
    for i, (no, name, tag, formula, desc, col) in enumerate(parts):
        c.add(f'<g class="anim-riseL" style="animation-delay:{.12*i}s">')
        c.add(panel(150, y, 1620, 104, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=16))
        c.add(f'<rect x="150" y="{y}" width="5" height="104" rx="2.5" fill="{col}"/>')
        c.add(T(190, y + 44, no, 26, col, "800", mono=True))
        c.add(T(252, y + 42, name, 24, C["ink"], "800", mono=True))
        c.add(T(252, y + 76, tag, 19, col, "700"))
        # 公式
        c.add(f'<rect x="600" y="{y+22}" width="620" height="60" rx="10" fill="#060A14" opacity="0.85"/>')
        c.add(T(620, y + 60, formula, 19, C["ink2"], "500", mono=True))
        c.add(T(1260, y + 60, desc, 18, C["ink3"], "500"))
        c.add("</g>")
        y += 116
    c.add(f'<g class="anim-rise" style="animation-delay:.75s">')
    c.add(panel(150, 946, 1620, 60, fill="#0C1424", stroke=AM, op=0.7, rx=14))
    c.add(T(186, 984, "另外：帧数 TS=6 · 时间窗 WIN=5 · 空间块 32×32 · 采样 30 帧 @10fps = 3 秒",
            20, AM, "600"))
    c.add("</g>")
    return c.render()
