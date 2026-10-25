# -*- coding: utf-8 -*-
"""cards_c.py — 卡片 12-18：第三幕「v3600 的关键改动」+ 第四幕「结果·原则·边界」"""
import math
from deck import *
from deck import CUR

TOTAL = 18


# ============================================================ 12 D=64 vs D=125
def card_12():
    c = Card(12, TOTAL, "THE ONE CHANGE", "v3600 只改了一件事：编解码器",
             "v3600 相对 v2500 的全部改动，就是把 rfft2 频谱从「截断 64 个」换成「全留 125 个」。"
             "物理核心、损失、优化器日程，一行未动。", accent=AM, uid="c12")
    c.addcss("""
@keyframes cellOn{from{opacity:0;transform:scale(.2)}to{opacity:1;transform:scale(1)}}
.co{animation:cellOn .5s cubic-bezier(.2,.9,.3,1) both;transform-box:fill-box;transform-origin:center}
@keyframes rowScan{0%{opacity:0}20%{opacity:1}80%{opacity:1}100%{opacity:0}}
.rs{animation:rowScan 3s ease-in-out infinite}
""")
    # 左：D=64 截断示意
    c.add(panel(150, 350, 620, 480, fill="#120C12", stroke=RD, op=0.72, rx=20))
    c.add(f'<rect x="150" y="350" width="620" height="3" rx="1.5" fill="{RD}" opacity="0.9"/>')
    c.add(T(186, 396, "v2500 · D = 64（截断）", 22, RD, "800"))
    c.add(T(186, 424, "行优先 flatten 后取前 64 个 —— 是不对称硬截断", 16, C["ink3"], "500"))
    # 5x25 网格，高亮哪些被保留
    gx, gy, cw, ch = 200, 456, 21, 40
    for j in range(5):
        kept_row = j < 2
        partial = (j == 2)
        for i in range(25):
            keep = kept_row or (partial and i <= 12)
            x = gx + i * cw; y = gy + j * ch
            if keep:
                c.add(f'<rect x="{x}" y="{y}" width="{cw-2}" height="{ch-2}" rx="2" fill="{RD}" '
                      f'opacity="0.62" class="co" style="animation-delay:{(j*25+i)*0.004:.2f}s"/>')
            else:
                c.add(f'<rect x="{x}" y="{y}" width="{cw-2}" height="{ch-2}" rx="2" fill="{C["ink3"]}" '
                      f'opacity="0.13"/>')
    c.add(f'<rect x="{gx}" y="{gy+2*ch-2}" width="{13*cw:.0f}" height="{ch-2}" fill="none" '
          f'stroke="{AM}" stroke-width="1.6" opacity="0.8" class="rs"/>')
    c.add(T(gx, gy + 5 * ch + 30, "h=0 ✓   h=1 ✓   h=2 只有 w≤12   h=3 ✗   h=4 ✗",
            16, C["ink3"], "600", mono=True))
    c.add(Twrap(186, 760, "丢掉的那一带约携带 10% 的片段能量。丢掉近 1/3 的频谱系数，"
                          "却把 h=3、h=4 整行扔掉——形状是不对称的。", 560, 16, C["ink2"], "500", 24))
    # 右：D=125 全谱
    c.add(panel(800, 350, 620, 480, fill="#0C1424", stroke=GR, op=0.72, rx=20))
    c.add(f'<rect x="800" y="350" width="620" height="3" rx="1.5" fill="{GR}" opacity="0.9"/>')
    c.add(T(836, 396, "v3600 · D = 125（全谱）", 22, GR, "800"))
    c.add(T(836, 424, "PATCH=4, WIN=5 → rfft2 网格恰好 5×25 = 125，一个不落", 16, C["ink3"], "500"))
    gx2, gy2, cw2, ch2 = 850, 456, 21, 40
    import random as _r
    rnd = _r.Random(5)
    for j in range(5):
        for i in range(25):
            v = rnd.uniform(0.08, 1.0)
            col = GR if v > 0.7 else (AM if v > 0.45 else (CY if v > 0.25 else C["ink3"]))
            c.add(f'<rect x="{gx2+i*cw2}" y="{gy2+j*ch2}" width="{cw2-2}" height="{ch2-2}" rx="2" '
                  f'fill="{col}" opacity="{0.25+v*0.68:.2f}" class="co" '
                  f'style="animation-delay:{(j*25+i)*0.004:.2f}s"/>')
    c.add(T(gx2, gy2 + 5 * ch2 + 30, "h=0..4 全部保留，w=0..24 一个不丢", 16, C["ink3"], "600", mono=True))
    c.add(Twrap(836, 760, "参数量 1,135,658 → 2,022,232。编解码天花板 28.63 dB → 49.82 dB，"
                          "整整抬高了约 21 dB 的可用余量。", 560, 16, C["ink2"], "500", 24))
    # 底部结论条
    c.add(f'<g class="anim-rise" style="animation-delay:.5s">')
    c.add(panel(150, 856, 1620, 90, fill="#0A1220", stroke=AM, op=0.75, rx=16))
    c.add(f'<rect x="150" y="856" width="4" height="90" rx="2" fill="{AM}"/>')
    c.add(T(186, 898, "这是一次干净的消融实验", 20, AM, "800"))
    c.add(T(186, 928, "物理核心一行没改，唯一的变量是编解码宽度——所以指标变化可以完全归因到 codec。",
            19, C["ink2"], "500"))
    c.add("</g>")
    return c.render()


# ============================================================ 13 LED 网格伪影
def card_13():
    c = Card(13, TOTAL, "BUG HUNT", "「LED 网格」伪影的真凶",
             "v3500 曾经对着这个伪影反复微调模型，进展甚微。真相是：缺陷根本不在物理，而在编解码器。",
             accent=RD, uid="c13")
    c.addcss("""
@keyframes jitter{0%,100%{transform:translateX(0)}25%{transform:translateX(3px)}75%{transform:translateX(-3px)}}
.jit{animation:jitter 1.1s ease-in-out infinite}
@keyframes gridFlash{0%,100%{opacity:.25}50%{opacity:.75}}
.gf{animation:gridFlash 2.4s ease-in-out infinite}
""")
    # 左：伪影示意
    c.add(panel(150, 350, 700, 400, fill="#120C12", stroke=RD, op=0.7, rx=20))
    c.add(T(186, 392, "伪影长这样：块边界出现不该有的跳变", 20, RD, "800"))
    # D=64 网格
    c.add(T(200, 432, "D=64（截断）", 16, C["ink3"], "700"))
    for j in range(5):
        for i in range(12):
            x = 200 + i * 44; y = 446 + j * 30
            c.add(f'<rect x="{x}" y="{y}" width="42" height="28" rx="2" fill="{RD}" opacity="0.10"/>')
            c.add(f'<rect x="{x}" y="{y}" width="42" height="28" rx="2" fill="none" stroke="{RD}" '
                  f'stroke-width="1.4" opacity="0.55" class="jit" '
                  f'style="animation-delay:{(i+j)*0.07:.2f}s"/>')
    c.add(T(740, 466, "← 网格线在", 15, RD, "700"))
    c.add(T(740, 492, "不该有的位置", 15, C["ink3"], "500"))
    c.add(T(740, 518, "出现", 15, C["ink3"], "500"))
    # D=125 对照
    c.add(T(200, 618, "D=125（全谱）", 16, C["ink3"], "700"))
    for j in range(2):
        for i in range(12):
            x = 200 + i * 44; y = 632 + j * 30
            c.add(f'<rect x="{x}" y="{y}" width="42" height="28" rx="2" fill="{GR}" opacity="0.08"/>')
            c.add(f'<rect x="{x}" y="{y}" width="42" height="28" rx="2" fill="none" stroke="{GR}" '
                  f'stroke-width="1" opacity="0.20"/>')
    c.add(T(740, 652, "边界消失", 15, GR, "700"))
    c.add(T(740, 676, "伪影源头", 15, C["ink3"], "500"))
    c.add(T(740, 700, "是截断", 15, C["ink3"], "500"))
    c.add(Twrap(186, 720, "丢掉的那一带约携带 10% 的片段能量——但真正致命的不是能量，"
                          "而是截断的形状不对称。", 620, 15, C["ink2"], "500", 22))
    # 右：数据表
    rx = 900
    c.add(T(rx, 380, "诊断指标：网格比 = 块边界处的列差 ÷ 块内部列差", 20, C["ink"], "800"))
    c.add(T(rx, 408, "源 GIF 自身的比值 ≈ 1.111 —— 这是「正常」的基准线", 16, C["ink3"], "500"))
    rows = [
        ("D = 64（截断，v2500）", "28.63 dB", "1.373", RD, True),
        ("D = 125（全谱，v3600）", "49.82 dB", "1.172", GR, True),
        ("全谱，未量化前", "~147 dB", "1.062", CY, False),
    ]
    y = 444
    for name, psnr, ratio, col, hl in rows:
        c.add(f'<g class="anim-rise" style="animation-delay:{.1+y/1000}s">')
        c.add(panel(rx, y, 870, 84, fill="#0A0E1B", stroke=col if hl else C["line"],
                    op=0.7, rx=14))
        c.add(f'<rect x="{rx}" y="{y}" width="4" height="84" rx="2" fill="{col}"/>')
        c.add(T(rx + 26, y + 40, name, 20, C["ink"], "700"))
        c.add(T(rx + 26, y + 68, "往返 PSNR", 15, C["ink3"], "500"))
        c.add(T(rx + 470, y + 54, psnr, 26, col, "800", mono=True))
        c.add(T(rx + 700, y + 40, "网格比", 15, C["ink3"], "500"))
        c.add(T(rx + 700, y + 68, ratio, 26, col, "800", mono=True))
        c.add("</g>")
        y += 96
    # 反直觉点
    c.add(f'<g class="anim-rise" style="animation-delay:.55s">')
    c.add(panel(rx, 736, 870, 210, fill="#0C1424", stroke=AM, op=0.75, rx=18))
    c.add(f'<rect x="{rx}" y="736" width="870" height="3" rx="1.5" fill="{AM}" opacity="0.9"/>')
    c.add(T(rx + 28, 780, "反直觉的一点：形状比数量更致命", 21, AM, "800"))
    c.add(Twrap(rx + 28, 816, "一个「正经」的 39 系数低通角（h<3, w<13），保留了 88.4% 的能量，"
                               "和 D=64 的 89.8% 几乎一样多——但它的网格比是 1.276，"
                               "比 D=64 的 1.192 型截断还差。", 810, 16, C["ink2"], "500", 24))
    c.add(Twrap(rx + 28, 890, "能量账算不出伪影。真正决定结果的是截断的形状——"
                               "只有让块边界连续，才是安全的。", 810, 16, AM, "600", 24))
    c.add(Twrap(rx + 28, 930, "这也解释了 v3500 为什么白干：缺陷在 codec，不在物理。",
                810, 16, C["ink3"], "500", 24))
    c.add("</g>")
    # 底部
    c.add(f'<g class="anim-rise" style="animation-delay:.7s">')
    c.add(panel(150, 774, 700, 172, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=18))
    c.add(T(186, 812, "两条结论", 20, C["ink"], "800"))
    c.add(badge_num(196, 862, "1", GR))
    c.add(Twrap(232, 868, "FFT 本身无噪声。125 个系数全留时往返误差 ~6e-7，"
                          "网格比等于源自身的值——伪影彻底消失。", 590, 16, C["ink2"], "500", 24))
    c.add(badge_num(196, 918, "2", GR))
    c.add(Twrap(232, 924, "它从来不是分块 FFT 造成的，是截断造成的。FFT 从一开始就是清白的。",
                590, 16, C["ink2"], "500", 24))
    c.add("</g>")
    return c.render()


# ============================================================ 14 训练协议
def card_14():
    c = Card(14, TOTAL, "TRAINING", "损失函数是被数据统计逼出来的",
             "FFT 系数的分布极不均匀：69.4% 的系数振幅小于 0.01，它们的目标相位纯粹是 atan2 的数值噪声。",
             accent=MG, uid="c14")
    # 上：损失公式
    c.add(code(150, 350, 1620, 250, [
        "pe       = wrap_to_pi(θ_model − θ_target)",
        "loss_amp = weighted_mean( (log1p(a) − log1p(a_tar))² ,  w = (a_tar / a_tar.max)^0.5 )",
        "loss_ph  = Σ ( a_tar² · pe² ) / Σ a_tar²",
        "loss     = loss_amp + loss_ph",
    ], accent=MG, size=21, lead=40, title="PLAN D 目标函数"))
    # 三个设计决策
    items = [
        ("为什么用 log1p？", "a → 0 处梯度有界", CY,
         "不用 log1p，20 / 393,216 个尖峰系数只能拿到约 0.67% 的线性域梯度，永远拟合不上。"),
        ("为什么权重开根号？", "软化 DC 的垄断", AM,
         "早期用 a_tar² 加权，DC 通道独占 99.9% 的监督，其他通道全部饿死。"),
        ("为什么相位按功率加权？", "主动丢弃数值噪声", GR,
         "69.4% 的系数振幅 < 0.01，其目标相位是纯 atan2 噪声。不加权就去拟合噪声。"),
    ]
    for i, (q, a, col, d) in enumerate(items):
        x = 150 + i * 545
        c.add(f'<g class="anim-rise" style="animation-delay:{.15+i*0.13}s">')
        c.add(panel(x, 630, 505, 200, fill="#0A0E1B", stroke=C["line"], op=0.72, rx=16))
        c.add(f'<rect x="{x}" y="630" width="4" height="200" rx="2" fill="{col}"/>')
        c.add(T(x + 28, 674, q, 21, col, "800"))
        c.add(T(x + 28, 710, "→ " + a, 18, C["ink"], "700"))
        c.add(Twrap(x + 28, 750, d, 450, 16, C["ink2"], "500", 24, maxlines=4))
        c.add("</g>")
    # 底部：日程 + 五频段读出
    c.add(f'<g class="anim-rise" style="animation-delay:.55s">')
    c.add(panel(150, 856, 800, 90, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=14))
    c.add(T(178, 894, "日程：单一 AdamW · 恒定 lr=3e-4 · 无 warmup · 2,500 步 · 固定种子 1234",
            18, C["ink2"], "600"))
    c.add(T(178, 924, "固定种子是协议的一部分：目标是验证通路容量，不是跨样本泛化。",
            16, C["ink3"], "500"))
    c.add("</g>")
    c.add(f'<g class="anim-rise" style="animation-delay:.68s">')
    c.add(panel(975, 856, 795, 90, fill="#0C1424", stroke=CY, op=0.72, rx=14))
    c.add(T(1003, 894, "五频段读出沿真实 rfft2 w 轴切分（W_FFT=25）", 18, CY, "700"))
    c.add(T(1003, 924, "均匀 0/16/32/48 切分会混进 DC，读数就失去意义。", 16, C["ink3"], "500"))
    c.add("</g>")
    return c.render()


# ============================================================ 15 结果
def card_15():
    c = Card(15, TOTAL, "RESULTS", "数字说话",
             "2,500 步之后，加权相位余弦 0.9991，频谱轮廓余弦 0.9967。物理通路走得通。",
             accent=GR, uid="c15")
    c.addcss("""
@keyframes barGrow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.bg2{animation:barGrow 1.8s cubic-bezier(.2,.8,.2,1) both;transform-box:fill-box;transform-origin:left center}
""")
    # 四个大指标
    kpis = [("0.9991", "功率加权相位余弦", "1.0 = 完全对齐", CY),
            ("0.9967", "频谱轮廓余弦", "0 = 频谱被压平", MG),
            ("0.9985", "cos_eff (a_tar>1)", "只算重要系数", AM),
            ("19.84 dB", "对编解码天花板 PSNR", "2500 步", GR)]
    for i, (v, l, s, col) in enumerate(kpis):
        x = 150 + i * 408
        c.add(f'<g class="anim-rise" style="animation-delay:{.1+i*0.1}s">')
        c.add(kpi(x, 350, 378, 210, v, l, s, col, vsize=58))
        c.add("</g>")
    # 对比表
    c.add(T(150, 606, "v2500 → v3600 对比", 21, C["ink"], "800"))
    c.add(T(390, 606, "训练片段不同，像素 PSNR 不可直接比", 16, C["ink3"], "500"))
    c.add(T(660, 606, "v2500", 17, C["ink3"], "700", mono=True))
    c.add(T(900, 606, "v3600", 17, AM, "700", mono=True))
    rows = [
        ("可训练参数", "1,135,658", "2,022,232", AM),
        ("功率加权相位余弦 wcos", "0.8707", "0.9991", CY),
        ("频谱轮廓余弦", "0.9899", "0.9967", MG),
        ("编解码往返天花板", "28.63 dB", "49.82 dB", GR),
    ]
    y = 628
    for name, a, b, col in rows:
        c.add(f'<g class="anim-riseL" style="animation-delay:{.3+y/3000:.2f}s">')
        c.add(f'<rect x="150" y="{y}" width="900" height="50" rx="10" fill="#0C1322" opacity="0.6"/>')
        c.add(T(174, y + 32, name, 19, C["ink2"], "600"))
        c.add(T(660, y + 32, a, 20, C["ink3"], "600", mono=True))
        c.add(T(900, y + 32, b, 20, col, "800", mono=True))
        c.add("</g>")
        y += 58
    # 右：频段读出
    rx = 1090
    c.add(panel(rx, 600, 680, 320, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=18))
    c.add(T(rx + 26, 640, "五频段读出 / 目标值", 20, CY, "800"))
    bands = [("DC", 1.02), ("低频", 1.13), ("中低", 1.48), ("中高", 2.18), ("高频", 1.71)]
    for i, (bn, v) in enumerate(bands):
        yy = 682 + i * 38
        c.add(T(rx + 26, yy, bn, 17, C["ink2"], "600"))
        c.add(f'<rect x="{rx+120}" y="{yy-13}" width="340" height="15" rx="7" fill="#101A2C"/>')
        c.add(f'<rect class="bg2" x="{rx+120}" y="{yy-13}" width="{340*min(v/2.4,1):.0f}" height="15" '
              f'rx="7" fill="{AM if v>1.4 else CY}" style="transform-origin:{rx+120}px {yy-5}px;'
              f'animation-delay:{.4+i*0.08:.2f}s"/>')
        c.add(T(rx + 490, yy, f"{v:.2f}", 18, AM if v > 1.4 else CY, "800", mono=True))
    c.add(Twrap(rx + 26, 878, "全部高于 1.0，因为水平项 μ 仍有偏——这正是 §5.4 记录的 "
                               "log 与 log1p 域不一致。", 620, 15, C["ink3"], "500", 20))
    # 底部提示
    c.add(f'<g class="anim-rise" style="animation-delay:.7s">')
    c.add(panel(150, 872, 900, 48, fill="#120C12", stroke=RD, op=0.7, rx=12))
    c.add(T(178, 903, "注意：未加权平均相位余弦从 0.9933 微降到 0.9885 —— 这是好事", 17, RD, "700"))
    c.add("</g>")
    c.add(f'<g class="anim-rise" style="animation-delay:.78s">')
    c.add(T(1075, 903, "因为模型在抬高弱系数的振幅，它们的噪声相位进入了平均。", 15, C["ink3"], "500"))
    c.add("</g>")
    return c.render()


# ============================================================ 16 八条原则
def card_16():
    c = Card(16, TOTAL, "DESIGN PRINCIPLES", "八条承重结论",
             "每一条都由一个被记录在案的失败模式换来。这是 AetherGIF 这条线最有价值的部分。",
             accent=CY, uid="c16")
    items = [
        ("ω 必须确定性", "每次前向重采样本征频率 = 每步换掉整个动力系统，什么都学不到。", CY),
        ("漂移场是唯一目标通道", "噪声初值 + 确定性 ω 之下，目标信息只能靠吸引子进入相位演化。", AM),
        ("振幅在 log 域乘性演化", "加性更新会锁死动态范围，把重尾频谱压平。", GR),
        ("相位链永不折叠", "折叠只属于损失；边界处梯度为零，会静默掐断 12 层的信用分配。", MG),
        ("相位按功率加权，振幅按 log 软化", "大多数 FFT 系数是数值噪声，无权重监督就是在拟合噪声。", CY),
        ("深度就是物理时间", "耦合课程 + 温度退火 + 频率扫频，让 12 层成为一条热力学叙事。", AM),
        ("编解码器是架构的一部分", "一行 spec[:, :D] 截断支配了所有下游视觉指标。", RD),
        ("均值指标会骗人", "验收必须看空间域统计，且以原始源为下限，绝不拿过度锐化的旧权重当基准。", GR),
    ]
    for i, (t, d, col) in enumerate(items):
        r, cc = divmod(i, 2)
        x = 150 + cc * 830
        y = 350 + r * 152
        c.add(f'<g class="anim-rise" style="animation-delay:{.08*i:.2f}s">')
        c.add(panel(x, y, 790, 132, fill="#0A0E1B", stroke=C["line"], op=0.7, rx=16))
        c.add(f'<rect x="{x}" y="{y}" width="4" height="132" rx="2" fill="{col}"/>')
        c.add(badge_num(x + 48, y + 44, f"{i+1:02d}", col, r=22, size=17))
        c.add(T(x + 88, y + 52, t, 22, C["ink"], "800"))
        c.add(Twrap(x + 30, y + 96, d, 730, 16, C["ink2"], "500", 23, maxlines=2))
        c.add("</g>")
    return c.render()


# ============================================================ 17 已知局限
def card_17():
    c = Card(17, TOTAL, "KNOWN LIMITATIONS", "诚实的边界",
             "这是一次单样本过拟合实验。它的价值在于验证通路，而不是成为一个产品。",
             accent=RD, uid="c17")
    items = [
        ("单样本过拟合", "跨片段泛化未经验证；固定种子是协议的一部分，不是疏忽。", RD),
        ("方案 D 的 γ 一身二职", "同时管水平收敛与形状衰减；自训到 ≈0.04 时 61% 初始水平撑过 12 层。", AM),
        ("log 与 log1p 域不一致", "架构在 log 域递推、损失在 log1p 域聚合，实测差 ≈3.05 个单位，μ 因此振荡。", MG),
        ("固定配置", "128×128、3 秒（30 帧 @10fps）。更长时间/变长生成有索引构建器支持，但没训过。", CY),
        ("诊断脚本路径硬编码", "verify_dmodel_ceiling.py 与 eval_generalize.py 写死了 D:/AetherGIF-V1 路径。", GR),
        ("成本不低", "分块两遍 drift + 梯度检查点才把 N=6,144 塞进 4GB 笔记本显卡。这不是快架构。", ORG),
    ]
    for i, (t, d, col) in enumerate(items):
        r, cc = divmod(i, 2)
        x = 150 + cc * 830
        y = 350 + r * 150
        c.add(f'<g class="anim-rise" style="animation-delay:{.1+i*0.1}s">')
        c.add(panel(x, y, 790, 130, fill="#0F0C12", stroke=C["line"], op=0.7, rx=16))
        c.add(f'<rect x="{x}" y="{y}" width="4" height="130" rx="2" fill="{col}"/>')
        c.add(f'<path d="M{x+34} {y+44} l16 26 h-32 z" fill="none" stroke="{col}" stroke-width="2" '
              f'stroke-linejoin="round"/>')
        c.add(f'<line x1="{x+42}" y1="{y+54}" x2="{x+42}" y2="{y+64}" stroke="{col}" stroke-width="2"/>')
        c.add(f'<circle cx="{x+42}" cy="{y+69}" r="1.6" fill="{col}"/>')
        c.add(T(x + 70, y + 52, t, 22, C["ink"], "800"))
        c.add(Twrap(x + 30, y + 94, d, 730, 16, C["ink2"], "500", 23, maxlines=2))
        c.add("</g>")
    # 底部：这个实验到底证明了什么
    c.add(f'<g class="anim-rise" style="animation-delay:.75s">')
    c.add(panel(150, 820, 1620, 126, fill="#0A1220", stroke=CY, op=0.75, rx=18))
    c.add(f'<rect x="150" y="820" width="4" height="126" rx="2" fill="{CY}"/>')
    c.add(T(186, 862, "那么，这个实验到底证明了什么？", 21, CY, "800"))
    c.add(Twrap(186, 900, "证明了「噪声 → 振荡器场 → 像素」这条纯物理通路在工程上是走得通的，"
                          "而且在单样本上可以拟合到极高保真度——编解码天花板抬高 21 dB 之后，"
                          "物理核心真的把这份额外余量吃掉了（wcos 0.8707 → 0.9991）。",
                1560, 18, C["ink2"], "500", 26))
    c.add("</g>")
    return c.render()


# ============================================================ 18 结语
def card_18():
    c = Card(18, TOTAL, "WHERE TO GO NEXT", "物理路线能走多远？",
             "v3600 把「能不能用物理替代注意力」这个问题，从想象推进到了有数字支撑的实验事实。",
             accent=MG, uid="c18", variant="hero")
    # 三个方向
    dirs = [
        ("多片段泛化", "把单样本过拟合换成跨片段训练，才能谈「生成」两个字。", CY),
        ("γ 拆分 + 域统一", "γ_level / γ_shape 分家，log 与 log1p 统一到同一个域。", AM),
        ("文本条件生成", "Qwen2.5-0.5B 的四个条件头已经接好，只是还没被训练。", MG),
    ]
    for i, (t, d, col) in enumerate(dirs):
        x = 150 + i * 545
        c.add(f'<g class="anim-rise" style="animation-delay:{.15+i*0.13}s">')
        c.add(panel(x, 356, 505, 180, fill="#0A0E1B", stroke=col, op=0.78, rx=18))
        c.add(f'<rect x="{x}" y="356" width="505" height="3" rx="1.5" fill="{col}" opacity="0.9"/>')
        c.add(badge_num(x + 44, 408, f"{i+1}", col, r=20, size=18))
        c.add(T(x + 82, 416, t, 22, C["ink"], "800"))
        c.add(Twrap(x + 28, 462, d, 450, 16, C["ink2"], "500", 24))
        c.add("</g>")
    # 复现路径
    c.add(T(150, 604, "复现路径", 22, C["ink"], "800"))
    c.add(code(150, 628, 1000, 300, [
        "python scripts/diag_grid_source.py        # 1. D=125 往返无损",
        "python scripts/verify_dmodel_ceiling.py   # 2. 截断消融 D=64/100/125",
        "python scripts/overfit.py 150  3D87B14F...gif   # 3. 冒烟",
        "python scripts/overfit.py 2500 3D87B14F...gif   # 4. 完整训练",
        "python scripts/eval_generalize.py         # 5. 空间域评估",
    ], accent=CY, size=17, lead=42, title="COMMANDS"))
    # 参数指纹
    c.add(panel(1180, 628, 590, 300, fill="#0C1424", stroke=AM, op=0.78, rx=18))
    c.add(T(1206, 672, "身份指纹（回滚 / 复现锚点）", 19, AM, "800"))
    c.add(f'<rect x="1206" y="692" width="538" height="86" rx="10" fill="#060A14"/>')
    c.add(T(1224, 722, "sd = torch.load(ckpt, map_location='cpu')", 15, C["ink2"], "500", mono=True))
    c.add(T(1224, 748, "print(sum(v.numel() for v in sd.values()))", 15, C["ink2"], "500", mono=True))
    c.add(T(1224, 770, "# 必须打印 2,022,232", 15, AM, "700", mono=True))
    c.add(Twrap(1206, 812, "训练日志、权重、输出 GIF 与本报告中的每一个数字都随快照一起发布，"
                           "任何人都能重新生成。", 540, 16, C["ink2"], "500", 24))
    c.add(T(1206, 892, "AetherGIF v3600 · 方案 D · D=125", 16, C["ink3"], "600", mono=True))
    # 收尾金句
    c.add(f'<g class="anim-rise" style="animation-delay:.8s">')
    c.add(panel(150, 952, 1620, 1, fill="none", stroke="none"))
    c.add(f'<rect x="150" y="944" width="1620" height="1" fill="{C["line2"]}"/>')
    c.add(T(150, 992, "把「注意」换成「共振」，把「层」换成「时间」——", 30, C["ink"], "800"))
    c.add(T(1195, 992, "这条路已经走通了第一段。", 30, MG, "800", anchor="end"))
    c.add("</g>")
    return c.render()
