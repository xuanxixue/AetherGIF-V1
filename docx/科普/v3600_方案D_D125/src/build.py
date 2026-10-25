# -*- coding: utf-8 -*-
"""build.py — 生成全部 18 张卡片 SVG + index.html 画廊"""
import os, sys, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import deck
from deck import W, H

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "cards")

TITLES = {
 1:  "封面 · 用物理造 GIF",
 2:  "研究问题 · 能不能不用注意力",
 3:  "全景架构 · 一帧 GIF 的旅程",
 4:  "状态空间 · 幅度与相位",
 5:  "五块物理 · 替掉 Transformer 栈",
 6:  "DriftField · 唯一的目标通道",
 7:  "Kuramoto · 替掉 self-attention",
 8:  "StandingWaveFFN · 替掉 FFN",
 9:  "OU 恢复力 · 替掉 LayerNorm",
 10: "TimeField · 深度即物理时间",
 11: "编解码器 · 零参数双向变换",
 12: "全谱之争 · D=64 vs D=125",
 13: "LED 网格伪影 · 形状比数量致命",
 14: "训练协议 · 加权损失设计",
 15: "结果 · 数字说话",
 16: "八条设计原则",
 17: "已知局限 · 诚实边界",
 18: "结语 · 物理路线能走多远",
}

def build():
    import cards_a, cards_b, cards_c
    mods = [cards_a, cards_b, cards_c]
    fns = []
    for m in mods:
        for i in range(1, 19):
            f = getattr(m, f"card_{i:02d}", None)
            if f:
                fns.append((i, f))
    fns.sort()
    os.makedirs(OUT, exist_ok=True)
    made = []
    for i, f in fns:
        svg = f()
        name = f"{i:02d}_{TITLES.get(i, 'card').split('·')[0].strip().replace(' ', '_')}.svg"
        path = os.path.join(OUT, name)
        with io.open(path, "w", encoding="utf-8") as fp:
            fp.write(svg)
        made.append((i, name, TITLES.get(i, ""), len(svg)))
        print(f"[ok] {name}  {len(svg)/1024:.1f} KB")
    # 导出清单供 studio.py 使用
    import json
    with io.open(os.path.join(os.path.dirname(HERE), "_cards.json"), "w", encoding="utf-8") as fp:
        json.dump([[i, n, t] for i, n, t, _ in made], fp, ensure_ascii=False)
    return made

def gallery(made):
    rows = "\n".join(
        f'<figure class="c"><div class="box"><img src="cards/{n}" alt="{t}" loading="lazy"></div>'
        f'<figcaption><b>{i:02d}</b> {t}</figcaption></figure>'
        for i, n, t, _ in made)
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AetherGIF v3600 · 科普卡片</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#05070E;color:#EDF3FF;font-family:'Inter','HarmonyOS Sans SC','PingFang SC','Microsoft YaHei',system-ui,sans-serif;padding:56px 40px 90px}}
h1{{font-size:40px;font-weight:800;letter-spacing:-1px;margin:0 0 8px}}
.sub{{color:#8E9CBC;font-size:17px;margin:0 0 44px;line-height:1.7}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:34px}}
.c{{margin:0}}
.box{{aspect-ratio:16/9;border-radius:16px;overflow:hidden;border:1px solid #1E2A44;background:#04060D;
 box-shadow:0 18px 50px rgba(0,0,0,.6);transition:transform .35s,border-color .35s,box-shadow .35s}}
.box:hover{{transform:translateY(-6px) scale(1.012);border-color:#2DD4F0;box-shadow:0 24px 70px rgba(45,212,240,.16)}}
img{{width:100%;height:100%;display:block}}
figcaption{{margin-top:12px;color:#7C8AA8;font-size:15px}}
figcaption b{{color:#2DD4F0;font-family:ui-monospace,monospace;margin-right:8px}}
</style></head><body>
<h1>AetherGIF v3600 · 科普卡片集</h1>
<p class="sub">18 张动态 SVG 卡片，统一 16:9。主题：用耦合振子物理替代 self-attention 的 GIF 生成实验。<br>
方案 D · D=125 全谱编解码 · 2,022,232 可训练参数 · 2,500 步单样本过拟合。</p>
<div class="grid">
{rows}
</div></body></html>"""
    p = os.path.join(os.path.dirname(HERE), "index.html")
    with io.open(p, "w", encoding="utf-8") as fp:
        fp.write(html)
    return p

if __name__ == "__main__":
    made = build()
    p = gallery(made)
    print(f"\n共 {len(made)} 张卡片 -> {OUT}")
    print(f"画廊 -> {p}")
