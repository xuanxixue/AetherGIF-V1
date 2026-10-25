# -*- coding: utf-8 -*-
"""
deck.py — AetherGIF v3600 科普卡片视觉套件
统一 16:9 (1920x1080) · 暗色科技风 · 纯 SVG 动画 (CSS keyframes + SMIL)
"""
import math, random

W, H = 1920, 1080
PAD = 104

CUR = {"uid": "x"}          # 当前卡的 id 前缀（供全局辅助函数引用 defs）

# ---------------------------------------------------------------- palette
C = dict(
    bg0="#04060D", bg1="#0A0F1E", bg2="#131B32",
    ink="#EDF3FF", ink2="#9DABC8", ink3="#5B6884",
    line="#22304E", line2="#18223A",
    cy="#2DD4F0", cyD="#0C4A5C",
    mg="#C084FC", mgD="#43246B",
    am="#FBBF24", amD="#4E3508",
    gr="#34D399", grD="#0B4634",
    rd="#FB7185", rdD="#4E1526",
    org="#FB923C", orgD="#4A2408",
)
CY, MG, AM, GR, RD, ORG = C["cy"], C["mg"], C["am"], C["gr"], C["rd"], C["org"]

FS = "'Inter','HarmonyOS Sans SC','PingFang SC','Microsoft YaHei','Noto Sans SC',system-ui,sans-serif"
FM = "'JetBrains Mono','Cascadia Mono','SF Mono',Consolas,ui-monospace,monospace"


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def text_w(s, size, mono=False):
    w = 0.0
    for ch in s:
        o = ord(ch)
        if o > 0x2E80:
            w += 1.0
        elif ch == " ":
            w += 0.30
        elif mono:
            w += 0.60
        elif ch.isupper() or ch.isdigit():
            w += 0.60
        else:
            w += 0.52
    return w * size


def wrap(s, size, maxw, mono=False):
    lines, cur = [], ""
    tokens, buf = [], ""
    for ch in s:
        if ord(ch) > 0x2E80 or ch == " ":
            if buf:
                tokens.append(buf); buf = ""
            tokens.append(ch)
        else:
            buf += ch
    if buf:
        tokens.append(buf)
    for tk in tokens:
        cand = cur + tk
        if text_w(cand.rstrip(), size, mono) > maxw and cur.strip():
            lines.append(cur.rstrip()); cur = "" if tk == " " else tk
        else:
            cur = cand
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


# ---------------------------------------------------------------- css
BASE_CSS = """
@keyframes drift1{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(90px,60px) scale(1.14)}}
@keyframes drift2{0%,100%{transform:translate(0,0) scale(1.1)}50%{transform:translate(-110px,40px) scale(0.94)}}
@keyframes drift3{0%,100%{transform:translate(0,0)}50%{transform:translate(60px,-80px)}}
@keyframes spinSlow{to{transform:rotate(360deg)}}
@keyframes spinRev{to{transform:rotate(-360deg)}}
@keyframes pulse{0%,100%{opacity:.35;transform:scale(1)}50%{opacity:.9;transform:scale(1.07)}}
@keyframes pulseSoft{0%,100%{opacity:.18}50%{opacity:.62}}
@keyframes scanline{0%{transform:translateY(-320px)}100%{transform:translateY(1300px)}}
@keyframes dashflow{to{stroke-dashoffset:-440}}
@keyframes floatUp{0%{transform:translate(0,0);opacity:0}
 14%{opacity:var(--op,.5)}82%{opacity:var(--op,.5)}100%{transform:translate(var(--dx,20px),-190px);opacity:0}}
@keyframes twinkle{0%,100%{opacity:.12;transform:scale(.55)}50%{opacity:1;transform:scale(1.3)}}
@keyframes gridScroll{to{transform:translateY(64px)}}
@keyframes glowPulse{0%,100%{filter:drop-shadow(0 0 5px var(--gc,#2DD4F0))}
 50%{filter:drop-shadow(0 0 24px var(--gc,#2DD4F0))}}
@keyframes sheenMove{0%{transform:translateX(-160%) skewX(-18deg)}100%{transform:translateX(360%) skewX(-18deg)}}
@keyframes slideIn{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideL{from{opacity:0;transform:translateX(-34px)}to{opacity:1;transform:translateX(0)}}
@keyframes growX{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes popIn{0%{opacity:0;transform:scale(.86)}68%{transform:scale(1.035)}100%{opacity:1;transform:scale(1)}}
@keyframes ringSpin{0%{transform:rotate(0) scale(1)}50%{transform:rotate(180deg) scale(1.05)}100%{transform:rotate(360deg) scale(1)}}
@keyframes dashdraw{from{stroke-dashoffset:var(--len,1000)}to{stroke-dashoffset:0}}
@keyframes breathe{0%,100%{transform:scale(1);opacity:.85}50%{transform:scale(1.05);opacity:1}}
@keyframes converge{0%{transform:translate(0,0) scale(1);opacity:.35}
 100%{transform:translate(var(--tx,0),var(--ty,0)) scale(.28);opacity:1}}
@keyframes orbit{from{transform:rotate(0) translateX(var(--rad,180px)) rotate(0)}
 to{transform:rotate(360deg) translateX(var(--rad,180px)) rotate(-360deg)}}
@keyframes waveUp{0%,100%{transform:translateY(0)}50%{transform:translateY(-13px)}}
@keyframes blink{0%,45%{opacity:1}50%,95%{opacity:.22}100%{opacity:1}}

text{font-family:__FS__}
.mono{font-family:__FM__}
.anim-rise{animation:slideIn .95s cubic-bezier(.2,.8,.2,1) both}
.anim-riseL{animation:slideL .95s cubic-bezier(.2,.8,.2,1) both}
.anim-pop{animation:popIn .8s cubic-bezier(.2,.9,.25,1) both}
.anim-breathe{animation:breathe 5.2s ease-in-out infinite}
.anim-spin{animation:spinSlow 28s linear infinite;transform-box:fill-box;transform-origin:center}
.anim-spinr{animation:spinRev 36s linear infinite;transform-box:fill-box;transform-origin:center}
.anim-spin-c{animation:spinSlow 22s linear infinite;transform-origin:center}
.anim-glow{animation:glowPulse 3.6s ease-in-out infinite}
.anim-d1{animation:drift1 19s ease-in-out infinite}
.anim-d2{animation:drift2 23s ease-in-out infinite}
.anim-d3{animation:drift3 17s ease-in-out infinite}
.anim-scan{animation:scanline 9.5s linear infinite}
.anim-sheen{animation:sheenMove 12s ease-in-out infinite}
.anim-float{animation:floatUp linear infinite}
.anim-twinkle{animation:twinkle 3.8s ease-in-out infinite}
.anim-growx{animation:growX 1.6s cubic-bezier(.2,.8,.2,1) both}
.anim-draw{animation:dashdraw 2.3s ease-out both}
.anim-converge{animation:converge 3.4s cubic-bezier(.2,.8,.2,1) both}
.anim-orbit{animation:orbit var(--dur,14s) linear infinite;transform-box:fill-box;transform-origin:center}
.anim-wave{animation:waveUp var(--dur,3.4s) ease-in-out infinite}
.anim-blink{animation:blink 2.4s steps(1) infinite}
.anim-dash{animation:dashflow 2.4s linear infinite}
""".replace("__FS__", FS).replace("__FM__", FM)


# ---------------------------------------------------------------- defs
def defs_common(uid, accent=CY):
    return f"""
<defs>
  <linearGradient id="{uid}_bg" x1="0" y1="0" x2="0.75" y2="1">
    <stop offset="0%" stop-color="{C['bg0']}"/>
    <stop offset="52%" stop-color="{C['bg1']}"/>
    <stop offset="100%" stop-color="#070B16"/>
  </linearGradient>
  <radialGradient id="{uid}_a1" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0%" stop-color="{accent}" stop-opacity="0.40"/>
    <stop offset="100%" stop-color="{accent}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="{uid}_a2" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0%" stop-color="{MG}" stop-opacity="0.24"/>
    <stop offset="100%" stop-color="{MG}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="{uid}_a3" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0%" stop-color="{AM}" stop-opacity="0.16"/>
    <stop offset="100%" stop-color="{AM}" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="{uid}_vig" cx="0.5" cy="0.48" r="0.74">
    <stop offset="52%" stop-color="#000" stop-opacity="0"/>
    <stop offset="100%" stop-color="#000" stop-opacity="0.82"/>
  </radialGradient>
  <linearGradient id="{uid}_sheen" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#fff" stop-opacity="0"/>
    <stop offset="48%" stop-color="#fff" stop-opacity="0.06"/>
    <stop offset="100%" stop-color="#fff" stop-opacity="0"/>
  </linearGradient>
  <linearGradient id="{uid}_accent" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="{accent}"/>
    <stop offset="100%" stop-color="{MG}"/>
  </linearGradient>
  <linearGradient id="{uid}_fade" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="{accent}" stop-opacity="0"/>
    <stop offset="50%" stop-color="{accent}" stop-opacity="0.85"/>
    <stop offset="100%" stop-color="{accent}" stop-opacity="0"/>
  </linearGradient>
  <filter id="{uid}_soft" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur stdDeviation="92"/>
  </filter>
  <filter id="{uid}_soft2" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur stdDeviation="36"/>
  </filter>
  <filter id="{uid}_glow" x="-70%" y="-70%" width="240%" height="240%">
    <feGaussianBlur stdDeviation="7" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="{uid}_glows" x="-70%" y="-70%" width="240%" height="240%">
    <feGaussianBlur stdDeviation="2.8" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="{uid}_titleglow" x="-60%" y="-60%" width="220%" height="220%">
    <feGaussianBlur stdDeviation="3.5" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <filter id="{uid}_grain" x="0" y="0" width="100%" height="100%">
    <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" seed="7" result="n"/>
    <feColorMatrix in="n" type="saturate" values="0"/>
  </filter>
  <pattern id="{uid}_dot" width="46" height="46" patternUnits="userSpaceOnUse">
    <circle cx="2" cy="2" r="1.1" fill="{C['ink3']}" opacity="0.30"/>
  </pattern>
  <pattern id="{uid}_fine" width="10" height="10" patternUnits="userSpaceOnUse">
    <path d="M10 0H0V10" fill="none" stroke="{accent}" stroke-width="0.5" opacity="0.11"/>
  </pattern>
  <marker id="{uid}_ar" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="6.5"
          markerHeight="6.5" orient="auto-start-reverse">
    <path d="M0 0 L10 5 L0 10 z" fill="{accent}"/>
  </marker>
</defs>"""


# ---------------------------------------------------------------- backdrop
def backdrop(uid, accent=CY, variant="aurora", seed=1):
    p = [f'<rect width="{W}" height="{H}" fill="url(#{uid}_bg)"/>']
    if variant in ("aurora", "rich", "hero"):
        p.append(f'<g filter="url(#{uid}_soft)" style="mix-blend-mode:screen">'
                 f'<ellipse class="anim-d1" cx="300" cy="200" rx="560" ry="400" fill="url(#{uid}_a1)"/>'
                 f'<ellipse class="anim-d2" cx="1720" cy="900" rx="430" ry="330" fill="url(#{uid}_a2)"/>'
                 f'<ellipse class="anim-d3" cx="1120" cy="90" rx="360" ry="270" fill="url(#{uid}_a3)"/>'
                 f'</g>')
    p.append(f'<rect width="{W}" height="{H}" fill="url(#{uid}_dot)" opacity="0.5"/>')
    p.append(f'<rect width="{W}" height="{H}" fill="url(#{uid}_fine)" opacity="0.5"/>')
    p.append(f'<rect x="0" y="0" width="{W}" height="2" fill="url(#{uid}_accent)" opacity="0.85"/>')
    return "\n".join(p)


def grain(uid, op=0.055):
    return (f'<rect width="{W}" height="{H}" filter="url(#{uid}_grain)" opacity="{op}" '
            f'style="mix-blend-mode:overlay"/>')


def vignette(uid):
    return f'<rect width="{W}" height="{H}" fill="url(#{uid}_vig)"/>'


def sheen(uid):
    return (f'<g style="mix-blend-mode:screen;opacity:.85">'
            f'<rect class="anim-sheen" x="0" y="0" width="430" height="{H}" '
            f'fill="url(#{uid}_sheen)"/></g>')


def scanbeam(uid, y=0, h=300, color=CY, op=0.13):
    return (f'<g style="mix-blend-mode:screen">'
            f'<rect class="anim-scan" x="0" y="{y}" width="{W}" height="{h}" '
            f'fill="url(#{uid}_accent)" opacity="{op}"/></g>')


def persp_grid(uid, horizon=690, color=CY, op=0.30, rows=16, cols=30, vx=960):
    p = [f'<g opacity="{op}" stroke="{color}" stroke-width="1" fill="none">']
    for i in range(rows + 1):
        t = i / rows
        y = horizon + (H - horizon) * (t ** 2.35)
        p.append(f'<line x1="0" y1="{y:.1f}" x2="{W}" y2="{y:.1f}" opacity="{0.22+t*0.62:.2f}"/>')
    for j in range(-cols, cols + 1):
        x = vx + j * 80
        p.append(f'<line x1="{vx}" y1="{horizon}" x2="{x:.0f}" y2="{H+70}" opacity="0.40"/>')
    p.append("</g>")
    p.append(f'<line x1="0" y1="{horizon}" x2="{W}" y2="{horizon}" stroke="{color}" '
             f'stroke-width="1.5" opacity="0.5"/>')
    return "\n".join(p)


def particles(uid, n=46, seed=7, color=CY, op=0.55, y0=200, y1=1080):
    rnd = random.Random(seed)
    p = []
    for i in range(n):
        x = rnd.uniform(0, W); y = rnd.uniform(y0, y1)
        r = rnd.uniform(1.1, 3.4)
        dur = rnd.uniform(11, 24); dl = -rnd.uniform(0, 24)
        dx = rnd.uniform(-70, 70)
        o = rnd.uniform(0.18, 0.85)
        p.append(f'<circle class="anim-float" cx="{x:.0f}" cy="{y:.0f}" r="{r:.1f}" '
                 f'fill="{color}" style="--op:{o:.2f};--dx:{dx:.0f}px;'
                 f'animation-duration:{dur:.1f}s;animation-delay:{dl:.1f}s"/>')
    return "\n".join(p)


def hud_frame(uid, accent=CY, op=0.28):
    L, T, S = 46, 46, 58
    p = [f'<g fill="none" stroke="{accent}" stroke-width="2" opacity="{op}">']
    for (x, y, dx, dy) in ((L, T, 1, 1), (W - L, T, -1, 1), (L, H - T, 1, -1), (W - L, H - T, -1, -1)):
        p.append(f'<path d="M{x} {y+dy*S} L{x} {y} L{x+dx*S} {y}"/>')
    p.append("</g>")
    return "\n".join(p)


# ---------------------------------------------------------------- chrome
def header(idx, total, kicker, title, lede, accent=CY, title_size=78, sub2=""):
    p = []
    p.append(f'<text x="{PAD}" y="132" font-size="22" font-weight="700" letter-spacing="7" '
             f'fill="{accent}">{esc(kicker.upper())}</text>')
    p.append(f'<rect x="{PAD}" y="152" width="86" height="4" rx="2" fill="{accent}"/>')
    p.append(f'<text x="{PAD}" y="{132+title_size*0.92:.0f}" font-size="{title_size}" '
             f'font-weight="800" letter-spacing="-1.5" fill="{C["ink"]}">{esc(title)}</text>')
    if lede:
        lines = wrap(lede, 27, 1180)
        y = 132 + title_size * 0.92 + 50
        for ln in lines[:3]:
            p.append(f'<text x="{PAD}" y="{y:.0f}" font-size="27" fill="{C["ink2"]}">{esc(ln)}</text>')
            y += 40
    p.append(f'<text x="{W-PAD}" y="128" text-anchor="end" font-size="19" font-weight="600" '
             f'letter-spacing="3" fill="{C["ink3"]}">AETHERGIF v3600</text>')
    p.append(f'<text x="{W-PAD}" y="{132+title_size*0.92:.0f}" text-anchor="end" font-size="52" '
             f'font-weight="800" fill="{C["ink3"]}">{idx:02d}'
             f'<tspan fill="{C["line"]}"> / {total:02d}</tspan></text>')
    if sub2:
        p.append(f'<text x="{W-PAD}" y="{132+title_size*0.92+40:.0f}" text-anchor="end" '
                 f'font-size="19" fill="{C["ink3"]}">{esc(sub2)}</text>')
    return "\n".join(p)


def footer(uid, idx, total, accent=CY, note=""):
    y = H - 62
    p = [f'<rect x="{PAD}" y="{y-34}" width="{W-2*PAD}" height="1" fill="{C["line2"]}"/>']
    frac = idx / total
    p.append(f'<rect x="{PAD}" y="{y-34}" width="{(W-2*PAD)*frac:.0f}" height="2" '
             f'fill="{accent}" opacity="0.75"/>')
    p.append(f'<text x="{PAD}" y="{y+4}" font-size="18" fill="{C["ink3"]}">'
             f'{esc(note or "AetherGIF v3600 · 方案 D · D=125 全谱编解码")}</text>')
    p.append(f'<text x="{W-PAD}" y="{y+4}" text-anchor="end" font-size="18" fill="{C["ink3"]}" '
             f'class="mono">v3600_方案D_D125 / TECHNICAL_REPORT</text>')
    return "\n".join(p)


# ---------------------------------------------------------------- atoms
def T(x, y, s, size=22, fill=None, weight="500", anchor="start", mono=False,
      ls=None, op=None, extra=""):
    fill = fill or C["ink"]
    a = f'font-size="{size}" font-weight="{weight}" fill="{fill}"'
    if anchor != "start":
        a += f' text-anchor="{anchor}"'
    if ls is not None:
        a += f' letter-spacing="{ls}"'
    if op is not None:
        a += f' opacity="{op}"'
    if mono:
        a += ' class="mono"'
    if extra:
        a += " " + extra
    return f'<text x="{x}" y="{y}" {a}>{esc(s)}</text>'


def Twrap(x, y, s, maxw, size=18, fill=None, weight="500", lead=26, mono=False,
          anchor="start", maxlines=99, ls=None):
    """自动换行多行文本"""
    fill = fill or C["ink2"]
    out = []
    for i, ln in enumerate(wrap(s, size, maxw, mono)):
        if i >= maxlines:
            break
        out.append(T(x, y + i * lead, ln, size, fill, weight, anchor=anchor, mono=mono, ls=ls))
    return "\n".join(out)


def panel(x, y, w, h, fill=None, stroke=None, rx=20, op=0.55, sw=1.2, accent=None):
    fill = fill or "#0C1322"
    stroke = stroke or C["line"]
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" opacity="{op}" '
         f'stroke="{stroke}" stroke-width="{sw}"/>')
    if accent:
        s += (f'<rect x="{x}" y="{y}" width="{w}" height="3" rx="1.5" fill="{accent}" opacity="0.9"/>')
    return s


def corners(x, y, w, h, color=CY, op=0.75, s=18):
    p = [f'<g fill="none" stroke="{color}" stroke-width="1.8" opacity="{op}">']
    for (cx, cy, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        p.append(f'<path d="M{cx} {cy+dy*s} L{cx} {cy} L{cx+dx*s} {cy}"/>')
    p.append("</g>")
    return "\n".join(p)


def chip(x, y, s, color=CY, size=17, solid=False, pad=15, h=33, mono=False):
    w = text_w(s, size, mono) + pad * 2
    if solid:
        r = (f'<rect x="{x}" y="{y}" width="{w:.0f}" height="{h}" rx="{h/2}" fill="{color}"/>'
             f'<text x="{x+w/2:.0f}" y="{y+h/2+size*0.36:.0f}" text-anchor="middle" '
             f'font-size="{size}" font-weight="700" fill="#05070E">{esc(s)}</text>')
    else:
        r = (f'<rect x="{x}" y="{y}" width="{w:.0f}" height="{h}" rx="{h/2}" fill="{color}" '
             f'opacity="0.12" stroke="{color}" stroke-width="1.2"/>'
             f'<text x="{x+w/2:.0f}" y="{y+h/2+size*0.36:.0f}" text-anchor="middle" '
             f'font-size="{size}" font-weight="700" fill="{color}">{esc(s)}</text>')
    return r, w


def code(x, y, w, h, lines, accent=CY, size=21, lead=34, title=None, pad=26, mono=True):
    p = [panel(x, y, w, h, fill="#070C17", stroke=C["line"], op=0.85, rx=18)]
    p.append(f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" fill="{accent}" opacity="0.9"/>')
    if title:
        p.append(T(x + pad, y + 36, title, 17, C["ink3"], "700", ls=2.4))
        ty = y + 78
    else:
        ty = y + 48
    for i, ln in enumerate(lines):
        col = C["ink"] if ln.startswith("  ") else C["ink2"]
        if ln.startswith("#"):
            col = C["ink3"]
        cls = ' class="mono"' if mono else ''
        p.append(f'<text x="{x+pad}" y="{ty+i*lead}" font-size="{size}"{cls} '
                 f'fill="{col}">{esc(ln)}</text>')
    return "\n".join(p)


def bullet(x, y, s, size=24, color=CY, maxw=760, lead=36, dot=True):
    p = []
    if dot:
        p.append(f'<circle cx="{x}" cy="{y-size*0.32:.0f}" r="4.5" fill="{color}" '
                 f'filter="url(#{CUR["uid"]}_glows)"/>')
    for i, ln in enumerate(wrap(s, size, maxw)):
        p.append(T(x + (20 if dot else 0), y + i * lead, ln, size, C["ink2"]))
    return "\n".join(p)


def odo(uid, n, value, x, y, size=104, color=CY, dur=2.0, delay=0.0, mono=True):
    """滚动数字（里程表）—— SMIL + clipPath"""
    s = str(value)
    lh = size * 1.16
    p = []
    cx = x
    for i, ch in enumerate(s):
        if not ch.isdigit():
            p.append(T(cx, y, ch, size, C["ink3"], "700", mono=mono))
            cx += text_w(ch, size, mono)
            continue
        d = int(ch)
        cid = f"{uid}_odo{n}_{i}"
        pw = text_w(ch, size, mono)
        p.append(f'<clipPath id="{cid}"><rect x="{cx-3}" y="{y-size*0.82:.0f}" '
                 f'width="{pw+6:.0f}" height="{size*1.06:.0f}"/></clipPath>')
        strip = []
        for k in range(30):
            strip.append(f'<text x="{cx}" y="{y+k*lh:.1f}" font-size="{size}" class="mono" '
                         f'font-weight="800" fill="{color}">{k%10}</text>')
        p.append(f'<g clip-path="url(#{cid})"><g>'
                 f'<animateTransform attributeName="transform" type="translate" '
                 f'from="0 0" to="0 {-(20+d)*lh:.1f}" dur="{dur}s" begin="{delay}s" '
                 f'calcMode="spline" keyTimes="0;1" keySplines="0.16 0.84 0.24 1" '
                 f'fill="freeze" repeatCount="indefinite"/>'
                 + "".join(strip) + "</g></g>")
        cx += pw
    return "\n".join(p), cx - x


def odo_w(value, size, mono=True):
    return text_w(str(value), size, mono)


def bar(x, y, w, h, frac, color=CY, label=None, value=None, dur=1.6, delay=0.0,
        track="#101A2C", mono_val=True):
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" fill="{track}" opacity="0.9"/>']
    p.append(f'<rect class="anim-growx" x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" '
             f'fill="{color}" style="transform-origin:{x}px {y+h/2}px;'
             f'animation-duration:{dur}s;animation-delay:{delay}s"/>')
    p.append(f'<rect class="anim-growx" x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2}" '
             f'fill="#fff" opacity="0.26" style="transform-origin:{x}px {y+h/2}px;'
             f'animation-duration:{dur}s;animation-delay:{delay}s"/>')
    if label:
        p.append(T(x, y - 14, label, 19, C["ink3"], "600"))
    if value is not None:
        p.append(T(x + w, y - 14, value, 21, color, "800", anchor="end", mono=mono_val))
    return "\n".join(p)


def spark(x, y, w, h, vals, color=CY, dur=2.4, fill=True, sw=2.8, dots=False, dur_a="2.3s"):
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    n = len(vals)
    pts = [(x + w * i / (n - 1), y + h - h * (v - lo) / rng) for i, v in enumerate(vals)]
    d = "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts)
    p = []
    if fill:
        p.append(f'<path d="{d} L{x+w} {y+h} L{x} {y+h} Z" fill="{color}" opacity="0.10"/>')
    L = w * 1.7
    p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
             f'stroke-linejoin="round" class="anim-draw" style="--len:{L:.0f};'
             f'stroke-dasharray:{L:.0f};animation-duration:{dur_a}"/>')
    if dots:
        for px, py in pts[::max(1, n // 6)]:
            p.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{C["bg0"]}" '
                     f'stroke="{color}" stroke-width="2.2"/>')
    return "\n".join(p)


def node(x, y, w, h, title, sub, color=CY, glow=False, tsize=25, ssize=18, rx=16):
    p = [panel(x, y, w, h, fill="#0A1020", stroke=C["line"], op=0.8, rx=rx)]
    if glow:
        p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" '
                 f'stroke="{color}" stroke-width="1.5" opacity="0.5" '
                 f'filter="url(#{CUR["uid"]}_glows)"/>')
    p.append(T(x + 20, y + 42, title, tsize, color, "800"))
    if sub:
        yy = y + 42
        for ln in wrap(sub, ssize, w - 40)[:4]:
            yy += 28
            p.append(T(x + 20, yy, ln, ssize, C["ink2"]))
    return "\n".join(p)


def arrow(x1, y1, x2, y2, color=CY, sw=2.4, op=0.9, dash=None):
    d = ""
    if dash:
        d = f' stroke-dasharray="{dash}" class="anim-dash"'
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" '
            f'opacity="{op}" marker-end="url(#{CUR["uid"]}_ar)"{d}/>')


def flowline(d, color=CY, sw=2.4, op=0.85, dash="14 11", dur=2.4):
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}" opacity="{op}" '
            f'stroke-dasharray="{dash}" class="anim-dash" '
            f'style="animation-duration:{dur}s"/>')


def lattice(x, y, cols, rows, cell, r=2.2, color=CY, dim=None, op=1.0):
    dim = dim or C["ink3"]
    p = []
    for j in range(rows):
        for i in range(cols):
            cx = x + i * cell
            cy = y + j * cell
            o = 0.22 + 0.5 * ((i * 7 + j * 13) % 5) / 5.0
            p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{dim}" opacity="{o*op:.2f}"/>')
    return "\n".join(p)


def kpi(x, y, w, h, value, label, sub, color=CY, vsize=74, radius=20):
    p = [panel(x, y, w, h, fill="#090F1D", stroke=C["line"], op=0.85, rx=radius)]
    p.append(f'<rect x="{x}" y="{y}" width="3" height="{h}" rx="1.5" fill="{color}"/>')
    p.append(T(x + 28, y + 32, label, 17, color, "700", ls=2.2))
    p.append(T(x + 28, y + 32 + vsize * 0.80, value, vsize, C["ink"], "800", mono=True))
    yy = y + 32 + vsize * 0.80 + 34
    for ln in wrap(sub, 18, w - 56)[:3]:
        p.append(T(x + 28, yy, ln, 18, C["ink3"]))
        yy += 26
    return "\n".join(p)


def badge_num(x, y, n, color=CY, r=19, size=19):
    return (f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" opacity="0.16" '
            f'stroke="{color}" stroke-width="1.4"/>'
            f'<text x="{x}" y="{y+size*0.36:.0f}" text-anchor="middle" font-size="{size}" '
            f'font-weight="800" fill="{color}">{n}</text>')


def quote(x, y, s, size=30, color=CY, maxw=1500, lead=46, mark=True):
    p = []
    if mark:
        p.append(f'<text x="{x-14}" y="{y+12}" font-size="{size*2.1}" font-weight="800" '
                 f'fill="{color}" opacity="0.30" font-family="Georgia,serif">“</text>')
    for i, ln in enumerate(wrap(s, size, maxw)):
        p.append(T(x + (32 if mark else 0), y + i * lead, ln, size, C["ink"]))
    return "\n".join(p)


def label_row(x, y, items, color=CY, size=17, gap=12):
    p, cx = [], x
    for it in items:
        s = it[0] if isinstance(it, tuple) else it
        col = it[1] if isinstance(it, tuple) else color
        r, ww = chip(cx, y, s, col, size)
        p.append(r); cx += ww + gap
    return "\n".join(p)


def table(x, y, w, cols, rows, widths, accent=CY, rh=52, hsize=19, msize=21,
          hi_col=None, hi_color=AM):
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="{rh}" rx="10" fill="{accent}" opacity="0.14"/>']
    tot = sum(widths)
    cx = x
    for i, c in enumerate(cols):
        cw = w * widths[i] / tot
        p.append(T(cx + 18, y + rh / 2 + 7, c, hsize, accent, "800", ls=1.2))
        cx += cw
    yy = y + rh
    for ri, row in enumerate(rows):
        if ri % 2 == 1:
            p.append(f'<rect x="{x}" y="{yy}" width="{w}" height="{rh}" fill="{C["ink"]}" opacity="0.032"/>')
        if hi_col is not None and ri in (hi_col if isinstance(hi_col, (list, tuple)) else [hi_col]):
            p.append(f'<rect x="{x}" y="{yy}" width="{w}" height="{rh}" fill="{hi_color}" opacity="0.10"/>')
        cx = x
        for ci, cell in enumerate(row):
            cw = w * widths[ci] / tot
            col = C["ink"] if ci == 0 else C["ink2"]
            bold = "700" if ci == 0 else "500"
            mono = ci > 0
            p.append(T(cx + 18, yy + rh / 2 + 7, str(cell), msize, col, bold, mono=mono))
            cx += cw
        yy += rh
    return "\n".join(p)


def axis(x, y, w, h, xlab="", ylab="", color=C["ink3"], ticks=5):
    p = [f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" stroke="{C["line"]}" stroke-width="1.4"/>',
         f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" stroke="{C["line"]}" stroke-width="1.4"/>']
    for i in range(ticks + 1):
        tx = x + w * i / ticks
        p.append(f'<line x1="{tx:.1f}" y1="{y+h}" x2="{tx:.1f}" y2="{y+h+8}" stroke="{C["line"]}" stroke-width="1.4"/>')
    if xlab:
        p.append(T(x + w, y + h + 30, xlab, 17, C["ink3"], "600", anchor="end"))
    if ylab:
        p.append(T(x - 12, y - 4, ylab, 17, C["ink3"], "600", anchor="end"))
    return "\n".join(p)


# ---------------------------------------------------------------- card shell
class Card:
    def __init__(self, idx, total, kicker, title, lede="", accent=CY, uid=None, seed=1,
                 variant="aurora", title_size=78, sub2="", use_header=True):
        self.idx, self.total = idx, total
        self.kicker, self.title, self.lede = kicker, title, lede
        self.accent = accent
        self.uid = uid or f"c{idx:02d}"
        self.seed = seed
        self.variant = variant
        self.title_size = title_size
        self.sub2 = sub2
        self.use_header = use_header
        self.parts = []
        self.defs_extra = ""
        self.css_extra = ""
        CUR["uid"] = self.uid

    def add(self, *s):
        for x in s:
            self.parts.append(x)
        return self

    def adddefs(self, s):
        self.defs_extra += s
        return self

    def addcss(self, s):
        self.css_extra += s
        return self

    def render(self):
        CUR["uid"] = self.uid
        parts = [defs_common(self.uid, self.accent), self.defs_extra,
                 f'<style>{BASE_CSS}{self.css_extra}</style>',
                 backdrop(self.uid, self.accent, self.variant, self.seed)]
        if self.use_header:
            parts.append(header(self.idx, self.total, self.kicker, self.title, self.lede,
                                self.accent, self.title_size, self.sub2))
        parts += self.parts
        parts.append(grain(self.uid))
        parts.append(vignette(self.uid))
        parts.append(sheen(self.uid))
        parts.append(hud_frame(self.uid, self.accent, 0.28))
        parts.append(footer(self.uid, self.idx, self.total, self.accent))
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
                f'height="{H}" role="img">\n' + "\n".join(parts) + "\n</svg>")
