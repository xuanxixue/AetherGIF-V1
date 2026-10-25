# -*- coding: utf-8 -*-
"""shot.py — 用 Edge headless 把 SVG 渲染成 PNG，用于视觉验收
默认注入 FREEZE_CSS，让所有 CSS 动画停在终态（便于看清最终排版）"""
import os, sys, subprocess, glob, re

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SHOT = os.path.join(BASE, "_shots")
TMP = os.path.join(BASE, "_tmp_render")

FREEZE_CSS = """
<style id="__freeze">
*,*::before,*::after{animation-play-state:paused !important;animation-delay:0s !important;
  animation-duration:0.001s !important;animation-iteration-count:1 !important;
  transition:none !important}
.anim-rise,.anim-riseL,.anim-pop,.anim-title{opacity:1 !important;transform:none !important;
  filter:none !important}
</style>
"""

def freeze(svg_text):
    # 把冻结样式插到 </svg> 前
    return svg_text.replace("</svg>", FREEZE_CSS + "\n</svg>")

def shot(svg_path, out_png, delay=1400):
    os.makedirs(SHOT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    with open(svg_path, "r", encoding="utf-8") as fp:
        txt = fp.read()
    tmp = os.path.join(TMP, os.path.basename(svg_path))
    with open(tmp, "w", encoding="utf-8") as fp:
        fp.write(freeze(txt))
    url = "file:///" + tmp.replace("\\", "/")
    cmd = [EDGE, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
           f"--window-size=1920,1080", f"--virtual-time-budget={delay}",
           f"--screenshot={out_png}", url]
    try:
        subprocess.run(cmd, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print("ERR", e, svg_path)
        return False
    return os.path.exists(out_png)

if __name__ == "__main__":
    args = sys.argv[1:]
    files = []
    for a in args:
        files.extend(glob.glob(a))
    if not files:
        files = sorted(glob.glob(os.path.join(BASE, "cards", "*.svg")))
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(SHOT, base + ".png")
        if os.path.exists(out):
            os.remove(out)
        ok = shot(f, out)
        print(("OK  " if ok else "FAIL"), base,
              (f"{os.path.getsize(out)//1024}KB") if ok else "")
