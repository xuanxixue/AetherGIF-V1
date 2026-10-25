# -*- coding: utf-8 -*-
"""矩阵结果汇总器：把mx_A..mx_E（及 fix_r3/r4）的日志拉成一张对照表。

★ 必须以**绝对判据**为主（PSNR/ac1/hilo），wcos/margin 只作辅助
  （记忆铁律：margin/wcos 是相对指标，全噪声图上同样成立）。

用法：
  C:/Python312/python.exe -u scripts/summarize_matrix.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

RUNS = [
    ("fix_r3", "基线(LN修复,clamp=pi/2,norm)"),
    ("fix_r4", "A: clamp=pi + BAND_ABS_W=0.3"),
    ("mx_A", "A: lowrank=16mode=norm"),
    ("mx_B", "B: lowrank=64  mode=norm"),
    ("mx_C", "C: lowrank=16  mode=uniform"),
    ("mx_D", "D: lowrank=64  mode=uniform"),
    ("mx_E", "E: lowrank=16  mode=band"),
]

STEP_RE = re.compile(r"\[\s*(\d+)\].*?wcos=([+-][\d.]+).*?\|dθ\|=([\d.]+)"
                     r".*?hi/lo比=([\d.]+)")
# 绝对判据模型行：三路全给|src0  PSNR  噪声基线  ac1  sobel  hilo  素材hilo  判决
# 只匹配「三路全给|srcN」开头且第2列是数字的行（排除「vs_A=」的相对判据行）
PSNR_RE = re.compile(r"^\s*三路全给\|src(\d)\s+([\d.]+)\s+[\d.]+dB\s+"
                     r"([+-][\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", re.M)


def parse(tag):
    p = ROOT / "outputs_real" / f"{tag}.log"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="ignore")
    steps = [(int(a), float(b), float(c), float(d))
             for a, b, c, d in STEP_RE.findall(txt)]
    # PSNR_RE 捕获: (src, psnr, ac1, sobel, hilo, mat_hilo)
    absq = [(int(g), float(psnr), float(ac1), float(hilo), float(mhilo))
            for g, psnr, ac1, _, hilo, mhilo in PSNR_RE.findall(txt)]
    return {"steps": steps, "abs": absq,
            "verdict": "PASS" if "总判定" in txt and "FAIL" not in
            txt.split("总判定")[-1] else ("FAIL" if "FAIL" in txt else "?")}


def main():
    rows = []
    for tag, desc in RUNS:
        r = parse(tag)
        if r is None:
            continue
        st = r["steps"]
        ab = r["abs"]
        best_wcos = max((s[1] for s in st), default=float("nan"))
        last_hilo = st[-1][3] if st else float("nan")
        a0 = next((x for x in ab if x[0] == 0), None)
        a1 = next((x for x in ab if x[0] == 1), None)
        rows.append((tag, desc, len(st), best_wcos, last_hilo,
                     a0[1] if a0 else float("nan"),
                     a0[2] if a0 else float("nan"),
                     a0[3] if a0 else float("nan"),
                     a1[3] if a1 else float("nan"),
                     r["verdict"]))

    print("=" * 118)
    print("  矩阵结果汇总（绝对判据为主，wcos 仅辅助）")
    print("=" * 118)
    print(f"  {'run':8s}{'步':>4s}{'wcos峰':>9s}{'hi/lo末':>9s}"
          f"{'PSNR0':>8s}{'ac1_0':>8s}{'hilo_0':>8s}{'hilo_1':>8s}{'判定':>6s}  说明")
    print("  " + "-" * 114)
    for tag, desc, n, bw, lh, p0, ac0, h0, h1, v in rows:
        print(f"  {tag:8s}{n:>4d}{bw:>+9.4f}{lh:>9.2f}{p0:>8.2f}"
              f"{ac0:>+8.3f}{h0:>8.2f}{h1:>8.2f}{v:>6s}  {desc}")
    print()
    print("  基准：素材0 PSNR>13.03 ac1=+0.666 hilo=0.01 | "
          "素材1 hilo=0.04")
    print("  PASS 条件：PSNR>噪声基线 且 ac1 接近素材 且 hilo<=素材x3")
    print("  ★ hilo 是当前最硬的瓶颈（越低越好），wcos 只作参考。")


if __name__ == "__main__":
    main()
