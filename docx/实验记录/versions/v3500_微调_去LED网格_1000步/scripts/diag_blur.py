# -*- coding: utf-8 -*-
"""诊断「为什么变模糊」：跳变的分布形状，不只是均值。

核心假设：素材是「少量大跳变(真实轮廓) + 大量极小跳变(干净背景)」，
而 v3500_t 可能是「中等跳变均匀分布」——均值相同，但轮廓被压平、噪点没去净。
均值指标区分不出这两者，必须看分布。"""
import sys
import numpy as np
from PIL import Image

IMG, PATCH = 128, 4
TARGET = r"D:/AetherGIF-V1/outputs_real/target_recon_ball.gif"
V2500 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_honest.gif"
V3500T = "outputs_real/v3500_t/overfit_honest.gif"


def load(path):
    im = Image.open(path)
    fr = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        fr.append(np.array(im.convert("L"), dtype=np.float32))
    return np.stack(fr)


def jump_distribution(g):
    """所有相邻像素差分的绝对值，展平成分布"""
    dh = np.abs(np.diff(g, axis=2)).ravel()
    dv = np.abs(np.diff(g, axis=1)).ravel()
    j = np.concatenate([dh, dv])
    qs = [50, 75, 90, 95, 99, 99.9]
    out = {f"p{q}": float(np.percentile(j, q)) for q in qs}
    out["mean"] = float(j.mean())
    out["max"] = float(j.max())
    # 尾部能量占比：大跳变(>p95)贡献了多少总能量
    thr = np.percentile(j, 95)
    out["tail_energy"] = float((j[j >= thr].sum() / j.sum()))
    # 高跳变像素占比（>20，大致是真实轮廓的量级）
    out["frac_gt20"] = float((j > 20).mean() * 100)
    out["frac_gt40"] = float((j > 40).mean() * 100)
    return out


print("=" * 100)
print("跳变分布诊断：区分「真平滑」与「假平滑(模糊)」")
print("=" * 100)
hdr = f"{'':14s}{'mean':>7s}{'p50':>7s}{'p75':>7s}{'p90':>7s}{'p95':>7s}{'p99':>7s}{'p99.9':>7s}{'max':>7s}{'尾能量%':>8s}{'>20占%':>8s}{'>40占%':>8s}"
print(hdr)
print("-" * 100)
for tag, p in (("素材(真平滑)", TARGET), ("v2500(起点)", V2500), ("v3500_t(本版)", V3500T)):
    d = jump_distribution(load(p))
    print(f"{tag:14s}{d['mean']:7.2f}{d['p50']:7.2f}{d['p75']:7.2f}{d['p90']:7.2f}"
          f"{d['p95']:7.2f}{d['p99']:7.2f}{d['p99.9']:7.2f}{d['max']:7.1f}"
          f"{d['tail_energy']*100:8.1f}{d['frac_gt20']:8.2f}{d['frac_gt40']:8.2f}")

print("""
判读方法：
- 素材若「p50 极小 + p99 大」= 干净背景 + 锐利轮廓 = 真平滑
- v3500_t 若「p50 抬高 + p99 压低」= 背景有噪 + 轮廓被压平 = 模糊
- 尾部能量%：真实轮廓应该集中较多能量；模糊版能量会被摊平""")
