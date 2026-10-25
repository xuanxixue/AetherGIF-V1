# -*- coding: utf-8 -*-
"""v2500 vs v3000 线性域诊断：为什么 v3000 视觉更差？

关键：训练用的是 log 域振幅 loss，只约束了 log1p(a) 的**均值**。
log 域均值校准 ≠ 线性域校准。必须看线性域的分位数分布。
"""
import os, sys, math
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "src"))
import numpy as np
import torch
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
V2500 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt"
V3000 = os.path.join(BASE, "outputs_real", "overfit_ckpt.pt")
device = "cuda"

field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D)
a_tar_cpu = torch.sqrt(field[:, :D] ** 2 + field[:, D:] ** 2 + 1e-12)  # [N,64] 留在CPU做统计
field = field.to(device)
a_tar = a_tar_cpu.to(device)


def pct(x):
    x = x.flatten().float()
    return "mean=%7.4f  p1=%8.5f  p10=%8.5f  p50=%8.5f  p90=%8.5f  p99=%8.4f  max=%8.3f" % (
        x.mean(), *[x.quantile(q) for q in (0.01, 0.10, 0.50, 0.90, 0.99)], x.max())


print("=" * 92)
print("线性域振幅分布对比（log 域 loss 只管均值，这里看全分布）")
print("=" * 92)
print("target   %s" % pct(a_tar_cpu))

fwd = {}
for tag, ck in [("v2500", V2500), ("v3000", V3000)]:
    if not os.path.exists(ck):
        print(tag, "缺失"); continue
    m = AetherGIFModel(device=device, use_qwen=False).to(device)
    sd = torch.load(ck, map_location=device)
    msd = m.state_dict()
    # 兼容：v2500 有 12 个独立 a_mu，v3000 共享 1 个
    miss = [k for k in msd if k not in sd]
    extra = [k for k in sd if k not in msd]
    m.load_state_dict(sd, strict=False)
    m.eval()
    t, s = build_spacetime_index(TS, GRID, GRID, device)
    with torch.inference_mode():
        a, th, _ = m(texts=["台球"], target_field=field.unsqueeze(0), t_coord=t, s_coord=s, batch_size=1)
    a = a[0]
    fwd[tag] = a
    print("%-8s %s" % (tag, pct(a)))
    print("         log域: mean(log1p a)=%.4f   target=%.4f   比值=%.3f"
          % (torch.log1p(a).mean(), torch.log1p(a_tar).mean(),
             (torch.log1p(a).mean() / torch.log1p(a_tar).mean()).item()))
    print("         线性域均值比 a_out/a_tar = %.3f    std比 = %.3f"
          % ((a.mean() / a_tar.mean()).item(), (a.std() / a_tar.std()).item()))
    # log 域 ratio 的"假校准"证据：log1p 域 ratio
    print("         log1p域逐通道均值比 = %.3f  (训练 loss 就是盯这个)"
          % (torch.log1p(a).mean(0) / torch.log1p(a_tar).mean(0)).mean().item())
    print("         线性域逐通道均值比 = %s"
          % " ".join("%.2f" % v for v in (a.mean(0) / a_tar.mean(0))[::8].tolist()))

print("=" * 92)
print("解码后像素域统计（固定 clamp 口径 = overfit_honest.gif）")
print("=" * 92)


def px_stats(fr):
    px = np.stack([np.asarray(f, dtype=np.float32) for f in fr])  # [T,H,W,C]
    return ("像素 mean=%6.2f std=%6.2f | p1=%6.1f p50=%6.1f p99=%6.1f max=%6.1f | 过曝(>=254)=%.2f%% 欠曝(<=1)=%.2f%%"
            % (px.mean(), px.std(), np.percentile(px, 1), np.percentile(px, 50),
               np.percentile(px, 99), px.max(), (px >= 254).mean() * 100, (px <= 1).mean() * 100))


for tag, a in fwd.items():
    a_cpu = a.cpu()
    # 溢出诊断：超过 target 全局 max 的系数有多少
    thr = a_tar_cpu.max().item()
    over = (a_cpu > thr).float()
    print("%-8s %s" % (tag, px_stats(decode_field_to_frames(
        torch.cat([a_cpu, torch.zeros_like(a_cpu)], 1), TS, GRID, PATCH, WIN, IMG, D, normalize=False))))
    print("         超target最大振幅(%.1f)的系数: %.4f%% 个数, 最大 %.1f (=%.1fx)"
          % (thr, over.mean().item() * 100, a_cpu.max().item(), a_cpu.max().item() / thr))
    # 溢出集中在哪些通道
    oc = (a_cpu > thr).sum(0)
    hot = [(int(c), int(oc[c])) for c in range(D) if oc[c] > 0]
    hot.sort(key=lambda z: -z[1])
    print("         溢出通道(top6, c=位置:个数): %s" % (hot[:6] if hot else "无"))
    # 高频溢出 = 空间域伪影(Gibbs)来源
    hf = [c for c, _ in hot if c % 25 >= 13]
    print("         其中高频(w>=13)占 %d/%d 通道 → Gibbs 伪影风险" % (len(hf), len(hot)))

pxt = decode_field_to_frames(field.cpu(), TS, GRID, PATCH, WIN, IMG, D, normalize=False)
print("%-8s %s" % ("codec顶", px_stats(pxt)))
print("=" * 92)