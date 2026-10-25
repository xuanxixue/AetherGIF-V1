# -*- coding: utf-8 -*-
"""最后一层诊断：分层相位误差 + 空间域锐度。

已确认的事实（diag_ablate / diag_linear）：
  v2500 振幅谱「扁平」p50=0.16（target 0.0015，重尾被抹平）→ 对比度不足 std 27.6，但能量分散→局部对比度保留→看着有细节+LED网格噪声
  v3000 振幅谱「重尾完美复现」p50=0.0018 / p99=5.13 → 但 std 只有 24.9，糊
  → 假设：v3000 把能量集中在少数大振幅系数上，而这些大系数的**相位错**，
         能量被摊到错误位置 → 结构模糊。
  wcos=0.91 骗人：它是功率加权平均，被少数大系数独占，不能反映分层精度。
本脚本按振幅分位数分层测相位误差，验证该假设。
"""
import os, sys, math
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "src"))
import numpy as np
from PIL import Image
import torch
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
device = "cuda"

field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D)
tgt_a = torch.sqrt(field[:, :D] ** 2 + field[:, D:] ** 2 + 1e-12)
tgt_th = torch.atan2(field[:, D:], field[:, :D])
t, sc = build_spacetime_index(TS, GRID, GRID, device)
fc = field.unsqueeze(0).to(device)
tgt_pix = np.stack([np.asarray(f, dtype=np.float32) for f in
                    decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)]).mean(-1)


def run(ck, seed=0):
    m = AetherGIFModel(device=device, use_qwen=False).to(device)
    m.load_state_dict(torch.load(ck, map_location=device), strict=False)
    m.eval()
    torch.manual_seed(seed)
    with torch.inference_mode():
        a, th, _ = m(texts=["台球"], target_field=fc, t_coord=t, s_coord=sc, batch_size=1)
    return a[0].cpu(), th[0].cpu()


def dec(a, th):
    z = torch.zeros(a.shape[0], 2 * D)
    z[:, :D] = a * torch.cos(th); z[:, D:] = a * torch.sin(th)
    return np.stack([np.asarray(f, dtype=np.float32) for f in
                     decode_field_to_frames(z, TS, GRID, PATCH, WIN, IMG, D, normalize=False)]).mean(-1)


def psnr(p):
    mse = ((p - tgt_pix) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


def sharp(p):
    """空间域锐度：Sobel 梯度均值 + 高频能量占比（拉普拉斯方差）
    p 可能是 [T,H,W,C] 或 [T,H,W]（gray）"""
    p = np.asarray(p, dtype=np.float32)
    if p.ndim == 4:
        p = p.mean(-1)
    H, W = p.shape[1], p.shape[2]
    p = p[:, :H - 1, :W - 1]              # 统一裁剪，H==W 时 diff 长度才一致
    gx = np.diff(p, axis=2)               # [T,H-1,W-2]
    gy = np.diff(p, axis=1)               # [T,H-2,W-1]
    mh = min(gx.shape[1], gy.shape[1]); mw = min(gx.shape[2], gy.shape[2])
    sob = np.sqrt(gx[:, :mh, :mw] ** 2 + gy[:, :mh, :mw] ** 2).mean()
    lap = (4 * p[:, 1:-1, 1:-1] - p[:, :-2, 1:-1] - p[:, 2:, 1:-1]
           - p[:, 1:-1, :-2] - p[:, 1:-1, 2:])
    return sob, lap.std()


print("=" * 104)
print("1. 分层相位误差（按 target 振幅分位数分层）—— 验证「能量集中但大系数相位错」假设")
print("=" * 104)
print("%-8s %10s %10s %10s %10s %12s" % ("模型", "顶层1%°", "1-10%°", "10-50%°", "50-99%°", "功率加权wcos"))
qs = [0.99, 0.90, 0.50, 0.01]
thr = [tgt_a.quantile(q) for q in qs]
print("  target 振幅分层阈值: " + "  ".join("p%d=%.4f" % (int(q * 100), v) for q, v in zip(qs, thr)))

for tag, ck in [("v2500", r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt"),
                ("v3000", os.path.join(BASE, "outputs_real", "overfit_ckpt.pt"))]:
    a, th = run(ck, seed=0)
    d = torch.atan2(torch.sin(th - tgt_th), torch.cos(th - tgt_th)).abs() * 57.3
    top1 = d >= thr[0]
    b1 = (d >= thr[1]) & (d < thr[0])
    b2 = (d >= thr[2]) & (d < thr[1])
    b3 = d < thr[2]
    wc = (torch.cos(th - tgt_th) * tgt_a ** 2).sum() / (tgt_a ** 2).sum()
    # 每个分层的能量占比（决定该层误差对图像的影响权重）
    e1 = (tgt_a ** 2)[top1].sum() / (tgt_a ** 2).sum()
    e2 = (tgt_a ** 2)[b1].sum() / (tgt_a ** 2).sum()
    print("%-8s %10.1f %10.1f %10.1f %10.1f %12.4f   | 能量占比: 顶1%%=%.1f%% 1-10%%=%.1f%%"
          % (tag, d[top1].mean(), d[b1].mean(), d[b2].mean(), d[b3].mean(), wc, e1 * 100, e2 * 100))

print()
print("=" * 104)
print("2. 空间域锐度与 PSNR（seed 固定 0）")
print("=" * 104)
for tag, ck in [("v2500", r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt"),
                ("v3000", os.path.join(BASE, "outputs_real", "overfit_ckpt.pt"))]:
    a, th = run(ck, seed=0)
    p = dec(a, th)
    s, l = sharp(p)
    print("%-8s PSNR=%5.2f dB  Sobel锐度=%6.2f  拉普拉斯std=%6.2f  像素std=%5.2f"
          % (tag, psnr(p), s, l, p.std()))
s, l = sharp(tgt_pix)
print("%-8s PSNR=  inf  Sobel锐度=%6.2f  拉普拉斯std=%6.2f  像素std=%5.2f" % ("codec顶", s, l, tgt_pix.std()))

print()
print("=" * 104)
print("3. 推理随机性：同一 ckpt 换 seed 的输出变化（a0/θ0 每步重采样）")
print("=" * 104)
for tag, ck in [("v2500", r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt"),
                ("v3000", os.path.join(BASE, "outputs_real", "overfit_ckpt.pt"))]:
    out = []
    for sd in range(4):
        a, th = run(ck, seed=sd)
        p = dec(a, th)
        s, _ = sharp(p)
        out.append((psnr(p), s, p.std(), a.max().item()))
    print("%s: PSNR %s | 锐度 %s | 像素std %s | a_max %s"
          % (tag,
             " ".join("%.1f" % o[0] for o in out),
             " ".join("%.1f" % o[1] for o in out),
             " ".join("%.1f" % o[2] for o in out),
             " ".join("%.0f" % o[3] for o in out)))
print("=" * 104)