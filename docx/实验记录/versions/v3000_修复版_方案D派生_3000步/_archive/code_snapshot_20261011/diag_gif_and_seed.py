# -*- coding: utf-8 -*-
"""直接分析用户实际看到的两个 GIF 文件本身（不重新推理，排除口径差异），
并检验「推理随机性」假设：同一 ckpt 连跑 5 次的输出方差。
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
G25 = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_honest.gif"
G30 = os.path.join(BASE, "outputs_real", "overfit_honest.gif")
device = "cuda"


def read_gif(p):
    im = Image.open(p)
    fr = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        fr.append(np.array(im.convert("RGB"), dtype=np.uint8))
    return np.stack(fr)


def st(p, name):
    g = p.astype(np.float32).mean(-1)
    # 逐帧统计，看是否某些帧崩了
    per = [("%3.0f/%4.1f" % (g[i].mean(), g[i].std())) for i in (0, 9, 19, 29) if i < len(g)]
    print("%-14s T=%d mean=%6.2f std=%6.2f p1=%5.1f p50=%5.1f p99=%6.1f 过曝=%.2f%% | 逐帧(mean/std) %s"
          % (name, len(p), g.mean(), g.std(), np.percentile(g, 1), np.percentile(g, 50),
             np.percentile(g, 99), (g >= 254).mean() * 100, " ".join(per)))
    return g


print("=" * 100)
print("A. 用户实际看到的 GIF（honest 口径）逐帧稳定性")
print("=" * 100)
g25 = st(read_gif(G25), "v2500-gif")
g30 = st(read_gif(G30), "v3000-gif")
raw = read_gif(GIF)
graw = st(raw, "原始GIF")

field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D)
tgt_pix = np.stack([np.asarray(f, dtype=np.float32) for f in
                    decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)]).mean(-1)
print("%-14s mean=%6.2f std=%6.2f" % ("codec顶", tgt_pix.mean(), tgt_pix.std()))


def psnr(a, b, n):
    n = min(len(a), len(b))
    mse = ((a[:n].astype(np.float32) - b[:n].astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


print("  v2500-gif vs 原始GIF PSNR=%.2f dB" % psnr(g25, graw, 30))
print("  v3000-gif vs 原始GIF PSNR=%.2f dB" % psnr(g30, graw, 30))

print()
print("=" * 100)
print("B. 推理随机性假设：同一 ckpt 连跑 5 次（ckpt 存的是训练末态，a0/θ0 每步重采样）")
print("=" * 100)
t, sc = build_spacetime_index(TS, GRID, GRID, device)
fc = field.unsqueeze(0).to(device)
tgt_a = torch.sqrt(field[:, :D] ** 2 + field[:, D:] ** 2 + 1e-12)
for tag, ck in [("v2500", r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt"),
                ("v3000", os.path.join(BASE, "outputs_real", "overfit_ckpt.pt"))]:
    m = AetherGIFModel(device=device, use_qwen=False).to(device)
    m.load_state_dict(torch.load(ck, map_location=device), strict=False)
    m.eval()
    print("%s:" % tag)
    for r in range(5):
        torch.manual_seed(r)
        with torch.inference_mode():
            a, th, _ = m(texts=["台球"], target_field=fc, t_coord=t, s_coord=sc, batch_size=1)
        a = a[0].cpu()
        z = torch.zeros(a.shape[0], 2 * D)
        z[:, :D] = a * torch.cos(th[0].cpu()); z[:, D:] = a * torch.sin(th[0].cpu())
        p = np.stack([np.asarray(f, dtype=np.float32) for f in
                      decode_field_to_frames(z, TS, GRID, PATCH, WIN, IMG, D, normalize=False)]).mean(-1)
        mse = ((p - tgt_pix) ** 2).mean()
        print("   seed=%d  PSNR=%5.2f dB  a_max=%7.2f  像素 mean=%6.2f std=%6.2f  过曝=%.2f%%"
              % (r, 10 * math.log10(255.0 ** 2 / max(mse, 1e-12)), a.max().item(),
                 p.mean(), p.std(), (p >= 254).mean() * 100))
print("=" * 100)