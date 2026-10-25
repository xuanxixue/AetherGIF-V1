# -*- coding: utf-8 -*-
"""【诊断】v3000J 频段能量失真 + PSNR 退化的根因定位。

矛盾点：
  训练日志  振幅比[模型/target] 各频段全部 < 1.0（欠幅）
  验收报告  频段能量占比 w13-18 高出 target 61 倍（11.61% vs 0.19%）
=> 均值欠幅、能量超载 => 分布极重尾：少数位置振幅巨大（尖峰/脉冲），
   被log 域 loss 的1/(1+a) 梯度压缩放���。

本脚本只读，不改任何权重。
"""
import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))
import torch
import numpy as np

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
W_FFT = PATCH * PATCH * 3 // 2 + 1
GIF_SRC = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
CKPT = os.path.join(BASE, "outputs_real", "v3000J", "overfit_ckpt.pt")

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

device = "cuda" if torch.cuda.is_available() else "cpu"
field = encode_gif_to_field(GIF_SRC, TS, GRID, PATCH, WIN, IMG, D)
re, im = field[:, :D], field[:, D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)

m = AetherGIFModel(device=device, use_qwen=False).to(device)
m.load_state_dict(torch.load(CKPT, map_location=device), strict=False)
m.eval()
t, s = build_spacetime_index(TS, GRID, GRID, device)
with torch.inference_mode():
    a, th, _ = m(texts=["台球"], target_field=field.unsqueeze(0).to(device),
                 t_coord=t, s_coord=s, batch_size=1)
a_out = a[0].cpu()
a_t = a_tar.cpu()

print("=" * 92)
print("【A】振幅分布：均值 vs 能量（大数定律检验）")
print("=" * 92)
for nm, x in [("target", a_t), ("model ", a_out)]:
    q = torch.quantile(x.flatten().float(), torch.tensor([0.5, 0.9, 0.99, 0.999, 1.0]))
    print(f"  {nm}: mean={x.mean():.5f}  median={q[0]:.5f} p90={q[1]:.5f} "
          f"p99={q[2]:.4f} p999={q[3]:.3f} max={q[4]:.2f}")
    print(f"          E[a^2]={ (x**2).mean():.5f}  std/mean={(x.std()/x.mean()):.2f}"
          f"  max/mean={(x.max()/x.mean()):.1f}")

print()
print("=" * 92)
print("【B】能量集中度：最高 K 个系数占总能量比例")
print("=" * 92)
for nm, x in [("target", a_t), ("model ", a_out)]:
    e = (x ** 2).flatten().float()
    tot = e.sum()
    srt, _ = torch.sort(e, descending=True)
    cum = torch.cumsum(srt, 0) / tot
    n = e.numel()
    print(f"  {nm}: total={n}系数")
    for frac in [1e-5, 1e-4, 1e-3, 1e-2]:
        k = max(1, int(n * frac))
        print(f"        top{frac:>7.0%} ({k:5d} 个) 占能量 {cum[k-1]*100:6.2f}%")

print()
print("=" * 92)
print("【C】逐通道能量：找出能量爆炸的具体通道")
print("=" * 92)
bands = {
    "w0(DC)": [c for c in range(D) if c % W_FFT == 0],
    "w1-6":   [c for c in range(D) if 1 <= c % W_FFT <= 6],
    "w7-12":  [c for c in range(D) if 7 <= c % W_FFT <= 12],
    "w13-18": [c for c in range(D) if 13 <= c % W_FFT <= 18],
    "w19-24": [c for c in range(D) if 19 <= c % W_FFT <= 24],
}
et = {k: (a_t[:, v] ** 2).sum().item() for k, v in bands.items()}
eo = {k: (a_out[:, v] ** 2).sum().item() for k, v in bands.items()}
tt, to = sum(et.values()), sum(eo.values())
for k in bands:
    print(f"  {k:9s} target={et[k]/tt*100:6.2f}%  model={eo[k]/to*100:6.2f}%  "
          f"倍数={eo[k]/max(et[k],1e-9):7.1f}x")

print()
print("  逐通道 model/target 能量比（>5x 标红为爆炸通道）：")
ch_e_t = (a_t ** 2).mean(0)
ch_e_o = (a_out ** 2).mean(0)
ratio = ch_e_o / ch_e_t.clamp(min=1e-12)
top = torch.argsort(ratio, descending=True)[:12]
for c in top.tolist():
    print(f"    c={c:2d} (h={c//W_FFT}, w={c%W_FFT:2d})  能量比={ratio[c]:9.1f}x"
          f"  target均幅={a_t[:,c].mean():.5f}  model均幅={a_out[:,c].mean():.4f}"
          f"  model_max={a_out[:,c].max():8.2f}")

print()
print("=" * 92)
print("【D】log1p域 loss 对尖峰的梯度惩罚能力（为何关不住）")
print("=" * 92)
print("  d/da log1p(a) = 1/(1+a)：")
for v in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 111.8]:
    print(f"    a={v:8.3f}-> 梯度={1/(1+v):.5f}  线性域梯度=1.0  比值={1/(1+v):.5f}")
print("  => 尖峰 a=111.8 处log1p 梯度只有线性域的 0.89%，几乎不受惩罚")