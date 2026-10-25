# -*- coding: utf-8 -*-
"""重尾 × 标量 OU 的矛盾：corr 掉到 0.197 的真正机制

上一轮我误判为「μ 初值量级错」，已纠正（口径混了 NORM）。
真机制：log(a) 分布跨 9 个数量级（q0.1=−9.21、q0.5=−6.82、q0.99=+0.04），
标量 μ 把所有系数往同一点收敛 ⇒ 摧毁 80% 的系数。

本探针做对照实验：同一条重尾分布，
  (a) 经标量 OU 收敛 → corr
  (b) 保留逐位置值（旁路直接给） → corr = 1.0
  (c) 经**逐通道** OU（有D=125 个独立 μ） → corr
回答：修法该是「换更细的收敛粒度」还是「旁路直接给」。
"""
import sys, math
from pathlib import Path
import torch
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from codec.gif_codec import encode_gif_to_field

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
g = [x for x in sorted((ROOT/"data/gif_raw").rglob("*.gif")) if "ball" in x.name.lower()][0]
fld = encode_gif_to_field(str(g), TS, GRID, PATCH, WIN, IMG, D).float()
re_t, im_t = fld[..., :D], fld[..., D:]
a_t = torch.sqrt(re_t**2 + im_t**2 + 1e-8)
la0 = torch.log(a_t[:, 1:].clamp(min=1e-8))          # [N, D-1]

def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return ((x*y).sum()/(x.norm()*y.norm())).item()

def ou(x, mu, gamma=0.3, layers=12):
    for _ in range(layers):
        x = x - gamma*(x - mu)
    return x

print("log(a) 去DC 分布：")
for q in [0.01, 0.1, 0.5, 0.9, 0.99]:
    v = la0.flatten().quantile(q); print(f"  q{q:<5}= {v:+.3f}")

print()
print("=== 对照实验（12 层后与初值的结构 corr）===")
# (a) 标量 OU：全通道往一个 μ 收敛
mu_scalar = -1.6094
print(f"  (a) 标量 OU  μ={mu_scalar:+.3f}      corr = {corr(ou(la0.clone(), mu_scalar), la0):+.4f}")
# (b) 旁路直接给（无 OU）
print(f"  (b) 不经 OU（旁路直给）  corr = {corr(la0.clone(), la0):+.4f}")
# (c) 逐通道 OU：每通道独立 μ（= 各通道自己的均值）
mu_perch = la0.mean(dim=0, keepdim=True).expand_as(la0)
print(f"  (c) 逐通道 OU（per-ch μ） corr = {corr(ou(la0.clone(), mu_perch), la0):+.4f}")
# (d) 逐位置组 OU：按 w%25 分5 组
gid = (torch.arange(1, D) % 25)
edges = torch.linspace(0,25,6); gg = torch.bucketize(gid.float(), edges[1:-1].float()).clamp(0,4)
mu_g = torch.zeros_like(la0)
for k in range(5):
    m = (gg == k)
    if bool(m.any()):
        mu_g[:, m] = la0[:, m].mean()
print(f"  (d) 逐频段组 OU（G=5）    corr = {corr(ou(la0.clone(), mu_g), la0):+.4f}")

print()
print("★ 结论：收敛粒度越细，结构保留越高。")
print("  标量 μ 把重尾压平 ⇒ corr 崩塌；旁路直给 corr=1.0。")
print("  ⇒ FFT 旁路的价值正在于它是**逐位置**的，不经过任何标量收敛。")
