# -*- coding: utf-8 -*-
"""振幅层级归因诊断（2500 步 checkpoint）。

问题：输出振幅全局偏高 2.3~3.0x，消融显示"只修振幅"就能把 frame-diff
从 10.26 拉到 5.17（target 5.56），即动画抖动 100% 来自振幅偏差。

假设：OU 稳态 log a* = mu_c + res_c / gamma。若 FFN 残差 res 对每个通道
都有一个系统性正偏 r̄_c，它会被 1/gamma（gamma 自学到 0.211 => 4.7x）
放大进最终层级。本脚本逐块验证：

  1. 逐块 a_gamma / a_mu 的实际取值
  2. 逐块 FFN 残差 res（tanh 后）在 N 维上的均值 r̄_i —— 是否系统性正
  3. 层级递推 m_{i+1} = (1-g_i) m_i + g_i mu_i + r̄_i 的各项贡献
  4. 预测层级 vs 实际 log a_out，定位偏差归谁
  5. a_mu 剖面与 target 对数谱的相关性（OU 到底背了多少谱形状）
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import torch
import numpy as np
from pathlib import Path
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = sys.argv[1] if len(sys.argv) > 1 else "8-ball_1f3b1.gif"
CKPT = sys.argv[2] if len(sys.argv) > 2 else "outputs_real/overfit_ckpt.pt"
device = "cuda"

gp = next(Path("data/gif_raw/noto").glob(GIF))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
th_tar = torch.atan2(im, re)

model = AetherGIFModel(device=device, use_qwen=False).to(device)
sd = torch.load(CKPT, map_location=device)
model.load_state_dict(sd)
model.eval()

# ---- 钩子：抓每块 FFN 原始输出 ----
caps = {}
def mk_hook(i):
    def hook(mod, inp, out):
        caps[i] = out.detach()
    return hook
for i, blk in enumerate(model.pmnncore.blocks):
    blk.ffn.register_forward_hook(mk_hook(i))

with torch.inference_mode():
    a_out, th_out, _ = model(texts=["台球"], target_field=target,
                             t_coord=t, s_coord=s, batch_size=1)

B, N, _ = a_out.shape
print(f"== 层级归因诊断 ==  N={N}  a_tar mean={a_tar.mean().item():.5f} "
      f"a_out mean={a_out.mean().item():.5f} "
      f"(比值 {a_out.mean().item()/a_tar.mean().item():.2f}x)")

# 逐通道对数谱
lt = torch.log(a_tar[0].mean(dim=0).clamp(min=1e-8))     # [D]
lo = torch.log(a_out[0].mean(dim=0).clamp(min=1e-8))     # [D]
print(f"[通道] target log 谱 span={lt.max()-lt.min():.3f}  "
      f"model log 谱 span={lo.max()-lo.min():.3f}")
print(f"[通道] log 均值差 mean(lo-lt)={(lo-lt).mean().item():+.4f}  "
      f"逐通道 log 比 std={(lo-lt).std().item():.4f}")

# ---- 逐块参数 + 残差均值 ----
print("\n-- 逐块 a_gamma / a_mu / FFN 残差均值 --")
g_list, mu_list, rb_list = [], [], []
for i, blk in enumerate(model.pmnncore.blocks):
    g = blk.a_gamma.detach()
    mu = blk.a_mu.detach()                                # [D]
    ffn_out = caps[i]                                     # [B,N,2D]
    log_res = ffn_out[..., :D]
    res = 2.0 * torch.tanh(log_res / 2.0)                 # [B,N,D]
    rb = res.mean(dim=(0, 1))                             # [D] N 维均值
    g_list.append(g); mu_list.append(mu); rb_list.append(rb)
    print(f"  blk{i:2d} gamma={g.item():.4f}  mu:mean={mu.mean().item():+.4f} "
          f"span={mu.max()-mu.min():.3f}  rbar:mean={rb.mean().item():+.4f} "
          f"std={rb.std().item():.4f}  rbar/gamma={(rb.mean()/g).item():+.4f}")

# ---- 层级递推验证 ----
# m_{i+1} = (1-g_i) m_i + g_i mu_i + rbar_i   （逐通道 [D]）
a0_log = None  # 用与模型一致的初值统计
# 重算 a0 的 log 均值（与 aether_gif_model.forward 同公式，eval 下 RNG 固定不了，
# 直接取实测：用第一次前向的输入近似不可能，这里改为"从 m_0 反推不可得"，
# 用递推残差方式：从最终 m_12 反解无意义；改为正向递推，m_0 用经验值）
print("\n-- 层级递推（逐通道 log 均值）--")
m = torch.full((D,), -2.18, device=device)   # log a0 均值经验值 exp(-2.5+0.32)
print(f"  m_0 (log a0 均值, 经验) = {m.mean().item():+.4f}")
for i in range(12):
    m = (1 - g_list[i]) * m + g_list[i] * mu_list[i] + rb_list[i]
    if i in (0, 3, 7, 11):
        print(f"  m_{i+1:2d} = {m.mean().item():+.4f}")
print(f"  递推终态 m_12 mean = {m.mean().item():+.4f}")
print(f"  实际 log a_out 均值 = {lo.mean().item():+.4f}")
print(f"  target log 均值      = {lt.mean().item():+.4f}")

# 各项对终态层级的贡献（把 12 块求和）
contrib_mu = sum((g_list[i] * (1 - sum(g_list[j] for j in range(i))) * 0
                  for i in range(0)))  # placeholder, 下面用展开式
# 展开：m_12 = prod(1-g) m_0 + sum_i [ prod_{j>i}(1-g_j) (g_i mu_i + rbar_i) ]
prod_all = torch.ones(1, device=device)
for i in range(12):
    prod_all = prod_all * (1 - g_list[i])
term_m0 = prod_all * (-2.18)
t_mu = torch.zeros(D, device=device)
t_rb = torch.zeros(D, device=device)
for i in range(12):
    w = torch.ones(1, device=device)
    for j in range(i + 1, 12):
        w = w * (1 - g_list[j])
    t_mu = t_mu + (w * g_list[i] * mu_list[i]).squeeze()
    t_rb = t_rb + (w * rb_list[i]).squeeze()
print(f"\n-- 终态层级贡献分解 --")
print(f"  初值 m_0 残留 : {term_m0.mean().item():+.4f}")
print(f"  OU 锚 mu 合计 : {t_mu.mean().item():+.4f}")
print(f"  FFN 残差 rbar : {t_rb.mean().item():+.4f}   <-- 若 >0 且大，即为放大源")
print(f"  三者之和      : {(term_m0+t_mu+t_rb).mean().item():+.4f}  vs 实际 {lo.mean().item():+.4f}")
print(f"  target        : {lt.mean().item():+.4f}")
print(f"  残差项/target 的层级超出 = {(t_mu+t_rb+term_m0-lt).mean().item():+.4f} "
      f"(= {math.exp((t_mu+t_rb+term_m0-lt).mean().item()):.2f}x)")

# ---- a_mu 剖面 vs target 对数谱 ----
mu_all = torch.stack(mu_list)                              # [12,D]
mu_last = mu_list[-1]
lt_c = lt - lt.mean()
for name, mu in (("a_mu[末块]", mu_last), ("a_mu 均值", mu_all.mean(0))):
    mc = mu - mu.mean()
    cos = (mc * lt_c).sum() / (mc.norm() * lt_c.norm() + 1e-12)
    print(f"[谱形状] {name} vs target log 谱 cos = {cos.item():+.4f}  "
          f"span mu={mu.max()-mu.min():.3f} vs target={lt.max()-lt.min():.3f}")

# ---- 分频段振幅比 ----
ar = (a_out[0].mean(dim=0) / a_tar[0].mean(dim=0).clamp(min=1e-6)).clamp(0, 20)
bands = [(0, 16, "低频"), (16, 32, "中低"), (32, 48, "中高"), (48, 64, "高频")]
print("\n-- 分频段振幅比（模型/target，1.0 = 校准）--")
print("  " + "  ".join(f"{n}:{ar[l:h].mean():.2f}" for l, h, n in bands))

# ---- 相位：大/小系数分开 ----
pe = torch.remainder(th_out - th_tar + math.pi, 2 * math.pi) - math.pi
c = torch.cos(pe)
mag = a_tar[0].mean(dim=0)                                  # [D] 通道平均幅度
big = mag > 0.1
small = (mag > 0.01) & (mag <= 0.1)
pow_tot = (a_tar ** 2).sum()
for name, msk in (("大系数(>0.1)", big), ("小系数(0.01-0.1)", small)):
    if msk.any():
        cc = c[0][:, msk].mean()
        pw = (a_tar[0][:, msk] ** 2).sum() / pow_tot
        print(f"[相位] {name}: ch数={int(msk.sum()):2d} cos={cc.item():+.4f} "
              f"功率占比={pw.item()*100:.3f}%")
