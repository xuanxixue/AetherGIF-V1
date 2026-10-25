# -*- coding: utf-8 -*-
"""【短验证前置】在 v3000J 训练后 ckpt 上测量三项 loss 的取值与梯度量级，
用于选择 EN_W / CST_W 权重。只读，不改任何权重。"""
import os, sys, math, importlib.util

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))
import torch

# 以模块方式载入 overfit.py 的定义部分（不跑训练循环）
src = open(os.path.join(BASE, "scripts", "overfit.py"), encoding="utf-8").read()
head = src.split("for step in range")[0]
head = head.replace('os.path.dirname(os.path.dirname(os.path.abspath(__file__)))', repr(BASE))
ns = {"__file__": os.path.join(BASE, "scripts", "overfit.py")}
sys.argv = ["overfit.py", "1"]
exec(compile(head, "overfit_head", "exec"), ns)

device = ns["device"]
model, a_tar, th_tar, target, t, s = (ns[k] for k in
                                      ["model", "a_tar", "th_tar", "target", "t", "s"])
amplitude_loss = ns["amplitude_loss"]
contrast_loss = ns["contrast_loss"]
energy_loss = ns["energy_loss"]
space_gray = ns["space_gray"]

print()
print("=" * 92)
print("三项 loss 在【v3000J 训练后 ckpt】上的取值与梯度量级")
print("=" * 92)
raw = torch.load(os.path.join(BASE, "outputs_real", "v3000J", "overfit_ckpt.pt"),
                 map_location=device)
model.load_state_dict(raw, strict=False)
model.eval()
with torch.no_grad():
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)

w_ph = (a_tar ** 2).clamp(min=1e-8)
pe = torch.remainder(th[0] - th_tar + math.pi, 2 * math.pi) - math.pi
print("  amp      = %.4f" % amplitude_loss(a[0], a_tar).item())
print("  phase    = %.4f" % ((w_ph * pe.pow(2)).sum() / w_ph.sum()).item())
print("  contrast = %.4f" % contrast_loss(a[0], th[0]).item())
print("  energy   = %.4f" % energy_loss(a[0], a_tar).item())

print()
print("=== 梯度量级对比（dL/da）——决定权重比例 ===")
a0 = a[0]
spike_mask = a0 > 100
print("  a>100 的尖峰系数个数 = %d / %d" % (int(spike_mask.sum()), a0.numel()))
for nm in ["amp", "energy", "contrast"]:
    aa = a0.detach().clone().requires_grad_(True)
    if nm == "amp":
        l = amplitude_loss(aa, a_tar)
    elif nm == "energy":
        l = energy_loss(aa, a_tar)
    else:
        l = contrast_loss(aa, th[0])
    g = torch.autograd.grad(l, aa)[0]
    gs = g[spike_mask].abs().mean().item() if bool(spike_mask.any()) else 0.0
    print("  %-9s loss=%8.4f  |grad|=%.3e  尖峰处平均|grad|=%.3e"
          % (nm, l.item(), g.norm().item(), gs))

print()
print("=== 关键验证：尖峰梯度压制比 ===")
print("  amp(log1p) 与 energy(log比值) 在尖峰处的梯度之比 =")
aa = a0.detach().clone().requires_grad_(True)
g_amp = torch.autograd.grad(amplitude_loss(aa, a_tar), aa)[0]
aa2 = a0.detach().clone().requires_grad_(True)
g_en = torch.autograd.grad(energy_loss(aa2, a_tar), aa2)[0]
if bool(spike_mask.any()):
    r = (g_en[spike_mask].abs().mean() / g_amp[spike_mask].abs().mean().clamp(min=1e-30)).item()
    print("    energy/amp = %.1f 倍  => energy 项对尖峰的压制力显著更强" % r)
print()
print("=== 逐帧 contrast loss 口径自检（修复后）===")
x = space_gray(a0, th[0])
s_dim = 4 * 4 * 3
TS, GRID, WIN = ns["TS"], ns["GRID"], ns["WIN"]
xx = x.reshape(TS * GRID * GRID, WIN, s_dim).mean(-1)
xx = xx.reshape(TS, GRID * GRID, WIN).permute(0, 2, 1).reshape(TS * WIN, GRID * GRID)
per = (xx - xx.mean(1, keepdim=True)).std(1)
tgt = ns["TGT_STD_PER_FRAME"].to(device)
print("  模型逐帧 std: " + " ".join("%.0f" % v for v in per[:10].tolist()) + " ...")
print("  target 逐帧:  " + " ".join("%.0f" % v for v in tgt[:10].tolist()) + " ...")
print("  比值(model/target) = " + " ".join("%.2f" % v for v in (per / tgt)[:10].tolist()) + " ...")
print("  min比值 = %.2f  max比值 = %.2f" % ((per / tgt).min(), (per / tgt).max()))