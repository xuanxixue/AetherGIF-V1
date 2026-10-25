# -*- coding: utf-8 -*-
"""漂移场灵敏度：cos(θ−θ_tar) 对 κ 与 τ 的响应（零训练，秒级）。

cos 钉在 0.4187 不动。若加大 κ 让 cos 上升 => 强度问题；不动 => g 失效。
同时直接测 g 的量级、softmax 权重集中度（熵）、τ 的效果。
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import torch
import torch.nn.functional as F
from pathlib import Path
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from physics.drift_field import DriftField

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
device = "cuda"

gp = next(Path("data/gif_raw/noto").glob("8-ball_1f3b1.gif"))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
th_tar = torch.atan2(im, re)

model = AetherGIFModel(device=device, use_qwen=False).to(device)
model.eval()

# ---- 1. 直接测 DriftField 的 g ----
torch.manual_seed(42)
a = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
th = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi
z = torch.cat([a * torch.cos(th), a * torch.sin(th)], dim=-1)

print("=== DriftField 的 g 诊断（零训练）===")
for tau in [0.01, 0.05, 0.1, 0.5, 1.0]:
    dr = DriftField(D, tau=tau).to(device)
    with torch.inference_mode():
        g = dr(a, th, target)
    # softmax 权重熵（取 512 子集，分块算距离避免 OOM）
    xs = F.normalize(z[0, :512], dim=-1)
    ys = F.normalize(target[0, :512], dim=-1)
    sub = torch.cdist(xs, ys) ** 2  # [512,512]
    w = F.softmax(-sub / tau, dim=-1)
    ent = -(w * (w + 1e-12).log()).sum(-1).mean()
    eff = w.sum(-1).reciprocal().mean()  # 有效近邻数
    pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
    print(f"  tau={tau:5.2f}  |g| mean={g.abs().mean():.5f} norm={g.norm():.3f}  "
          f"有效近邻={eff:.1f}/512  熵={ent:.2f}")

# ---- 2. cos 对 κ_max 的响应 ----
print("\n=== cos 对 κ 的响应（零训练，固定 seed）===")
from physics.time_field import TimeField
tf = TimeField(n_blocks=12)

def run(kappa_scale):
    torch.manual_seed(1234)
    a0 = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
    th0 = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi
    with torch.inference_mode():
        text_embs = model.text_param.detach()
        ob = model.omega_bias_head(text_embs)
        omega = model._init_omega(t, s) + ob.unsqueeze(1)
        ks = torch.sigmoid(model.kappa_scale_head(text_embs)).squeeze(-1)
        Ts = torch.sigmoid(model.T_scale_head(text_embs)).squeeze(-1)
        a, th = a0, th0
        for i in range(model.n_blocks):
            c = model.pmnncore.time_field.forward(i, device)
            a, th, _ = model.pmnncore.blocks[i](
                a, th, omega, t, s,
                kappa=c["kappa"] * kappa_scale,
                T=c["T"] * Ts,
                omega_d=c["omega_d"],
                targets=target)
        pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
        return torch.cos(pe).mean().item(), a.mean().item()

for ks in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
    c, am = run(ks)
    print(f"  kappa_scale={ks:4.1f}  cos={c:+.4f}  a_out={am:.4f}")

# ---- 3. 对照：把 g 换成"直接对应" sin(θ_i^tar − θ_i) ----
print("\n=== 对照：漂移场换成直接目标吸引 sin(θ_i^tar−θ_i) ===")

class DirectDrift(torch.nn.Module):
    """诊断对照：跳过空间近邻平均，直接逐点吸引到对应 target 相位。"""
    def forward(self, a, theta, targets):
        re_t, im_t = targets[..., :D], targets[..., D:]
        th_tar = torch.atan2(im_t, re_t)
        return torch.sin(th_tar - theta)

direct = DirectDrift().to(device)

def run_direct(kappa_scale):
    torch.manual_seed(1234)
    a0 = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
    th0 = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi
    with torch.inference_mode():
        text_embs = model.text_param.detach()
        ob = model.omega_bias_head(text_embs)
        omega = model._init_omega(t, s) + ob.unsqueeze(1)
        Ts = torch.sigmoid(model.T_scale_head(text_embs)).squeeze(-1)
        a, th = a0, th0
        for i in range(model.n_blocks):
            c = model.pmnncore.time_field.forward(i, device)
            blk = model.pmnncore.blocks[i]
            g = direct(a, th, target)
            th, sd = blk.kuramoto(th, a, omega, g, c["kappa"] * kappa_scale,
                                  c["T"] * Ts, t, s)
            ffn_out = blk.ffn(a, th, c["omega_d"], t, s)
            log_res, ph_res = ffn_out.chunk(2, dim=-1)
            a = a * torch.exp(4.0 * torch.tanh(log_res / 4.0))
            th = th + ph_res.clamp(-1.5, 1.5)
        pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
        return torch.cos(pe).mean().item(), a.mean().item()

for ks in [0.0, 0.1, 0.25, 0.5, 1.0]:
    c, am = run_direct(ks)
    print(f"  kappa_scale={ks:4.2f}  cos={c:+.4f}  a_out={am:.4f}")
