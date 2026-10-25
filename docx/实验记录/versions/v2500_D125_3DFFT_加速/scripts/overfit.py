# -*- coding: utf-8 -*-
"""单样本过拟合训练 —— 方案 D（2500 步，历史最优基线，参数量指纹 1,135,658）。

【本文件是 2500 步那次的完整代码，已从方案 H 脚手架中剥离还原，供回退/复现。】

loss 口径（方案 D，AMP_P=0.5）：
  振幅: loss_amp = weighted_mean((log1p(a) - log1p(a_tar))^2,
                                  w = (a_tar / a_max)^0.5)
  相位: loss_ph  = sum(w_ph * pe^2) / sum(w_ph),  w_ph = a_tar^2

架构（配合本目录 block.py：单可训练 gamma + 每 block 独立 per-channel a_mu，log 域）：
  log_a_new = log_a + res - gamma * (log_a - a_mu)

历史结果（overfit_out21.txt）：2500 步 PSNR 22.46 dB（clamp 口径）、
  profile_cos 0.9899、wcos 0.8707、frame-diff 10.26。用户评价 85% 还原。

运行：cd 到本快照根目录后  python scripts/overfit.py 2500
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
GIF = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
AMP_P = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
OUT = "outputs_real/overfit.gif"
device = "cuda"

gp = next(Path("data/gif_raw/noto").glob(GIF))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)                      # [1,N,2D]
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
th_tar = torch.atan2(im, re)
print(f"[DATA] {gp.name}  N={target.shape[1]}  "
      f"a_tar mean={a_tar.mean().item():.5f} std={a_tar.std().item():.5f}  "
      f"std/mean={a_tar.std().item()/max(a_tar.mean().item(),1e-9):.3f}")

model = AetherGIFModel(device=device, use_qwen=False, d_model=D, ffn_hidden=512).to(device)
# 方案 D：统一优化器，无分组 lr、无 warmup（2500 步当时即如此）
opt = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

# 固定初始条件 seed：只记忆一条轨迹，先验证通路记忆能力
# （每步重设 seed 会逼模型学"任意起点→target"，12 块深度不够，不可行）
SEED_MODE = sys.argv[5] if len(sys.argv) > 5 else "fixed"
if SEED_MODE == "fixed":
    torch.manual_seed(1234)

for step in range(1, STEPS + 1):
    if SEED_MODE == "step":
        torch.manual_seed(step)
    opt.zero_grad()
    # 混合精度训练 (AMP) - 显著加速，保持质量
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
    pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
    # 振幅 loss：log1p 压重尾 + (a_max 相对)^0.5 软化权重（方案 D 最终口径）
    la = torch.log1p(a.clamp(min=1e-8))
    lt = torch.log1p(a_tar.clamp(min=1e-8))
    w = (a_tar / a_tar.max().clamp(min=1e-8)).pow(AMP_P)
    loss_amp = (w * (la - lt).pow(2)).mean()
    # 相位按 FFT 功率 a_tar^2 加权：69.4% 系数幅度<0.01，其相位是 atan2 数值噪声
    w_ph = (a_tar ** 2).clamp(min=1e-8)
    loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
    loss = loss_amp + loss_ph
    loss.backward()
    opt.step()
    if step % 25 == 0 or step == 1:
        with torch.no_grad():
            c = torch.cos(pe)
            cos = c.mean().item()
            wcos = (w_ph * c).sum().item() / w_ph.sum().item()
            m = a_tar > 1.0
            cos_eff = c[m].mean().item() if bool(m.any()) else float("nan")
            am = a.mean().item()
        ch = c[0].mean(dim=0)                       # [D] 每通道 cos
        ar = (a[0].mean(dim=0) / a_tar[0].mean(dim=0).clamp(min=1e-6)).clamp(0, 20)  # [D] 振幅比
        # 频段四等分（诊断性打印，不影响训练；D=125，每段约 31 通道）
        bands = [("低频", slice(0, 31)), ("中低", slice(31, 62)),
                 ("中高", slice(62, 93)), ("高频", slice(93, 125))]
        bd = " ".join(f"{n}:{ch[s].mean():+.2f}/{ar[s].mean():.2f}" for n, s in bands)
        # 谱形状指标：模型逐通道 log1p 振幅剖面与 target 的相关（谱被压平≈0）
        am_ch = a[0].mean(dim=0)                    # [D]
        at_ch = a_tar[0].mean(dim=0)                # [D]
        lm, lt_ = torch.log1p(am_ch), torch.log1p(at_ch)
        lm_c, lt_c = lm - lm.mean(), lt_ - lt_.mean()
        prof_cos = (lm_c * lt_c).sum() / (lm_c.norm() * lt_c.norm() + 1e-12)
        prof_ratio_std = torch.log(ar.clamp(min=1e-3)).std()   # 0=逐通道全校准
        print(f"[{step:4d}] loss={loss.item():.4f} amp={loss_amp.item():.4f} "
              f"phase={loss_ph.item():.4f} cos={cos:+.4f} wcos={wcos:+.4f} "
              f"cos_eff={cos_eff:+.4f} a_out={am:.4f}", flush=True)
        print(f"        频段[cos/振幅比] {bd}", flush=True)
        print(f"        谱形状 profile_cos={prof_cos:+.4f} ratio_logstd={prof_ratio_std:.3f} "
              f"(target 逐通道跨 {at_ch.max()/at_ch.min():.0f}x, 模型 {am_ch.max()/am_ch.min():.0f}x)", flush=True)

# 生成 GIF（不喂 target，走推理路径）
model.eval()
with torch.inference_mode():
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
# 主出图 = normalize=True（min-max，可视化口径）；副出图 = normalize=False（固定 clamp）
frames = np.stack(decode_field_to_frames(
    fld, TS, GRID, PATCH, WIN, IMG, D, normalize=True)).astype(np.uint8)
pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
pil[0].save(OUT, save_all=True, append_images=pil[1:], loop=0, duration=100)
frames_h_raw = np.stack(decode_field_to_frames(
    fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
frames_h = [f.astype(np.uint8) for f in frames_h_raw]
pil_h = [Image.fromarray(np.ascontiguousarray(f)) for f in frames_h]
OUT_H = OUT.replace(".gif", "_honest.gif")
pil_h[0].save(OUT_H, save_all=True, append_images=pil_h[1:], loop=0, duration=100)
print(f"[GEN] {OUT}  frames={len(pil)}")
print(f"[GEN] {OUT_H}  (固定 clamp 诚实重建)")

# 存瘦身 checkpoint（只存 PMNN 物理核心 + 文本头，不含冻结 Qwen）
sd = {k: v for k, v in model.state_dict().items()
      if not k.startswith("qwen.")}
torch.save(sd, "outputs_real/overfit_ckpt.pt")
print(f"[CKPT] outputs_real/overfit_ckpt.pt  ({sum(v.numel() for v in sd.values()):,} params)")

# 定量对比：clamp 口径，与 overfit_out21.txt 一致
with torch.inference_mode():
    zt = (a_tar[0] * torch.exp(1j * th_tar[0])).to(torch.complex64)
    ft = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()
arr_t_h = np.stack(decode_field_to_frames(
    ft, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
arr_m_h = np.stack(frames_h).astype(np.float32)
psnr_h = 10 * math.log10(255.0 ** 2 / max(((arr_m_h - arr_t_h) ** 2).mean(), 1e-12))
fd_h = float(np.abs(np.diff(arr_m_h, axis=0)).mean())
fd_t_h = float(np.abs(np.diff(arr_t_h, axis=0)).mean())
print(f"[CMP] vs target-recon PSNR={psnr_h:.2f} dB  frame-diff model={fd_h:.2f} target={fd_t_h:.2f}")