# -*- coding: utf-8 -*-
"""单样本过拟合训练：验证模型能否真正重建一个真实 GIF。

全程监控 amp / phase 分解 + cos(θ−θ_tar)，最后生成 GIF。
振幅量级失配（模型 a≈0.75 vs target a≈0.008）是当前 loss 主导项，
本脚本同时检验它能否被压下来。
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'src')
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
GIF = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
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
model = AetherGIFModel(device=device, use_qwen=False).to(device)
# 分组学习率：谱包络 a_mu 是 64 维低维凸参数（log 域最小二乘），动态范围 6.6 个
# log 单位，梯度近线性 (μ−lt)，统一 lr=3e-4 时几万步才铺得开谱形状（实测 2500 步
# span 只到 0.1）。给 a_mu 单独放大 lr。其余参数（FFN 形状、漂移场、相位头）保持
# 原 lr，避免放大后震动破坏已收敛的 frame-diff（2.71 已低于 target 5.56）。
# 【第十三死穴修复后的倍率】架构搬到 log1p 域后，μ 的目标跨度从无界的 6.6 个
# log 单位降到**有界的 2.17 个 log1p 单位**（ch0=2.179 → ch55=0.032）。
# ×80是为 log 域 6.6 跨度设计的，在 2.17 跨度上等于过冲 20 倍（实测 μ 从 2.18
# 冲到 8.99、震荡 4~16）。×5 时爬完 2.17 需约 1450 步，在 3000 步预算内从容收敛。
MU_LR_MULT = float(sys.argv[6]) if len(sys.argv) > 6 else 5.0
# μ warmup 步数：起步阶段防止大跨度通道（ch0 的 log1p 均值 2.09 vs ch55 的 0.029）
# 在 ×80 lr 下冲过头。取总步数的 5%，最少 50 步。
WARMUP = max(50, int(STEPS * 0.05))
mu_params = [p for n, p in model.named_parameters() if ".a_mu" in n and p.requires_grad]
other = [p for n, p in model.named_parameters() if ".a_mu" not in n and p.requires_grad]
opt = torch.optim.AdamW([
    {"params": mu_params, "lr": LR * MU_LR_MULT},
    {"params": other, "lr": LR},
], lr=LR)
print(f"[OPT] 分组 lr: a_mu x{MU_LR_MULT:g} ({len(mu_params)} params), "
      f"其余 x1 ({len(other)} params)")
model.train()

# 判别实验：固定初始条件 seed。
# 每步重设 manual_seed(step) => a0/θ0 每步随机 => 模型必须学"任意起点→target"
# 的全局吸引子，12 个 block 的演化深度（零训练 cos 只到 0.11，指数趋近估算需
# ~300 步深度）根本不够。固定 seed => 只需记忆一条轨迹，先验证通路记忆能力。
SEED_MODE = sys.argv[4] if len(sys.argv) > 4 else "fixed"
if SEED_MODE == "fixed":
    torch.manual_seed(1234)

for step in range(1, STEPS + 1):
    if SEED_MODE == "step":
        torch.manual_seed(step)
    # 【μ warmup】log1p 域的层级跨度是 ch0=2.09 → ch55=0.029（73×），
    # 统一 ×80 lr 在起步阶段会冲过头（150 步实测 ch0 从 +0.38 恶化到 -2.13）。
    # 线性 warmup：前 WARMUP 步从 ×1 线性升到 ×MU_LR_MULT，让 μ 先稳住
    # 大跨度的 DC 通道，再逐步加速小幅度通道的收敛。
    if step <= WARMUP:
        mult = 1.0 + (MU_LR_MULT - 1.0) * (step / max(WARMUP, 1))
    else:
        mult = MU_LR_MULT
    for gparam, base_lr in zip(opt.param_groups, [LR * MU_LR_MULT, LR]):
        gparam["lr"] = base_lr * (mult if gparam is opt.param_groups[0] else 1.0)
    opt.zero_grad()
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
    pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
    # 振幅 loss：log 域相对误差 + 软截断（方案 G，最终形态）。
    # 历史（diag_blur.py 消融）：振幅是主要瓶颈（振幅→target +5.6~10.5 dB，
    # 相位→target 仅 +0.07~0.32 dB）。方案对照（150/1000 步同 seed）:
    #   A. log1p 无权重       → PSNR 17.60，中低频 ratio 1.4~4.3×（偏高=糊）
    #   B. a_tar² 功率加权    → DC 独占 99.9% 监督，a_out 飙 0.66，ratio 14.9×（崩）
    #   C. 纯 log 相对误差    → PSNR 11.13（旧单 γ 架构：通道耦合→"低频饱和→高频补偿"）
    #   D. log1p + (a/a_max)^0.5 → PSNR 20.74（1000 步）/ 22.46（2500 步）
    #   F. log 域 + (a/a_max)^p 权重 → 谱形状接近解决（profile_cos 0.994），但频段比
    #      依然整体偏高 1.3~1.4×（p=0.5），或偏低 0.6×（p=0.25，训练不足）
    #
    # 【F 的根因：log 域不需要幅度权重】log 域 (la−lt) 已经是"倍数"度量——DC 与
    # 高频通道在 log 域被自动放到同一尺度。此时再叠 (a_tar/a_max)^p 幅度权重，等于
    # 重新按"幅度占比"给 DC 加权（p=0.5 时 DC 权重是高频通道的 ~5.8 倍），μ_c 的
    # 稳态被大通道主导、全局系统性偏高。p 越小只是把同一个偏置摊平、让收敛更慢
    # （p=0.25 时 150 步谱跨只展开 25x，是"没训够"不是"方向对"）。两个端点的方向
    # 效应（0.5→偏高 1.4×、0.25→偏低 0.6×）都是同一个偏置的两种表现。
    #
    # ======================================================================
    # 【第十二死穴（经三轮消融定位，方案 H = log1p 无截断无权重）】
    #
    # 【错误假设的修正链，务必保留】
    # ① 最初以为「近零系数梯度是噪声」→ 加 EPS_A=0.01 软截断屏蔽（方案 G）。
    #    实测：软截断把 62/64 通道的 target 真值钉在 log(0.01)=-4.605，
    #    "屏蔽噪声梯度"变成"屏蔽真值"，非 DC 通道 μ 系统性偏高 4.1 个 log 单位。
    # ② 改为纯 log 无截断 → DC ch25/ch50 误差从 2.90 降到 0.31，方向对了，
    #    但 ch0 从 +0.38 恶化到 -2.13，且非 DC 仍差 2.50。
    # ③ **真正的根因：log 在 a→0 处无界。**
    #    实测 ch25/ch50 上各有 1748/6144 个位置 a<1e-4（FFT 数值零，
    #    不是真零），这些位置 log(a)=-13.8，比 ch0 的 +1.94 低 15.7 个单位，
    #    把整条通道的 log 域均值硬拽到 -5.5 / -5.9。
    #    log1p 在 a→0 时→0（梯度=1，有界），log1p 域逐通道均值：
    #      ch0=2.093  ch25=0.344  ch50=0.234  ch1=0.154  ch55=0.0285
    #    动态范围 73×（有界、可学），而非 log 域的无界负值。
    #
    # 【方案 H 最终口径】
    #   la = log1p(a),  lt = log1p(a_tar)
    #   loss_amp = (la - lt)²  跨通道等权，无幅度权重、无截断
    # 为何不加 (a/a_max)^p 权重：log1p 域已是"倍数"度量，加权等于重新按幅度
    # 占比给 DC 加权（p=0.5 时 DC 权重是高频的 5.8 倍），会把谱压平（第七死穴）。
    # 为何不用 log：见上，a≈0 处无界会把层级拽飞。
    # ======================================================================
    EPS_A = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0   # 保留 argv 兼容
    AMP_P = float(sys.argv[5]) if len(sys.argv) > 5 else -1.0  # 保留 argv 兼容
    la = torch.log1p(a.clamp(min=0.0))
    lt = torch.log1p(a_tar)
    loss_amp = (la - lt).pow(2).mean()
    # 相位按 FFT 功率 a_tar² 加权：69.4% 的系数幅度<0.01，其相位是 atan2 的
    # 数值噪声（无监督意义）。平等 MSE 会让随机相位误差的梯度淹没有效位置，
    # 200 步 cos 纹丝不动。功率权重把监督集中在携带图像结构的低频相位上。
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
        # 分频段诊断：必须按 **真实 rfft2 网格** 切，不能按0/16/32/48 均分。
        # rfft2(win=5, patch²·3=48) -> h_fft=3, w_fft=25，展平成D=64 通道，
        # 通道 c 对应 (h=c//25, w=c%25)。空间频率 w 才是"细节 vs 轮廓"的真正维度。
        # 旧代码按 0/16/32/48 均分把 w=0 的 DC（ch0/25/50，占 99.4% 能量）
        # 和高频混在一起，频段比读数完全失真。
        ch = c[0].mean(dim=0)                       # [D] 每通道 cos
        ar = (a[0].mean(dim=0) / a_tar[0].mean(dim=0).clamp(min=1e-6)).clamp(0, 20)  # [D] 振幅比
        W_FFT = (PATCH * PATCH * 3) // 2 + 1
        w_idx = torch.arange(D, device=ar.device) % W_FFT
        # 按真实空间频率分 4 段：w=0(DC) / w=1-3 / w=4-9 / w=10+
        bands = [(w_idx == 0, "DC"), ((w_idx >= 1) & (w_idx <= 3), "低频"),
                 ((w_idx >= 4) & (w_idx <= 9), "中频"), (w_idx >= 10, "高频")]
        bd = " ".join(f"{n}:{ch[m].mean():+.2f}/{ar[m].mean():.2f}"
                      for m, n in bands if bool(m.any()))
        # 谱形状指标（第七个死穴的直接读数）：
        # 细节 = 振幅谱的重尾形状（逐通道），不是全局均值。标量 OU 先验会把
        # 谱压平——全局均值对了但逐通道剖面崩掉。profile_cos 衡量模型逐通道
        # 振幅剖面与 target 的相关：1.0=谱形状完全一致，~0=谱被压平。
        # 【口径统一到 log1p 域，与 loss_amp 同一量纲】用 log 会因 a≈0 无界而失真。
        am_ch = a[0].mean(dim=0)                    # [D] 模型逐通道振幅
        at_ch = a_tar[0].mean(dim=0)                # [D] target 逐通道振幅
        lm, lt_ = torch.log1p(am_ch), torch.log1p(at_ch)
        lm_c, lt_c = lm - lm.mean(), lt_ - lt_.mean()
        prof_cos = (lm_c * lt_c).sum() / (lm_c.norm() * lt_c.norm() + 1e-12)
        prof_ratio_std = torch.log(ar.clamp(min=1e-3)).std()     # 振幅比对数的离散度，0=每通道都校准
        # 【第十二死穴专项】μ 层级误差，log1p 域（与 loss 同口径）
        mu_ = model.pmnncore.a_mu.detach()
        l1t = torch.log1p(at_ch)
        mu_err = (mu_ - l1t).abs()
        dc_err = mu_err[w_idx == 0].max().item()
        nd_err = mu_err[w_idx != 0].mean().item()
        print(f"[{step:4d}] loss={loss.item():.4f} amp={loss_amp.item():.4f} "
              f"phase={loss_ph.item():.4f} cos={cos:+.4f} wcos={wcos:+.4f} "
              f"cos_eff={cos_eff:+.4f} a_out={am:.4f}", flush=True)
        print(f"        频段[cos/振幅比] {bd}", flush=True)
        print(f"        谱形状 profile_cos={prof_cos:+.4f} ratio_logstd={prof_ratio_std:.3f} "
              f"(target 逐通道跨 {at_ch.max()/at_ch.min():.0f}x, 模型 {am_ch.max()/am_ch.min():.0f}x)", flush=True)
        # 早停判据（第十二死穴）：DC 层级误差应 <0.3，非 DC 应 <0.3
        print(f"        [死穴12] μ层级误差(log1p域) DC_max={dc_err:.4f} 非DC均值={nd_err:.4f}"
              f"   (判据: 均<0.3)", flush=True)

# 生成 GIF（不喂 target，走推理路径）
model.eval()
with torch.inference_mode():
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()

# 【出图口径，2026-10-11 第十一死穴修正】
# 主口径 = 分位裁剪（0.5%, 99.5%）：对离群系数稳健，同时把模型振幅对齐到 target 量程，
# 是唯一可用于「视觉质量评估」的口径。
# 副口径 = 固定 clamp ±1.2：与 encode 严格互逆，反映绝对量纲（含亮度标度）。
def _pct_norm(arr, lo=0.5, hi=99.5):
    out = []
    for fr in arr:
        a_, b_ = np.percentile(fr, [lo, hi])
        out.append(np.clip((fr.astype(np.float32) - a_) * 255.0 / max(b_ - a_, 1e-6), 0, 255))
    return np.stack(out)

frames_h_raw = np.stack(decode_field_to_frames(
    fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
frames = _pct_norm(frames_h_raw).astype(np.uint8)
pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
pil[0].save(OUT, save_all=True, append_images=pil[1:], loop=0, duration=100)
frames_h = [f.astype(np.uint8) for f in frames_h_raw]
pil_h = [Image.fromarray(np.ascontiguousarray(f)) for f in frames_h]
OUT_H = OUT.replace(".gif", "_honest.gif")
pil_h[0].save(OUT_H, save_all=True, append_images=pil_h[1:], loop=0, duration=100)
print(f"[GEN] {OUT}  frames={len(pil)}  (分位裁剪口径, 主口径)")
print(f"[GEN] {OUT_H}  (固定 clamp 诚实重建, 副口径)")

# 存瘦身 checkpoint（只存 PMNN 物理核心 + 文本头，不含冻结 Qwen）
sd = {k: v for k, v in model.state_dict().items()
      if not k.startswith("qwen.")}
torch.save(sd, "outputs_real/overfit_ckpt.pt")
print(f"[CKPT] outputs_real/overfit_ckpt.pt  ({sum(v.numel() for v in sd.values()):,} params)")

# 定量对比：双口径报告，缺一不可。
#   主口径 = 分位裁剪：对离群系数稳健，把模型振幅对齐到 target 量程，
#     PSNR 反映「结构/纹理/对比度」的真实质量 —— 判断模型好坏必须看它。
#   副口径 = 固定 clamp：与 encode 严格互逆，反映绝对量纲（含亮度标度）。
#     模型振幅偏小会让 pix_mean 偏离 target、PSNR 被亮度偏移拖垮 2~4 dB，
#     这反映振幅标度问题，不是纹理重建能力。
# 【历史口径污染记录】min-max 曾被当作"真实重建质量"，实测它对重尾 FFT 场
#   极不稳健（离群系数撑满量程，std 从 33.9 塌到 4.8），已废弃禁用。
with torch.inference_mode():
    zt = (a_tar[0] * torch.exp(1j * th_tar[0])).to(torch.complex64)
    ft = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()

arr_t_h = np.stack(decode_field_to_frames(
    ft, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
arr_m_h = np.stack(frames_h).astype(np.float32)
psnr_h = 10 * math.log10(255.0 ** 2 / max(((arr_m_h - arr_t_h) ** 2).mean(), 1e-12))
fd_h = float(np.abs(np.diff(arr_m_h, axis=0)).mean())
fd_t_h = float(np.abs(np.diff(arr_t_h, axis=0)).mean())

# 主口径：分位裁剪（抗离群，对齐量程）—— 视觉质量评估唯一可信口径
arr_t_p = _pct_norm(arr_t_h)
arr_m_p = np.stack(frames).astype(np.float32)
psnr_p = 10 * math.log10(255.0 ** 2 / max(((arr_m_p - arr_t_p) ** 2).mean(), 1e-12))

print(f"[CMP] 主口径 (分位裁剪, 视觉质量)          PSNR={psnr_p:.2f} dB  "
      f"对比度 std={arr_m_p.std():.1f} (target {arr_t_p.std():.1f})")
print(f"[CMP] 副口径 (固定clamp, 绝对量纲)          PSNR={psnr_h:.2f} dB  "
      f"pix_mean={arr_m_h.mean():.1f} (target {arr_t_h.mean():.1f})")
print(f"[CMP] frame-diff  model={fd_h:.2f} target={fd_t_h:.2f} (绝对量纲, "
      f"噪声/抖动指标)")

# 【第十二死穴专项监控】DC 通道（w=0，占 99.4% 振幅能量）层级误差
# 口径：log1p 域，与 loss_amp 完全一致（log 域在 a≈0 处无界会失真）
W_FFT = (PATCH * PATCH * 3) // 2 + 1
DC = [c for c in range(D) if c % W_FFT == 0]
lt1 = torch.log1p(a_tar[0].mean(0))
lm1 = torch.log1p(a[0].mean(0))
mu_ = model.pmnncore.a_mu.detach()
print(f"[DC] 通道 {DC}（占 99.4% 振幅能量）层级误差，log1p 域：")
for c in DC:
    print(f"[DC]   ch{c:>3}  target={lt1[c].item():>8.4f}  mu={mu_[c].item():>8.4f}  "
          f"mdl={lm1[c].item():>8.4f}  mu_err={mu_[c].item() - lt1[c].item():>+7.4f}")
nd = [c for c in range(D) if c not in DC]
nd_err = (mu_[nd] - lt1[nd]).abs().mean().item()
print(f"[DC] 非DC 通道 mu 误差均值 = {nd_err:.4f}   "
      f"（判据：DC_max 与非DC均值均应 <0.3；≈2~4 说明 log 无界问题仍在）")
