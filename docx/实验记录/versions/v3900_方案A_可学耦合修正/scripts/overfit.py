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

# D=125：rfft2 真实网格 = WIN(5) x (patch^2*3/2+1 = 25) = 125，全保留不截断。
# 这是 v3600 相对 v2500 的唯一改动 —— 网格感的根因是 codec 里 spec[:, :D] 的
# 非对称硬截断（D=64 只留 h0,h1 全行 + h2 的 w0~13，h3,h4 整行丢弃）。
# 实测 D=64→125：往返 PSNR 28.63→49.82 dB，网格比 1.373→1.172（素材原值 1.111）。
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
# 真实频域网格尺寸（用于诊断分频段，不再用 0/16/32/48 均分——那会混淆 DC 与高频）
H_FFT, W_FFT = WIN, (PATCH * PATCH * 3) // 2 + 1
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
GIF = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
AMP_P = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
# ---------- v3700 新增：高频增益约束（治烧焦感）----------
# 只约束中高频(w12-18) 与 高频(w19-24)，DC/低频/中低不碰（它们由原 loss 负责）
HF_SEGS = [(12, 19), (19, 25)]
GAIN_TGT = 1.0         # 目标增益：输出/目标 的分段能量比 = 1.0（完全对齐）
GAIN_W = 0.5           # 该 loss 权重
HUBER_DELTA = 0.5      # 过冲 0.5 以内用 L2，超过用线性（防梯度爆炸）
# ---------- v3700 新增：径向 Nyquist 过冲约束（治烧焦的物理成因）----------
# 128px 网格的 Nyquist 半径 = 90.5px，取最后 3 个等宽 bin（r >= 72.4）作为「最高频环」。
# ⚠️ 口径修正（第一版踩坑）：不能用 r>=11（那是像素半径，对应 bin 1/2，不是高频），
#   也不能说素材在r>=11 处能量为0 —— 之前那张功率谱表**归一化到半径0**才显示 0.0000，
#   实际能量是非零的。真正接近 Nyquist 的是最后几个 bin。
# 素材在 r>=72.4 的能量=1.37e12（占总能量 99.99%），可安全作分母。
RAD_LO_FRAC = 0.80       # 取半径 >= 0.80 * rmax 的那一圈
RAD_CAP = 1.2# 能量比上限（素材=1.0，模型不得超过 1.2倍）
RAD_W = 1.0
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

# ---------- v3700：径向约束的预计算（只需算一次）----------
# ★ 口径修正（第二版踩坑）：RING_TAR 必须与模型输出**同量纲**。
#   第一版拿「素材像素图(0~255)」的能量当分母，模型输出却是「归一化 [-1,1]」的图，
#   两者差 4 个数量级 (1.71e12 vs 8.45e7) → 比值恒为 0，约束永不激活。
#   正确做法：用 **target 场解码出来的图**当基准 —— 走与模型输出完全相同的 _decodable 链，
#   只差振幅值，口径 100% 对齐。
from codec.gif_codec import load_gif_frames as _lfg
_yy, _xx = np.indices((IMG, IMG))
_rmax = float(np.sqrt((IMG // 2) ** 2 + (IMG // 2) ** 2))
RAD_R = torch.from_numpy(
    (np.sqrt((_yy - IMG // 2) ** 2 + (_xx - IMG // 2) ** 2) >= RAD_LO_FRAC * _rmax).astype(np.float32)
).to(device)


def _decodable(a_, th_):
    """纯 torch 解码到 [T*WIN, IMG, IMG, 3]，**保持梯度**。

    不能用 codec.decode_field_to_frames：它返回 numpy uint8（不可微），
    且每步解码 6144 块会拖垮训练速度。这里严格复刻它的变换链，但全程 torch。

    原始链（codec.decode_field_to_frames）：
        z = complex(a,th)[N,D]
        flat = irfft2(z.reshape(N,WIN,W_FFT), s=(WIN,s_dim)) -> [N,WIN,s_dim]
        flat.reshape(N, WIN*s_dim)     # 注意这是把 WIN 和 s_dim 摊平成一维
        blocks = flat.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
        xt = blocks.permute(0,3,1,4,2,5,6)          # [t,w,gy,py,gx,px,c]
        x  = xt.reshape(TS*WIN, IMG, IMG, 3)
    """
    z = torch.complex(a_ * torch.cos(th_), a_ * torch.sin(th_))       # [N,D]
    N = z.shape[0]
    z = z * 10.0                                                      # encode标定常数
    flat = torch.fft.irfft2(z.reshape(N, WIN, W_FFT), s=(WIN, PATCH * PATCH * 3),
                            dim=(-2, -1)).real# [N,WIN,s_dim]
    flat = flat.reshape(N, WIN * PATCH * PATCH * 3)
    blocks = flat.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)                          # [t,w,gy,py,gx,px,c]
    return xt.reshape(TS * WIN, IMG, IMG, 3)


# 素材侧径向能量：直接由 target 场解码，口径与模型输出完全一致
with torch.no_grad():
    _tg = _decodable(a_tar[0], th_tar[0]).mean(dim=-1)            # [T,H,W]
    _Fg = torch.fft.fft2(_tg - _tg.mean(dim=(1, 2), keepdim=True))
    RING_TAR = (_Fg.abs().pow(2) * RAD_R).sum(dim=(1, 2)).mean().clamp(min=1e-12)
print(f"[RAD] Nyquist 环(r>={RAD_LO_FRAC*_rmax:.1f}) 素材基准能量 = {RING_TAR.item():.4e}")

model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)
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
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
    pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
    # 振幅 loss：log1p 压重尾 + (a_max 相对)^AMP_P 软化权重
    la = torch.log1p(a.clamp(min=1e-8))
    lt = torch.log1p(a_tar.clamp(min=1e-8))
    w = (a_tar / a_tar.max().clamp(min=1e-8)).pow(AMP_P)
    # ==============================================================
    # 【v3700 修复「烧焦感」· 改动 1：高频增益归一 loss】
    #
    # 病因（v3600 实测）：频段振幅比 中高 2.18 / 高频 1.71，模型把中高频放大近一倍。
    # 根因量化：a_tar 在 DC(w=0) 均值 4.798，中高(w12-18) 仅 0.011 —— **差 435 倍**。
    # 原权重 w=(a_tar/a_max)^0.5 只把这个差距压到约 20 倍，中高频绝对权重仍是 DC 的 1/20，
    # 所以模型**放大高频几乎不付代价**，loss 从头到尾不管这件事
    # （实测中高频振幅比从 25 步到 2500 步一直在 1.69~2.55 震荡，从未收敛到 1.0）。
    #
    # ★ 口径修正（第一版踩坑，能量比口径完全不能用）：
    #   第一版用 sum(a²)/sum(a_tar²) 做增益，150 步实测恒为 0.016，约束永不激活。
    #   原因：中高段 a_tar 的 std/mean = 2.1（极度重尾），少数极大值主导 sum(a²)，
    #   分母被撑大 → 任何模型输出都会被误判成「欠幅」。
    #   同一时刻日志的「频段振幅比」（mean_N 口径）读数是 2.54（过冲）。
    #   两个指标差 150 倍。**必须用与诊断读数一致的 mean_N 口径**，
    #   否则约束和观测互相打架，永远学不会。
    #
    # 修法：g = mean_N(a_out) / mean_N(a_tar)，按 w 分段，偏离 1 就是过冲/欠幅。
    # 用 **Huber**：过冲侧斜率常数，欠幅侧平方，避免大过冲时梯度爆炸。
    # relu 单侧性：只罚过冲方向 —— 欠幅留给 AMP_P 和原 loss 处理。
    # ==============================================================
    a_bar_out = a.clamp(min=1e-8).mean(dim=0)               # [D] mean_N(a_out)
    a_bar_tar = a_tar.clamp(min=1e-8).mean(dim=0)           # [D] mean_N(a_tar)
    gain_seg, loss_gain = [], torch.zeros((), device=a.device)
    for lo, hi in HF_SEGS:                # 只管中高频 + 高频
        idx = torch.tensor([c for c in range(D) if lo <= (c % W_FFT) < hi],
                           device=a.device)
        go = (a_bar_out[idx].mean() / a_bar_tar[idx].mean().clamp(min=1e-12)).clamp(0.1, 20.0)
        excess = (go - GAIN_TGT).clamp(min=0.0)             # 单侧：只罚过冲
        huber = torch.where(excess <= HUBER_DELTA,
                            0.5 * excess.pow(2) / HUBER_DELTA,
                            excess - 0.5 * HUBER_DELTA)
        loss_gain = loss_gain + huber
        gain_seg.append(go.detach())
    loss_gain = loss_gain / len(HF_SEGS)
    # ==============================================================
    # 【v3700 修复「烧焦感」· 改动 2：径向 Nyquist 过冲约束】
    #
    # 病因（v3600 径向功率谱实测，用户提出假设后验证）：
    #   半径   素材      codec顶     模型      模型/素材
    #   0-4     1.0000   1.0000   1.0000    1.00x
    #   4-5     0.0106   0.0105   0.0142    1.34x
    #   8-9     0.0005   0.0005   0.0007    1.34x
    #   11-12   0.0001   0.0001   0.0001    1.84x
    #   12-13   0.0000   0.0000   0.0001    2.21x
    #   13-14   0.0000   0.0000   0.0001    2.88x
    #   14-15   0.0000   0.0000   0.0000    4.23x
    # → **素材和 codec 顶在半径 11+ 已是有效数字以下，模型却在那里放了 2~4 倍能量。**
    #   模型生成了素材里「从未被表示过」的高频（128px 网格上超过 Nyquist 的信息不存在），
    #   这就是烧焦颗粒感的**物理成因**。
    # 注意降采样本身无罪：LANCZOS 已是最优插值，实测 NEAREST/BILINEAR/BICUBIC/LANCZOS
    # 四种方式的 Sobel 几乎相同（1042~1045），素材与 codec 顶的功率谱逐位吻合。
    #
    # 修法：在**空间域 FFT** 上直接约束最高频环的能量不得超过素材的 RAD_CAP 倍。
    # 与改动 1（频域逐通道增益）互补：那条管「已有频段能量偏大」，
    # 这条管「不该有频段凭空出现」。
    # ==============================================================
    # ---- 约束 2：径向 Nyquist 过冲（纯 torch 可微，全程不 numpy）----
    frames = _decodable(a[0], th[0])                     # [T*WIN,128,128,3] 可微
    gray = frames.mean(dim=-1)                            # [T*WIN,128,128]
    Fg = torch.fft.fft2(gray - gray.mean(dim=(1, 2), keepdim=True))
    ring_out = (Fg.abs().pow(2) * RAD_R).sum(dim=(1, 2)).mean()
    rad_ratio = ring_out / RING_TAR.clamp(min=1e-12)
    loss_rad = torch.clamp(rad_ratio - RAD_CAP, min=0.0)   # 单侧：只罚超
    loss_amp = (w * (la - lt).pow(2)).mean()
    # 相位按 FFT 功率 a_tar^2 加权：69.4% 系数幅度<0.01，其相位是 atan2 数值噪声
    w_ph = (a_tar ** 2).clamp(min=1e-8)
    loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
    loss = loss_amp + loss_ph + GAIN_W * loss_gain + RAD_W * loss_rad
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
        # 分频段按真实 rfft2 网格切（通道 c ↔ (h=c//W_FFT, w=c%W_FFT)，W_FFT=25）。
        # 必须按 w 切：w=0 列承载 target 绝大部分能量（DC），按 0/16/32/48 均分
        # 会把 DC 和高频混在一段，读数失真。
        def _band(lo, hi):
            idx = [c for c in range(D) if lo <= (c % W_FFT) < hi]
            return torch.tensor(idx, device=ch.device)
        bands = [("DC(w0)", _band(0, 1)), ("低频", _band(1, 6)),
                 ("中低", _band(6, 12)), ("中高", _band(12, 19)),
                 ("高频", _band(19, W_FFT))]
        bd = " ".join(f"{n}:{ch[ix].mean():+.2f}/{ar[ix].mean():.2f}" for n, ix in bands)
        # 谱形状指标：模型逐通道 log1p 振幅剖面与 target 的相关（谱被压平≈0）
        am_ch = a[0].mean(dim=0)                    # [D]
        at_ch = a_tar[0].mean(dim=0)                # [D]
        lm, lt_ = torch.log1p(am_ch), torch.log1p(at_ch)
        lm_c, lt_c = lm - lm.mean(), lt_ - lt_.mean()
        prof_cos = (lm_c * lt_c).sum() / (lm_c.norm() * lt_c.norm() + 1e-12)
        prof_ratio_std = torch.log(ar.clamp(min=1e-3)).std()   # 0=逐通道全校准
        print(f"[{step:4d}] loss={loss.item():.4f} amp={loss_amp.item():.4f} "
              f"phase={loss_ph.item():.4f} gain={loss_gain.item():.4f} "
              f"cos={cos:+.4f} wcos={wcos:+.4f} "
              f"cos_eff={cos_eff:+.4f} a_out={am:.4f}", flush=True)
        print(f"        频段[cos/振幅比] {bd}", flush=True)
        # v3700：分段能量增益读数（烧焦感的直接指标，1.0 = 对齐，越大越"焦"）
        print(f"        高频增益 中高={gain_seg[0]:.3f} 高频={gain_seg[1]:.3f} "
              f"| Nyquist环={rad_ratio.item():.2f} (上限 {RAD_CAP})", flush=True)
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