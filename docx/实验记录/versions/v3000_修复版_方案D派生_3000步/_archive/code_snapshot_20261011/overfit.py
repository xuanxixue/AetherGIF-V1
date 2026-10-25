# -*- coding: utf-8 -*-
"""单样本过拟合训练 —— V3000 修复版（方案 D 派生，共享 μ + 分组 lr + 可选 loss 域）。

在 v2500 方案 D（视觉最优基线）基础上做三个针对性修复（均有诊断依据）：
  1. 共享 μ：诊断(diag_amu.py)显示 12 个 block 独立 μ 全程停在初值 log(0.2)≈-1.6，
     谱包络没学到，导致 AC 通道能量过冲 2~3.6 倍（噪声感来源）。
  2. 分组 lr：μ 单独放大（补偿 log 域 μ 追 log1p 域 loss 时 a/(1+a) 的梯度压缩）。
  3. 可选 loss 域：log1p（方案 D 原样）或 log+eps（修第十三死穴量纲错位）。

运行：cd 到本快照根目录后
  python scripts/overfit.py 2500               # 全量
  python scripts/overfit.py 150 8-ball_1f3b1.gif 3e-4 0.5 fixed log1p 20
参数：STEPS GIF LR AMP_P SEED_MODE LOSS_DOMAIN MU_LR_MULT
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

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
GIF = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
AMP_P = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
SEED_MODE = sys.argv[5] if len(sys.argv) > 5 else "fixed"
LOSS_DOMAIN = sys.argv[6] if len(sys.argv) > 6 else "log1p"   # "log1p" | "log"
MU_LR_MULT = float(sys.argv[7]) if len(sys.argv) > 7 else 20.0
DC_WEIGHT = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0  # 真 DC 通道 c=0 额外权重（0=等权）
# 【第十五死穴】空间域对比度 loss 权重（0=关闭）
CST_W = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
LOG_EPS = 1e-3
OUT = "outputs_real/overfit.gif"
device = "cuda"

# 【流程铁律】试跑必须与正式产物隔离，否则会覆盖已训好的 ckpt（已发生过一次）
# 用 --tag 给每次运行指定独立子目录，ckpt/gif/log 全部落在各自目录里，互不污染。
TAG = os.environ.get("OVERFIT_TAG", "").strip()
if TAG:
    OUT_DIR = os.path.join("outputs_real", TAG)
    os.makedirs(OUT_DIR, exist_ok=True)
else:
    OUT_DIR = "outputs_real"
OUT = os.path.join(OUT_DIR, "overfit.gif")

gp = next(Path("data/gif_raw/noto").glob(GIF))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)
re, im = target[0, ..., :D], target[0, ..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
th_tar = torch.atan2(im, re)

# 【第十五死穴】target 的空间域灰度 std（对比度目标）
# 必须用与 space_gray 完全相同的口径，否则 loss 目标是错位的
_s_dim = PATCH * PATCH * 3
_full = WIN * (_s_dim // 2 + 1)
_zc = torch.complex(re, im)
if D < _full:
    _zc = torch.cat([_zc, torch.zeros(_zc.shape[0], _full - D,
                                     device=_zc.device, dtype=_zc.dtype)], dim=-1)
elif D > _full:
    _zc = _zc[:, :_full]
_zc = _zc * 10.0
_tgt_flat = torch.fft.irfft2(_zc.reshape(_zc.shape[0], WIN, _s_dim // 2 + 1),
                             s=(WIN, _s_dim), dim=(-2, -1)).real
# 与 codec.decode_field_to_frames 的 clamp 口径完全对齐：
# flat 归一化到 [-1,1]（normalize=False 时是 clamp(flat,-1,1)），再 ×127.5 映到 0~255
_tgt_px = _tgt_flat.clamp(-1.0, 1.0) * 127.5
TGT_STD = float(_tgt_px.mean(-1).std())
print(f"[DATA] {gp.name}  N={target.shape[1]}  "
      f"a_tar mean={a_tar.mean().item():.5f}  loss_domain={LOSS_DOMAIN}  "
      f"MU_LR_MULT={MU_LR_MULT}  AMP_P={AMP_P}  CST_W={CST_W}  TGT_STD={TGT_STD:.2f}")

model = AetherGIFModel(device=device, use_qwen=False).to(device)

# 分组优化器：a_mu（谱包络，64 维低维参数）单独放大学习率
mu_params = []
other_params = []
for name, p in model.named_parameters():
    if "a_mu" in name:
        mu_params.append(p)
    else:
        other_params.append(p)
opt = torch.optim.AdamW([
    {"params": mu_params, "lr": LR * MU_LR_MULT},
    {"params": other_params, "lr": LR},
])
print(f"[OPT] 分组 lr: a_mu x{MU_LR_MULT} ({len(mu_params)} params), "
      f"其他 x1 ({len(other_params)} params)")
model.train()

if SEED_MODE == "fixed":
    torch.manual_seed(1234)

# 历史最优 ckpt 追踪：[偏离度, step]，对比度最接近 target 的那一步会被单独留档
BEST_GAP = [1e9, 0]


def amplitude_loss(a, at):
    """振幅 loss：两种域可选。返回 (loss, 诊断梯度比)。"""
    if LOSS_DOMAIN == "log":
        # log 域 + eps：修第十三死穴（量纲错位）—— log 域 loss 与架构 log 域 μ 严格同量纲，
        # AC 小通道梯度 = a/(a+eps)≈0.96（无 a/(1+a) 的 20 倍压缩）。
        la = torch.log(a.clamp(min=0.0) + LOG_EPS)
        lt = torch.log(at.clamp(min=0.0) + LOG_EPS)
    else:
        # log1p 域（方案 D 原样）：a→0 有界，但与 log 域架构量纲错位（十三死穴）
        la = torch.log1p(a.clamp(min=1e-8))
        lt = torch.log1p(at.clamp(min=1e-8))
    if AMP_P > 0:
        w = (at / at.max().clamp(min=1e-8)).pow(AMP_P)
        w = w / w.mean().clamp(min=1e-8)   # 归一化到 mean=1（保持 loss 尺度）
    else:
        w = torch.ones_like(at)
        # DC_WEIGHT>0 时：只给真 DC 通道 c=0（空间DC×时间DC，唯一承载 96% 能量的大通道）加权，
        # 避免等权下 1 个 DC 被 63 个 AC 稀释而发暗。c=25/50 是时间基频，量级只有 c0 的 1/9~1/17，不给加权。
        if DC_WEIGHT > 0:
            w = w.clone()
            w[:, :, 0] = DC_WEIGHT
    return (w * (la - lt).pow(2)).mean()


def space_gray(a, th):
    """可微的空间域灰度解码（不走 numpy，保留梯度）。

    与 codec.decode_field_to_frames 的 clamp 口径一致，但全在 torch 里做，
    因为 contrast_loss 需要梯度回传到 (a, th)。
    """
    N = a.shape[0]
    z = torch.zeros(N, 2 * D, device=a.device, dtype=a.dtype)
    z[:, :D] = a * torch.cos(th)
    z[:, D:] = a * torch.sin(th)
    zc = torch.complex(z[:, :D], z[:, D:])
    s_dim = PATCH * PATCH * 3
    full = WIN * (s_dim // 2 + 1)
    if D < full:
        zc = torch.cat([zc, torch.zeros(N, full - D, device=a.device, dtype=zc.dtype)], dim=-1)
    elif D > full:
        zc = zc[:, :full]
    zc = zc * 10.0                       # encode 的标定常数（与 codec 一致）
    flat = torch.fft.irfft2(zc.reshape(N, WIN, s_dim // 2 + 1),
                        s=(WIN, s_dim), dim=(-2, -1)).real     # [N, WIN, s_dim]
    # 与 codec 的 clamp 口径对齐到 0~255（对比度即此口径下的灰度标准差）
    flat = flat.clamp(-1.0, 1.0) * 127.5
    return flat.reshape(N, WIN * s_dim)   # 像素域


def contrast_loss(a, th):
    """【第十五死穴修复】空间域对比度直接监督。

    诊断（diag_contrast_gap.py, seed=0）给出的因果链：
      振幅能量占比  v3000 已极准（DC 96.2% vs target 97.2%，均匀度 0.094）
      μ 与真值      corr=0.9953，μ 本身没问题
      振幅溢出      max 111.8 vs target 24，但削峰后对比度几乎不变（25.55→25.31）→ 不背锅
      换 target 相位 对比度 25.55→30.03、动态范围 158→185.3（target 188.3）
    ⇒ 「糊」的根因是**相位精度不足**：irfft2 干涉后能量被摊平，
      而 log 域振幅 loss + a_tar² 功率加权相位 loss 都只能看「加权平均」，
      这个平均恰好会掩盖对比度损失（第十五死穴：log 域校准与视觉质量脱钩）。

    修法：把监督搬到**空间域**——直接对齐灰度 std。
    空间域损失对「能量被摊平」是线性的，不像功率加权平均那样互相抵消。
    """
    x = space_gray(a, th).mean(-1)        # 灰度 [N, pix]
    xc = x - x.mean()
    std_m = xc.std()
    # 用 log 比值保证尺度对称，避免 std_m→0 时的数值爆炸
    return (torch.log(std_m.clamp(min=1e-3)) - math.log(TGT_STD)) ** 2


for step in range(1, STEPS + 1):
    if SEED_MODE == "step":
        torch.manual_seed(step)
    opt.zero_grad()
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
    pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
    loss_amp = amplitude_loss(a, a_tar)
    w_ph = (a_tar ** 2).clamp(min=1e-8)
    loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
    loss = loss_amp + loss_ph
    if CST_W > 0:
        loss_cst = contrast_loss(a[0], th[0])
        loss = loss + CST_W * loss_cst
    else:
        loss_cst = torch.zeros((), device=a.device)
    loss.backward()
    opt.step()
    if step % 25 == 0 or step == 1:
        with torch.no_grad():
            c = torch.cos(pe[0])                      # a/th 带 batch 维，a_tar 不带
            cos = c.mean().item()
            wcos = (w_ph * c).sum().item() / w_ph.sum().item()
            m = a_tar > 1.0
            cos_eff = c[m].mean().item() if bool(m.any()) else float("nan")
            am = a.mean().item()
        am_ch = a[0].mean(dim=0)
        at_ch = a_tar.mean(dim=0)
        ar = (am_ch / at_ch.clamp(min=1e-6)).clamp(0, 20)
        # 【第十五死穴】实时对比度（= 空间域灰度 std），target 基准 = TGT_STD
        with torch.no_grad():
            cstd = float(space_gray(a[0], th[0]).mean(-1).std())
        # 真实 rfft2 网格：h_fft=5, w_fft=25，按 w 维切分
        bands = {
            "DC(w0)": [c_ for c_ in range(D) if c_ % 25 == 0],
            "低频w1-6": [c_ for c_ in range(D) if 1 <= c_ % 25 <= 6],
            "中频w7-16": [c_ for c_ in range(D) if 7 <= c_ % 25 <= 16],
            "高频w17-24": [c_ for c_ in range(D) if 17 <= c_ % 25 <= 24],
        }
        bd = " ".join(f"{n}:{ar[idxs].mean():.2f}" for n, idxs in bands.items())
        print(f"[{step:4d}] loss={loss.item():.4f} amp={loss_amp.item():.4f} "
              f"phase={loss_ph.item():.4f} cst={loss_cst.item():.4f} "
              f"cos={cos:+.4f} wcos={wcos:+.4f} "
              f"a_out={am:.4f} 对比度std={cstd:.1f}/{TGT_STD:.1f}", flush=True)
        print(f"        振幅比[模型/target] {bd}  (1.0=校准)", flush=True)

        # 【防丢权重】周期性保存 + 历史最优单独留档。
        # 起因：早先只 在训练结束时存 ckpt，3000 步跑 3.5h，中途崩溃= 全丢；
        # 更糟的是试跑会直接覆盖正式 ckpt（已发生过一次）。
        # 现在：① 每 250 步落 _ckpt_stepNNNN.pt，可断点续训
        #      ② 对比度最接近 target 的那一步单独存 _ckpt_best.pt，永不被后续覆盖
        if step % 250 == 0 or step == 1:
            _sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
            torch.save(_sd, os.path.join(OUT_DIR, f"overfit_ckpt_step{step:04d}.pt"))
            _gap = abs(cstd / TGT_STD - 1.0)
            if _gap < BEST_GAP[0]:
                BEST_GAP[0] = _gap
                BEST_GAP[1] = step
                torch.save(_sd, os.path.join(OUT_DIR, "overfit_ckpt_best.pt"))
            print(f"[SAVE] step{step} ckpt  对比度={cstd:.1f}/{TGT_STD:.1f} "
                  f"(偏离 {_gap * 100:.1f}%, 历史最优 step={BEST_GAP[1]})", flush=True)

# 生成 GIF（不喂 target，走推理路径 —— 实际单样本过拟合仍喂 target 作漂移场）
model.eval()
with torch.inference_mode():
    a, th, _ = model(texts=["台球"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
frames = np.stack(decode_field_to_frames(
    fld, TS, GRID, PATCH, WIN, IMG, D, normalize=True)).astype(np.uint8)
pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
pil[0].save(OUT, save_all=True, append_images=pil[1:], loop=0, duration=100)
frames_h_raw = np.stack(decode_field_to_frames(
    fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
pil_h = [Image.fromarray(f.astype(np.uint8)) for f in frames_h_raw]
OUT_H = OUT.replace(".gif", "_honest.gif")
pil_h[0].save(OUT_H, save_all=True, append_images=pil_h[1:], loop=0, duration=100)
print(f"[GEN] {OUT}  frames={len(pil)}")
print(f"[GEN] {OUT_H}  (固定 clamp 诚实重建)")

sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
CKPT_PATH = os.path.join(OUT_DIR, "overfit_ckpt.pt")
torch.save(sd, CKPT_PATH)
print(f"[CKPT] {CKPT_PATH}  ({sum(v.numel() for v in sd.values()):,} params)")
print(f"[BEST] 历史最优 step={BEST_GAP[1]}  对比度偏离 target {BEST_GAP[0] * 100:.1f}%  "
      f"→ {os.path.join(OUT_DIR, 'overfit_ckpt_best.pt')}")

with torch.inference_mode():
    zt = (a_tar * torch.exp(1j * th_tar)).to(torch.complex64)
    ft = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()
arr_t_h = np.stack(decode_field_to_frames(
    ft, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
arr_m_h = np.stack(frames_h_raw).astype(np.float32)
psnr_h = 10 * math.log10(255.0 ** 2 / max(((arr_m_h - arr_t_h) ** 2).mean(), 1e-12))
print(f"[CMP] vs target-recon PSNR={psnr_h:.2f} dB")