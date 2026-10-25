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
# 【第十六死穴】逐通道能量比 loss 权重（0=关闭）——治log1p 关不住的尖峰
EN_W = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0
# 【第十八死穴】patch 边缘跳变惩罚权重（0=关闭）——治「像被火烧糊了」的真凶
# 病因：v3000J 边界跳变 26.88 / 内部跳变 1.30（网格比 20.66），
#内部细节被抽干、能量全堆在 4x4 patch 边界上。目标比值 1.229（codec 天花板实测）。
EDGE_W = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0
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
# 注意：逐帧 std 的定义见下方【2026-10-12 修正·逐帧口径】段（第 92-94 行），
# 那里的 reshape 顺序才与 decode_field_to_frames 的真实 permute 一致（实测 corr=0.99999）。
# 不要用简单的 [TS*WIN, GRID^2, pix] reshape——那把 win 维当成了帧序，口径是错的。

# 【2026-10-12 修正·逐帧口径】N = TS * GRID * GRID（不是 TS*WIN*GRID*GRID）。
# build_spacetime_index 用 meshgrid(t,y,x) 且 grid=32=WIN，N=6*32*32=6144。
# decode_field_to_frames 把 flat.reshape(TS,GRID,GRID,WIN,patch,patch,3) 后
# permute(0,3,1,4,2,5,6) -> (t,win,...) 再reshape(TS*WIN,...)，
# 所以帧序号 f = t*WIN + w，帧 f 对应的块是
#   px[t*GRID^2 : (t+1)*GRID^2 , w*s_dim : (w+1)*s_dim]
# 实测该口径 vs 真实解码逐帧 std：maxabs=0.26corr=0.99999。
_tgt_pf = _tgt_px.reshape(TS * GRID * GRID, WIN, _s_dim).mean(-1)   # [TS*GRID^2, WIN]
_tgt_pf = _tgt_pf.reshape(TS, GRID * GRID, WIN).permute(0, 2, 1).reshape(TS * WIN, GRID * GRID)
TGT_STD_PER_FRAME = _tgt_pf.std(dim=1).contiguous()# [TS*WIN=30]
TGT_STD = float(TGT_STD_PER_FRAME.mean())# 全局均值，仅用于日志显示
# 【第十八死穴】patch 边缘跳变/内部跳变 的目标值 = codec 天花板的实测值。
# 原图是 1.010，但 rfft2 截断决定了下限约 1.229 —— 目标必须用架构可达值，不能用原图值。
TGT_EDGE_RATIO = 1.229
print(f"[DATA] target 逐帧 std: " +
      " ".join(f"{v:.0f}" for v in TGT_STD_PER_FRAME.tolist()) +
      f"  | 全局均值 {TGT_STD:.2f}  | 边缘比目标 {TGT_EDGE_RATIO}")
print(f"[DATA] {gp.name}  N={target.shape[1]}  "
      f"a_tar mean={a_tar.mean().item():.5f}  loss_domain={LOSS_DOMAIN}  "
      f"MU_LR_MULT={MU_LR_MULT}  AMP_P={AMP_P}  CST_W={CST_W}  EN_W={EN_W}  "
      f"TGT_STD={TGT_STD:.2f}")

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


def energy_loss(a, at):
    """【2026-10-12 新增·第十六死穴】逐通道能量比loss——治log1p 关不住的尖峰。

    实测病灶（diag_outlier_energy.py, v3000J 3000 步）：
      频段能量占比 w13-18 = 11.61%，target 只有 0.19% → **61 倍过载**
      定位到单通道 c=16(h0,w16)：能量比 **235.7x**，模型均幅 3.25 vs target 0.304
      a_out.max=148.41 —— 恰好等于架构 clamp 上限 exp(5.0)，说明**尖峰被顶在硬边界上**
      而非自然收敛（block.py: log_a_new.clamp(-20,5)）

    为什么现有 loss 关不住：
      d/da log1p(a) = 1/(1+a)。a=148 时梯度 = 0.0067，只有线性域的 **0.67%**。
      重尾分布下尖峰只占极少数系数，被海量近零系数淹没 → 尖峰几乎不受惩罚。
      这正是「均值欠幅（四段全<1.0）但能量超载 61 倍」这个矛盾读数的成因：
      log1p 看的是**位置级相对幅度**，验收看的是**频段能量占比**，两者不是一回事。

    修法（不加数据先验、不改架构递推）：
      直接监督逐通道能量比E_m/E_t的 **log 差**。
      为什么用 log 比值而不是线性比值：
        - 尺度不变：尖峰 10x 过载和100x 过载给出同量级梯度，不会被小误差淹没
        - **梯度不随尖峰幅度衰减**：d/da log(E_m) = 2a/E_m，尖峰主导 E_m 时 ≈ 2/N，
          与 log1p 的 1/(1+a) 正好相反——这才是真正「关得住」的域
        - 直接对齐验收第 3 项（频段能量占比），不是代理指标
      目标端加相对下限floor，避免 target 数值零通道 log 发散。
    """
    e_m = (a ** 2).mean(dim=0)                # [D] 逐通道能量（a 无 batch 维）
    e_t = (at ** 2).mean(dim=0)               # [D]
    floor = (e_t.max() * 1e-4).clamp(min=1e-12)
    return (torch.log(e_m + floor) - torch.log(e_t + floor)).pow(2).mean()


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

    **必须逐帧，不能用全局 std**（2026-10-12 v3000J 实测教训）：
    全局 std 版跑满 3000 步后全局达标（97.8% of 天花板），但**帧 10~19 全崩塌**
    （std 5~6，target 该区是 38→16 的低对比度区），同时正常帧被冲到 55~57（target 44~46）。
    原因：目标序列第 13~19 帧本身就是低对比度区，全局 std 被它们拖累，
    模型为了抬高全局值而**牺牲低对比度帧的真实性**——平均值度量掩盖局部问题，在空间域重演。

    修法：逐帧算 std，每帧对齐自己的 target 值。

    【2026-10-12 修正】上面的 reshape 口径是错的：
      错：x.reshape(TS*WIN, GRID*GRID, pix)   —— N 里根本没有 WIN 维
      对：帧 f = t*WIN+w，块 = x[t*G^2:(t+1)*G^2, w*s_dim:(w+1)*s_dim]
    错的reshape 会把 5 个win 帧混在一起算 std，等于又退回全局口径。
    """
    x = space_gray(a, th)                # [N, WIN*s_dim]，N = TS*GRID*GRID
    s_dim = PATCH * PATCH * 3
    x = x.reshape(TS * GRID * GRID, WIN, s_dim).mean(-1)      # [TS*G^2, WIN] 灰度
    x = x.reshape(TS, GRID * GRID, WIN).permute(0, 2, 1).reshape(TS * WIN, GRID * GRID)
    xc = x - x.mean(dim=1, keepdim=True)          # 逐帧零均值
    std_m = xc.std(dim=1)                # [30] 逐帧 std
    tgt = TGT_STD_PER_FRAME.to(std_m.device)
    return (torch.log(std_m.clamp(min=1e-3)) - torch.log(tgt)).pow(2).mean()


def patch_edge_loss(a, th):
    """【第十八死穴修复】patch 边缘跳变惩罚 —— 直击「像被火烧糊了」的真凶。

    空间频率域取证（diag_detail_vs_led.py, 2026-10-12）：

      版本      边界跳变   内部跳变   网格比
      原始GIF      0.6350.622     1.021
      v2500        9.961     6.148     1.620   ← 内外都有内容 → 有质感
      v3000J      13.440     0.650    20.663   ← 内部是空的 → 糊

    **真相**：v3000J 的内部跳变 0.650 ≈ 原始 GIF 的 0.622，
    即它**根本没有内部细节**；而边界跳变 13.44 是原始 GIF 的 21 倍。
    所以「糊」不是对比度不足，而是**能量全堆在 patch 边界上，内部被抽干**——
    一层剧烈的网格边界撑起画面，中间是空的。

    修法：惩罚边界跳变相对内部跳变的比例，把高频能量从边界引导进内部。
    目标 = **codec 天花板 1.229**（架构固有下限，原图是 1.010，rfft2 截断注定到不了）。

    实测（差分口径，2026-10-12校准）：
      版本       边界跳变   内部跳变   网格比
      原始素材     4.99     4.94     1.010
      codec顶     6.94     5.65     1.229   <- 目标
      v2500      19.92    12.30     1.620
      v3000J     26.88     1.30    20.663   <- 内部被抽干
    """
    s_dim = PATCH * PATCH * 3
    x = space_gray(a, th)                # [N=TS*G*G, WIN*s_dim]
    # 每个位置拆成 [WIN][PATCH][3]（s_dim = PATCH*PATCH*3 的 layout 见 space_gray）
    # space_gray 输出 reshape(N, WIN*s_dim) 是 irfft2 的 [WIN, s_dim] 展平，
    # s_dim = PATCH*PATCH*3，所以拆成 [WIN, PATCH, PATCH, 3] 后对最后两维(PATCH*3)求灰度
    x = x.reshape(TS * GRID * GRID, WIN, PATCH, 3 * PATCH).mean(-1)   # [N, WIN, PATCH]
    x = x.reshape(TS, GRID, GRID, WIN, PATCH)                # [TS,G,G,WIN,PATCH]
    # patch 内部相邻差分
    dh = (x[..., 1:] - x[..., :-1]).abs()          # [TS,G,G,WIN,PATCH-1]
    dv = (x[..., 1:, :] - x[..., :-1, :]).abs()    # [TS,G,G,WIN-1,PATCH]
    inner = dh.mean() + dv.mean()
    # 跨 patch 边界：x[...,PATCH-1]（本patch 最右/最下）与 roll 后的 x[...,0]
    eh = (x[..., PATCH - 1:] - torch.roll(x[..., 0:1], -1, dims=-2)).abs()
    ev = (x[..., -1:, :] - torch.roll(x[..., 0:1, :], -1, dims=-3)).abs()
    edge = eh.mean() + ev.mean()
    return (torch.log(edge.clamp(min=1e-3)) - torch.log(inner.clamp(min=1e-3))
            - math.log(TGT_EDGE_RATIO)) ** 2


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
    if EN_W > 0:
        loss_en = energy_loss(a[0], a_tar)
        loss = loss + EN_W * loss_en
    else:
        loss_en = torch.zeros((), device=a.device)
    if EDGE_W > 0:
        loss_edge = patch_edge_loss(a[0], th[0])
        loss = loss + EDGE_W * loss_edge
    else:
        loss_edge = torch.zeros((), device=a.device)
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
        # 【第十五死穴】实时对比度（逐帧口径）—— target 基准 = TGT_STD_PER_FRAME
        with torch.no_grad():
            _x = space_gray(a[0], th[0])                      # [N, WIN*s_dim]
            # 与 target 同口径：先按 [TS*GRID^2, WIN, s_dim] 拆，再沿 s_dim 求灰度
            _xf = _x.reshape(TS * GRID * GRID, WIN, -1).mean(-1)   # [TS*GRID^2, WIN]
            _xf = _xf.reshape(TS, GRID * GRID, WIN).permute(0, 2, 1).reshape(TS * WIN, GRID * GRID)
            _xf = _xf - _xf.mean(dim=1, keepdim=True)
            cstd_f = _xf.std(dim=1)                            # [30]
            cstd = float(cstd_f.mean())
            # 崩塌检测：某帧低于/高于自身 target 的 50%（相对判据，不用全局均值）
            _bad = (cstd_f < TGT_STD_PER_FRAME * 0.5).sum().item()
            cbad = (cstd_f > TGT_STD_PER_FRAME * 1.5).sum().item()
            # 【第十八死穴】patch 边缘比实时监控（差分口径，与验收脚本一致）
            _gx = _x.reshape(TS * GRID * GRID, WIN, PATCH, 3 * PATCH).mean(-1).reshape(
                TS, GRID, GRID, WIN, PATCH)
            _dh = (_gx[..., 1:] - _gx[..., :-1]).abs()
            _dv = (_gx[..., 1:, :] - _gx[..., :-1, :]).abs()
            _inner = float(_dh.mean() + _dv.mean())
            _eh = (_gx[..., PATCH - 1:] - torch.roll(_gx[..., 0:1], -1, dims=-2)).abs()
            _ev = (_gx[..., -1:, :] - torch.roll(_gx[..., 0:1, :], -1, dims=-3)).abs()
            _edge = float(_eh.mean() + _ev.mean())
            edge_ratio = _edge / max(_inner, 1e-6)
        # 真实 rfft2 网格：h_fft=5, w_fft=25，按 w 维切分
        bands = {
            "DC(w0)": [c_ for c_ in range(D) if c_ % 25 == 0],
            "低频w1-6": [c_ for c_ in range(D) if 1 <= c_ % 25 <= 6],
            "中频w7-16": [c_ for c_ in range(D) if 7 <= c_ % 25 <= 16],
            "高频w17-24": [c_ for c_ in range(D) if 17 <= c_ % 25 <= 24],
        }
        bd = " ".join(f"{n}:{ar[idxs].mean():.2f}" for n, idxs in bands.items())
        # 【第十六死穴】实时频段能量占比（验收第3 项的直接指标）+ 尖峰监控
        with torch.no_grad():
            eo = (a[0] ** 2).sum(dim=0).cpu()
            et = (a_tar ** 2).sum(dim=0).cpu()
            bnd = {}
            for nm, idxs in [("w0", [c_ for c_ in range(D) if c_ % 25 == 0]),
                             ("w1-6", [c_ for c_ in range(D) if 1 <= c_ % 25 <= 6]),
                             ("w7-12", [c_ for c_ in range(D) if 7 <= c_ % 25 <= 12]),
                             ("w13-18", [c_ for c_ in range(D) if 13 <= c_ % 25 <= 18]),
                             ("w19-24", [c_ for c_ in range(D) if 19 <= c_ % 25 <= 24])]:
                bnd[nm] = (eo[idxs].sum() / eo.sum().clamp(min=1e-12) * 100).item()
            amax_c = int((a[0] ** 2).sum(dim=0).argmax().item())
        bds = " ".join(f"{k}:{v:.2f}%" for k, v in bnd.items())
        print(f"[{step:4d}] loss={loss.item():.4f} amp={loss_amp.item():.4f} "
              f"phase={loss_ph.item():.4f} cst={loss_cst.item():.4f} "
              f"en={loss_en.item():.4f} edge={loss_edge.item():.4f} "
              f"cos={cos:+.4f} wcos={wcos:+.4f} "
              f"a_out={am:.4f} 对比度std={cstd:.1f}/{TGT_STD:.1f} 崩塌={_bad} 过冲={cbad} "
              f"边缘比={edge_ratio:.2f}/{TGT_EDGE_RATIO} 内部跳变={_inner:.2f}", flush=True)
        print(f"        振幅比[模型/target] {bd}  (1.0=校准)", flush=True)
        print(f"        频段能量占比 {bds}  (target w0:97.20 w1-6:2.13 w7-12:0.36 w13-18:0.19 w19-24:0.12)"
              f"  amax=c{amax_c} a_max={a[0].max().item():.1f}", flush=True)

        # 【防丢权重】周期性保存 + 历史最优单独留档。
        # 起因：早先只 在训练结束时存 ckpt，3000 步跑 3.5h，中途崩溃= 全丢；
        # 更糟的是试跑会直接覆盖正式 ckpt（已发生过一次）。
        # 现在：① 每 250 步落 _ckpt_stepNNNN.pt，可断点续训
        #      ② 对比度最接近 target 的那一步单独存 _ckpt_best.pt，永不被后续覆盖
        if step % 250 == 0 or step == 1:
            _sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
            torch.save(_sd, os.path.join(OUT_DIR, f"overfit_ckpt_step{step:04d}.pt"))
            # 最佳判据用**逐帧最大相对偏差 + 边缘比偏差** 的较大者。
            # 只看对比度不够：v3000J 的对比度达标但内部细节被抽干，
            # 必须让 best 的判据能同时捕捉「逐帧崩塌」和「内部空洞」两类退化。
            _rel = ((cstd_f - TGT_STD_PER_FRAME).abs() / TGT_STD_PER_FRAME)
            _gap_contrast = float(_rel.max())
            _gap_edge = abs(math.log(max(edge_ratio, 1e-3) / TGT_EDGE_RATIO))
            _gap = max(_gap_contrast, _gap_edge)
            if _gap < BEST_GAP[0]:
                BEST_GAP[0] = _gap
                BEST_GAP[1] = step
                torch.save(_sd, os.path.join(OUT_DIR, "overfit_ckpt_best.pt"))
            print(f"[SAVE] step{step} ckpt  对比度={cstd:.1f}/{TGT_STD:.1f} "
                  f"逐帧偏差={_gap_contrast * 100:.0f}% 边缘比={edge_ratio:.2f}/{TGT_EDGE_RATIO} "
                  f"内部={_inner:.2f} 崩塌={_bad} 过冲={cbad} "
                  f"| 历史最优 step={BEST_GAP[1]}", flush=True)

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
print(f"[BEST] 历史最优 step={BEST_GAP[1]}  逐帧对比度最大偏差 {BEST_GAP[0] * 100:.0f}%  "
      f"→ {os.path.join(OUT_DIR, 'overfit_ckpt_best.pt')}")

with torch.inference_mode():
    zt = (a_tar * torch.exp(1j * th_tar)).to(torch.complex64)
    ft = torch.cat([zt.real, zt.imag], dim=-1).float().cpu()
arr_t_h = np.stack(decode_field_to_frames(
    ft, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
arr_m_h = np.stack(frames_h_raw).astype(np.float32)
psnr_h = 10 * math.log10(255.0 ** 2 / max(((arr_m_h - arr_t_h) ** 2).mean(), 1e-12))
print(f"[CMP] vs target-recon PSNR={psnr_h:.2f} dB")