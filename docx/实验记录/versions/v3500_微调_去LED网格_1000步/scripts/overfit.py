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

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
GIF = sys.argv[2] if len(sys.argv) > 2 else "8-ball_1f3b1.gif"
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-5
AMP_P = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

# ============================================================================
# v3500 微调专用参数（2026-10-12）
#
# 目标极窄：**只去掉 LED 拼接屏网格感 + 降低噪声**，其他一律不动。
# 做法：从 v2500 的 2500 步权重继续训练 1000 步，**架构（方案 D）与原有 loss
#      （log1p 加权振幅 + a_tar^2 加权相位）完全不变**，只追加两个空间域正则项。
#      微调 lr 远低于原训练（3e-5 vs 3e-4），保证既有细节不被大改。
# ============================================================================
# 边缘损失权重：压低 patch 边界跳变相对内部跳变的比例（= 去 LED 网格）
EDGE_W = float(os.environ.get("EDGE_W", "1.0"))# 高频噪声损失权重：压低 AC 通道（非 DC）能量占比的过冲部分（= 降噪）
# 注意：已证实会经共享参数误伤 DC（std 掉 19%），默认关闭，改用 TEXTURE_W
NOISE_W = float(os.environ.get("NOISE_W", "0.0"))
# 纹理噪声权重：把空间域绝对跳变（边界+内部）压到 target 水平（= 真正的降噪）
TEXTURE_W = float(os.environ.get("TEXTURE_W", "1.0"))
# std 守卫下限 = target_std × STD_FLOOR。0.75 ≈ v2500 现状，即「不许比现在更差」
STD_FLOOR = float(os.environ.get("STD_FLOOR", "0.75"))
STD_W = float(os.environ.get("STD_W", "2.0"))
# 分位数降噪（第十八死穴）：LOW 压背景噪点 q75，HIGH 保护轮廓锐度 q99
JUMP_LOW_W = float(os.environ.get("JUMP_LOW_W", "1.0"))
JUMP_HIGH_W = float(os.environ.get("JUMP_HIGH_W", "1.0"))
# 基线权重（v2500 的 2500 步产物，只读）
BASE_CKPT = os.environ.get(
    "BASE_CKPT",
    r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt")
# 输出隔离：所有产物落outputs_real/<tag>/，绝不覆盖基线
TAG = os.environ.get("OVERFIT_TAG", "v3500").strip()
OUT_DIR = os.path.join("outputs_real", TAG)
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "overfit.gif")
device = "cuda"

gp = next(Path("data/gif_raw/noto").glob(GIF))
field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)
target = field.unsqueeze(0)                      # [1,N,2D]
# 【bugfix 2026-10-12】原代码 re/im/a_tar/th_tar 都带 batch 维 [1,...]，
# 靠广播与模型输出 [1,N,D] 对齐。**新增的loss_noise 必须用无 batch 维的版本**，
# 否则 (a_tar**2).mean(dim=0) 会在错误的维上求均值 → loss 爆到1.5e6（实测）。
# 这里显式给出 a_tar_n / th_tar_n（N,D 口径）供新 loss 使用；原有 loss 一行未改。
re, im = target[..., :D], target[..., D:]
a_tar = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
th_tar = torch.atan2(im, re)
a_tar_n = a_tar[0].contiguous()                   # [N,D]
th_tar_n = th_tar[0].contiguous()                 # [N,D]
print(f"[DATA] {gp.name}  N={target.shape[1]}  "
      f"a_tar mean={a_tar.mean().item():.5f} std={a_tar.std().item():.5f}  "
      f"std/mean={a_tar.std().item()/max(a_tar.mean().item(),1e-9):.3f}")

model = AetherGIFModel(device=device, use_qwen=False).to(device)
# 方案 D：统一优化器，无分组 lr、无 warmup（2500 步当时即如此）
opt = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

# 固定初始条件 seed：只记忆一条轨迹，先验证通路记忆能力
# （每步重设 seed 会逼模型学"任意起点→target"，12 块深度不够，不可行）
SEED_MODE = sys.argv[5] if len(sys.argv) > 5 else "fixed"
if SEED_MODE == "fixed":
    torch.manual_seed(1234)


# ============================================================================
# 可微空间域解码（微调用）
# ============================================================================
def space_gray(a, th):
    """与 codec.decode_field_to_frames 同口径的可微解码，保留梯度。
    输出 [N, WIN*s_dim]，N = TS*GRID*GRID，数值域 0~255（clamp 后）。"""
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
    zc = zc * 10.0                       # encode的标定常数
    flat = torch.fft.irfft2(zc.reshape(N, WIN, s_dim // 2 + 1),
                            s=(WIN, s_dim), dim=(-2, -1)).real
    return flat.clamp(-1.0, 1.0) * 127.5


def edge_ratio_t(a, th):
    """返回 (边缘跳变, 内部跳变) —— patch 网格感的直接度量。

    **口径必须是「图像空间」**：先把 [N, WIN*s_dim] 还原成完整图像
    [TS*WIN, GRID*PATCH, GRID*PATCH]，再按 PATCH 网格切边界。
    踩过的坑：早先在 patch 空间 [TS,G,G,WIN,PATCH] 上算，因s_dim=PATCH*PATCH*3
    需要沿 3*PATCH=12 个值求均值，算的并不是完整 48 像素 patch，
    导致同一份target 算出 2.966 而 numpy 口径是 1.229 —— 目标值完全失去意义。

    图像空间口径的实测值（已校准）：
      原始素材= 1.010，codec 天花板 = 1.229，v2500 = 1.620
    """
    x = space_gray(a, th)                                   # [N=TS*G*G, WIN*s_dim]
    # 与 decode_field_to_frames 相同的还原顺序：
    # [TS,G,G,WIN,patch,patch,3] -> permute(0,3,1,4,2,5,6)
    #      -> [TS,WIN,G,patch,G,patch,3] -> reshape(TS*WIN, G*patch, G*patch)
    x = x.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3).permute(0, 3, 1, 4, 2, 5, 6)
    x = x.reshape(TS * WIN, GRID * PATCH, GRID * PATCH, 3).mean(-1)   # 灰度 [T,128,128]
    dh = (x[..., 1:] - x[..., :-1]).abs()                    # 水平差分 [T,H,W-1]
    dv = (x[..., 1:, :] - x[..., :-1, :]).abs()              # 垂直差分 [T,H-1,W]
    dev = PATCH
    hb = ((torch.arange(dh.shape[-1], device=a.device) + 1) % dev == 0)
    vb = ((torch.arange(dv.shape[-2], device=a.device) + 1) % dev == 0)
    edge = dh[..., hb].mean() + dv[..., vb, :].mean()
    inner = dh[..., ~hb].mean() + dv[..., ~vb, :].mean()
    return edge, inner


def loss_edge(a, th, tgt_ratio):
    """去 LED 网格：把边缘/内部比压到 codec 天花板的水平。
    只约束「边界相对内部有多跳」，不约束绝对强度 —— 因此不会削弱细节。"""
    e, i = edge_ratio_t(a, th)
    return (torch.log(e.clamp(min=1e-3)) - torch.log(i.clamp(min=1e-3))
            - math.log(tgt_ratio)) ** 2


def loss_texture(a, th, tgt_e, tgt_i):
    """降噪（绝对强度）：把边界/内部跳变的绝对值压到 target（原始素材）水平。

    与 loss_edge 的区别：loss_edge 只管「比值」，噪点多没关系、只要边界不比内部更跳；
    loss_texture 直接把两者的绝对强度都拉到素材水平 —— 这才是肉眼看到的「颗粒感/网点」。

    关键：差分运算天然不含 DC，所以**不会误伤对比度**（这是上一版 loss_noise 崩盘的根因）。
    用 log 域：跳变跨 3 倍时线性 loss 会被高频主导，log 域梯度均匀。
    """
    e, i = edge_ratio_t(a, th)
    return ((torch.log(e.clamp(min=1e-3)) - math.log(tgt_e)) ** 2
            + (torch.log(i.clamp(min=1e-3)) - math.log(tgt_i)) ** 2)


def _jump_tensor(a, th):
    """所有相邻像素差分的绝对值，展平成一维（用于分位数约束）"""
    x = _image_gray(a, th)
    dh = (x[:, :, 1:] - x[:, :, :-1]).abs().reshape(-1)
    dv = (x[:, 1:, :] - x[:, :-1, :]).abs().reshape(-1)
    return torch.cat([dh, dv])


def loss_jump_dist(a, th, tgt_q75, tgt_q99):
    """分位数感知的降噪：把「背景噪点」和「真实轮廓」分开约束。

    **为什么不能只用均值**（第十八死穴）：素材是极端重尾分布——
    p50=0（一半相邻像素完全相同）、p99=47（轮廓极锐）、尾部 5% 像素占 61.6% 能量。
    用均值约束时，模型为了把均值压到素材水平，会选择「把所有跳变压到中等」：
    轮廓被压平 45%（p99 47→26）、背景噪点还没清干净（p50 0→2）—— 就是「模糊」。

    正确做法是分别约束两端：
      · 低端 q75 → 压到 target 水平（清背景噪点）
      · 高端 q99 → **不许低于** target（保住轮廓锐度）
    高端用 relu 单侧保护：轮廓可以更锐，但不能更糊。
    """
    j = _jump_tensor(a, th)
    q75 = torch.quantile(j, 0.75)
    q99 = torch.quantile(j, 0.99)
    low = (torch.log(q75.clamp(min=1e-3)) - math.log(tgt_q75)) ** 2
    high = torch.relu(math.log(tgt_q99) - torch.log(q99.clamp(min=1e-3))) ** 2
    return low, high, float(q75.item()), float(q99.item())


def _image_gray(a, th):
    """space_gray 输出还原成完整灰度图 [TS*WIN, 128, 128]（0~255 域）。"""
    x = space_gray(a, th)
    x = x.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3).permute(0, 3, 1, 4, 2, 5, 6)
    return x.reshape(TS * WIN, GRID * PATCH, GRID * PATCH, 3).mean(-1)


def loss_std_guard(a, th, tgt_std, floor):
    """守卫：空间域 std 不许低于 target×floor。

    **为什么必须加**：texture/noise 类降噪 loss 的共同失败模式是模型用
    「整体压低振幅」来降低差分 —— 噪点降了，但 std 也崩了（光影更平）。
    本项把这条捷径堵死，逼模型走「降高频 + 提低频」的正路。
    relu 保证只罚「低于下限」，不限制 std 往上走。
    """
    s = _image_gray(a, th).std()
    return torch.relu(math.log(tgt_std * floor) - torch.log(s.clamp(min=1e-3))) ** 2


def loss_noise(a, at):
    """降噪：压低 AC 通道（c 不是 w=0 的空间 DC）能量的**过冲部分**。
    target 的能量 97.2% 在 DC；v2500 把 82% 塞进 AC（放大 60~120 倍）= 噪声源。
    只惩罚「超过 target 的部分」，欠幅不惩罚 —— 细节不会被削弱。

    实现要点：用 **log 比值**而非线性比值，否则 AC 通道 e_t 极小（~1e-4），
    除出来上万会爆（实测 noise loss=4.4e9）。
    再除以 AC 通道数做归一化，让 loss 量级保持在 O(1)~O(10)，
    避免压过主loss（amp+phase 实测量级 0.24~0.001）。"""
    e_m = (a ** 2).mean(dim=0)                     # [D] 模型逐通道能量
    e_t = (at ** 2).mean(dim=0).clamp(min=1e-12)   # [D] target 逐通道能量
    ac = torch.tensor([1.0 if c % 25 != 0 else 0.0 for c in range(D)],
                      device=a.device)              # w=0 是空间 DC 列（c=0/25/50）
    r = torch.log((e_m + 1e-12) / e_t)# log 能量比：>0 即过冲
    n_ac = max(int(ac.sum().item()), 1)
    return (torch.relu(r) * ac).pow(2).sum() / n_ac


# ============================================================================
# 目标边缘比：用**同口径**从 target（codec 天花板）现场计算，不引用外部硬编码值。
# 口径 = 本脚本 edge_ratio_t 的口径（patch 空间：[TS,G,G,WIN,PATCH]）。
# 为什么必须现场算：edge_ratio_t 在 patch 空间测量，与图像空间口径数值不同
#（v2500：patch 空间 2.217 / 图像空间 1.620）。用错口径会让目标值失去意义。
# ============================================================================
def _edge_ratio_of(a, th):
    with torch.no_grad():
        e, i = edge_ratio_t(a, th)
        return float(e.item()), float(i.item()), float(e / i.clamp(min=1e-6))


_a_tgt = a_tar_n# [N,D]
_th_tgt = th_tar_n
TGT_EDGE_ABS, TGT_INNER_ABS, TGT_EDGE_RATIO = _edge_ratio_of(_a_tgt, _th_tgt)
TGT_STD = float(_image_gray(_a_tgt, _th_tgt).std().item())
# 分位数目标（第十八死穴：均值约束会压平轮廓，必须按分位数分开约束）
with torch.no_grad():
    _j_tgt = _jump_tensor(_a_tgt, _th_tgt)
    TGT_Q75 = float(torch.quantile(_j_tgt, 0.75).item())
    TGT_Q99 = float(torch.quantile(_j_tgt, 0.99).item())
print(f"[DATA] target 绝对跳变: 边界={TGT_EDGE_ABS:.3f} 内部={TGT_INNER_ABS:.3f} "
      f"比值={TGT_EDGE_RATIO:.4f}")
print(f"[DATA] target 空间域 std={TGT_STD:.2f}  守卫下限={TGT_STD * STD_FLOOR:.2f}")
print(f"[DATA] target 跳变分位数: q75={TGT_Q75:.2f}(背景噪点) q99={TGT_Q99:.2f}(轮廓锐度)")

# ============================================================================
# 载入 v2500 基线权重（微调起点）
# ============================================================================
_target_edge_ratio = TGT_EDGE_RATIO
_base_sd = torch.load(BASE_CKPT, map_location=device)
_missing, _unexpected = model.load_state_dict(_base_sd, strict=False)
print(f"[BASE] 载入v2500 基线权重：{BASE_CKPT}")
print(f"[BASE]   missing={len(_missing)} unexpected={len(_unexpected)}"
      + (f"  (missing: {_missing[:4]})" if _missing else ""))
print(f"[OPT] 微调 lr={LR}  EDGE_W={EDGE_W}  TEXTURE_W={TEXTURE_W}  STD_W={STD_W}  NOISE_W={NOISE_W}\n"
      f"      目标边缘比={_target_edge_ratio:.4f}  目标绝对跳变 边界={TGT_EDGE_ABS:.3f} 内部={TGT_INNER_ABS:.3f}")
print(f"[OUT] 产物目录 {OUT_DIR}（与基线完全隔离，可随时回退）")
print(f"[SAVE] 基线权重只读，不会被本脚本修改：{BASE_CKPT}")
# 网格比最优留档：[偏离度, step]
BEST = [1e9, 0]

for step in range(1, STEPS + 1):
    if SEED_MODE == "step":
        torch.manual_seed(step)
    opt.zero_grad()
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
    # ===== v3500 微调：只追加两个空间域正则项，原有 loss 一行未改 =====
    if EDGE_W > 0:
        loss_edge_v = loss_edge(a[0], th[0], _target_edge_ratio)
        loss = loss + EDGE_W * loss_edge_v
    else:
        loss_edge_v = torch.zeros((), device=a.device)
    if NOISE_W > 0:
        loss_noise_v = loss_noise(a[0], a_tar_n)     # 必须用 [N,D] 口径（见上方 bugfix）
        loss = loss + NOISE_W * loss_noise_v
    else:
        loss_noise_v = torch.zeros((), device=a.device)
    if TEXTURE_W > 0:
        loss_tex_v = loss_texture(a[0], th[0], TGT_EDGE_ABS, TGT_INNER_ABS)
        loss = loss + TEXTURE_W * loss_tex_v
    else:
        loss_tex_v = torch.zeros((), device=a.device)
    if JUMP_LOW_W > 0 or JUMP_HIGH_W > 0:
        _lo, _hi, _q75n, _q99n = loss_jump_dist(a[0], th[0], TGT_Q75, TGT_Q99)
        loss = loss + JUMP_LOW_W * _lo + JUMP_HIGH_W * _hi
        loss_jlow_v, loss_jhigh_v = _lo, _hi
    else:
        loss_jlow_v = torch.zeros((), device=a.device)
        loss_jhigh_v = torch.zeros((), device=a.device)
        _q75n = _q99n = float("nan")
    if STD_FLOOR > 0:
        loss_std_v = loss_std_guard(a[0], th[0], TGT_STD, STD_FLOOR)
        loss = loss + STD_W * loss_std_v
    else:
        loss_std_v = torch.zeros((), device=a.device)
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
            # 网格感实时监控（本次微调的唯一核心指标）
            _e, _i = edge_ratio_t(a[0], th[0])
            eratio = float(_e / _i.clamp(min=1e-6))
            einner = float(_i)
            _eabs = float(_e)
            _std_now = float(_image_gray(a[0], th[0]).std().item())
        ch = c[0].mean(dim=0)                       # [D] 每通道 cos
        ar = (a[0].mean(dim=0) / a_tar[0].mean(dim=0).clamp(min=1e-6)).clamp(0, 20)  # [D] 振幅比
        # 旧口径：通道四等分（诊断性打印，不影响训练；真实 rfft2 网格 h_fft=3 x
        # w_fft=25 的分段切分已在 v3000 修复，见 3000 步快照）
        bands = [("低频", slice(0, 16)), ("中低", slice(16, 32)),
                 ("中高", slice(32, 48)), ("高频", slice(48, 64))]
        bd = " ".join(f"{n}:{ch[s].mean():+.2f}/{ar[s].mean():.2f}" for n, s in bands)
        # 谱形状指标：模型逐通道 log1p 振幅剖面与 target 的相关（谱被压平≈0）
        am_ch = a[0].mean(dim=0)                    # [D]
        at_ch = a_tar[0].mean(dim=0)                # [D]
        lm, lt_ = torch.log1p(am_ch), torch.log1p(at_ch)
        lm_c, lt_c = lm - lm.mean(), lt_ - lt_.mean()
        prof_cos = (lm_c * lt_c).sum() / (lm_c.norm() * lt_c.norm() + 1e-12)
        prof_ratio_std = torch.log(ar.clamp(min=1e-3)).std()   # 0=逐通道全校准
        print(f"[{step:4d}] loss={loss.item():.4f} amp={loss_amp.item():.4f} "
              f"phase={loss_ph.item():.4f} edge={loss_edge_v.item():.4f} "
              f"tex={loss_tex_v.item():.4f} "
              f"stdg={loss_std_v.item():.4f} "
              f"noise={loss_noise_v.item():.4f} cos={cos:+.4f} wcos={wcos:+.4f} "
              f"cos_eff={cos_eff:+.4f} a_out={am:.4f} "
              f"网格比={eratio:.3f}/{_target_edge_ratio} 边界={_eabs:.2f}/{TGT_EDGE_ABS:.2f} "
              f"内部={einner:.2f}/{TGT_INNER_ABS:.2f} std={_std_now:.1f}/{TGT_STD:.1f} "
              f"q75={_q75n:.1f}/{TGT_Q75:.1f} q99={_q99n:.1f}/{TGT_Q99:.1f}", flush=True)
        print(f"        频段[cos/振幅比] {bd}", flush=True)
        print(f"        谱形状 profile_cos={prof_cos:+.4f} ratio_logstd={prof_ratio_std:.3f} "
              f"(target 逐通道跨 {at_ch.max()/at_ch.min():.0f}x, 模型 {am_ch.max()/am_ch.min():.0f}x)", flush=True)

        # 周期保存 + 最佳留档（微调也必须防丢：每 250 步 + 网格比最优单独存档）
        if step % 250 == 0 or step == 1:
            _sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
            torch.save(_sd, os.path.join(OUT_DIR, f"overfit_ckpt_step{step:04d}.pt"))
            _g = abs(math.log(max(eratio, 1e-3) / _target_edge_ratio))
            if _g < BEST[0]:
                BEST[0] = _g; BEST[1] = step
                torch.save(_sd, os.path.join(OUT_DIR, "overfit_ckpt_best.pt"))
            print(f"[SAVE] step{step}  网格比={eratio:.3f} 边界={_eabs:.2f} "
                  f"内部={einner:.2f} std={_std_now:.1f} q99={_q99n:.1f} "
                  f"| 网格最优 step={BEST[1]}", flush=True)

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
_FINAL_CKPT = os.path.join(OUT_DIR, "overfit_ckpt.pt")
torch.save(sd, _FINAL_CKPT)
print(f"[CKPT] {_FINAL_CKPT}  ({sum(v.numel() for v in sd.values()):,} params)")
print(f"[BEST] 网格比最优 step={BEST[1]} 偏离目标 {BEST[0] * 100:.1f}%  "
      f"→ {os.path.join(OUT_DIR, 'overfit_ckpt_best.pt')}")

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