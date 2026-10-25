# -*- coding: utf-8 -*-
"""频谱形状监督 loss（A + B 项，2026-10-18）

【为什么需要】绝对质量诊断（abs_quality_probe.py）判决：
    codec 天花板 51.09/51.97 dB，模型只做到 13.10 dB；
    生成 hi/lo = 0.23，素材 hi/lo = 0.01（23 倍）；
    生成 sobel 28.31，素材 7.09（4 倍）；ac1 0.587 保住了低频结构。
  → 生成 = **低频结构 + 大量高频噪声纹理**。
  → 且「重建模式」与「生成」逐位相同（13.09 vs 13.10），
    说明问题**不在条件通路、不在物理核**，而在**训练目标/频谱层面**：
    现有 loss_amp 是逐元素 log1p MSE，它对「单个系数的小幅随机偏离」
    几乎没有惩罚（高频系数目标本身极小，中位数 0.0015），
    但这些系数在 decode 时会**叠加成可见的空间域高频噪声**。
    逐元素损失看不见这种「每点都小、整体成灾」的失真。

【两项对策】
  A) 高频能量单侧守卫 loss_hf_guard
     罚「高频占比过高」，单侧 relu（允许更锐，不允许更噪）。
  B) 幅度谱形状监督 loss_spec_shape
     约束 **分带能量占比分布** 匹配 target。
     ★ 关键性质：占比是「除以各自总和」的**形状量**，
       总量不变 → 模型无法用「整体压低振幅」走捷径
       （对比：绝对能量 MSE 会被整体缩放作弊，正是第十八死穴）。

【频带切法：必须按 w 切】
  rfft2 真实网格 = WIN(5) × s_fft(25)，通道 c ↔ (h=c//25, w=c%25)。
  w=0 是横向直流，承载 target 97.2% 能量 —— 与其他 w 混在一起算
  会被 DC 单点主导。按 h 均分会把 DC 和高频混淆（第十五/十六死穴同源）。
"""
import math

import torch

# 分带边界（w 方向，s_fft = PATCH*PATCH*3//2+1 = 25）
# ★ 2026-10-18 修正：原先 band0 只含 w=0，把 w=1~24 全算高频 →
#   模型为满足守卫把 w≥1 压光 → 空间域过平滑（sobel 4.33 vs 素材 6.97，
#   ac1 0.305 vs 0.666）。低频带必须给够宽度，与空间域「中心低频」对齐。
#   现在 band0 = w∈[0,6) 六个低频列（含直流），其余 19 列分4 段。
BAND_EDGES = (0, 6, 12, 18, 22, 25)
N_BANDS = len(BAND_EDGES) - 1          # 5
# hi/lo 的低频列数（= BAND_EDGES[1]）：w <此值算低频
N_LO_COLS = BAND_EDGES[1]


def band_energy(a: torch.Tensor, win: int = 5, w_fft: int = 25):
    """把 [B, N, D] 的幅度谱拆成 [B, win, w_fft]，算每格点平均能量。

    a: [B, N, D]，D = win * w_fft
    返回 energy: [B, win, w_fft]（已对 N 求均值）
    """
    B, N, D = a.shape
    assert D == win * w_fft, f"D={D} != {win}*{w_fft}"
    e = a.pow(2).reshape(B, N, win, w_fft).mean(dim=1)     # [B, win, w_fft]
    return e


def band_profile(a: torch.Tensor, win: int = 5, w_fft: int = 25,
                 edges=BAND_EDGES) -> torch.Tensor:
    """分带能量占比 -> [B, N_BANDS]，每行和为 1。

    ★ 占比 = 形状量，与总量无关 → 抗「整体缩放」作弊。
    """
    e = band_energy(a, win, w_fft)                          # [B, win, w_fft]
    prof = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        prof.append(e[:, :, lo:hi].sum(dim=(1, 2)))
    p = torch.stack(prof, dim=-1)                           # [B, N_BANDS]
    return p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def loss_spec_shape(a_pred: torch.Tensor, a_tar: torch.Tensor,
                    win: int = 5, w_fft: int = 25) -> torch.Tensor:
    """B 项：分带能量占比的 log 比值 L1。

    用 log 空间（ratio = log p_pred - log p_tar）而不是绝对占比差：
    target 占比极度重尾（DC 带 ~0.97，其余带各 ~1e-3~1e-2），
    线性空间里小占比的误差会被 DC 完全淹没。
    DC 带自身也进 log（p_pred 有下界 clamp），防止模型把 DC 能量偷偷
    挪到高频带（那正是当前 23 倍 hi/lo 的成因）。
    """
    pp = band_profile(a_pred, win, w_fft).clamp(min=1e-12)
    pt = band_profile(a_tar, win, w_fft).clamp(min=1e-12)
    return (pp.log() - pt.log()).abs().mean()


def h_profile(a: torch.Tensor, win: int = 5, w_fft: int = 25) -> torch.Tensor:
    """时间频率 h 维度能量占比 -> [B, win]，每行和为 1。

    与 band_profile（按 w 切空间频带）正交：这一维对应 WIN 帧窗的
    时间频谱，h=0 是「帧间恒定」（准静态 GIF 的主体），h>0 是真实运动。
    """
    e = band_energy(a, win, w_fft)                          # [B, win, w_fft]
    p = e.sum(dim=2)                                        # [B, win]
    return p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)


def loss_h_spec(a_pred: torch.Tensor, a_tar: torch.Tensor,
                win: int = 5, w_fft: int = 25) -> torch.Tensor:
    """时间频率 h 维度能量分布的 log 比值 L1（2026-10-21，diag_bandbreak）。

    ★ 起因（第三十三个发现）：现有全部频谱 loss（spec/hf_guard/lo_floor/
    band_abs）都只按**空间频率 w** 切带，时间频率 h 维度**零监督**。
    diag_bandbreak.py 实测（fix_clamp_150）：
        素材  逐h = [84.2, 6.1, 1.8, 1.8, 6.1]（84% 在 h=0，准静态）
        生成  逐h = [23.1, 21.2, 18.0, 18.1, 19.6]（被摊平）
        重建  逐h = [31.2, 25.3, 7.8, 10.1, 25.6]（重建也散）
    h 被摊平 = 帧间能量随机闪烁 = ac1 直接到负数（实测 -0.47）。
    占比形状量（与 loss_spec_shape 同构）：与总量无关，
    模型无法用「整体压低/抬高振幅」走捷径（第十八死穴）。
    """
    pp = h_profile(a_pred, win, w_fft).clamp(min=1e-12)
    pt = h_profile(a_tar, win, w_fft).clamp(min=1e-12)
    return (pp.log() - pt.log()).abs().mean()


def hf_ratio(a: torch.Tensor, hi_lo: float = 6.0,
             win: int = 5, w_fft: int = 25) -> torch.Tensor:
    """A 项度量：高频/低频能量比。

    低频 = w < N_LO_COLS（默认 6 列，含直流，对齐空间域「中心低频」）；
    其余 w 全算高频。
    """
    e = band_energy(a, win, w_fft)                          # [B, win, w_fft]
    lo = e[:, :, :N_LO_COLS].sum(dim=(1, 2))
    hi = e[:, :, N_LO_COLS:].sum(dim=(1, 2))
    return hi / lo.clamp(min=1e-12)


def loss_hf_guard(a_pred: torch.Tensor, a_tar: torch.Tensor,
                  slack: float = 1.5, win: int = 5, w_fft: int = 25) -> torch.Tensor:
    """A 项：高频占比单侧守卫。

    只罚「比 target 高出 slack 倍以上」的部分（单侧 relu）：
      ratio_pred <= ratio_tar * (1+slack) → 0（允许比素材更锐，不罚）
      ratio_pred >  ratio_tar * (1+slack) → 罚超额
    ★ 单侧是必须的：双边约束会把模型往「过平滑」推，
      那正是第十七/十八死穴（压平轮廓=糊）的路。
    """
    r_pred = hf_ratio(a_pred, win=win, w_fft=w_fft)
    r_tar = hf_ratio(a_tar, win=win, w_fft=w_fft)
    cap = (r_tar * (1.0 + slack)).detach()
    return (r_pred - cap).clamp(min=0.0).mean()


# ======================================================================
# 反向守卫：防「过平滑」（第十八/十九死穴的频域版）
# ======================================================================
# A 项是**单侧**的（只罚高频超标），单独用会把模型往「全低频」推 ——
# 实测 150 步：空间域 sobel 28.31 → 4.33（素材 6.97），ac1 0.587 → 0.305，
# 即**从「噪点过锐」翻到了「糊」**。两个极端都是死。
# 所以必须同时给低频侧一个下限：低频占比不能低于 target 的 floor 倍。
# 同样是单侧（relu 只罚下限），不会把模型推过头。

def lo_ratio(a: torch.Tensor, win: int = 5, w_fft: int = 25) -> torch.Tensor:
    """低频能量 / 总能量（= 1 − hi_ratio 的补）。"""
    e = band_energy(a, win, w_fft)
    tot = e.sum(dim=(1, 2))
    return e[:, :, :N_LO_COLS].sum(dim=(1, 2)) / tot.clamp(min=1e-12)


def loss_lo_floor(a_pred: torch.Tensor, a_tar: torch.Tensor,
                  floor: float = 0.7, win: int = 5, w_fft: int = 25) -> torch.Tensor:
    """低频占比下限守卫（单侧）：低于 target×floor 才罚。

    ★ 这是 A 项的必要配套：没有它，A 项会把模型从「高频噪点」推到「全糊」，
      那是同一个错误的另一端（第十八死穴：压平轮廓 = 模糊）。
    """
    l_pred = lo_ratio(a_pred, win, w_fft)
    l_tar = (lo_ratio(a_tar, win, w_fft) * floor).detach()
    return (l_tar - l_pred).clamp(min=0.0).mean()

# ======================================================================
# C) per-band 绝对幅度约束 —— 补「形状量」的盲区（第二十死穴修复）
# ======================================================================
# 【病灶实测，diag_amp_phase.py，素材0】
#     a_pred/a_tar： DC列0.0745、w1-5 **217889×**、w6+ **98587×**
#   而同一时刻训练 hi/lo比 = 0.21（形状是对的）。
#   → 形状对了、绝对幅度爆 10^5。loss_spec_shape 是占比，
#     对「每个系数多大」完全盲（单元测试已证明 a*=2 时它恒为 0）。
#   → 占比正确 + 总量全堆高频 = 另一种噪声。
#   修法：**同时约束每个频段的 log1p 域平均绝对幅度**，且
#     用 per-band 归一化（每个 band 自己算自己的均值再比），
#     避免 DC 段的巨值淹没高频段的信号。

def _band_mean_log1p(a: torch.Tensor, win: int, w_fft: int) -> torch.Tensor:
    """[B,N,D] → [B, N_BANDS]，每band 的 log1p 域平均幅度。"""
    B, N, D = a.shape
    L = torch.log1p(a.clamp(min=1e-8)).reshape(B, N, win, w_fft)
    out = []
    for i in range(len(BAND_EDGES) - 1):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        out.append(L[:, :, :, lo:hi].mean(dim=(1, 2, 3)))
    return torch.stack(out, dim=-1)                # [B, N_BANDS]


def loss_band_abs(a_pred: torch.Tensor, a_tar: torch.Tensor,
                  win: int = 5, w_fft: int = 25, eps: float = 1e-3) -> torch.Tensor:
    """per-band log1p 幅度对齐（补形状项盲区）。

    loss = mean_band | mean_log1p(a_pred) − mean_log1p(a_tar) |

    ★ 为什么在 log1p 域：FFT 系数重尾（跨 5+ 数量级），线性域 MSE 会被
      DC 段的巨值独占；log1p 后每band 都在 O(0.01~2) 的可比较区间。
    ★ 为什么 per-band 各自求均值再比：不这样做的话，DC band 的绝对值
      （~2.09）会把高频 band（~0.001）的小误差在 mean 里抹平，
      等于又回到「DC 淹没一切」。
    """
    lp = _band_mean_log1p(a_pred, win, w_fft)
    lt = _band_mean_log1p(a_tar, win, w_fft)
    return (lp - lt).abs().mean()


def loss_color_band(a_pred: torch.Tensor, a_tar: torch.Tensor,
                    win: int = 5, w_fft: int = 25, col_w: int = 16) -> torch.Tensor:
    """颜色频点专项幅度对齐（只打 w=col_w 这一列）。

    ★ 为什么需要单独一项（2026-10-24 实测定位）
      s_dim = 4×4×3 = 48，颜色通道是这 48 维的**最后一位**，周期 = 3，
      沿 48 维做 rfft 后，三通道差异落在 **w = 48/3 = 16**。
      去掉/只留 w=16 的重建实验（铁证）：
          8-ball    完整 sat 6.36 → 去掉 w16 后 0.51  → 只留 w16(+DC) 6.85
          前辈前辈  完整 sat 9.87 → 去掉 w16 后 2.85  → 只留 w16(+DC) 9.79
      即 **w=16 是颜色的唯一载体**，其余 120 个通道对颜色几乎没有贡献。

    ★ 为什么不能用 loss_band_abs 代替
      后者把 25 个 w 压成 5 个 BAND，w=16 落在其中一个带里，被同带其他频点
      摊薄 → 梯度不足以把它拉起来。实测 color4 生成场 w=16 只有 target 的
      12~17%（8-ball 0.103/0.616、前辈前辈 0.139/1.140），而 w=0 亮度反而是
      target 的 1.35~1.43 倍。所以必须**单频点、独立权重**地打。

    与 loss_band_abs 同口径：log1p 域、对 N 取均值（不用线性域 MSE——
    FFT 系数跨 5+ 数量级，线性域会被 DC 巨值独占）。
    """
    B, N, D = a_pred.shape
    Lp = torch.log1p(a_pred.clamp(min=1e-8)).reshape(B, N, win, w_fft)
    Lt = torch.log1p(a_tar.clamp(min=1e-8)).reshape(B, N, win, w_fft)
    return (Lp[:, :, :, col_w].mean() - Lt[:, :, :, col_w].mean()).abs()


def loss_color_phase(pe: torch.Tensor, d_model: int = 125, w_fft: int = 25,
                     col_w: int = 16) -> torch.Tensor:
    """颜色频点**相位**对齐（与 loss_color_band 配对使用，缺一不可）。

    ★ 为什么必须补这一项（2026-10-24 第一轮验证教训）
      只约束 w=16 的**幅度**时，sat 确实从 63%/31% 拉到 110%/66%，
      但生成出来是**彩色噪点**而不是正确的颜色：
        8-ball 本该几乎无彩（纯灰球），却被染上青蓝斑点；
        前辈前辈该是黑白+局部红缎带，却满屏红蓝紫乱点。
      原因：颜色 = 幅度 + 相位。幅度对了只说明"这里有色差"，
      **色相（往哪个颜色偏）由相位决定**，相位随机 → 色相随机 → 彩色噪点。
      所以 amplitude 与 phase 必须成对约束。

    参数：
      pe       : 相位误差 [B,N,D]，已 wrap 到 [-π, π]（训练脚本里的同名列）
      d_model  : 通道数 125
      w_fft    : 空间频点数 25
      col_w    : 颜色频点 16（48 维里 ch 周期=3 → 48/3=16）

    返回 (1 - cos(Δθ)) 在 5 个颜色通道上的均值，范围 [0,2]，0=完全对齐。
    """
    m = (torch.arange(d_model, device=pe.device) % w_fft) == col_w
    return (1.0 - torch.cos(pe[..., m])).mean()


def loss_band_abs_cap(a_pred: torch.Tensor, a_tar: torch.Tensor,
                      cap_mult: float = 3.0, win: int = 5,                      w_fft: int = 25) -> torch.Tensor:
    """per-band 绝对幅度**上限**（单侧硬顶，防止某一 band 爆掉）。

    只罚「超过 target×cap_mult 倍」的部分（log1p 域单侧 relu）。
    与 loss_band_abs 的区别：后者双向对齐（也罚过小），
    本项只防爆——因为「过小」已由 loss_amp 和 loss_lo_floor 覆盖，
    再罚一次会导致低频段被压死（同第十九死穴）。
    """
    lp = _band_mean_log1p(a_pred, win, w_fft)
    lt = _band_mean_log1p(a_tar, win, w_fft)
    cap = lt + math.log1p(cap_mult)                 # log1p 域的等比上限
    return (lp - cap).clamp(min=0.0).mean()


# ======================================================================
# D) 相位监督 —— 真凶（diag_amp_phase.py 四路交叉判决）
# ======================================================================
# 【判决证据】
#   ①模型a+模型th      13.52 dB / ac1 0.318 / sobel  6.09
#   ②模型a+target th   14.03 dB / ac1 0.411 / sobel  7.40← 换相位能救
#   ③target a+模型th    9.72 dB / ac1 0.641 / sobel  8.06（素材0 wcos **−0.456**）
#   ④target a+target th 51.10 dB（天花板）
#   → **相位是主凶**。素材0 相位 wcos = −0.456 比随机还差（反相），
#     平均相位误差 1.13 rad。
#
# 【为什么现有 loss_ph 完全无效】
#   loss_ph = Σ(a_tar²·pe²) / Σ(a_tar²)，而 **DC 承载 97.2% 能量**
#   → 99.99% 的相位监督砸在 DC 一个点上，高频相位的梯度被稀释到 ~0。
#   实测 phase loss 三次打点纹丝不动（2.7653 / 2.7481 / 2.7599）。
#   训练日志 wcos 恒定 ~0.22。
#
# 【修法：per-band 相位分组监督】
#   每个 band 内**独立**用 cos 误差 + 该band 内的均匀权重，
#   再对band 平均 → 每个频段获得同等相位监督权。
#   用 cos(pe) 而非 pe²：相位是圆周量，pe 有 ±π 跳变，
#   pe² 在 ±π 边界不可导且会给出错误梯度；cos 是周期连续的。

def loss_phase_band(a_pred: torch.Tensor, a_tar: torch.Tensor,
                    win: int = 5, w_fft: int = 25,
                    skip_dc: bool = True) -> torch.Tensor:
    """per-band 相位分组监督（cos 误差，band 内均匀权重）。

    skip_dc=True 时跳过 band0（w<6）—— DC 相位由低频结构主导，
    交给 loss_amp 和 block 的层级 μ 管；相位专项专注中高频，
    避免 DC 的 97.2% 能量再次淹没梯度。
    """
    B, N, D = a_pred.shape
    pe = torch.remainder(a_pred - a_tar + math.pi, 2 * math.pi) - math.pi
    cost = (1.0 - torch.cos(pe)).reshape(B, N, win, w_fft)   # 周期连续
    out = []
    for i in range(len(BAND_EDGES) - 1):
        if skip_dc and i == 0:
            continue
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        out.append(cost[:, :, :, lo:hi].mean(dim=(1, 2, 3)))
    return torch.stack(out, dim=-1).mean()


def loss_phase_band_floor(a_pred: torch.Tensor, a_tar: torch.Tensor,
                          win: int = 5, w_fft: int = 25,
                          skip_dc: bool = True) -> torch.Tensor:
    """相位分组监督的**下限守卫**：模型相位不许烂到低于随机水平。

    cos 误差在随机相位时 ≈ 1.0。要求 mean_band cos ≥ 0.15
    （即 wcos 必须为正，明显好于随机）。
    单侧 relu —— 不设上限，允许学得比 target 更规整。
    ★ 这是与「梯度被淹没」配套的硬约束：loss 只给梯度不给保证，
      梯度可能长期为零 → 需要一个直接罚「低于随机」的项兜底。
    """
    B, N, D = a_pred.shape
    pe = torch.remainder(a_pred - a_tar + math.pi, 2 * math.pi) - math.pi
    c = torch.cos(pe).reshape(B, N, win, w_fft)
    out = []
    for i in range(len(BAND_EDGES) - 1):
        if skip_dc and i == 0:
            continue
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        out.append(c[:, :, :, lo:hi].mean(dim=(1, 2, 3)))
    cb = torch.stack(out, dim=-1).mean(dim=-1)               # [B]
    return (0.15 - cb).clamp(min=0.0).mean()
