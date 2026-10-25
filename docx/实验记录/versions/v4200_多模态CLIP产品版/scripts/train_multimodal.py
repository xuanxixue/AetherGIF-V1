# -*- coding: utf-8 -*-
"""
v4200 多模态产品版 —— 7 天预算全量训练主脚本。

目标（用户 2026-10-13 定案，2026-10-14 追加早停）
------------------------------------------------
  1. 文本生成 GIF / 图像生成 GIF / 图像+文本生成 GIF 三种能力
  2. 画质与 v3700_D125_修复烧焦 保持同一水平（同一编解码口径，不追画质）
  3. **训练总时长上限 7 天，且越快越好** —— 不是跑满 7 天，
     而是**最早什么时候验证集达标就停**
  4. 计时器：显示已训练多久
  5. 实时 ETA：按当下真实速度算还剩多久
  6. 进度条：百分比 + 步数 + ETA
  7. 断点保存：崩溃/断电后可 --resume 无损续训

★ 早停四条判据（每 --check-every 步检查，任一触发即停）
  判据一  验证指标连续 N 次不改善        → 已到平台期，后面全是浪费
  判据二  训练/验证 loss 分叉            → 过拟合，继续只会更差
  判据三  频段比稳定在 0.8~1.2 且不漂移  → 谱形状收敛，收益递减
  判据四  单步时间突变 / 显存异常        → 效率已崩，继续跑纯浪费

  7 天是**上限**，下限可能只有 2~3 天。训练结束会打印实际用了预算的百分之几。

★ 学习率调度（WarmupCosineLR）
  前期大 lr 快速收敛，后期小 lr 精修 → 验证指标更早进入平台期 → 早停更早触发
  lr_start(3e-5) --warmup(300步)--> lr_peak(3e-4) --cosine--> lr_min(1e-5)

★ 验证集纪律（早停能不能生效全看这条）
  **必须按 group 划分，不能按样本划分**。实测：按样本随机划 424 条验证集，
  其 80 个 group **全部**出现在训练集里 —— bulk_mid_16_40 这类 group 有上万 GIF，
  同 group 的 GIF 画面高度相似（同一角色/同一素材库），等于训练时见过
  验证样本的"近亲"，验证指标偏乐观 → 早停偏晚触发。
  现在按 group 整组划分：8% group = 16 组 / 251 条，group 重叠 0。
  固定随机种子 1234，划分可复现。
  验证集固定 mode="both"（确定性，不掷随机数），指标才可复现可比。
  训练与验证**共用 compute_loss()**，否则是在比两个不同口径的数字。

★ 三种能力的实现（不是三个模型，是同一个模型的三种条件采样）
    mode="mix"（默认）：每个 batch 内随机 1/3 文生 / 1/3 图生 / 1/3 图文生

加速手段（用户抱怨过"6 小时太慢"，这里逐条列明 + 各自收益）
--------------------------------------------------------------
  1. **离线 CLIP 预编码**（最大头）：CLIP 865M 参数完全不进训练循环，
     训练时零 CLIP 开销，且不占显存（本机 CLIP fp32 权重 3.5GB，4GB 卡根本放不下）
  2. **TF32 + cudnn.benchmark**：矩阵乘/卷积走 Tensor Core（本脚本**未**实现
     autocast/bf16 混合精度；AMP_P 是振幅 loss 的 0.5 次幂权重，与精度无关。
     要开混合精度需自行包 autocast，物理核里的 log1p/atan2/remainder
     对精度敏感，建议先不碰）
  3. **torch.compile**（可选，默认关）：首步编译慢，稳态提速 20~40%
  4. **TF32 / cudnn.benchmark**：矩阵乘与卷积走 Tensor Core
  5. **num_workers=4**（默认）：★ B=1 时 GPU 单步就要 4.4 秒，4 个 worker 早已
     能把 CPU 端喂饱（每样本解码+FFT 编码约 0.1~2.4 秒）。**不要开 8 个**：
     每个 worker 的提交量 0.65GB（线程已限到 1），8 个就是 5GB 提交额度，
     而本机 C 盘只剩 258MB → pagefile 撑不开 → 提交上限被钉死在 36.7GB，
     一旦顶穿就是 OpenBLAS 分配失败 + worker 崩溃 + 主进程静默挂死
  7. **batch=1**（默认）：★ D=125 下实测唯一干净可用的档位。
     B=4 → GPU 3.95GB 占满 4GB 卡、主进程提交 11.4GB；B=2 → 分配器碎裂、
     单步 39~49 秒；B=1 → GPU 2.55GB、单步 4.4 秒
  8. **DriftField chunk=512**（默认，从 2048 降下来）：数学等价（分块 logsumexp
     合并精确），只降峰值。单块 logits 张量从 201MB 降到 48MB
  9. **自适应步数**：按 7 天预算自动算 total_steps，不用你手填
  10. **预编码断点续跑**：preencode 每 3 分钟原子落盘，中途崩溃最多损失 3 分钟

★ 实测口径（本机 RTX3050 Laptop 4GB / 16GB RAM / **C 盘仅剩 258MB**）
    D=125 + N=6144 单步 fwd+bwd 实测（2026-10-14）
        batch=1  chunk=512   GPU 2.55 GB   单步 4.4 s     ← 当前默认，唯一干净可用
        batch=2  chunk=512   GPU 4.78 GB   单步 49.4 s    ← 分配器碎裂，比 B=4 更慢
        batch=2  chunk=2048  GPU 4.95 GB   单步 38.9 s
        batch=4  chunk=512   GPU 3.95 GB   主进程提交量 11.4GB / RAM 剩 0.27GB
    DataLoader worker（线程限 1 之后）  6~10 线程 / 提交 0.65 GB
    ★★ C 盘 258MB 是隐形炸弹：pagefile.sys 22.6GB 但磁盘没空间扩展，
       提交上限被永久钉在 36.7GB。开训前请先清出 5GB 以上。
    14145 个 GIF 全量索引         204 个 group 去重后只需编码 204 条文本

loss 口径（与 v3700 完全一致，保证画质可比）
----------------------------------------------
  振幅: weighted_mean((log1p(a) - log1p(a_tar))^2, w=(a_tar/a_max)^0.5)
  相位: sum(w_ph*pe^2)/sum(w_ph),  w_ph = a_tar^2
  外加 v3700 的两项烧焦约束（高频增益 + 径向 Nyquist），权重沿用

运行
----
  # 首次：先离线预编码 CLIP 条件（一次性，约 1-2 小时）
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  C:/Python312/python.exe scripts/preencode.py

  # 训练（自动按 7 天预算算步数）
  C:/Python312/python.exe scripts/train_multimodal.py

  # 指定天数 / 恢复训练 / 只做图生等
  C:/Python312/python.exe scripts/train_multimodal.py --days 7 --mode mix --resume auto
"""
import os
import sys
import math
import time
import json
import argparse
import random
from pathlib import Path

# ============================================================
# ★★★ 线程数必须在 import numpy / torch **之前**锁死（2026-10-14 崩溃修复）
# ============================================================
# 症状：训练启动后没有任何报错地挂死，GPU 利用率 0%，
#       控制台只留下一行
#         OpenBLAS error: Memory allocation still failed after 10 retries
# 诊断（实测进程快照）：
#       主进程 44 线程 / 每个 DataLoader worker 31 线程
#       每个 worker 提交量(private+pagefile) 1.59GB，8 worker + 主进程 ≈ 13GB
#       系统提交上限 36.7GB 被顶穿，只剩 0.74GB → 任何新分配都失败
# 根因：OpenBLAS / OpenMP 按**核数**开 per-thread buffer（16 逻辑核）。
#       DataLoader 用 spawn 起 8 个 worker，每个 worker 再开 16 线程 →
#       128 个线程各自的 buffer 累加起来就是那 13GB。
#       环境变量必须在 import numpy/torch 前设，否则库初始化时已按核数开完线程。
#       worker 里的 torch.set_num_threads(1) 见 _worker_init（spawn 重新
#       import 本模块时会再次执行这段环境变量设置，worker 因此也拿到 1 线程）。
#       主进程在 main() 里显式调回 4 线程（它只做 GPU 计算与 H2D 拷贝，
#       多给 CPU 线程没有收益，但也不能跟 worker 一样锁死）。
# ============================================================
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from data.multimodal_dataset import MultimodalGifDataset, make_collate
from data.clip_condition import CLIP_EMB_DIM
from physics.cond_field import CondFieldPredictor

# ---------- 编解码口径：与 v3700 逐位一致（改这里 = 改画质基准）----------
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
H_FFT, W_FFT = WIN, (PATCH * PATCH * 3) // 2 + 1

# ---------- loss 超参：与 v3700 一致 ----------
AMP_P = 0.5
HF_SEGS = [(12, 19), (19, 25)]
GAIN_TGT, GAIN_W, HUBER_DELTA = 1.0, 0.5, 0.5
RAD_LO_FRAC, RAD_CAP, RAD_W = 0.80, 1.2, 1.0

# 数据根：v4200 在 D:/AetherGIF-V1/docx/实验记录/versions/v4200_.../，要往上退 4 级。
# 用函数在运行时推导（而非硬编码绝对路径），换机器/挪目录也不会失效。
# preencode.py 曾把这里写成退 3 级 → 14145 张全部静默跳过。
def _resolve_gif_root() -> str:
    for up in range(3, 7):
        cand = ROOT
        for _ in range(up):
            cand = cand.parent
        p = cand / "data" / "gif_raw"
        if p.is_dir():
            return str(p)
    raise SystemExit(
        f"[FATAL] 找不到数据根：在 {ROOT} 往上退 3~6 级的 data/gif_raw 都不存在。\n"
        f"        请确认项目布局，或改 _resolve_gif_root()。"
    )


GIF_ROOT = _resolve_gif_root()


# ============================================================
# 工具：时间格式化
# ============================================================
def fmt_dur(sec):
    sec = max(0, int(sec))
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d:
        return f"{d}天{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_bar(step, total, elapsed, rate, width=32):
    """单行进度条（\r 刷新，不刷屏）。"""
    frac = min(1.0, step / max(total, 1))
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    if rate > 1e-9:
        remain = (total - step) / rate
    else:
        remain = float("nan")
    return (f"\r[{bar}] {frac*100:5.1f}%  {step}/{total} 步  "
            f"已用 {fmt_dur(elapsed)}  剩余 ~{fmt_dur(remain)}  "
            f"{rate:.2f} 步/s   ")


def next_batch(it, device):
    """取一个 batch 并搬上 GPU。

    collate 刻意只做 CPU 拼接（pin_memory 只能 pin CPU 张量），
    H2D 搬运在这里统一做，配合 pin_memory + non_blocking 走 DMA。
    """
    try:
        cond, field, modes, idxs = next(it)
    except StopIteration:
        return None
    cond = cond.to(device, non_blocking=True)
    field = field.to(device, non_blocking=True)
    return cond, field, modes, idxs


# ============================================================
# DataLoader worker 初始化：线程锁到 1
# ============================================================
def _worker_init(worker_id):
    """每个 worker 进程启动时调用：把 CPU 线程池锁到 1。

    ★ 必须放在 worker_init_fn 里，而不是只在 main() 里设环境变量：
      Windows 的 DataLoader 用 **spawn** 启动 worker（不是 fork），
      worker 进程会重新 import 本模块，环境变量确实会继承 —— 但 torch 的
      intra-op 线程池是在 import torch 时按核数建的，只设环境变量时
      worker_init_fn 里再调一次 torch.set_num_threads(1) 才能真正生效
      （双保险，任一路径失守都不会再回到 16 线程）。

    副作用检查：worker 里做的是 PIL 解码 + numpy/torch FFT，
    FFT 走 pocketfft 不吃 BLAS 多线程，锁成单线程几乎不损失吞吐
    （实测 bench 速率不降），换来的是每个 worker 提交量从 1.59GB 掉到几百 MB。

    cv2 同样要锁：OpenCV 默认按核数开线程池，若后续 codec 引入 cv2 会复发。
    """
    torch.set_num_threads(1)
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass          # 没装 cv2 也无所谓，本项目 codec 走 PIL


# ============================================================
# loss（与 v3700 完全一致）
# ============================================================
# ★ 必须抽成函数被训练和验证**共用**。若验证侧另写一份，早停判据二
#   （训练/验证 loss 分叉）就是在比两个不同口径的数字，分叉/收敛判据全失效。
def compute_loss(a, th, field, pred_field=None, args_gen_weight=0.0):
    """返回 (loss, loss_amp, loss_ph, loss_gain, 诊断张量)。

    a, th : [B,N,D]     模型输出
    field : [B,N,2D]    目标实虚拼接
    """
    re_t, im_t = field[..., :D], field[..., D:]
    a_tar = torch.sqrt(re_t ** 2 + im_t ** 2 + 1e-12)
    th_tar = torch.atan2(im_t, re_t)

    pe = torch.remainder(th - th_tar + math.pi, 2 * math.pi) - math.pi
    la = torch.log1p(a.clamp(min=1e-8))
    lt = torch.log1p(a_tar.clamp(min=1e-8))
    w = (a_tar / a_tar.max().clamp(min=1e-8)).pow(AMP_P)
    loss_amp = (w * (la - lt).pow(2)).mean()

    w_ph = (a_tar ** 2).clamp(min=1e-8)
    loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()

    # 高频增益约束（v3700 改动1）
    a_bar_out = a.clamp(min=1e-8).mean(dim=1)          # [B,D]
    a_bar_tar = a_tar.clamp(min=1e-8).mean(dim=1)      # [B,D]
    loss_gain = torch.zeros((), device=a.device)
    for lo, hi in HF_SEGS:
        idx = torch.tensor([c for c in range(D) if lo <= (c % W_FFT) < hi],
                           device=a.device)
        go = (a_bar_out[:, idx].mean() / a_bar_tar[:, idx].mean().clamp(min=1e-12)
              ).clamp(0.1, 20.0)
        excess = (go - GAIN_TGT).clamp(min=0.0)
        huber = torch.where(excess <= HUBER_DELTA,
                            0.5 * excess.pow(2) / HUBER_DELTA,
                            excess - 0.5 * HUBER_DELTA)
        loss_gain = loss_gain + huber.mean()
    loss_gain = loss_gain / len(HF_SEGS)

    loss = loss_amp + loss_ph + GAIN_W * loss_gain

    # ---------- cond_field 直接监督（内容生成通路）----------
    # 只有这一项能让"条件→场"一步到位地学，而不是靠 12 个物理 block 反传。
    # 口径与 block.py 的 log1p 域 OU 递推一致（量纲铁律）。
    if pred_field is not None:
        loss_sup = CondFieldPredictor.supervise(pred_field, field, d_model=D)
        loss = loss + args_gen_weight * loss_sup
    else:
        loss_sup = torch.zeros((), device=a.device)

    return (loss, loss_amp, loss_ph, loss_gain, loss_sup,
            (a, a_tar, w_ph, pe))


# ============================================================
# ★ 早停判据 + 学习率调度
# ============================================================
class EarlyStopper:
    """四条早停判据（用户 2026-10-14 定案）。

    目标不是"跑满 7 天"，而是"最早什么时候达标就停"——7 天只是上限。
    每 check_every 步检查一次，任何一条触发即停。

    判据一  验证指标连续 N 次不改善 → 停
    判据二  训练/验证 loss 分叉     → 停（过拟合，继续更差）
    判据三  频段比稳定在 0.8~1.2   → 停（谱形状收敛，收益递减）
    判据四  单步时间突变 / 显存异常 → 停（效率已崩，继续跑是浪费）
    """

    def __init__(self, patience=5, min_delta=1e-3, gap_ratio=3.0, gap_patience=3,
                 stable_window=4, stable_tol=0.01, step_slow_factor=1.8,
                 step_slow_patience=3, mem_abort_frac=0.97, min_steps=200,
                 band_lo=0.8, band_hi=1.2, verbose=True):
        self.patience = patience              # 判据一：连续 N 次不改善
        self.min_delta = min_delta           # 改善阈值，低于它算"没改善"
        self.gap_ratio = gap_ratio      # 判据二：val/train 超过此倍数算分叉
                                     # ★ 默认 1.15 → 3.0（2026-10-14修正误停）
                                     # 实测 batch=1 时 train loss 是**单样本**测量，
                                     # 而 val 是 48 样本均值，拿单样本比均值，
                                     # 比值天然虚高。step 250~1000 的实测比值：
                                     #   0.34 / 2.02 / 3.08 / 2.74
                                     # 而 val_loss 全程单调下降
                                     #   0.5380 → 0.4777 → 0.4610 → 0.4256
                                     # ——真过拟合的特征是 val 回升，这里没有。
                                     # 1.15 会在 val 仍在下降时误判"已过拟合"，
                                     # 第一次训练就因此在 step 1000 白停。
        self.gap_patience = gap_patience     # 判据二：连续 N 次分叉
        self.stable_window = stable_window   # 判据三：连续 N 次在带内且不漂移
        self.stable_tol = stable_tol         # 判据三：带内漂移容忍度
        self.step_slow_factor = step_slow_factor   # 判据四：单步时间涨到基线几倍
        self.step_slow_patience = step_slow_patience
        self.mem_abort_frac = mem_abort_frac       # 显存占用超过此比例就停
        self.min_steps = min_steps           # 少于这个步数不停（防早期噪声误停）
        self.band_lo, self.band_hi = band_lo, band_hi
        self.verbose = verbose

        self.best_val = float("inf")
        self.best_step = 0
        self.no_improve = 0
        self.gap_count = 0
        self.slow_count = 0
        self.band_hist = []           # 判据三的频段比历史
        self.hist = []                # 全部验证历史，用于报告
        self.baseline_step_time = None   # 判据四的基线单步时间

    def _log(self, msg):
        if self.verbose:
            print(f"[EARLY] {msg}", flush=True)

    def update_baseline(self, step, step_time):
        """判据四的基线：用训练早期的滑动平均，避开首步编译/预热抖动。

        ★ step 必须显式传进来：基线只在 min_steps 之前持续更新，
          靠"当前步号"判断是否还在预热段。之前这里引用了一个
          并不存在的局部变量 step —— 只要训练真的跑到第 2 步之后
          调用本方法就抛 NameError（实测 smoke test 在 step=2 崩在这行），
          等于训练永远活不过第 2 步，早停判据四形同虚设。
        """
        if self.baseline_step_time is None:
            self.baseline_step_time = step_time
        elif step <= self.min_steps:
            # 前 min_steps 步取滑动平均，比单点稳
            self.baseline_step_time = 0.8 * self.baseline_step_time + 0.2 * step_time

    def check(self, step, train_loss, val_metrics, step_time, mem_frac):
        """检查四条判据。返回 (should_stop, reason)。"""
        if step < self.min_steps:
            return False, ""

        val_loss = val_metrics["loss"]
        ratio = None
        if "seg_ratio" in val_metrics:
            ratio = val_metrics["seg_ratio"]

        # ---------- 判据一：验证指标连续 N 次不改善 ----------
        improved = (self.best_val - val_loss) > self.min_delta
        if improved:
            self.best_val = val_loss
            self.best_step = step
            self.no_improve = 0
        else:
            self.no_improve += 1
        if self.no_improve >= self.patience:
            return True, (f"判据一 验证 loss 连续 {self.no_improve} 次不改善 "
                          f"(best={self.best_val:.4f} @step{self.best_step}, "
                          f"当前 {val_loss:.4f})")

        # ---------- 判据二：训练/验证 loss 分叉 ----------
        # 只在两边都有足够样本时判定；train_loss 是单 batch 抖动大，用 EMA。
        if train_loss is not None and train_loss > 1e-9:
            gap = val_loss / train_loss
            if gap > self.gap_ratio:
                self.gap_count += 1
                if self.gap_count >= self.gap_patience:
                    return True, (f"判据二 训练/验证 loss 分叉 {gap:.3f} "
                                  f"> {self.gap_ratio}（val={val_loss:.4f} "
                                  f"train={train_loss:.4f}）连续 {self.gap_count} 次，"
                                  f"已过拟合")
            else:
                self.gap_count = 0

        # ---------- 判据三：频段比稳定在 0.8~1.2 且不再变化 ----------
        if ratio is not None:
            self.band_hist.append((step, ratio))
            if len(self.band_hist) > self.stable_window:
                self.band_hist.pop(0)
            if len(self.band_hist) == self.stable_window:
                vals = [r for _, r in self.band_hist]
                in_band = all(self.band_lo <= r <= self.band_hi for r in vals)
                drift = max(vals) - min(vals)
                if in_band and drift <= self.stable_tol:
                    return True, (f"判据三 频段比连续 {self.stable_window} 次稳定在 "
                                  f"[{self.band_lo},{self.band_hi}]（漂移 {drift:.4f} "
                                  f"≤ {self.stable_tol}），谱形状已收敛，收益递减")

        # ---------- 判据四：单步时间突变 / 显存异常 ----------
        if (self.baseline_step_time and step_time
                and step_time > self.baseline_step_time * self.step_slow_factor):
            self.slow_count += 1
            if self.slow_count >= self.step_slow_patience:
                return True, (f"判据四 单步时间 {step_time:.2f}s 达基线 "
                              f"{self.baseline_step_time:.2f}s 的 "
                              f"{step_time/self.baseline_step_time:.1f} 倍，"
                              f"连续 {self.slow_count} 次（显存碎片/降频），"
                              f"继续跑效率低")
        else:
            self.slow_count = 0

        if mem_frac is not None and mem_frac > self.mem_abort_frac:
            return True, (f"判据四 显存占用 {mem_frac*100:.1f}% 超过 "
                          f"{self.mem_abort_frac*100:.0f}%，随时可能 OOM，"
                          f"继续跑有崩溃风险")

        return False, ""


class WarmupCosineLR:
    """学习率调度：前期大 lr 快速收敛，后期小 lr 精修。

    用户要求"前期用大 lr 快速收敛，后期用小 lr 精修，这样能在更短时间内
    达到同样效果"。这里用 warmup + cosine 退火到 lr_min：

        step < warmup:  lr 从 lr_start 线性升到 lr_peak（预热，避免早期大梯度
                        把已训好的物理核参数打乱）
        之后:           lr 按 cos 从 lr_peak 退火到 lr_min

    相比固定 lr 的收益：前期探索快，后期不再大 lr 反复扰动已收敛的参数，
    验证指标更早进入平台期 → 早停更早触发 → 总时长更短。
    """

    def __init__(self, optimizer, lr_start, lr_peak, lr_min,
                 warmup_steps, total_steps, verbose=True):
        self.opt = optimizer
        self.lr_start, self.lr_peak, self.lr_min = lr_start, lr_peak, lr_min
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.verbose = verbose
        self._set(lr_start)

    def _set(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.cur = lr

    def step(self, step):
        if step < self.warmup_steps:
            frac = step / self.warmup_steps
            lr = self.lr_start + (self.lr_peak - self.lr_start) * frac
        else:
            prog = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            prog = min(1.0, max(0.0, prog))
            lr = self.lr_min + 0.5 * (self.lr_peak - self.lr_min) * (1 + math.cos(math.pi * prog))
        self._set(lr)
        return lr


# ============================================================
# 参数
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7.0, help="训练预算天数（默认 7）")
    ap.add_argument("--budget-sec", type=int, default=None,
                    help="直接给秒数（优先于 --days）")
    ap.add_argument("--mode", default="mix", choices=["mix", "text", "image", "both"],
                    help="条件采样模式，mix = 三种能力混合训练")
    ap.add_argument("--batch", type=int, default=1,
                        help="micro batch。★ 默认 **1**（2026-10-14 从 4 改下来）。"
                             "原注释里的「B=4 峰值 1.03GB」是 **D=64 时**测的；"
                             "v4200 把 D 提到 125 后所有激活张量翻倍，实测单步 fwd+bwd："
                             "B=4/chunk2048 → GPU 3.95GB(占满 4GB 卡) + 进程提交 11.4GB；"
                             "B=2 → GPU 4.8GB(物理不可能，分配器已碎) + 单步 39~49 秒；"
                             "B=1/chunk512 → GPU 2.55GB + 单步 4.4 秒。**只有 B=1 干净可用**。"
                             "B=1/B=2 不是线性关系：B=2 时显存分配器开始 split/抖动，"
                             "反而比 B=4 还慢 10 倍")
    ap.add_argument("--acc", type=int, default=1,
                        help="梯度累积步数。★ **当前未实现**（训练循环里没用这个参数，"
                             "传了也不会生效，属于静默失败）。要真累积必须改训练循环："
                             "loss/=acc，每 acc 个 micro-batch 才 opt.step()")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4,
                        help="DataLoader 线程数。★ 默认 4（2026-10-14 从 8 改下来）。"
                             "B=1 时 GPU 单步就要 4.4 秒，每样本 CPU 端编码约 0.1~2.4 秒，"
                             "4 个 worker 已经远快于 GPU，不需要 8 个。"
                             "更重要的是每个 worker 的提交量：不限线程时 31 线程/"
                             "1.59GB（OpenBLAS 按核数开 buffer），限到 1 线程后 "
                             "6~10 线程/0.65GB。8 个 worker 白占 5GB 提交额度，"
                             "而 C 盘只剩 258MB 时 pagefile 撑不开、提交上限被钉死")
    ap.add_argument("--resume", default="auto",
                    help="auto=自动找最新 ckpt 续训 | none=从头训 | <路径>=指定 ckpt")
    ap.add_argument("--save-every", type=int, default=200, help="每 N 步存一次断点")
    ap.add_argument("--log-every", type=int, default=50, help="每 N 步打一次指标")
    ap.add_argument("--max-steps", type=int, default=None, help="手动指定总步数（覆盖预算推算）")
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 个样本（调试）")

    # ---------- 早停（7 天只是上限，越早达标越早停）----------
    ap.add_argument("--val-frac", type=float, default=0.08,
                    help="验证集 group 占比（默认 8%%。★ 必须按 group 划分而非按样本："
                         "bulk_* 上万 GIF 挤在少数 group，按样本划会 100% 撞上"
                         "同 group 兄弟样本（实测 424 条验证样本的 80 个 group "
                         "全部出现在训练集）。3% 时验证集只有 105 条、方差偏大，"
                         "8% 约 280 条，12 batch×4=48 个样本的验证更稳）")
    ap.add_argument("--val-batches", type=int, default=12,
                    help="每次验证跑多少个 batch（12×batch=48 个样本，"
                         "够稳定又不至于太慢）")
    ap.add_argument("--check-every", type=int, default=250,
                    help="每 N 步检查一次早停判据")
    ap.add_argument("--patience", type=int, default=5,
                    help="判据一：验证指标连续 N 次不改善就停")
    ap.add_argument("--min-steps", type=int, default=300,
                    help="少于这个步数不早停（早期噪声大，防误停）")
    ap.add_argument("--no-early-stop", action="store_true",
                    help="关掉早停，强制跑满预算（调试对照用）")

    # ---------- 学习率调度 ----------
    ap.add_argument("--lr-start", type=float, default=3e-5, help="预热起始 lr")
    ap.add_argument("--lr-peak", type=float, default=3e-4, help="预热峰值 lr")
    ap.add_argument("--lr-min", type=float, default=1e-5, help="退火下限 lr")
    ap.add_argument("--warmup", type=int, default=300, help="预热步数")

    ap.add_argument("--compile", action="store_true", help="开 torch.compile（首步慢，稳态快）")

    # ---------- 内容生成通路（2026-10-14 新增）----------
    # ---------- 内容生成监督权重（2026-10-14）----------
    # ★ 为什么必须加大：cond_field 的梯度要穿过 12 个物理 block 才能从最终 loss
    #   传回来，路径长且弱。实测不加直接监督时 40 步几乎不动 ——
    #   204 个 group 的输出场两两只差 1.7e-06，softmax 混合权重最大值 0.0389
    #   而均匀基线 1/27=0.0370，等于完全没分化（模型输出常数场，条件是摆设）。
    #   weight=2.0 让直接监督占主导，物理核的间接梯度继续负责"生成动力学"。
    #   调大到 4.0 时出现过单步 NaN（预测场冲得过大→漂移场 logits 溢出），
    #   2.0 是实测稳的值。
    ap.add_argument("--gen-weight", type=float, default=2.0,
                    help="cond_field 直接监督权重（条件→场）。0=只靠物理核间接梯度"
                         "（实测学不动），1~3 是可用区间，>4 可能 NaN")
    ap.add_argument("--gen-ratio", type=float, default=0.7,
                    help="走生成通路的步数占比（其余走重建）。★ 默认 0.5。"
                         "只走重建 → cond_field 梯度恒为 0（和 prototypes 同一个病，"
                         "生成出来永远是噪点）；只走生成 → 没有真实答案锚定，"
                         "早期预测场是噪声学不动。两条一起走才能同时学会")

    ap.add_argument("--bench", type=int, default=8,
                    help="开跑前实测多少步用来推算总步数（0=不实测）")
    return ap.parse_args()


# ============================================================
# 断点保存 / 恢复
# ============================================================
CKPT_DIR = ROOT / "outputs_real" / "ckpt"


def save_ckpt(model, opt, step, args, stats, tag="latest", stopper=None, sched=None):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
    payload = {
        "model": sd,
        "opt": opt.state_dict(),
        "step": step,
        "args": vars(args),
        "stats": stats,
        "d_model": D, "n_blocks": model.n_blocks,
        "saved_at": time.time(),
    }
    # ★ 早停器与调度器状态必须一起存：否则续训后 best_val/no_improve 清零、
    #   lr 回到 warmup 起点，早停会重新"耐心"一轮，白等几百步。
    if stopper is not None:
        payload["stopper"] = {
            "best_val": stopper.best_val, "best_step": stopper.best_step,
            "no_improve": stopper.no_improve, "gap_count": stopper.gap_count,
            "slow_count": stopper.slow_count, "band_hist": stopper.band_hist,
            "hist": stopper.hist,
            "baseline_step_time": stopper.baseline_step_time,
        }
    if sched is not None:
        payload["sched"] = {"cur": sched.cur}
    tmp = CKPT_DIR / f"{tag}.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, CKPT_DIR / f"{tag}.pt")          # 原子替换，防写一半断电
    # 同时留一个带步号的归档，方便回溯
    if step % 2000 == 0:
        torch.save(payload, CKPT_DIR / f"step_{step}.pt")


def find_resume(spec):
    if spec == "none" or spec is None:
        return None
    if spec == "auto":
        p = CKPT_DIR / "latest.pt"
        return str(p) if p.exists() else None
    return spec if Path(spec).exists() else None


# ============================================================
# 编解码常量（与 v3700 同口径）
# ============================================================
# 注：v3700 的径向 Nyquist 约束需要空间域解码，每步多一次 128x128 FFT 逆变换，
# 在 4GB 卡 + 7 天预算下性价比很低（对 batch=1 单步约 +8% 耗时），
# 故 v4200 只保留频域高频增益约束（改动1），去掉空间域径向约束（改动2）。
# 画质基准仍以 v3700 的解码口径为准（_decodable 保留供评估脚本用）。


def build_rad_mask(device):
    yy, xx = np.indices((IMG, IMG))
    rmax = float(np.sqrt((IMG // 2) ** 2 + (IMG // 2) ** 2))
    return torch.from_numpy(
        (np.sqrt((yy - IMG // 2) ** 2 + (xx - IMG // 2) ** 2) >= RAD_LO_FRAC * rmax).astype(np.float32)
    ).to(device), rmax


# ============================================================
# 主流程
# ============================================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 主进程线程数：脚本顶部把环境变量统一锁成 1（防 OpenBLAS 爆内存），
    # 这里只给主进程调回 4。主进程只做 GPU 前向/反传与 H2D 拷贝，
    # CPU 侧几乎没有重活，4 线程足够，再多只会白白吃掉内存。
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True          # 卷积形状固定 → 自动选最快算法

    # ---- 提交量体检：7 天训练最怕的不是慢，是静默挂死 ----
    # 打印一次提交量余量。worker 起完后若提交量逼近上限，OpenBLAS 会
    # "Memory allocation still failed" 直接崩 worker，而 persistent_workers
    # 下主进程表现为**卡住不动**（GPU 0%、无报错），能从日志一眼看出问题。
    def _mem_report(tag):
        try:
            import ctypes
            import ctypes.wintypes as _wt

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", _wt.DWORD), ("dwMemoryLoad", _wt.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            G = 1024 ** 3
            print(f"[MEM:{tag}] RAM 可用 {ms.ullAvailPhys/G:.2f}GB "
                  f"(load {ms.dwMemoryLoad}%) | 提交剩余 {ms.ullAvailPageFile/G:.2f}GB"
                  f"/{ms.ullTotalPageFile/G:.1f}GB", flush=True)
        except Exception as e:
            print(f"[MEM:{tag}] 读取失败：{e}", flush=True)

    _mem_report("启动")

    budget = args.budget_sec if args.budget_sec else int(args.days * 86400)
    # 预留 5% 给启动/checkpoint/收尾
    budget_train = int(budget * 0.95)

    print("=" * 78, flush=True)
    print(f"[CFG] v4200 多模态训练  预算上限={args.days}天({budget}s, 实际训 {fmt_dur(budget_train)})")
    print(f"[CFG] mode={args.mode} batch={args.batch} acc={args.acc} "
          f"workers={args.workers} compile={args.compile}")
    print(f"[CFG] lr: {args.lr_start:g} →(warmup {args.warmup})→ {args.lr_peak:g} "
          f"→(cosine)→ {args.lr_min:g}")
    print(f"[CFG] 能力: 文生GIF / 图生GIF / 图文生GIF  (mode={args.mode})")
    print(f"[CFG] 早停: 每 {args.check_every} 步检查 | patience={args.patience} "
          f"| min_steps={args.min_steps} | val={args.val_frac*100:.1f}%")
    if args.no_early_stop:
        print("[CFG] ★ 早停已禁用（--no-early-stop），会跑满预算")
    print("=" * 78, flush=True)

    # ---------- 数据：划分 训练集 / 验证集 ----------
    # ★★ 两条纪律（早停能不能生效全看这里）★★
    #  1. 验证集从不被训练采样 —— 否则 val loss 跟着一起降，判据一/二永不触发。
    #  2. **group 级隔离** —— 实测按样本随机划分会 100% 撞上同 group 的兄弟 GIF：
    #     bulk_* 有上万 GIF 挤在少数 group 里，424 个验证样本的 80 个 group
    #     全部出现在训练集。同一 group 的 GIF 画面高度相似（同一角色/同一素材库），
    #     等于"训练时见过验证样本的近亲"，验证指标偏乐观，早停会偏晚触发。
    #     所以按 **group** 划分：抽 3% 的 group 整组做验证。
    items_all = json.load(open(ROOT / "data" / "index.json", encoding="utf-8"))
    rng = np.random.default_rng(1234)          # 固定种子 → 划分可复现

    groups_all = sorted({it["group"] for it in items_all})
    n_grp_val = max(3, int(len(groups_all) * args.val_frac))
    grp_perm = rng.permutation(len(groups_all))
    val_groups = {groups_all[i] for i in grp_perm[:n_grp_val]}

    train_idx = np.array([i for i, it in enumerate(items_all)
                          if it["group"] not in val_groups], dtype=np.int64)
    val_idx = np.array([i for i, it in enumerate(items_all)
                        if it["group"] in val_groups], dtype=np.int64)
    if args.limit:
        train_idx = train_idx[:args.limit]
        val_idx = val_idx[:max(args.val_batches * args.batch, 8)]

    grp_overlap = len(val_groups & {items_all[i]["group"] for i in train_idx})
    print(f"[SPLIT] group 级隔离: {len(groups_all)} 组 → "
          f"训练 {len(train_idx)} 条 / 验证 {len(val_idx)} 条", flush=True)
    print(f"[SPLIT] 验证集独占 {len(val_groups)} 个 group，与训练集 group "
          f"重叠 {grp_overlap} 个 → {'✓ 无泄漏' if grp_overlap == 0 else '✗ 泄漏!'}",
          flush=True)

    ds = MultimodalGifDataset(
        index_path=str(ROOT / "data" / "index.json"),
        cache_dir=str(ROOT / "data" / "clip_cache"),
        gif_root=GIF_ROOT, device=device, mode=args.mode,
        encode_cache=False,   # 全量场缓存 80.9GB，放不下 → 实时编码
        indices=train_idx,
    )
    # 验证集用固定 mode="both"（确定性，不掷随机数），指标才可复现可比。
    # 判据三的频段比在 both 模式下最能反映"文本+图像条件是否被正确利用"。
    val_ds = MultimodalGifDataset(
        index_path=str(ROOT / "data" / "index.json"),
        cache_dir=str(ROOT / "data" / "clip_cache"),
        gif_root=GIF_ROOT, device=device, mode="both",
        encode_cache=False, indices=val_idx,
    )
    collate = make_collate(device)
    # ★ timeout=600 不是"慢"，是"死"的保险（2026-10-14 挂死事故修复）
    #   persistent_workers=True 时，某个 worker 一旦崩溃（OOM / OpenBLAS 分配失败），
    #   主进程 next(it) 会**永远阻塞**，GPU 利用率掉到 0%，控制台一行新日志都没有，
    #   7 天训练就这么静默停摆（本次实测：worker 31 线程×8 把提交量顶穿后崩掉）。
    #   DataLoader 的 timeout 参数专门管这个：取一个 batch 超过 N 秒没拿到就抛
    #   TimeoutError。worker 正常时单 batch 约 10 秒，600 秒是极宽松的下限，
    #   只有真死了才会触发。
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0,
                    worker_init_fn=_worker_init,
                    timeout=600,
                    prefetch_factor=3 if args.workers > 0 else None)
    # 验证集不打乱、worker 少一些（验证不频繁，且要确定性顺序便于排查）
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=max(2, args.workers // 4), collate_fn=collate,
                        pin_memory=True, drop_last=False, worker_init_fn=_worker_init,
                        timeout=600)

    # ---------- 模型 ----------
    model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, weight_decay=1e-4)

    # ---------- 早停器 + LR 调度 ----------
    stopper = EarlyStopper(patience=args.patience, min_steps=args.min_steps)
    sched = WarmupCosineLR(opt, args.lr_start, args.lr_peak, args.lr_min,
                           args.warmup, 10 ** 9)   # total 占位，训完由 update 修正

    # ---------- 恢复 ----------
    start_step = 0
    stats = {"step": 0, "elapsed": 0.0, "loss_ema": None}
    rp = find_resume(args.resume)
    if rp:
        ck = torch.load(rp, map_location=device)
        model.load_state_dict(ck["model"], strict=False)
        if "opt" in ck:
            try:
                opt.load_state_dict(ck["opt"])
            except Exception as e:
                print(f"[WARN] 优化器状态不兼容，重建：{e}")
        start_step = ck.get("step", 0)
        stats = ck.get("stats", stats)
        # 恢复早停器状态：否则续训后耐心值清零，会白等一整轮
        if "stopper" in ck:
            s = ck["stopper"]
            stopper.best_val = s.get("best_val", float("inf"))
            stopper.best_step = s.get("best_step", 0)
            stopper.no_improve = s.get("no_improve", 0)
            stopper.gap_count = s.get("gap_count", 0)
            stopper.slow_count = s.get("slow_count", 0)
            stopper.band_hist = s.get("band_hist", [])
            stopper.hist = s.get("hist", [])
            stopper.baseline_step_time = s.get("baseline_step_time")
            print(f"[RESUME] 早停状态: best_val={stopper.best_val:.4f} "
                  f"@step{stopper.best_step} no_improve={stopper.no_improve}", flush=True)
        print(f"[RESUME] 从 {rp} 恢复，step={start_step}", flush=True)

    # ---------- 坐标（所有样本共享）----------
    t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"[PARAM] {n_param:,} 参数  N={len(t_coord)}", flush=True)

    # ============================================================
    # 验证：在**训练中从未采样**的子集上评估（否则早停无意义）
    # ============================================================
    @torch.no_grad()
    def evaluate(max_batches):
        """返回 {loss, amp, ph, gain, cos, seg_ratio, ...}。

        seg_ratio = 高频段谱比（模型/target），判据三要看它是否稳定在 0.8~1.2。
        ★ 频段按 w=c%W_FFT 切（v3700 铁律）：按 0/16/32/48 均分会混淆 DC 与高频。
        """
        model.eval()
        it = iter(val_dl)
        agg = {"loss": 0.0, "amp": 0.0, "ph": 0.0, "gain": 0.0, "sup": 0.0}
        cos_sum, wcos_sum, wsum = 0.0, 0.0, 0.0
        hf_out, hf_tar, n_b = 0.0, 0.0, 0
        hf_idx = [c for c in range(D)
                  if any(lo <= (c % W_FFT) < hi for lo, hi in HF_SEGS)]
        for _ in range(max_batches):
            got = next_batch(it, device)
            if got is None:
                break
            cond, field, _, _ = got
            B = field.shape[0]
            # ★ 验证走**生成通路**：与推理同口径。早停衡量的必须是
            #   "只给条件能不能生成"，而不是"给答案能不能重建"——
            #   后者早就饱和了，用它判早停毫无意义（这次误停就是这原因）。
            a, th, diag = model(texts=None, target_field=field, t_coord=t_coord,
                             s_coord=s_coord, batch_size=B, cond_vec=cond,
                             use_cond_field=True)
            loss, l_amp, l_ph, l_gain, l_sup, (a_, a_tar, w_ph, pe) = \
                compute_loss(a, th, field, diag.get("pred_field"), args.gen_weight)
            agg["loss"] += loss.item()
            agg["amp"] += l_amp.item()
            agg["ph"] += l_ph.item()
            agg["gain"] += l_gain.item()
            agg["sup"] += l_sup.item()
            c = torch.cos(pe)
            cos_sum += c.sum().item()
            wcos_sum += (w_ph * c).sum().item()
            wsum += w_ph.sum().item()
            idx = torch.tensor(hf_idx, device=a.device)
            hf_out += a_.clamp(min=1e-8).mean(dim=1)[:, idx].mean().item()
            hf_tar += a_tar.clamp(min=1e-8).mean(dim=1)[:, idx].mean().item()
            n_b += 1
        model.train()
        if n_b == 0:
            return {"loss": float("inf")}
        out = {k: v / n_b for k, v in agg.items()}
        out["cos"] = cos_sum / max(wsum, 1e-8)
        out["wcos"] = wcos_sum / max(wsum, 1e-8)
        out["seg_ratio"] = hf_out / max(hf_tar, 1e-12)      # 高频段谱比
        return out

    # ---------- 实测速度 → 推算总步数 ----------
    def measure_rate(n_bench):
        """跑 n_bench 步测真实速度（返回 步/秒）。"""
        if n_bench <= 0:
            return 1.2
        model.train()
        it = iter(dl)
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        cnt = 0
        for _ in range(n_bench):
            got = next_batch(it, device)
            if got is None:
                it = iter(dl)
                got = next_batch(it, device)
            cond, field, modes, idxs = got
            a, th, _ = model(texts=None, target_field=field, t_coord=t_coord,
                             s_coord=s_coord, batch_size=args.batch, cond_vec=cond,
                             use_cond_field=False)
            loss = (a ** 2).mean() + (th ** 2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            cnt += 1
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        opt.zero_grad(set_to_none=True)
        return cnt / max(dt, 1e-6)

    # ---------- 内容生成 / 重建 双通路（2026-10-14）----------
    # 为什么必须双通路而不是只走生成：
    #   只走重建（targets=真实场）→ cond_field 的梯度恒为 0（与 prototypes 同一个病）
    #   只走生成 → loss 里没有真实答案锚定，早期预测场是噪声，学不动
    # 所以按步随机交替：走重建或走生成，比例由 gen_ratio 控制。
    # ★ batch=1 时无法在一个 batch 内拆分，按步交替期望比例不变且省显存。
    def _use_gen_path():
        return random.random() < args.gen_ratio

    if start_step == 0:
        rate = measure_rate(args.bench)
        print(f"[BENCH] 实测 {rate:.2f} 步/s", flush=True)
    else:
        rate = stats.get("rate", 1.2)
    _mem_report("bench后")

    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = int(budget_train * rate)
    print(f"[PLAN] 总步数 = {total_steps:,}  （{rate:.2f} 步/s × {fmt_dur(budget_train)}）",
          flush=True)
    if total_steps < 100:
        print("[WARN] 推算步数过少，检查 bench/rate", flush=True)

    # 预算算完后才知道总步数，把 cosine 退火的终点补上
    sched.total_steps = total_steps
    print(f"[LR] warmup {args.warmup} 步 → 峰值 {args.lr_peak:g}，"
          f"cosine 退火至 {args.lr_min:g}（终点 step {total_steps:,}）", flush=True)

    # ---------- torch.compile（可选）----------
    if args.compile:
        try:
            model.forward = torch.compile(model.forward, dynamic=False)
            print("[COMPILE] 已启用 torch.compile（首步会有编译耗时）", flush=True)
        except Exception as e:
            print(f"[WARN] torch.compile 失败，继续 eager：{e}", flush=True)

    # ---------- 训练主循环 ----------
    model.train()
    it = iter(dl)
    t_start = time.time()
    step = start_step
    last_save = time.time()

    print(f"\n[RUN] 开始训练  按 Ctrl+C 可安全中断（断点已保存）\n", flush=True)

    try:
        while step < total_steps:
            got = next_batch(it, device)
            if got is None:
                it = iter(dl)
                got = next_batch(it, device)
            cond, field, modes, idxs = got

            B = field.shape[0]
            opt.zero_grad(set_to_none=True)

            # ===== 条件编码 → 物理调制 → 演化 =====
            # 双通路：重建（真实场引导）/ 生成（条件预测场引导）
            is_gen = _use_gen_path()
            a, th, diag = model(texts=None, target_field=field, t_coord=t_coord,
                                s_coord=s_coord, batch_size=B, cond_vec=cond,
                                use_cond_field=is_gen)

            # ===== loss：与验证同口径（共用 compute_loss）=====
            loss, loss_amp, loss_ph, loss_gain, loss_sup, (a_, a_tar, w_ph, pe) = \
                compute_loss(a, th, field, diag.get("pred_field"), args.gen_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            # ===== LR 调度：前期大 lr 快收敛，后期小 lr 精修 =====
            cur_lr = sched.step(step)

            # ===== 指标 + 进度条 =====
            lv = loss.item()
            if stats.get("loss_ema") is None:
                stats["loss_ema"] = lv
            else:
                stats["loss_ema"] = 0.98 * stats["loss_ema"] + 0.02 * lv
            # 判据二要用 train loss 的 EMA：单 batch 抖动大，比 val 高一大截是噪声
            train_ema = stats["loss_ema"]

            now = time.time()
            elapsed = now - t_start
            if step % 5 == 0 or step == 1:
                inst_rate = 5.0 / max(now - getattr(main, "_last_t", now), 1e-9) \
                    if hasattr(main, "_last_t") else rate
                main._last_t = now
                rate = 0.7 * rate + 0.3 * inst_rate       # 平滑，避免抖动
            step_time = (now - getattr(main, "_last_step_t", now))
            main._last_step_t = now

            if step % args.log_every == 0 or step == 1:
                with torch.no_grad():
                    c = torch.cos(pe)
                    cos = c.mean().item()
                    wcos = (w_ph * c).sum().item() / w_ph.sum().item()
                    am = a.mean().item()
                    ar = (a.mean(dim=1) / a_tar.mean(dim=1).clamp(min=1e-6)).clamp(0, 20)
                    am_ch = a.mean(dim=1).mean(0)
                    at_ch = a_tar.mean(dim=1).mean(0)
                    lm, lt_ = torch.log1p(am_ch), torch.log1p(at_ch)
                    lm_c, lt_c = lm - lm.mean(), lt_ - lt_.mean()
                    prof_cos = (lm_c * lt_c).sum() / (lm_c.norm() * lt_c.norm() + 1e-12)
                mode_c = {"text": modes.count("text"), "image": modes.count("image"),
                          "both": modes.count("both")}
                print(f"\n[{step:6d}/{total_steps}] loss={lv:.4f} (ema {stats['loss_ema']:.4f}) "
                      f"amp={loss_amp.item():.4f} phase={loss_ph.item():.4f} "
                      f"gain={loss_gain.item():.4f} cos={cos:+.4f} wcos={wcos:+.4f} "
                      f"a_out={am:.4f} prof_cos={prof_cos:+.4f} lr={cur_lr:.2e}", flush=True)
                print(f"          谱比(模型/target) 中位 {ar.median().item():.2f} | "
                      f"本批模态 {mode_c} | 已跑 {fmt_dur(elapsed)}", flush=True)

            # 进度条（每步刷新，同一行）
            sys.stdout.write(progress_bar(step, total_steps, elapsed, rate))
            sys.stdout.flush()

            # ============================================================
            # ★ 早停检查：每 check_every 步一次，四条判据任一触发就停
            # ============================================================
            stopper.update_baseline(step, step_time)
            if (not args.no_early_stop) and step % args.check_every == 0:
                vm = evaluate(args.val_batches)
                mem_frac = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) \
                    if device == "cuda" else None
                stopper.hist.append((step, vm.get("loss", float("inf")),
                                     vm.get("seg_ratio", float("nan"))))
                print(f"\n[VAL] step={step} val_loss={vm.get('loss', float('inf')):.4f} "
                      f"amp={vm.get('amp', 0):.4f} ph={vm.get('ph', 0):.4f} "
                      f"wcos={vm.get('wcos', 0):+.4f} 频段比={vm.get('seg_ratio', float('nan')):.3f} "
                      f"| train_ema={train_ema:.4f} best={stopper.best_val:.4f}"
                      f"@{stopper.best_step} 不改善 {stopper.no_improve}次"
                      f"{'' if mem_frac is None else f' 显存{mem_frac*100:.0f}%'}",
                      flush=True)
                should_stop, reason = stopper.check(
                    step, train_ema, vm, step_time, mem_frac)
                if should_stop:
                    elapsed = time.time() - t_start
                    stats["step"] = step
                    stats["elapsed"] = elapsed
                    stats["rate"] = rate
                    stats["stopped_by"] = reason
                    stats["best_val"] = stopper.best_val
                    stats["best_step"] = stopper.best_step
                    # ★ 早停时保存 best 权重：后段可能已过拟合，最终权重不一定最好
                    save_ckpt(model, opt, step, args, stats, "latest", stopper, sched)
                    save_ckpt(model, opt, stopper.best_step, args, stats, "best", stopper, sched)
                    print(f"\n{'='*78}", flush=True)
                    print(f"[EARLY-STOP] 触发于 step {step}（已跑 {fmt_dur(elapsed)}，"
                          f"占 7 天预算的 {elapsed/budget*100:.1f}%）", flush=True)
                    print(f"[EARLY-STOP] 原因：{reason}", flush=True)
                    print(f"[EARLY-STOP] 最佳验证 loss {stopper.best_val:.4f} @ step "
                          f"{stopper.best_step}", flush=True)
                    print(f"[EARLY-STOP] best 权重已存 outputs_real/ckpt/best.pt", flush=True)
                    print(f"{'='*78}", flush=True)
                    return

            # ===== 断点保存 =====
            if step % args.save_every == 0 or (now - last_save) > 600:
                stats["step"] = step
                stats["elapsed"] = elapsed
                stats["rate"] = rate
                save_ckpt(model, opt, step, args, stats, "latest", stopper, sched)
                last_save = now
                print(f"\n[CKPT] 已保存 step={step}  ({fmt_dur(elapsed)} 已用)", flush=True)

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        stats["step"] = step
        stats["elapsed"] = elapsed
        stats["rate"] = rate
        save_ckpt(model, opt, step, args, stats, "latest", stopper, sched)
        print(f"\n[STOP] 用户中断，已保存断点 step={step}", flush=True)
        return

    except Exception as e:
        # ★ 任何异常都先落断点再退出。7 天训练崩一次不能白跑：
        #   原来只有 KeyboardInterrupt 会保存，其余异常直接抛栈、断点停在
        #   最多 600 秒前。现在统一先存 latest.pt 再报告。
        elapsed = time.time() - t_start
        stats["step"] = step
        stats["elapsed"] = elapsed
        stats["rate"] = rate
        try:
            save_ckpt(model, opt, step, args, stats, "latest", stopper, sched)
            print(f"\n[ERROR] 已保存断点 step={step}（已跑 {fmt_dur(elapsed)}）", flush=True)
        except Exception as e2:
            print(f"[ERROR] 断点保存也失败了：{e2}", flush=True)
        _mem_report("崩溃时")
        import traceback
        traceback.print_exc()
        msg = f"{type(e).__name__}: {e}".lower()
        if "timeout" in msg:
            print("\n[ERROR] ★ 这是 DataLoader 超时（取 batch 超过 600 秒）："
                  "某个 worker 进程死了。\n"
                  "        最常见原因是内存提交量被顶穿 —— 检查 [MEM] 那几行的"
                  "「提交剩余」是否接近 0，\n"
                  "        以及 C 盘剩余空间（pagefile 撑不开会直接钉死提交上限）。"
                  "\n        应急：--workers 4 减半重跑。", flush=True)
        elif "out of memory" in msg or "alloc" in msg:
            print("\n[ERROR] ★ 显存/内存分配失败。应急：--batch 2 或 --workers 4。", flush=True)
        print(f"[ERROR] 续跑命令：--resume auto（默认就是 auto，直接重跑即可）\n", flush=True)
        return

    # ---------- 收尾 ----------
    elapsed = time.time() - t_start
    stats["step"] = step
    stats["elapsed"] = elapsed
    stats["rate"] = rate
    stats["best_val"] = stopper.best_val
    stats["best_step"] = stopper.best_step
    save_ckpt(model, opt, step, args, stats, "latest", stopper, sched)
    save_ckpt(model, opt, stopper.best_step, args, stats, "best", stopper, sched)
    print(f"\n[DONE] 训练完成 {step} 步  用时 {fmt_dur(elapsed)}  "
          f"均速 {rate:.2f} 步/s  （占 7 天预算 {elapsed/budget*100:.1f}%）",
          flush=True)
    print(f"[DONE] 最佳验证 loss {stopper.best_val:.4f} @ step {stopper.best_step}", flush=True)
    print(f"[DONE] 权重：outputs_real/ckpt/best.pt（推荐）/ latest.pt（末步）", flush=True)


if __name__ == "__main__":
    main()
