# 10 · train_multimodal 训练脚本源码逐行分析

源文件：`docx/实验记录/versions/v4200_多模态CLIP产品版/scripts/train_multimodal.py`（1112行）

---

## 1. 文件头声明（第1-92行）

```python
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
"""
```

---

## 2. 线程数锁定（第102-123行）

```python
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
```

崩溃实测数据：
- 主进程 44 线程 / 每个 worker 31 线程
- 每 worker 提交量 1.59GB
- 8 worker + 主进程 ≈ 13GB
- 系统提交上限 36.7GB，顶穿后只剩 0.74GB
- 16 逻辑核 × 8 worker = 128 线程

---

## 3. 路径与导入（第125-139行）

```python
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
```

---

## 4. 编解码与 loss 口径常量（第141-149行）

```python
# ---------- 编解码口径：与 v3700 逐位一致（改这里 = 改画质基准）----------
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
H_FFT, W_FFT = WIN, (PATCH * PATCH * 3) // 2 + 1

# ---------- loss 超参：与 v3700 一致 ----------
AMP_P = 0.5
HF_SEGS = [(12, 19), (19, 25)]
GAIN_TGT, GAIN_W, HUBER_DELTA = 1.0, 0.5, 0.5
RAD_LO_FRAC, RAD_CAP, RAD_W = 0.80, 1.2, 1.0
```

- D=125：rfft2 真实网格
- W_FFT=25：(4×4×3)//2+1 = 48//2+1 = 25
- HF_SEGS：高频段按 w=c%25 切，段 (12,19) 和 (19,25)
- AMP_P=0.5：振幅 loss 权重指数
- GAIN_TGT=1.0：高频增益目标
- GAIN_W=0.5：高频增益 loss 权重
- HUBER_DELTA=0.5
- RAD_*：径向 Nyquist 约束参数（v4200 未使用，见第618-622行注释）

---

## 5. 数据根解析（第151-168行）

```python
# 数据根：v4200 在 D:/AetherGIF-V1/docx/实验记录/versions/v4200_.../、要往上退 4 级。
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
    raise SystemExit(...)
```

历史坑：preencode.py 曾把路径写成退 3 级，导致 14145 张全部静默跳过。

---

## 6. 工具函数（第174-212行）

### 6.1 `fmt_dur`（第174-183行）

时间格式化：秒 → 「X天HH:MM:SS」或「H:MM:SS」或「MM:SS」。

### 6.2 `progress_bar`（第186-197行）

```python
def progress_bar(step, total, elapsed, rate, width=32):
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
```

### 6.3 `next_batch`（第200-212行）

```python
def next_batch(it, device):
    try:
        cond, field, modes, idxs = next(it)
    except StopIteration:
        return None
    cond = cond.to(device, non_blocking=True)
    field = field.to(device, non_blocking=True)
    return cond, field, modes, idxs
```

H2D 搬运，pin_memory + non_blocking。

---

## 7. DataLoader worker 初始化（第218-239行）

```python
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
        pass
```

- Windows DataLoader 用 spawn 不是 fork
- worker 重新 import 本模块，环境变量继承，但 torch intra-op 线程池在 import 时已建
- worker_init_fn 里再调 set_num_threads(1) 双保险
- 实测：锁单线程后 bench 速率不降，提交量从 1.59GB 降到几百 MB

---

## 8. `compute_loss` 函数（第247-294行）

```python
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
    if pred_field is not None:
        loss_sup = CondFieldPredictor.supervise(pred_field, field, d_model=D)
        loss = loss + args_gen_weight * loss_sup
    else:
        loss_sup = torch.zeros((), device=a.device)

    return (loss, loss_amp, loss_ph, loss_gain, loss_sup,
            (a, a_tar, w_ph, pe))
```

**loss 组成**：

1. **振幅 loss**：
   - `pe = remainder(th − th_tar + π, 2π) − π`（圆度相位差）
   - `la = log1p(a)`, `lt = log1p(a_tar)`（log1p 域）
   - `w = (a_tar/a_tar.max())^0.5`（软化权重，AMP_P=0.5）
   - `loss_amp = mean(w × (la−lt)²)`

2. **相位 loss**：
   - `w_ph = a_tar²`（功率加权）
   - `loss_ph = Σ(w_ph × pe²) / Σ(w_ph)`

3. **高频增益 loss**：
   - 按 w=c%25 切频段 [(12,19), (19,25)]
   - `go = mean(a_out[idx]) / mean(a_tar[idx])`，clamp [0.1, 20]
   - huber 惩罚 go > GAIN_TGT=1.0 的部分
   - `loss_gain = mean over 2 segments`

4. **总 loss**：
   - `loss = loss_amp + loss_ph + 0.5 × loss_gain + gen_weight × loss_sup`

**频段切分铁律**：`c % W_FFT`（W_FFT=25），按 w 切。按 0/16/32/48 均分会混淆 DC 与高频（train_multimodal.py:811 注释）。

---

## 9. `EarlyStopper` 类（第300-436行）

### 9.1 `__init__`（第312-346行）

```python
def __init__(self, patience=5, min_delta=1e-3, gap_ratio=3.0, gap_patience=3,
             stable_window=4, stable_tol=0.01, step_slow_factor=1.8,
             step_slow_patience=3, mem_abort_frac=0.97, min_steps=200,
             band_lo=0.8, band_hi=1.2, verbose=True):
```

关键参数修正记录（代码注释第318-328行）：
- `gap_ratio` 默认 1.15 → 3.0（2026-10-14 修正误停）
- 实测 batch=1 时 train loss 是单样本测量，val 是 48 样本均值
- step 250~1000 的实测比值：0.34 / 2.02 / 3.08 / 2.74
- val_loss 全程单调下降：0.5380 → 0.4777 → 0.4610 → 0.4256
- 1.15 会在 val 仍下降时误判「已过拟合」，第一次训练因此在 step 1000 白停

### 9.2 `update_baseline`（第352-365行）

```python
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
        self.baseline_step_time = 0.8 * self.baseline_step_time + 0.2 * step_time
```

历史坑：之前引用未定义变量 `step`，smoke test 在 step=2 崩，训练永远活不过第 2 步。

### 9.3 `check` 方法（第367-436行）

四条判据：

**判据一**（第377-388行）：验证 loss 连续 patience 次不改善
```python
improved = (self.best_val - val_loss) > self.min_delta
if improved:
    self.best_val = val_loss
    self.best_step = step
    self.no_improve = 0
else:
    self.no_improve += 1
if self.no_improve >= self.patience:
    return True, ...
```

**判据二**（第390-402行）：train/val loss 分叉
```python
if train_loss is not None and train_loss > 1e-9:
    gap = val_loss / train_loss
    if gap > self.gap_ratio:
        self.gap_count += 1
        if self.gap_count >= self.gap_patience:
            return True, ...
```

**判据三**（第404-416行）：频段比稳定在 [0.8, 1.2] 且漂移 ≤ stable_tol
```python
if ratio is not None:
    self.band_hist.append((step, ratio))
    if len(self.band_hist) > self.stable_window:
        self.band_hist.pop(0)
    if len(self.band_hist) == self.stable_window:
        vals = [r for _, r in self.band_hist]
        in_band = all(self.band_lo <= r <= self.band_hi for r in vals)
        drift = max(vals) - min(vals)
        if in_band and drift <= self.stable_tol:
            return True, ...
```

**判据四**（第418-434行）：单步时间突变 / 显存异常
```python
if (self.baseline_step_time and step_time
        and step_time > self.baseline_step_time * self.step_slow_factor):
    self.slow_count += 1
    ...
if mem_frac is not None and mem_frac > self.mem_abort_frac:
    return True, ...
```

---

## 10. `WarmupCosineLR` 类（第439-476行）

```python
class WarmupCosineLR:
    """学习率调度：前期大 lr 快速收敛，后期小 lr 精修。"""
    def __init__(self, optimizer, lr_start, lr_peak, lr_min,
                 warmup_steps, total_steps, verbose=True):
        ...
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
```

- lr_start=3e-5, lr_peak=3e-4, lr_min=1e-5, warmup=300
- 前 300 步线性 warmup
- 之后 cosine 退火

---

## 11. 参数解析（第482-565行）

关键参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--days` | 7.0 | 训练预算天数 |
| `--mode` | mix | 条件采样模式 |
| `--batch` | 1 | ★ 唯一干净可用档 |
| `--acc` | 1 | ★ 未实现（静默失败） |
| `--lr` | 3e-4 | |
| `--workers` | 4 | ★ 从 8 改下来 |
| `--resume` | auto | 断点续训 |
| `--save-every` | 200 | 断点保存间隔 |
| `--log-every` | 50 | 日志间隔 |
| `--val-frac` | 0.08 | 验证集 group 占比 |
| `--val-batches` | 12 | 每次验证 batch 数 |
| `--check-every` | 250 | 早停检查间隔 |
| `--patience` | 5 | 判据一耐心 |
| `--min-steps` | 300 | 少于此步数不停 |
| `--lr-start` | 3e-5 | warmup 起始 |
| `--lr-peak` | 3e-4 | warmup 峰值 |
| `--lr-min` | 1e-5 | 退火下限 |
| `--warmup` | 300 | warmup 步数 |
| `--gen-weight` | 2.0 | cond_field 监督权重 |
| `--gen-ratio` | 0.7 | 生成通路步数占比 |
| `--bench` | 8 | 速度实测步数 |

batch 参数的历史修正（第490-497行注释）：
- 原注释「B=4 峰值 1.03GB」是 D=64 时测的
- D=125 后：B=4 → GPU 3.95GB + 提交 11.4GB；B=2 → GPU 4.8GB（物理不可能）+ 单步 39~49s；B=1 → GPU 2.55GB + 单步 4.4s
- 「B=1/B=2 不是线性关系：B=2 时显存分配器开始 split/抖动，反而比 B=4 还慢 10 倍」

---

## 12. 断点保存/恢复（第571-612行）

```python
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
    if stopper is not None:
        payload["stopper"] = {...}
    if sched is not None:
        payload["sched"] = {"cur": sched.cur}
    tmp = CKPT_DIR / f"{tag}.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, CKPT_DIR / f"{tag}.pt")          # 原子替换，防写一半断电
    if step % 2000 == 0:
        torch.save(payload, CKPT_DIR / f"step_{step}.pt")
```

- 排除 qwen.* 权重
- 早停器和调度器状态一起存
- 原子替换（tmp + os.replace）
- 每 2000 步归档

---

## 13. 主流程 `main`（第635-1107行）

### 13.1 环境设置（第636-644行）

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

### 13.2 提交量体检（第646-675行）

```python
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
```

Windows API GlobalMemoryStatusEx，监控 RAM 可用和提交剩余。

### 13.3 数据集划分（第694-723行）

```python
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
```

**group 级隔离**（第695-701行注释）：
- 按样本随机划分会 100% 撞上同 group 兄弟 GIF
- bulk_* 有上万 GIF 挤在少数 group
- 424 个验证样本的 80 个 group 全部出现在训练集
- 按 group 划分：8% group = 16 组 / 251 条，group 重叠 0
- 固定种子 1234

### 13.4 DataLoader 配置（第725-758行）

```python
ds = MultimodalGifDataset(
    index_path=str(ROOT / "data" / "index.json"),
    cache_dir=str(ROOT / "data" / "clip_cache"),
    gif_root=GIF_ROOT, device=device, mode=args.mode,
    encode_cache=False,   # 全量场缓存 80.9GB，放不下 → 实时编码
    indices=train_idx,
)
val_ds = MultimodalGifDataset(..., mode="both", ...)
collate = make_collate(device)
dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                collate_fn=collate, pin_memory=True, drop_last=True,
                persistent_workers=args.workers > 0,
                worker_init_fn=_worker_init,
                timeout=600,
                prefetch_factor=3 if args.workers > 0 else None)
val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    num_workers=max(2, args.workers // 4), collate_fn=collate,
                    pin_memory=True, drop_last=False, worker_init_fn=_worker_init,
                    timeout=600)
```

- `encode_cache=False`：全量场缓存 80.9GB 放不下
- `timeout=600`：防 worker 死了主进程永久阻塞
- `persistent_workers=True`
- 验证集固定 mode="both"（确定性）

### 13.5 模型与优化器（第760-767行）

```python
model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr_peak, weight_decay=1e-4)
stopper = EarlyStopper(patience=args.patience, min_steps=args.min_steps)
sched = WarmupCosineLR(opt, args.lr_start, args.lr_peak, args.lr_min,
                       args.warmup, 10 ** 9)   # total 占位，训完由 update 修正
```

### 13.6 恢复（第769-796行）

```python
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
    if "stopper" in ck:
        s = ck["stopper"]
        stopper.best_val = s.get("best_val", float("inf"))
        ...
```

### 13.7 坐标构建（第798-801行）

```python
t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)
n_param = sum(p.numel() for p in model.parameters())
print(f"[PARAM] {n_param:,} 参数  N={len(t_coord)}", flush=True)
```

N = 6×32×32 = 6144

### 13.8 `evaluate` 函数（第806-854行）

```python
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
        ...
        idx = torch.tensor(hf_idx, device=a.device)
        hf_out += a_.clamp(min=1e-8).mean(dim=1)[:, idx].mean().item()
        hf_tar += a_tar.clamp(min=1e-8).mean(dim=1)[:, idx].mean().item()
        n_b += 1
    model.train()
    ...
    out["seg_ratio"] = hf_out / max(hf_tar, 1e-12)      # 高频段谱比
    return out
```

验证走生成通路（use_cond_field=True），与推理同口径。

### 13.9 `measure_rate` 函数（第857-884行）

```python
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
    ...
    return cnt / max(dt, 1e-6)
```

bench 用 dummy loss（a²+th²），只测速度。

### 13.10 双通路（第886-893行）

```python
# 为什么必须双通路而不是只走生成：
#   只走重建（targets=真实场）→ cond_field 的梯度恒为 0（与 prototypes 同一个病）
#   只走生成 → loss 里没有真实答案锚定，早期预测场是噪声，学不动
# 所以按步随机交替：走重建或走生成，比例由 gen_ratio 控制。
# ★ batch=1 时无法在一个 batch 内拆分，按步交替期望比例不变且省显存。
def _use_gen_path():
    return random.random() < args.gen_ratio
```

### 13.11 训练主循环（第933-1054行）

```python
while step < total_steps:
    got = next_batch(it, device)
    if got is None:
        it = iter(dl)
        got = next_batch(it, device)
    cond, field, modes, idxs = got

    B = field.shape[0]
    opt.zero_grad(set_to_none=True)

    is_gen = _use_gen_path()
    a, th, diag = model(texts=None, target_field=field, t_coord=t_coord,
                        s_coord=s_coord, batch_size=B, cond_vec=cond,
                        use_cond_field=is_gen)

    loss, loss_amp, loss_ph, loss_gain, loss_sup, (a_, a_tar, w_ph, pe) = \
        compute_loss(a, th, field, diag.get("pred_field"), args.gen_weight)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1

    cur_lr = sched.step(step)
    ...
```

- 梯度裁剪 max_norm=1.0
- loss EMA（0.98/0.02）
- 进度条每步刷新
- 早停每 check_every=250 步检查
- 断点每 save_every=200 步或 600 秒保存

### 13.12 异常处理（第1056-1092行）

三种收尾：
1. `KeyboardInterrupt`：保存断点
2. `Exception`：先保存断点再报错，含 timeout/OOM 诊断
3. 正常完成：保存 latest + best

---

## 14. 未验证项

1. `--acc`（梯度累积）参数未实现，传了不生效（静默失败）
2. 全量训练从未启动，所有早停判据的实际触发行为无数据
3. `torch.compile` 可选，默认关，无实测数据
4. 径向 Nyquist 约束（RAD_*）已移除（第618-622行注释：「对 batch=1 单步约 +8% 耗时，性价比很低」），无对比实验
5. `_mem_report` 的 Windows API 调用在其他平台会失败（有 try/except 兜底）
6. 验证集 8% group = 16 组 / 251 条，验证指标的统计稳定性无多次运行数据

---

## 15. 代码事实汇总

| 项 | 值 | 来源 |
|---|---|---|
| D, GRID, PATCH, WIN, IMG, TS | 125, 32, 4, 5, 128, 6 | train_multimodal.py:142 |
| W_FFT | 25 | train_multimodal.py:143 |
| AMP_P | 0.5 | train_multimodal.py:146 |
| HF_SEGS | [(12,19),(19,25)] | train_multimodal.py:147 |
| GAIN_TGT/W/DELTA | 1.0/0.5/0.5 | train_multimodal.py:148 |
| batch | 1 | train_multimodal.py:489 |
| workers | 4 | train_multimodal.py:503 |
| lr_peak | 3e-4 | train_multimodal.py:502 |
| warmup | 300 | train_multimodal.py:541 |
| lr_start/min | 3e-5 / 1e-5 | train_multimodal.py:538, 540 |
| val_frac | 0.08 | train_multimodal.py:519 |
| val_batches | 12 | train_multimodal.py:526 |
| check_every | 250 | train_multimodal.py:528 |
| patience | 5 | train_multimodal.py:531 |
| min_steps | 300 | train_multimodal.py:532 |
| gen_weight | 2.0 | train_multimodal.py:554 |
| gen_ratio | 0.7 | train_multimodal.py:557 |
| save_every | 200 | train_multimodal.py:513 |
| log_every | 50 | train_multimodal.py:514 |
| bench | 8 | train_multimodal.py:563 |
| timeout | 600 | train_multimodal.py:752 |
| 随机种子 | 1234 | train_multimodal.py:703 |
| weight_decay | 1e-4 | train_multimodal.py:762 |
| grad_clip | 1.0 | train_multimodal.py:956 |
| loss EMA | 0.98/0.02 | train_multimodal.py:968 |
| gap_ratio | 3.0（原1.15） | train_multimodal.py:312 |
| 线程锁 | 1（import前） | train_multimodal.py:121-123 |
| 主进程线程 | 4 | train_multimodal.py:641 |
| 单步时间(B=1) | 4.4s | train_multimodal.py:66 |
| GPU显存(B=1) | 2.55GB | train_multimodal.py:66 |
