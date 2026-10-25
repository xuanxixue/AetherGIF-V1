# -*- coding: utf-8 -*-
"""guidance-dropout 条件生成训练 —— 第三十五发现后的阶段二。

【背景】
五路探针判死逐位置压缩后，结论是：3D-FFT 本身已是物理 VAE 的编码器
（无损 51.7dB、紧凑、零参数），真正缺的是**条件生成**。

现行 v4200 的训练/推理口径错配（第三十四发现的现形态）：
  - 训练时 forward 拿 target_field → fft_bypass 把**真答案的 FFT 场**
    直接注入 θ₀/a₀（teacher 旁路）→ 条件通路（cond_init）从来不是必需，
    甚至收不到有效梯度压力；
  - 推理/评测时 target_field=None 且无 _fft_cond_cache → 旁路整体消失，
    模型只剩 cond_init一条路，但它从没被强制学过。
  实测后果（fix_r3 最终评测）：条件生成 PSNR 10.35/7.50 dB，**低于**
  同 std 高斯噪声基线 13.03/7.44 dB，五种条件的判别 margin 几乎全同
  （+2.2 / -1.9 dB）= 输出层仍是与条件无关的通用场。

【guidance dropout（本脚本核心，同 VAE 的 condition-dropout / CFG 逻辑）】
每步以概率 GUID_P 把 target_field 丢掉（传 None）：
  - dropped 步的前向与推理**逐位一致**（fft_bypass 关闭、cond_anchor
    只含条件量），loss 仍对真 target 计算 → 强制 cond→场通路学习；
  - 保留步（1-p）维持 teacher 旁路训练（fft_bypass 是 GIF→GIF 生成
    的机制载体，不能丢）与物理栈的精化能力。
损失、优化器分组、交叉训练（七种条件组合）与 overfit_cond3 完全一致，
唯一机制性变量 = GUID_P，可与 fix_r3/fix_r4 严格 A/B 对照。

【判据】
  - dropped 步的 wcos（对 target 的功率加权相位余弦）: 冒烟期看它是否
    离开随机地板并持续上升 —— 这是「条件通路是否开始携带 target 相位
    结构」的直接度量；
  - content 层（w=0 列 c=0/25/50/75/100，承载 97.2% 能量、每位置 10 dof）
    的相位对齐 wcos_c: 该层天然适合条件监督，应最先逼近精确；
  - 最终评测: 五条件 × 两素材交叉生成 PSNR + 绝对判据
    （噪声基线 / ac1 / hilo，一票否决）。

【运行】
cd 到 v4200 根目录后
  C:/Python312/python.exe -u scripts/train_guided.py [STEPS]
  OVERFIT_TAG=guid0_150 GUID_P=0.5 ...   # 产物落 outputs_real/<tag>/
默认旋钮对齐 fix_r4（HF_GUARD_W=1.0 LO_FLOOR_W=0.5 BAND_ABS_W=0.3 ACC_K=2），
只多了 GUID_P=0.5。单步 ~6s（dropped 步更快），300 步 ≈ 25~30 分钟。
"""
import math
import os
import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir():
        break
    if _PROJ.parent == _PROJ:
        break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))
_DEPS = _PROJ / "deps312"
_VENV = _PROJ / ".venv" / "Lib" / "site-packages"
if _DEPS.is_dir():
    # ★ deps312 插前；.venv（cp313 的 numpy）必须 append，否则遮蔽 cp312
    sys.path.insert(0, str(_DEPS))
if _VENV.is_dir():
    sys.path.append(str(_VENV))
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass

try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] torch_accel_shim 加载失败：{ex}")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ★ 环境默认对齐 run_sep_fix.cmd（fix_r 系列基线，保证 A/B 唯一变量 = GUID_P）。
#   这些变量在本地模块**导入时**被读取（KAPPA_MAX 在 aether_gif_model、
#   PHASE_ACT/IDEAL_CLAMP 在 cond_init、RES_LIMIT/W2_NORM_MAX 在物理核），
#   必须在 import 本地模块之前 setdefault。命令行显式设置仍可覆盖。
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")
os.environ.setdefault("IDEAL_CLAMP", str(math.pi))
# ★ 2026-10-23 加固（本次审查挖出的最严重静默失败）：
#   v4200 产品版的 ckpt（color1_long 起）全部是 CondInitPC 逐通道结构
#   （cond_init.to_amp.1.weight = [768000, 8]，总参数 23,044,534）。
#   此前本脚本没有默认值，**漏设 COND_INIT_PC 就会静默退化为旧的
#   CondInitModulator（10,273,206 参数）**，而 RESUME 又会把 6 个
#   cond_init.* 层当成"shape 不匹配"跳过 —— 条件通路整体随机初始化，
#   训练几小时等于白跑，且日志里只有一行不起眼的"跳过 N 层"。
#   这里给默认值（显式设 COND_INIT_PC=0 仍可覆盖回旧架构）。
os.environ.setdefault("COND_INIT_PC", "1")
os.environ.setdefault("COND_PC_RANK", "8")

import random
import numpy as np
import torch
from PIL import Image

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames, load_gif_frames
from physics.kuramoto_sync import build_spacetime_index
# ★ 条件初始态类按 COND_INIT_PC 环境开关选择（2026-10-21 per-channel 扩容）。
#   两个类的 forward/expand_to_d/supervise 签名兼容，调用处统一用 _CondInitCls。
if os.environ.get("COND_INIT_PC", "").lower() in ("1", "true", "yes", "on"):
    from physics.cond_init_pc import CondInitPC as _CondInitCls
else:
    from physics.cond_init import CondInitModulator as _CondInitCls
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import CLIP_EMB_DIM, N_COND3, build_cond_vector3
from train.spectral_loss import (band_profile, hf_ratio, lo_ratio,
                                 loss_band_abs, loss_band_abs_cap,
                                 loss_hf_guard, loss_h_spec, loss_lo_floor,
                                 loss_phase_band, loss_phase_band_floor,
                                 loss_spec_shape, loss_color_band,
                                 loss_color_phase)

# ---------------- 编解码口径（与 overfit_cond3.py 完全一致） ----------------
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1  # 25
# content 层 = w=0 列（c%25==0，含 DC），承载 97.2% 能量，每位置 10 dof
_CONTENT_MASK = torch.arange(D) % W_FFT == 0

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
LR = 3e-4
COND_LR = 1e-3
AMP_P = 0.5
GEN_W = 2.0
INIT_MODE = os.environ.get("INIT_MODE", "init")
N_MODULES = int(os.environ.get("N_MODULES", "8"))
MOD_GAMMA = float(os.environ.get("MOD_GAMMA", "0.15"))
# ★★★ 本脚本核心旋钮：guidance dropout 概率 ★★★
#   每步以 GUID_P 概率丢掉 target_field（前向与推理逐位一致），
#   强制 cond_init→场通路学习。GUID_P=0 退化为 overfit_cond3 旧行为。
GUID_P = float(os.environ.get("GUID_P", "0.5"))
# ★ 初态级教师强制概率（2026-10-22，第三十七死穴修复实验）：
#   每步以 FORCE_INIT_P 概率把初态强制成目标场（force_init=(a_tar,θ_tar)），
#   loss 照常监督输出 → 逼 12 层动力学学会「对好结构只做小修正」。
#   FORCE_INIT_P=0 完全退化为旧行为（不使用 force_init）。
FORCE_INIT_P = float(os.environ.get("FORCE_INIT_P", "0.0"))
# ★ 保持性 loss 权重（2026-10-22，第三十七死穴加强修复）：
#   仅在 forced 步生效。主 loss_ph 用 a_tar² 加权（DC 独占 ~99.9% 监督），
#   逐通道细节相位缺监督；loss_pres 用均匀加权补上这个缺口，
#   直接训练动力学「对好结构只做小修正」。幅度侧用 log1p 域 MSE 对齐主 loss 口径。
#   PRES_W=0 完全退化为旧行为。
PRES_W = float(os.environ.get("PRES_W", "0.0"))
# 与 physics.cond_init_pc.py 同口径（run_sep_fix.cmd 设 IDEAL_CLAMP=π）
_IDEAL_CLAMP = float(os.environ.get("IDEAL_CLAMP", "") or math.pi)
# ★ 颜色 loss 权重（2026-10-22，「就差颜色」专项修复）：
#   空间域逐帧分位数约束，直击「颜色被噪点稀释」
#   （生成饱和度 22.5/34.3 vs 素材 5.8/9.9；hilo src0 超素材 20 倍）。
#   三部分（第十七/十八死穴教训：均值约束会压平轮廓、必须逐帧、判据相对 target）：
#   ① sat（每像素 RGB 通道散布）q50/q75 单侧压制 → 清颜色噪点
#   ② 相邻像素差分：q75 压制（清噪点）+ q99 单侧 relu 保护（不许轮廓变糊）
#   ③ 逐帧 std 下限守卫（std_t×0.85） → 堵死「整体压暗/压平」捷径
#   COLOR_W=0 完全退化为旧行为。
COLOR_W = float(os.environ.get("COLOR_W", "0.0"))
# ★★★ 颜色频点专项 loss（2026-10-24 定位后新增，本轮解决颜色的主力）★★★
#   s_dim=4×4×3=48，颜色通道是 48 维的最后一位、周期=3 → 沿 48 维 rfft 后
#   三通道差异落在 w=48/3=16。去掉/只留 w=16 的重建实验铁证：
#     8-ball    完整 sat 6.36 → 去掉 w16 后 0.51  → 只留 w16(+DC) 6.85
#     前辈前辈  完整 sat 9.87 → 去掉 w16 后 2.85  → 只留 w16(+DC) 9.79
#   即 w=16 是颜色的唯一载体；而实测生成场在 w=16 只有 target 的 12~17%
#   （w=0 亮度反而 1.35~1.43 倍 = 过亮）。故对 w=16 单独高权重对齐。
#   COL_BAND_W=0 完全退化为旧行为。
COL_BAND_W = float(os.environ.get("COL_BAND_W", "0.0"))
COL_BAND = int(os.environ.get("COL_BAND", "16"))
DC_BAND_W = float(os.environ.get("DC_BAND_W", "0.0"))
#   ★ DC_BAND_W：w=0 过亮（实测 1.35~1.43×）时用来压亮度，与 COL_BAND_W 配对使用。
#     只压「超过 target」的部分（单侧 relu），不压欠亮的样本。
#   ★ COL_PH_W：颜色频点的**相位**权重（2026-10-24 第一轮教训）。
#     只打幅度会得到「彩色噪点」（8-ball 被染青蓝、前辈前辈满屏红蓝紫乱点），
#     因为色相由相位决定。幅度+相位成对约束，才是「正确的颜色」。
COL_PH_W = float(os.environ.get("COL_PH_W", "0.0"))
# ★ 其余旋钮默认对齐 fix_r4（与 fix_r3/fix_r4 严格 A/B，唯一变量 = GUID_P）
FFN_HIDDEN = int(os.environ.get("FFN_HIDDEN", "256"))
GUARD_W = float(os.environ.get("GUARD_W", "1.0"))
SPEC_W = float(os.environ.get("SPEC_W", "0.0"))
H_SPEC_W = float(os.environ.get("H_SPEC_W", "0.0"))
HF_GUARD_W = float(os.environ.get("HF_GUARD_W", "1.0"))
HF_SLACK = float(os.environ.get("HF_SLACK", "1.5"))
LO_FLOOR_W = float(os.environ.get("LO_FLOOR_W", "0.5"))
LO_FLOOR = float(os.environ.get("LO_FLOOR", "0.7"))
ACC_K = max(1, int(os.environ.get("ACC_K", "2")))
WARM_BLOCKS = max(0, int(os.environ.get("WARM_BLOCKS", "0")))
BAND_ABS_W = float(os.environ.get("BAND_ABS_W", "0.3"))
BAND_CAP_W = float(os.environ.get("BAND_CAP_W", "0.0"))
BAND_CAP_MULT = float(os.environ.get("BAND_CAP_MULT", "3.0"))
PHASE_BAND_W = float(os.environ.get("PHASE_BAND_W", "0.0"))
PHASE_FLOOR_W = float(os.environ.get("PHASE_FLOOR_W", "0.0"))
DISC_W = float(os.environ.get("DISC_W", "0.0"))
DISC_TGT_COS = float(os.environ.get("DISC_TGT_COS", "0.60"))
DISC_DTGT_COS = float(os.environ.get("DISC_DTGT_COS", "2.0"))
MARGIN_W = float(os.environ.get("MARGIN_W", "0.0"))
MARGIN_DELTA = float(os.environ.get("MARGIN_DELTA", "0.05"))
WCS_W = float(os.environ.get("WCS_W", "0.0"))
WCS_TGT = float(os.environ.get("WCS_TGT", "0.30"))
DISC_MARGIN = float(os.environ.get("DISC_MARGIN", "0.10"))
# ★ 对齐 fix_r 基线（run_sep_fix.cmd 设 PH_ABS_W=1.0、IDEAL_CLAMP=π）：
#   绝对相位项是唯一直接监督「θ0+dθ ≈ θ_tar」的项，
#   dropout 下它是 cond_init 相位预测的主要教师。
PH_ABS_W = float(os.environ.get("PH_ABS_W", "1.0"))
FFT_W = float(os.environ.get("FFT_W", "1.0"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 素材清单：N_SRC=2（原始） / N_SRC=22（扩充后） ----------------
#   ★ 用环境变量切换，默认 2 完全保持旧行为，不破坏历史复现。
#   N_SRC=22 时从 训练素材22/ 读 22 个 GIF 与 vectors/ 下三路 22 条向量
#   （由 tools/encode_22.py 用 INT8 Qwen3-VL-Embedding-2B 产出，
#    文本 / 首帧[image分支] / 视频[PyAV 真fps] 三路严格区分）。
N_SRC = int(os.environ.get("N_SRC", "2"))
_DEF_VEC = ROOT / "data" / "clip_cache"
_DEF_GIF = ROOT / "data" / "gif_raw" / "noto"
if N_SRC == 22:
    VEC_DIR = Path(os.environ.get("VEC_DIR") or (_PROJ / "训练素材22" / "vectors"))
    GIF_DIR = Path(os.environ.get("GIF_DIR") or (_PROJ / "训练素材22"))
    _meta22 = json.loads((VEC_DIR / "meta22.json").read_text(encoding="utf-8"))
    GIF_SPECS = [(it["file"], "") for it in _meta22["items"]]
    assert len(GIF_SPECS) == 22, f"meta22 条数 {len(GIF_SPECS)} != 22"
    print(f"[DATA] N_SRC=22 扩充数据集：{len(GIF_SPECS)} 个素材")
    print(f"       素材目录 {GIF_DIR}")
    print(f"       向量目录 {VEC_DIR}")
else:
    VEC_DIR = Path(os.environ.get("VEC_DIR") or _DEF_VEC)
    GIF_DIR = Path(os.environ.get("GIF_DIR") or _DEF_GIF)
    # 两素材（描述与 encode_cond3.py 一致）
    GIF_SPECS = [
        ("8-ball_1f3b1.gif", "8号球图标"),
        ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "前辈前辈表情包"),
    ]

OUT_DIR = ROOT / "outputs_real" / (os.environ.get("OVERFIT_TAG") or "guided")
OUT_DIR.mkdir(parents=True, exist_ok=True)
if os.environ.get("OVERFIT_TAG"):
    print(f"[OUT] OVERFIT_TAG={os.environ['OVERFIT_TAG']}，产物落 {OUT_DIR}（隔离防覆盖）")


def _safe_torch_save(obj, path, retries=6):
    """原子保存 checkpoint（2026-10-23 加固，替代裸 torch.save）。

    为什么必须换掉直接 torch.save：
      1) Windows 上目标文件可能被杀软 / 搜索索引 / 上一进程残留句柄短暂占用，
         直接写会抛 PermissionError —— color4_long 正是在 200 步保存时这样崩的，
         这次崩掉让 100~200 步（约 20 分钟）的训练白跑；
      2) torch.save 直接写目标文件，一旦中途失败（磁盘满 / 进程被杀），会在原地
         留下**截断的 ckpt**，把上一份完好的权重覆盖掉 —— 这正是"模型损坏"的来源。

    做法：先写同目录 .tmp，成功后 os.replace 原子替换。任何时刻磁盘上要么是旧的
    完整文件、要么是新的完整文件，不会出现半截。失败时**保留 .tmp**（不删除），
    人工可改名恢复。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    last = None
    for i in range(retries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)          # 同一分区内原子替换
            return path
        except (PermissionError, OSError) as ex:
            last = ex
            time.sleep(1.0 + 1.5 * i)
        except Exception:
            print(f"[SAVE][ERROR] 保存 {path.name} 失败，完整临时文件保留在：{tmp}",
                  flush=True)
            raise
    print(f"[SAVE][FATAL] {path.name} 重试 {retries} 次仍失败：{last}\n"
          f"            完整数据仍在 {tmp}，可手动改名为 {path.name}", flush=True)
    raise RuntimeError(f"保存 {path} 失败：{last}")


def _decodable(a_, th_):
    z = torch.complex(a_ * torch.cos(th_), a_ * torch.sin(th_))
    N = z.shape[0]
    z = z * 10.0
    flat = torch.fft.irfft2(z.reshape(N, WIN, W_FFT), s=(WIN, PATCH * PATCH * 3),
                            dim=(-2, -1)).real
    flat = flat.reshape(N, WIN * PATCH * PATCH * 3)
    blocks = flat.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)
    return xt.reshape(TS * WIN, IMG, IMG, 3)


def to_gif(a, th, path, tag):
    z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    frames = np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=True)).astype(np.uint8)
    pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    pil[0].save(path, save_all=True, append_images=pil[1:], loop=0, duration=100)
    return pil


def psnr(a_img, b_img):
    mse = ((a_img.astype(np.float32) - b_img.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255.0 ** 2 / max(mse, 1e-12))


# ---------------- 绝对质量指标（D 项，与 overfit_cond3 一致） ----------------
def ac1(x):
    """相邻像素自相关：真实图像 >0.5，纯噪声 ≈0。"""
    a = x - x.mean()
    return float((a[..., :-1] * a[..., 1:]).sum() / max((a * a).sum(), 1e-12))


def sobel_mag(x):
    g = x.mean(axis=-1)
    gx = np.abs(np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) / 2
    gy = np.abs(np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0)) / 2
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def hilo(x):
    """高频/低频能量比：图像 ≈0.01~0.05，白噪声 ≈8。"""
    g = x.mean(axis=-1)
    F = np.abs(np.fft.fftshift(np.fft.fft2(
        g - g.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
    H, W_ = F.shape[1], F.shape[2]
    yy, xx = np.mgrid[0:H, 0:W_]
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W_ // 2) ** 2)
    return float(F[:, r > H * 0.35].sum() / max(F[:, r < H * 0.15].sum(), 1e-12))


def noise_baseline(ref):
    """同 std 高斯噪声对 ref 的 PSNR —— PSNR 低于它 = 图就是噪声。"""
    rng = np.random.default_rng(0)
    n = np.clip(rng.normal(ref.mean(), ref.std(), ref.shape), 0, 255).astype(np.float32)
    return psnr(n, ref), ac1(n), sobel_mag(n), hilo(n)


def wcos_of(pe, w_ph, mask=None):
    """功率加权相位余弦。pe=[B,N,D] 相位误差，w_ph=[B,N,D] 权重。
    mask 可选 [D] bool，只看部分通道（content 层用）。"""
    if mask is not None:
        pe = pe[..., mask]
        w_ph = w_ph[..., mask]
    return float((w_ph * torch.cos(pe)).sum() / w_ph.sum().clamp(min=1e-12))


# ---------------------------------------------------------------- 素材加载
print("=" * 74)
print(f"[1] 加载 {len(GIF_SPECS)} 张素材 -> 频域场")
gif_dir = GIF_DIR
fields, a_tars, th_tars = [], [], []
gif_paths = []
for name, desc in GIF_SPECS:
    gp = gif_dir / name
    if not gp.exists():
        alt = ROOT.parent.parent / "data" / "gif_raw" / "noto" / name
        gp = alt if alt.exists() else None
    if gp is None or not gp.exists():
        raise FileNotFoundError(name)
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    gif_paths.append(str(gp))
    fields.append(f)
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]))
    print(f"    {name}: N={f.shape[1]} a_mean={a_tars[-1].mean():.5f}")

target = torch.stack(fields, 0)
a_tar = torch.stack(a_tars, 0)
th_tar = torch.stack(th_tars, 0)
t, s = build_spacetime_index(TS, GRID, GRID, device)
# 颜色 loss 预计算：target 空间帧（与 _decodable 同口径）
# _decodable(a_tar[i]) ≈ 素材帧（FFT 往返误差 ~6e-07）
if COLOR_W > 0:
    with torch.no_grad():
        _TAR_FR = torch.stack(
            [_decodable(a_tar[i], th_tar[i]) for i in range(len(GIF_SPECS))])
    print(f"    [颜色 loss] target 空间帧预计算完成 shape={tuple(_TAR_FR.shape)}")

from codec.gif_codec import resample_frames
ref_imgs = []
raw_imgs = []
for i in range(len(GIF_SPECS)):
    z = (a_tar[i] * torch.exp(1j * th_tar[i])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    ref_imgs.append(np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32))
    _raw = load_gif_frames(str(gif_paths[i]), resize_to=(IMG, IMG))
    raw_imgs.append(resample_frames(_raw, TS * WIN).astype(np.float32))

# ---------------------------------------------------------------- 条件加载
print(f"\n[2] 加载三路原始向量（{VEC_DIR.name}，gif=真视频分支）")
CLIP_CACHE = VEC_DIR
if N_SRC == 22:
    text3 = np.load(CLIP_CACHE / "text22.npy")
    first3 = np.load(CLIP_CACHE / "first22.npy")
    gif3v = np.load(CLIP_CACHE / "gif22.npy")
else:
    text3 = np.load(CLIP_CACHE / "text3.npy")
    first3 = np.load(CLIP_CACHE / "first3.npy")
    gif3v = np.load(CLIP_CACHE / "gif3_video.npy")
assert text3.shape == first3.shape == gif3v.shape == (N_SRC, CLIP_EMB_DIM), \
    f"三路向量形状错: {text3.shape} {first3.shape} {gif3v.shape}，期望 ({N_SRC},{CLIP_EMB_DIM})"
for n_, v_ in [("文本", text3), ("首帧", first3), ("GIF真视频", gif3v)]:
    _mn = v_ / (np.linalg.norm(v_, axis=1, keepdims=True) + 1e-9)
    _C = _mn @ _mn.T
    _off = _C[~np.eye(N_SRC, dtype=bool)]
    print(f"    {n_:8s}: {v_.shape}  素材间余弦 min={_off.min():+.3f} "
          f"mean={_off.mean():+.3f} max={_off.max():+.3f}")
    if float(_off.max()) > 0.999:
        raise RuntimeError(f"{n_} 向量疑似退化成 NULL（素材间余弦全≈1.0）")

# 条件向量 cv_* 在 build_one_cv 定义之后构造（见下方）

# ★ 交叉训练的七种条件组合（单路×3 / 双路×3 / 三路×1，等概率）
COND_COMBOS = [
    ("三路全给", (1, 1, 1)),
    ("文生",     (1, 0, 0)),
    ("图生",     (0, 1, 0)),
    ("GIF本身",  (0, 0, 1)),
    ("图文生",   (1, 1, 0)),
    ("文+GIF",   (1, 0, 1)),
    ("图+GIF",   (0, 1, 1)),
]


def build_one_cv(src_i: int, has) -> np.ndarray:
    """从三路原始向量现拼单个 cond_vec [6147]，缺省路置零。"""
    z = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
    t = text3[src_i:src_i + 1] if has[0] else z
    f = first3[src_i:src_i + 1] if has[1] else z
    g = gif3v[src_i:src_i + 1] if has[2] else z
    return build_cond_vector3(t, f, g, has)          # [1, N_COND3]


# ★ 条件向量：N_SRC=22 用 build_one_cv 现构造（自适应条数）；
#   N_SRC=2 仍读历史预编码缓存，保持旧行为逐位一致。
if N_SRC == 22:
    cv_full = np.concatenate([build_one_cv(i, (1, 1, 1)) for i in range(N_SRC)], 0)
    cv_text = np.concatenate([build_one_cv(i, (1, 0, 0)) for i in range(N_SRC)], 0)
    cv_first = np.concatenate([build_one_cv(i, (0, 1, 0)) for i in range(N_SRC)], 0)
    cv_gif = np.concatenate([build_one_cv(i, (0, 0, 1)) for i in range(N_SRC)], 0)
    cv_tf = np.concatenate([build_one_cv(i, (1, 1, 0)) for i in range(N_SRC)], 0)
else:
    cv_full = np.load(CLIP_CACHE / "cv3_full.npy")
    cv_text = np.load(CLIP_CACHE / "cv3_text.npy")
    cv_first = np.load(CLIP_CACHE / "cv3_first.npy")
    cv_gif = np.load(CLIP_CACHE / "cv3_gif.npy")
    cv_tf = np.load(CLIP_CACHE / "cv3_tf.npy")
assert cv_full.shape[1] == N_COND3, f"缓存维度错 {cv_full.shape}"
assert cv_full.shape[0] == N_SRC, f"条件条数 {cv_full.shape[0]} != N_SRC {N_SRC}"
print(f"    cond_vec3 维度 = {cv_full.shape[1]}，条数 = {cv_full.shape[0]}")


# ---------------------------------------------------------------- 训练
print(f"\n[3] 训练 {STEPS} 步（batch=1，随机素材 × 随机条件组合 × guidance dropout）")
print(f"    ★ GUID_P={GUID_P}（每步以此概率丢掉 target_field，前向与推理一致）")
print(f"    ★ FORCE_INIT_P={FORCE_INIT_P}（每步以此概率强制初态=目标场=教师强制）")
print(f"    环境口径（对齐 run_sep_fix.cmd）：COND_GROUPS={os.environ.get('COND_GROUPS')}"
      f" COND_LOWRANK={os.environ.get('COND_LOWRANK')}"
      f" PHASE_ACT={os.environ.get('PHASE_ACT')}"
      f" IDEAL_CLAMP={float(os.environ.get('IDEAL_CLAMP', str(math.pi))):.4f}"
      f" PH_ABS_W={PH_ABS_W}")
print(f"    init_mode={INIT_MODE}, n_modules={N_MODULES}, mod_gamma={MOD_GAMMA}")
print(f"    ffn_hidden={FFN_HIDDEN}（物理核容量）, guard_w={GUARD_W}（dtheta 守卫）")
print(f"    SPEC_W={SPEC_W}（B 分带能量形状）, HF_GUARD_W={HF_GUARD_W}"
      f"（A 高频单侧守卫, slack={HF_SLACK}）")
print(f"    LO_FLOOR_W={LO_FLOOR_W}（低频占比下限守卫, floor={LO_FLOOR}）")
print(f"    ACC_K={ACC_K}（梯度累积窗口，1=关闭）")
print(f"    WARM_BLOCKS={WARM_BLOCKS}（只训最后 K 个block；0=全训）")
print(f"    BAND_ABS_W={BAND_ABS_W}（C 绝对幅度）, BAND_CAP_W={BAND_CAP_W}"
      f"（单侧上限×{BAND_CAP_MULT}）")
print(f"    PHASE_BAND_W={PHASE_BAND_W}（D 相位分组监督）, "
      f"PHASE_FLOOR_W={PHASE_FLOOR_W}（相位下限守卫）")
print(f"    DISC_W={DISC_W}, MARGIN_W={MARGIN_W}, WCS_W={WCS_W}, "
      f"PH_ABS_W={PH_ABS_W}, FFT_W={FFT_W}")
print(f"    条件组合（等概率）：{[n_ for n_, _ in COND_COMBOS]}")
torch.manual_seed(1234)
random.seed(1234)
_CONDPFX = ("cond_field.", "cond_init.", "fft_bypass.")

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=FFN_HIDDEN, init_mode=INIT_MODE,
                       n_modules=N_MODULES, mod_gamma=MOD_GAMMA).to(device)
# ★ 断点续训（与 overfit_cond3 同逻辑；shapes 不匹配的层跳过）
RESUME = os.environ.get("RESUME", "").strip()
if RESUME:
    _rp = Path(RESUME) if Path(RESUME).is_absolute() else ROOT / RESUME
    if _rp.exists():
        _rsd = torch.load(_rp, map_location=device, weights_only=True)
        _rsd = {k: v for k, v in _rsd.items() if not k.startswith("qwen.")}
        _own = model.state_dict()
        _drop = [k for k, v in _rsd.items()
                 if k in _own and tuple(_own[k].shape) != tuple(v.shape)]
        _R_OLD = int(os.environ.get("RESUME_R_OLD", "") or 0)
        if _R_OLD > 0:
            for _k in list(_drop):
                _v = _rsd.get(_k)
                if _v is None:
                    continue
                _tgt = _own[_k]
                if _v.dim() != _tgt.dim():
                    continue
                try:
                    if _k.endswith(".0.weight"):
                        _new = torch.zeros_like(_tgt)
                        _new[:_v.shape[0]] = _v
                        _rsd[_k] = _new
                    elif _k.endswith(".1.weight"):
                        _new = torch.zeros_like(_tgt)
                        _new[:, :_v.shape[1]] = _v
                        _rsd[_k] = _new
                    else:
                        continue
                    _drop.remove(_k)
                except Exception:
                    pass
        for _k in _drop:
            _rsd.pop(_k, None)
        _rret = model.load_state_dict(_rsd, strict=False)
        print(f"[RESUME] 载入 {_rp.name}  missing={len(_rret.missing_keys)}"
              f" unexpected={len(_rret.unexpected_keys)}")
        if _drop:
            print(f"[RESUME] ★跳过 {len(_drop)} 个 shape 不匹配的层：{_drop}")
            # ★ 2026-10-23 加固：条件通路层被跳过 = 该通路从随机初始化开始训练，
            #   续训语义已被破坏（最常见原因就是 COND_INIT_PC / COND_PC_RANK
            #   与 ckpt 口径不一致）。这种"看起来在续训、实际核心模块是新的"
            #   是最烧算力的静默失败，直接中止并要求显式确认。
            #   注意：`.0.weight` 是 RankLinear 的低秩投影，随 COND_LOWRANK 变动
            #   而跳过属于**设计意图**（见 RESUME_R_OLD 扩展逻辑），不算致命。
            _crit = [k for k in _drop
                     if k.startswith(("cond_init.", "cond_field.", "fft_bypass."))
                     and not k.endswith(".0.weight")]
            if _crit and os.environ.get("RESUME_ALLOW_DROP_COND", "0").lower() not in (
                    "1", "true", "yes", "on"):
                print(f"[RESUME][中止] 关键条件层 shape 不匹配 {len(_crit)} 个："
                      f"{_crit[:8]}", flush=True)
                print(f"[RESUME][中止] 常见原因：COND_INIT_PC / COND_PC_RANK 与 ckpt 口径不一致。"
                      f" 当前 COND_INIT_PC={os.environ.get('COND_INIT_PC')}"
                      f" COND_PC_RANK={os.environ.get('COND_PC_RANK')}", flush=True)
                print(f"[RESUME][中止] 继续会让条件通路随机初始化（几小时白跑），已中止。"
                      f" 确实要这么做请设 RESUME_ALLOW_DROP_COND=1", flush=True)
                raise SystemExit(3)
    elif os.environ.get("RESUME_STRICT", "1").lower() in ("0", "false", "no", "off"):
        # 显式关闭严格模式才允许"找不到就当没给"（旧行为）
        print(f"[RESUME] ★ 路径不存在，忽略（RESUME_STRICT=0 已关闭严格模式）：{_rp}")
    else:
        # ★ 2026-10-23 加固：指定了续训却找不到文件，如果继续就是**从头随机初始化**，
        #   与预期严重不符（几小时算力白跑 + 结果不是你以为的那个模型）。直接中止。
        print(f"[RESUME][中止] 指定了续训但路径不存在：{_rp}", flush=True)
        print(f"[RESUME][中止] 严格模式下不会静默从头训练。请检查 RESUME 环境变量。", flush=True)
        print(f"[RESUME][中止] 当前 ROOT = {ROOT}", flush=True)
        print(f"[RESUME][中止] 如需旧行为（找不到就从头训），设 RESUME_STRICT=0", flush=True)
        raise SystemExit(2)
else:
    print("[RESUME] 未指定，从随机初始化开始")
_cond_params = [p for n, p in model.named_parameters()
                if n.startswith(_CONDPFX)]
_other_params = [p for n, p in model.named_parameters()
                 if not n.startswith(_CONDPFX)]
if WARM_BLOCKS > 0:
    _nblk = len(model.pmnncore.blocks)
    _keep = {f"pmnncore.blocks.{i}." for i in range(_nblk - WARM_BLOCKS, _nblk)}

    def _warm(n):
        if n.startswith(_CONDPFX):
            return True
        return any(n.startswith(k) for k in _keep)

    for n, p in model.named_parameters():
        if not _warm(n):
            p.requires_grad_(False)
    _cond_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith(_CONDPFX)]
    _other_params = [p for n, p in model.named_parameters()
                     if p.requires_grad
                     and not n.startswith(_CONDPFX)]
# ★ 8bit 优化器（借鉴 DeepSeek 的「细粒度低精度存储」思路，但不搬它的公式）：
#   本架构的更新量本质是**动力学系数的增量**，二阶矩状态用 8bit 存足够，
#   省一半优化器显存。注意：**不能**用 fp16 混合精度——相位是周期量、
#   振幅跨 5~6 个数量级，fp16 会把高频系数下溢成 0（等于删掉中高频）。
_OPT8 = os.environ.get("OPT8BIT", "0") == "1"
_OptCls = torch.optim.AdamW
if _OPT8:
    try:
        if str(_PROJ / "int8_deps") not in sys.path:
            sys.path.insert(0, str(_PROJ / "int8_deps"))
        import bitsandbytes as _bnb
        _OptCls = _bnb.optim.AdamW8bit
        print("    [OPT] 使用 AdamW8bit（优化器状态 4x→2x）")
    except Exception as _e:
        print(f"    [OPT] 8bit 不可用（{type(_e).__name__}），回退 AdamW")

opt = _OptCls([
    {"params": _other_params, "lr": LR},
    {"params": _cond_params, "lr": COND_LR},
], lr=LR)
_n_par = sum(p.numel() for p in model.parameters())
print(f"    参数量 = {_n_par:,}（{_n_par/1e6:.2f}M，上限 500M）")
print(f"    优化器分组：其他={len(_other_params)} lr={LR}，条件通路={len(_cond_params)} lr={COND_LR}")
model.train()


def loss_discrim(d0, d1, th_target, h0=None, h1=None):
    """对比判别损失（与 overfit_cond3 同逻辑，详见其 docstring）。"""
    _loss = torch.zeros((), device=device)
    if h0 is not None and h1 is not None:
        _f0, _f1 = h0.reshape(-1), h1.reshape(-1)
        _n = _f0.norm() * _f1.norm()
        if float(_n) > 1e-9:
            _loss = _loss + torch.relu((_f0 @ _f1) / _n - DISC_TGT_COS)
    _f0, _f1 = d0.reshape(-1), d1.reshape(-1)
    _n = _f0.norm() * _f1.norm()
    if float(_n) > 1e-9 and DISC_DTGT_COS < 1.5:
        _loss = _loss + 0.2 * torch.relu((_f0 @ _f1) / _n - DISC_DTGT_COS)
    return _loss


COND_ONLY_STEPS = max(0, int(os.environ.get("COND_ONLY_STEPS", "0")))
if COND_ONLY_STEPS > 0:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.cond_init.parameters():
        p.requires_grad_(True)
    _co = [p for p in model.cond_init.parameters() if p.requires_grad]
    _co_opt = torch.optim.AdamW(_co, lr=COND_LR)
    print(f"★ 预热阶段：前 {COND_ONLY_STEPS} 步只训 cond_init"
          f"（{sum(p.numel() for p in _co)/1e6:.2f}M 参数），物理栈全冻")
    model.eval()
    for _s in range(1, COND_ONLY_STEPS + 1):
        _co_opt.zero_grad(set_to_none=True)
        _src = random.randrange(N_SRC)
        _has = COND_COMBOS[random.randrange(len(COND_COMBOS))][1]
        _cv = torch.from_numpy(build_one_cv(_src, _has)).to(device)
        _dt, _dl, _pp = model.cond_init(_cv, return_parts=True, n_pos=len(t))
        _tgt = target[_src:_src + 1]
        _th0 = (torch.rand(1, len(t), D, device=device) * 2 * math.pi
                - math.pi)
        _l = _CondInitCls.supervise(_dt, _dl, _tgt, d_model=D,
                                          w_ph=(a_tar[_src:_src + 1] ** 2).clamp(min=1e-8),
                                          guard_w=GUARD_W,
                                          theta0=_th0, abs_w=PH_ABS_W,
                                          theta_hat=_pp.get("theta_hat"),
                                          a_hat=_pp.get("a_hat"))
        _dc = torch.zeros((), device=device)
        if DISC_W > 0:
            _other = 1 - _src
            _cv2 = torch.from_numpy(build_one_cv(_other, _has)).to(device)
            _dt2, _, _ = model.cond_init(_cv2, return_parts=True, n_pos=len(t))
            _h0 = model.cond_init.proj(_cv)
            _h1 = model.cond_init.proj(_cv2)
            _dc = loss_discrim(_dt, _dt2, th_tar[_src:_src + 1],
                               h0=_h0, h1=_h1)
            _l = _l + DISC_W * _dc
        _l.backward()
        _co_opt.step()
        if _s % 50 == 0 or _s == 1:
            with torch.no_grad():
                if DISC_W > 0:
                    _hh = float((_h0.reshape(-1) @ _h1.reshape(-1))
                                / (_h0.norm() * _h1.norm() + 1e-12))
                    _dd = float((_dt.reshape(-1) @ _dt2.reshape(-1))
                                / (_dt.norm() * _dt2.norm() + 1e-12))
                else:
                    _hh = _dd = 0.0
            print(f"    [warm {_s:4d}] cond_only={float(_l):.4f} "
                  f"disc={float(_dc):.4f} |dθ|={float(_dt.abs().mean()):.4f} "
                  f"cos_h={_hh:+.4f} cos_dθ={_dd:+.4f}", flush=True)
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(_CONDPFX)
                         or (WARM_BLOCKS == 0
                             or any(n.startswith(f"pmnncore.blocks.{i}.")
                                    for i in range(len(model.pmnncore.blocks) - WARM_BLOCKS,
                                                    len(model.pmnncore.blocks)))))
    model.train()
    print("★ 预热结束，进入联合训练")

# ★ dropped 步指标的滑动记录（每 50 步汇总打印一次）
_drop_hist = {"n": 0, "wcos": 0.0, "wcos_c": 0.0, "amp": 0.0, "phase": 0.0}
_keep_hist = {"n": 0, "wcos": 0.0, "wcos_c": 0.0, "amp": 0.0, "phase": 0.0}

_t0 = time.time()
for step in range(1, STEPS + 1):
    src = random.randrange(N_SRC)
    combo_i = random.randrange(len(COND_COMBOS))
    combo_name, has = COND_COMBOS[combo_i]
    one_target = target[src:src + 1]
    one_cv = torch.from_numpy(build_one_cv(src, has)).to(device)
    # ★★★ guidance dropout：本脚本唯一机制性改动 ★★★
    #   dropped=True 时 target_field=None → 前向与推理逐位一致
    #   （fft_bypass 关闭、cond_anchor 只含条件量），loss 仍对真 target 算，
    #   强制 cond_init→场通路承担生成。
    dropped = random.random() < GUID_P
    fwd_target = None if dropped else one_target
    # ★ 初态级教师强制：以 FORCE_INIT_P 概率强制初态=目标场（覆盖 cond/旁路
    #   的初态调制，见 aether_gif_model.py 同名注释）。loss 仍对真 target 算。
    forced = FORCE_INIT_P > 0 and random.random() < FORCE_INIT_P
    _fi = (a_tar[src:src + 1], th_tar[src:src + 1]) if forced else None
    dth = None
    a, th, diag = model(texts=None, target_field=fwd_target, t_coord=t, s_coord=s,
                        batch_size=1, cond_vec=one_cv, use_cond_field=True,
                        force_init=_fi)
    pe = torch.remainder(th - th_tar[src:src + 1] + math.pi, 2 * math.pi) - math.pi
    la = torch.log1p(a.clamp(min=1e-8))
    lt = torch.log1p(a_tar[src:src + 1].clamp(min=1e-8))
    w = (a_tar[src:src + 1] / a_tar[src:src + 1].max().clamp(min=1e-8)).pow(AMP_P)
    loss_amp = (w * (la - lt).pow(2)).mean()
    w_ph = (a_tar[src:src + 1] ** 2).clamp(min=1e-8)
    loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
    wcos_self_t = (((w_ph * torch.cos(pe)).sum() / w_ph.sum())
                   if (MARGIN_W > 0 or WCS_W > 0) else None)
    # content 层（w=0 列）相位对齐 —— 每位置 10 dof，应最先逼近精确
    with torch.no_grad():
        _wc_now = wcos_of(pe, w_ph, _CONTENT_MASK.to(device))
    at = a_tar[src:src + 1]
    loss_spec = loss_spec_shape(a, at, win=WIN, w_fft=W_FFT)
    loss_hf = loss_hf_guard(a, at, slack=HF_SLACK, win=WIN, w_fft=W_FFT)
    loss_lo = loss_lo_floor(a, at, floor=LO_FLOOR, win=WIN, w_fft=W_FFT)
    loss_ba = loss_band_abs(a, at, win=WIN, w_fft=W_FFT)
    loss_hs = loss_h_spec(a, at, win=WIN, w_fft=W_FFT)
    loss_bc = loss_band_abs_cap(a, at, cap_mult=BAND_CAP_MULT, win=WIN, w_fft=W_FFT)
    th_t = th_tar[src:src + 1]
    loss_pb = loss_phase_band(th, th_t, win=WIN, w_fft=W_FFT)
    loss_pf = loss_phase_band_floor(th, th_t, win=WIN, w_fft=W_FFT)
    loss_gen = torch.zeros((), device=device)
    if INIT_MODE == "init":
        dth, dln = diag.get("dtheta"), diag.get("dlna")
        if dth is not None:
            loss_gen = _CondInitCls.supervise(
                dth, dln, one_target, d_model=D, w_ph=w_ph,
                guard_w=GUARD_W,
                theta0=diag.get("theta0"), abs_w=PH_ABS_W,
                theta_hat=diag.get("theta_hat"),
                a_hat=diag.get("a_hat"))
            # 第 4 路 FFT 旁路监督（仅非 dropped 步有 fft_dth）
            _fd = diag.get("fft_dth")
            if _fd is not None:
                from physics.fft_bypass import FFTBypass
                loss_gen = loss_gen + FFT_W * FFTBypass.supervise(
                    _fd, diag.get("fft_dla"), one_target, d_model=D,
                    n_groups=5, w_shape=1.0, w_scale=0.3)
    else:
        pred = diag.get("pred_field")
        if pred is not None:
            from physics.cond_field import CondFieldPredictor
            loss_gen = CondFieldPredictor.supervise(pred, one_target, d_model=D)
    # 对比判别项（与 overfit_cond3 同逻辑）
    loss_disc = torch.zeros((), device=device)
    if DISC_W > 0 and INIT_MODE == "init" and dth is not None:
        # ★ 2026-10-24 修复：原为 `_oth = 1 - src`，只在 2 素材下等价于"另一素材"，
        #   N_SRC=22 时会算出负数索引（src=14 → _oth=-13）→ 静默取错素材甚至报错。
        #   改为随机取一个 ≠ src 的素材；2 素材时行为与原式完全等价。
        _oth = random.randrange(N_SRC - 1)
        if _oth >= src:
            _oth += 1
        _cv2 = torch.from_numpy(build_one_cv(_oth, has)).to(device)
        _dt2, _, _ = model.cond_init(_cv2, return_parts=True, n_pos=len(t))
        if _dt2.shape != dth.shape:
            _dt2 = model.cond_init.expand_to_d(_dt2)
        _h0 = model.cond_init.proj(one_cv)
        _h1 = model.cond_init.proj(_cv2)
        loss_disc = loss_discrim(dth, _dt2, th_tar[src:src + 1],
                                 h0=_h0, h1=_h1)
    # wcos margin 判别（no-grad 前向本身就是 dropped 口径）
    loss_margin = torch.zeros((), device=device)
    _mgo = float("nan")
    if MARGIN_W > 0 and wcos_self_t is not None:
        with torch.no_grad():
            _oth = random.randrange(N_SRC - 1)
            if _oth >= src:
                _oth += 1
            _cv_o = torch.from_numpy(build_one_cv(_oth, has)).to(device)
            a_o, th_o, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                                 batch_size=1, cond_vec=_cv_o, use_cond_field=True)
            pe_o = torch.remainder(th_o - th_tar[_oth:_oth + 1] + math.pi,
                                   2 * math.pi) - math.pi
            _ph_o = (a_tar[_oth:_oth + 1] ** 2).clamp(min=1e-8)
            _mgo = float((_ph_o * torch.cos(pe_o)).sum() / _ph_o.sum())
        loss_margin = torch.relu(_mgo + MARGIN_DELTA - wcos_self_t)
    loss_wcs = torch.zeros((), device=device)
    if WCS_W > 0 and wcos_self_t is not None:
        loss_wcs = torch.relu(WCS_TGT - wcos_self_t)
    # ★ 保持性 loss：仅 forced 步。均匀加权 wrap 相位 MSE + log1p 幅度 MSE，
    #   补主 loss_ph（a_tar² 加权 → DC 独占监督）在逐通道细节上的缺口。
    loss_pres = torch.zeros((), device=device)
    if forced and PRES_W > 0:
        _pe_p = torch.remainder(th - th_tar[src:src + 1] + math.pi,
                                2 * math.pi) - math.pi
        _la_p = torch.log1p(a.clamp(min=1e-8))
        _lt_p = torch.log1p(a_tar[src:src + 1].clamp(min=1e-8))
        loss_pres = (_pe_p.clamp(max=_IDEAL_CLAMP).pow(2).mean()
                     + (_la_p - _lt_p).pow(2).mean())
    # ★ 颜色 loss：空间域逐帧分位数（直击"颜色被噪点稀释"）
    loss_color = torch.zeros((), device=device)
    if COLOR_W > 0:
        _xf = _decodable(a[0], th[0])               # [F,H,W,3] 生成帧（可微）
        _xt = _TAR_FR[src]                           # [F,H,W,3] target 帧
        _sat_g = _xf.max(-1).values - _xf.min(-1).values
        _sat_t = _xt.max(-1).values - _xt.min(-1).values
        _hj_g = (_xf[:, :, 1:] - _xf[:, :, :-1]).abs()
        _hj_t = (_xt[:, :, 1:] - _xt[:, :, :-1]).abs()
        _vj_g = (_xf[:, 1:, :] - _xf[:, :-1, :]).abs()
        _vj_t = (_xt[:, 1:, :] - _xt[:, :-1, :]).abs()
        _F = _xf.shape[0]
        _lcs = torch.zeros((), device=device)
        _lcj = torch.zeros((), device=device)
        for _fi in range(_F):
            _sg = _sat_g[_fi].reshape(-1)
            _st = _sat_t[_fi].reshape(-1)
            _lcs = _lcs + (torch.relu(torch.quantile(_sg, 0.50)
                                      - torch.quantile(_st, 0.50))
                           + torch.relu(torch.quantile(_sg, 0.75)
                                        - torch.quantile(_st, 0.75)))
            _jg = torch.cat([_hj_g[_fi].reshape(-1), _vj_g[_fi].reshape(-1)])
            _jt = torch.cat([_hj_t[_fi].reshape(-1), _vj_t[_fi].reshape(-1)])
            _lcj = _lcj + (torch.relu(torch.quantile(_jg, 0.75)
                                      - torch.quantile(_jt, 0.75))
                           + torch.relu(torch.quantile(_jt, 0.99)
                                        - torch.quantile(_jg, 0.99)))
        _lcs = _lcs / _F
        _lcj = _lcj / _F
        _lstd = torch.relu(_xt.std(dim=(1, 2)) * 0.85
                           - _xf.std(dim=(1, 2))).mean()
        loss_color = _lcs + _lcj + _lstd

    # ★ 颜色频点专项（w=COL_BAND 是颜色的唯一载体；w=0 过亮时用 DC_BAND_W 压）
    #   幅度项（COL_BAND_W）决定「这里有没有色差」，
    #   相位项（COL_PH_W）决定「色相往哪个颜色偏」——必须成对，否则只有彩色噪点。
    loss_cb = torch.zeros((), device=device)
    loss_cbp = torch.zeros((), device=device)
    loss_dcb = torch.zeros((), device=device)
    if COL_BAND_W > 0:
        loss_cb = loss_color_band(a, at, win=WIN, w_fft=W_FFT, col_w=COL_BAND)
    if COL_PH_W > 0:
        loss_cbp = loss_color_phase(pe, d_model=D, w_fft=W_FFT, col_w=COL_BAND)
    if DC_BAND_W > 0:
        # 单侧：只在生成 DC 超过 target 时罚（不压欠亮），避免整体压暗
        _lp0 = torch.log1p(a.clamp(min=1e-8)).reshape(1, -1, WIN, W_FFT)[:, :, :, 0].mean()
        _lt0 = torch.log1p(at.clamp(min=1e-8)).reshape(1, -1, WIN, W_FFT)[:, :, :, 0].mean()
        loss_dcb = torch.relu(_lp0 - _lt0)

    loss = (loss_amp + loss_ph + GEN_W * loss_gen
        + SPEC_W * loss_spec + HF_GUARD_W * loss_hf + LO_FLOOR_W * loss_lo
        + BAND_ABS_W * loss_ba + BAND_CAP_W * loss_bc
        + H_SPEC_W * loss_hs
        + PHASE_BAND_W * loss_pb + PHASE_FLOOR_W * loss_pf
        + DISC_W * loss_disc
        + MARGIN_W * loss_margin
        + WCS_W * loss_wcs
        + PRES_W * loss_pres
        + COLOR_W * loss_color
        + COL_BAND_W * loss_cb
        + COL_PH_W * loss_cbp
        + DC_BAND_W * loss_dcb)

    with torch.no_grad():
        _h = _drop_hist if dropped else _keep_hist
        _h["n"] += 1
        _h["wcos"] += float((w_ph * torch.cos(pe)).sum() / w_ph.sum())
        _h["wcos_c"] += _wc_now
        _h["amp"] += float(loss_amp)
        _h["phase"] += float(loss_ph)

    (loss / ACC_K).backward()
    if (step % ACC_K) == 0:
        opt.step()
        model.pmnncore.project_w2()
        opt.zero_grad()
    if step % 100 == 0:
        with torch.no_grad():
            _sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
            _safe_torch_save(_sd, OUT_DIR / "guided_ckpt.pt")
            del _sd
    if step % 600 == 0:
        with torch.no_grad():
            _sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
            _safe_torch_save(_sd, OUT_DIR / f"guided_ckpt_step{step}.pt")
            del _sd
    if step % 50 == 0 or step == 1:
        with torch.no_grad():
            wcos = (w_ph * torch.cos(pe)).sum().item() / w_ph.sum().item()
            dabs = dth.abs().mean().item() if dth is not None else float("nan")
            r_pred = hf_ratio(a, win=WIN, w_fft=W_FFT).mean().item()
            r_tar_ = hf_ratio(at, win=WIN, w_fft=W_FFT).mean().item()
            ratio_hi = r_pred / max(r_tar_, 1e-12)
            # 两素材 dtheta 余弦（★ 与 overfit_cond3 不同：传 target_field=None，
            #   与推理口径一致；diag["dtheta"] 本就只由 cond_init 产生，
            #   与 target_field 无关，同时省掉 fft_bypass 的算力）
            dcos = float("nan")
            if INIT_MODE == "init":
                _dts = []
                for _dsrc in range(2):
                    _dcv = torch.from_numpy(build_one_cv(_dsrc, (1, 1, 1))).to(device)
                    _, _, _dd = model(texts=None, target_field=None,
                                      t_coord=t, s_coord=s, batch_size=1,
                                      cond_vec=_dcv, use_cond_field=True)
                    _dt = _dd.get("dtheta")
                    if _dt is not None:
                        _dts.append(_dt.reshape(-1).float())
                if len(_dts) == 2:
                    _n = float(_dts[0].norm() * _dts[1].norm())
                    if _n > 1e-8:
                        dcos = float(_dts[0] @ _dts[1] / _n)
            # dropped / kept 滑动汇总
            _dh, _kh = _drop_hist, _keep_hist
            _fmt = lambda h: (f"n={h['n']:3d} wcos={h['wcos']/max(h['n'],1):+.3f}"
                              f" wcos_c={h['wcos_c']/max(h['n'],1):+.3f}"
                              f" amp={h['amp']/max(h['n'],1):.4f}"
                              f" ph={h['phase']/max(h['n'],1):.3f}")
            _el = time.time() - _t0
            print(f"  [{step:4d}] src={src} {combo_name:5s} "
                  f"{'DROP' if dropped else 'keep'}"
                  f"{' FORCE' if forced else ''} loss={loss.item():.4f} "
                  f"amp={loss_amp.item():.4f} phase={loss_ph.item():.4f} "
                  f"gen={loss_gen.item():.4f} spec={loss_spec.item():.3f} "
                  f"hfg={loss_hf.item():.4f} lo={loss_lo.item():.4f} "
                  f"ba={loss_ba.item():.4f} hs={loss_hs.item():.3f} pb={loss_pb.item():.4f} "
                  f"disc={float(loss_disc):.4f} "
                  f"mg={float(loss_margin):.3f}(o{_mgo:+.2f}) "
                  f"wcs={float(loss_wcs):.3f} "
                  f"pres={float(loss_pres):.4f} "
                  f"color={float(loss_color):.3f} "
                  f"wcos={wcos:+.4f} wcos_c={_wc_now:+.4f} |dθ|={dabs:.3f} "
                  f"dθcos={dcos:+.3f} hi/lo比={ratio_hi:.2f} "
                  f"[{int(_el)}s/{step*max(_el/step,0.1):.0f}s]", flush=True)
            print(f"          drop: {_fmt(_dh)}", flush=True)
            print(f"          keep: {_fmt(_kh)}", flush=True)
            _drop_hist = {"n": 0, "wcos": 0.0, "wcos_c": 0.0, "amp": 0.0, "phase": 0.0}
            _keep_hist = {"n": 0, "wcos": 0.0, "wcos_c": 0.0, "amp": 0.0, "phase": 0.0}

sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
_safe_torch_save(sd, OUT_DIR / "guided_ckpt.pt")
_bak = OUT_DIR / f"guided_ckpt_{STEPS}_{time.strftime('%H%M')}.pt"
_safe_torch_save(sd, _bak)
print(f"[BACKUP] ckpt 备份 -> {_bak}")

# ---------------------------------------------------------------- 条件生成评测
CONDS = [
    ("文生", cv_text),
    ("图生", cv_first),
    ("图文生", cv_tf),
    ("GIF本身", cv_gif),
    ("三路全给", cv_full),
]

print(f"\n[4] 五种条件 × 两素材 交叉生成（全部 target_field=None，纯条件）")
model.eval()
results = {}
absm = {}
with torch.inference_mode():
    for cv_name, cv in CONDS:
        for src_i in range(N_SRC):
            one = torch.from_numpy(cv[src_i:src_i + 1]).to(device)
            a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=one, use_cond_field=True)
            key = f"{cv_name}|src{src_i}"
            to_gif(a, th, OUT_DIR / f"{key.replace('|', '_')}.gif", key)
            z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
            fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
            gen = np.stack(decode_field_to_frames(
                fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
            ps = [psnr(gen, raw_imgs[k]) for k in range(N_SRC)]
            results[key] = ps
            # content 层 wcos（生成场 vs target 场；★ 两侧都要 .cpu()）
            _a_t = a_tar[src_i].cpu()
            _th_t = th_tar[src_i].cpu()
            _pe_g = torch.remainder(th[0].cpu() - _th_t + math.pi, 2 * math.pi) - math.pi
            _wp = (_a_t ** 2).clamp(min=1e-8)
            absm[key] = dict(psnr_self=ps[src_i], ac1=ac1(gen),
                             sobel=sobel_mag(gen), hilo=hilo(gen),
                             wcos_c=wcos_of(_pe_g, _wp, _CONTENT_MASK))
            winner = "自己" if ps[src_i] > ps[1 - src_i] else "错了"
            print(f"    {key}: vs_A={ps[0]:6.2f} dB  vs_B={ps[1]:6.2f} dB  "
                  f"wcos_c={absm[key]['wcos_c']:+.3f} -> {winner}")

# ---------------------------------------------------------------- 绝对判据
print(f"\n[5] ★ 绝对质量判据（一票否决；基线 = 原始素材）")
print("-" * 104)
print(f"{'对象':<22}{'PSNR':>8}{'噪声基线':>10}{'ac1':>8}{'sobel':>8}{'hilo':>8}{'素材hilo':>9}{'判决':>8}")
print("-" * 104)
abs_ok = True
nb_cache = {}
for i in range(len(GIF_SPECS)):
    ref = raw_imgs[i]
    nb = noise_baseline(ref)
    nb_cache[i] = nb
    print(f"{'素材' + str(i) + ' 原图':<22}{'∞':>8}{nb[0]:>9.2f}dB{ac1(ref):>+8.3f}"
          f"{sobel_mag(ref):>8.2f}{hilo(ref):>8.2f}{hilo(ref):>9.2f}{'基准':>8}")
    k = f"三路全给|src{i}"
    if k not in absm:
        continue
    m = absm[k]
    p_ok = m["psnr_self"] > nb[0] + 2.0
    h_ok = m["hilo"] <= hilo(ref) * 3.0
    a_ok = m["ac1"] >= ac1(ref) * 0.85
    ok = p_ok and h_ok and a_ok
    abs_ok = abs_ok and ok
    why = []
    if not p_ok:
        why.append("PSNR≈噪声")
    if not h_ok:
        why.append("高频爆")
    if not a_ok:
        why.append("低频塌")
    print(f"{k:<22}{m['psnr_self']:>8.2f}{nb[0]:>9.2f}dB{m['ac1']:>+8.3f}"
          f"{m['sobel']:>8.2f}{m['hilo']:>8.2f}{hilo(ref):>9.2f}"
          f"{'PASS' if ok else 'FAIL':>8}  {'/'.join(why)}")
print("-" * 104)

print(f"\n[6] 相对判据（margin/wcos —— 只能作辅助）")
all_ok = True
for cv_name, _ in CONDS:
    k0, k1 = f"{cv_name}|src0", f"{cv_name}|src1"
    if k0 not in results:
        continue
    d0 = results[k0][0] - results[k0][1]
    d1 = results[k1][1] - results[k1][0]
    ok = d0 > 0 and d1 > 0
    all_ok = all_ok and ok
    print(f"    [{'PASS' if ok else 'FAIL'}] {cv_name}: "
          f"margin_A={d0:+6.2f} dB  margin_B={d1:+6.2f} dB")

print(f"\n[7] ★ 总判定（绝对判据一票否决）")
print(f"    相对判据（margin 全对）    : {'PASS' if all_ok else 'FAIL'}")
print(f"    绝对判据（非噪声/不爆高频）: {'PASS' if abs_ok else 'FAIL'}")
final_ok = all_ok and abs_ok
print(f"    总判定: {'PASS ✅' if final_ok else 'FAIL ❌'}")

# ---------------------------------------------------------------- summary.json
summary = {
    "tag": os.environ.get("OVERFIT_TAG") or "guided",
    "steps": STEPS, "guid_p": GUID_P, "lr": LR, "cond_lr": COND_LR,
    "params": _n_par, "gen_w": GEN_W, "fft_w": FFT_W, "acc_k": ACC_K,
    "warm_blocks": WARM_BLOCKS, "init_mode": INIT_MODE,
    "ffn_hidden": FFN_HIDDEN, "n_modules": N_MODULES,
    "hf_guard_w": HF_GUARD_W, "lo_floor_w": LO_FLOOR_W, "band_abs_w": BAND_ABS_W,
    "spec_w": SPEC_W, "phase_band_w": PHASE_BAND_W, "disc_w": DISC_W,
    "margin_w": MARGIN_W, "wcs_w": WCS_W, "ph_abs_w": PH_ABS_W,
    "margin_ok": bool(all_ok), "abs_ok": bool(abs_ok),
    "final_ok": bool(final_ok),
    "results": results,
    "absm": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in absm.items()},
    "noise_baseline": {str(i): nb_cache[i][0] for i in nb_cache},
}
with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as _f:
    json.dump(summary, _f, ensure_ascii=False, indent=2)
print(f"\n[SUMMARY] -> {OUT_DIR / 'summary.json'}")
print(f"    GIF 输出目录: {OUT_DIR}")
print("=" * 74)
