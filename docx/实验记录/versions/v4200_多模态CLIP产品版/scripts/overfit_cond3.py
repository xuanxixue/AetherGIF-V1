# -*- coding: utf-8 -*-
"""三向量耦合过拟合测试 —— 文本 / 首帧 / GIF本身 三路向量，交叉训练 + 缺省生成评测。

【用户的方案（2026-10-16）】
用 int8 量化 Qwen3-VL-Embedding-2B 对同一 GIF 提取三路对齐向量：
  ① 文本描述向量  ② 首帧静图向量  ③ GIF 本身（真视频分支）向量

【交叉训练（2026-10-16 任务 #70，用户拍板）】
训练时每步**随机选一种条件组合**（七种：单路×3 / 双路×3 / 三路全给×1），
缺省模态置零向量、has 标志位如实置 0。这样模型在训练里见过所有缺省
形态，推理时单独输入任何一路都不会分布外 —— 这正是「加权融合但单独
输入还能生成」的解法（训练分布覆盖推理分布）。
  ★ gif 一路用 gif3_video.npy（真视频分支：PyAV 真 fps + 时序重采样），
    不用旧的 gif3.npy（帧序列，无时序重采样）。

【判据：交叉还原矩阵】
             生成目标
条件源GIF_A  A_psnr  B_psnr
条件源GIF_B  B_psnr  A_psnr
对角（自己条件→自己目标）必须显著大于非对角。

【运行】
cd 到 v4200 根目录后
  C:/Python312/python.exe -u scripts/overfit_cond3.py [STEPS]
  OVERFIT_TAG=xxx ...         # 产物落 outputs_real/<tag>/，隔离防覆盖
"""
import math
import os
import sys
import time
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

import random
import numpy as np
import torch
from PIL import Image

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames, load_gif_frames
from physics.kuramoto_sync import build_spacetime_index
from physics.cond_init import CondInitModulator
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import CLIP_EMB_DIM, N_COND3, build_cond_vector3
from train.spectral_loss import (band_profile, hf_ratio, lo_ratio,
                                 loss_band_abs, loss_band_abs_cap,
                                 loss_hf_guard, loss_h_spec, loss_lo_floor,
                                 loss_phase_band, loss_phase_band_floor,
                                 loss_spec_shape)

# ---------------- 编解码口径（与 overfit_cond2.py 完全一致） ----------------
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1  # 25

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
LR = 3e-4
COND_LR = 1e-3
AMP_P = 0.5
GEN_W = 2.0
INIT_W = 2.0
INIT_MODE = os.environ.get("INIT_MODE", "init")
N_MODULES = int(os.environ.get("N_MODULES", "8"))
MOD_GAMMA = float(os.environ.get("MOD_GAMMA", "0.15"))
# ★ 用户三步扩容旋钮（2026-10-16）：
#   1) FFN_HIDDEN 物理核容量（StandingWaveFFN 宽度，256→512 探针）
#   2) GUARD_W    dtheta 幅度守卫权重（1.0 太弱 |dθ| 顶满 0.785，待提到 5~10）
#   3) STEPS      训练长度（命令行参数）
#   按用户定调顺序逐步激活：先容量、再守卫、再延长。
FFN_HIDDEN = int(os.environ.get("FFN_HIDDEN", "256"))
GUARD_W = float(os.environ.get("GUARD_W", "1.0"))
# ★ A/B 频谱监督旋钮（2026-10-18，绝对质量诊断后新增）：
#   诊断判决：codec 天花板 51 dB，模型只 13.10 dB（= 同 std 高斯噪声 13.03）；
#   生成 hi/lo = 0.23 vs 素材 0.01（23 倍），重建与生成逐位相同
#   → 问题在训练目标/频谱层，高频完全无监督。
#   SPEC_W   B 项权重：分带能量占比 log 比值 L1（形状量，抗整体缩放作弊）
#   HF_GUARD_W  A 项权重：高频占比单侧守卫（只罚超标，允许更锐）
#   HF_SLACK    单侧容差：ratio 超过 target×(1+slack) 才罚
#   LO_FLOOR_W  反向守卫权重：低频占比下限（A 项单侧的必要配套，
#               否则 A 会把模型从「高频噪点」推到「全糊」——实测 150 步
#               sobel 28.31→4.33、ac1 0.587→0.305，从过锐翻到糊）
#   LO_FLOOR    下限系数：低频占比不得低于 target×此值
SPEC_W = float(os.environ.get("SPEC_W", "0.0"))
H_SPEC_W = float(os.environ.get("H_SPEC_W", "0.0"))
HF_GUARD_W = float(os.environ.get("HF_GUARD_W", "0.0"))
HF_SLACK = float(os.environ.get("HF_SLACK", "1.5"))
LO_FLOOR_W = float(os.environ.get("LO_FLOOR_W", "0.0"))
LO_FLOOR = float(os.environ.get("LO_FLOOR", "0.7"))
# ★ 梯度累积步数（第二十一死穴修复）：相邻步梯度余弦≈0 → 方向随机 → 随机行走。
#   前向已验证确定（S0），故平均 K 步梯度可把随机方向抵消。
#   ACC_K=1 退化为原行为。K 越大噪声越低但等效 lr 越小（已除以 K 补偿）。
ACC_K = max(1, int(os.environ.get("ACC_K", "1")))
# ★ 只训练最后 K 个 block（第二十二死穴候选：深层梯度一致、浅层梯度是噪声）
#   实测块内跨步梯度余弦：block9/10/11≈+0.63~0.65（健康），
#   block0~7 ≈ 0.00（纯随机游走，更新它们只会破坏前向变换）。
#   WARM_BLOCKS=0 → 全训（现行）；=4 → 只训 block 8~11。
WARM_BLOCKS = max(0, int(os.environ.get("WARM_BLOCKS", "0")))
# ★ C/D 绝对幅度与相位专项权重（2026-08-18）
#   BAND_ABS_W   per-band log1p 绝对幅度对齐（补spec 的分辨率盲区）
#   BAND_CAP_W   per-band 绝对幅度单侧上限（防某一 band 爆）
#   BAND_CAP_MULT   上限倍数（target×此值才罚）
#   PHASE_BAND_W per-band 相位分组监督（高频梯度提升 14~21 倍）
#   PHASE_FLOOR_W  相位下限守卫（罚「低于随机」，兜底）
BAND_ABS_W = float(os.environ.get("BAND_ABS_W", "0.0"))
BAND_CAP_W = float(os.environ.get("BAND_CAP_W", "0.0"))
BAND_CAP_MULT = float(os.environ.get("BAND_CAP_MULT", "3.0"))
PHASE_BAND_W = float(os.environ.get("PHASE_BAND_W", "0.0"))
PHASE_FLOOR_W = float(os.environ.get("PHASE_FLOOR_W", "0.0"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# 两素材（描述与 encode_cond3.py 一致）
GIF_SPECS = [
    ("8-ball_1f3b1.gif", "8号球图标"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "前辈前辈表情包"),
]

OUT_DIR = ROOT / "outputs_real" / (os.environ.get("OVERFIT_TAG") or "cond3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
if os.environ.get("OVERFIT_TAG"):
    print(f"[OUT] OVERFIT_TAG={os.environ['OVERFIT_TAG']}，产物落 {OUT_DIR}（与 cond3 隔离）")


def _safe_torch_save(obj, path, retries=6):
    """原子保存 checkpoint（2026-10-23 加固，替代裸 torch.save）。

    为什么必须换掉直接 torch.save：
      1) Windows 上目标文件可能被杀软 / 搜索索引 / 上一进程残留句柄短暂占用，
         直接写会抛 PermissionError（color4_long 就是这么崩在 200 步保存的）；
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
            os.replace(tmp, path)
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


# ---------------- D 项：绝对质量指标（2026-10-18） ----------------
# margin/wcos 是**相对**指标，全噪声图上同样成立 → 必须配绝对判据。
# 铁律：任何验收都要同时报「噪声基线」，否则 PSNR 数字无意义。
def ac1(x):
    """相邻像素自相关：真实图像 >0.5，纯噪声 ≈0。切片法不用 roll（环绕伪影）。"""
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


# ---------------------------------------------------------------- 素材加载
print("=" * 74)
print(f"[1] 加载 {len(GIF_SPECS)} 张素材 -> 频域场")
gif_dir = ROOT / "data" / "gif_raw" / "noto"
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

ref_imgs = []
# ★ 绝对判据基准（2026-10-18 D 项）：raw_imgs 才是**原始素材帧**。
#   ref_imgs 是 codec 天花板（field 往返），只能当 PSNR 上限参照，
#   拿它当"素材"会让所有数字虚高。必须与 encode 同口径 linspace 对齐。
from codec.gif_codec import resample_frames
raw_imgs = []
for i in range(len(GIF_SPECS)):
    z = (a_tar[i] * torch.exp(1j * th_tar[i])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    ref_imgs.append(np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32))
    _raw = load_gif_frames(str(gif_paths[i]), resize_to=(IMG, IMG))
    raw_imgs.append(resample_frames(_raw, TS * WIN).astype(np.float32))

# ---------------------------------------------------------------- 条件加载
print(f"\n[2] 加载三路原始向量（data/clip_cache/，gif=真视频分支）")
CLIP_CACHE = ROOT / "data" / "clip_cache"
text3 = np.load(CLIP_CACHE / "text3.npy")
first3 = np.load(CLIP_CACHE / "first3.npy")
gif3v = np.load(CLIP_CACHE / "gif3_video.npy")
assert text3.shape == first3.shape == gif3v.shape == (2, CLIP_EMB_DIM), \
    f"三路向量形状错: {text3.shape} {first3.shape} {gif3v.shape}"
for n_, v_ in [("文本", text3), ("首帧", first3), ("GIF真视频", gif3v)]:
    print(f"    {n_:8s}: src0-vs-src1 cos={float(v_[0] @ v_[1]):+.4f}")

# 评测用的预拼缓存（cv3_* 已用 gif3_video 重建，见 rebuild_cv3_video.py）
cv_full = np.load(CLIP_CACHE / "cv3_full.npy")      # 三路全给
cv_text = np.load(CLIP_CACHE / "cv3_text.npy")      # 只文本
cv_first = np.load(CLIP_CACHE / "cv3_first.npy")    # 只首帧
cv_gif = np.load(CLIP_CACHE / "cv3_gif.npy")        # 只 GIF本身
cv_tf = np.load(CLIP_CACHE / "cv3_tf.npy")          # 文本+首帧
assert cv_full.shape[1] == N_COND3, f"缓存维度错 {cv_full.shape}"
print(f"    cond_vec3 维度 = {cv_full.shape[1]}")

# ★ 交叉训练的七种条件组合（单路×3 / 双路×3 / 三路×1，等概率）
#   has = (has_text, has_first, has_gif)；缺省路用零向量 + flag=0，
#   模型训练里见过所有缺省形态 → 推理时单独输入任何一路都不分布外。
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

# ---------------------------------------------------------------- 训练
print(f"\n[3] 训练 {STEPS} 步（batch=1，随机素材 × 随机条件组合 = 交叉训练）")
print(f"    init_mode={INIT_MODE}, n_modules={N_MODULES}, mod_gamma={MOD_GAMMA}")
print(f"    ffn_hidden={FFN_HIDDEN}（物理核容量）, guard_w={GUARD_W}（dtheta 守卫）")
print(f"    SPEC_W={SPEC_W}（B 分带能量形状）, HF_GUARD_W={HF_GUARD_W}"
      f"（A 高频单侧守卫, slack={HF_SLACK}）")
print(f"    LO_FLOOR_W={LO_FLOOR_W}（低频占比下限守卫, floor={LO_FLOOR}，"
      f"防 A 项把模型推成「糊」）")
print(f"    ACC_K={ACC_K}（梯度累积窗口，第二十一死穴修复：相邻梯度余弦≈0 "
      f"→ 方向随机 → 随机行走；1=关闭）")
print(f"    WARM_BLOCKS={WARM_BLOCKS}（只训最后 K 个block；实测块内跨步梯度余弦 "
      f"block9-11≈+0.65 健康、block0-7≈0.00 纯噪声；0=全训）")
print(f"    BAND_ABS_W={BAND_ABS_W}（C 绝对幅度）, BAND_CAP_W={BAND_CAP_W}"
      f"（单侧上限×{BAND_CAP_MULT}）")
print(f"    PHASE_BAND_W={PHASE_BAND_W}（D 相位分组监督）, "
      f"PHASE_FLOOR_W={PHASE_FLOOR_W}（相位下限守卫）")
print(f"    条件组合（等概率）：{[n_ for n_, _ in COND_COMBOS]}")
torch.manual_seed(1234)
random.seed(1234)
# ★ 条件通路白名单（优化器分组 + warm 阶段共用）。
#   fft_bypass. 必须包含 —— 它是第 4 路条件的载体，
#   漏了会出现「旁路参数不训练」和「|dθ|=0.000」的假阴性。
#   （此前 WARM_BLOCKS 只匹配 pmnncore.blocks.*，把条件通路一起冻了，
#     教训见记忆：_warm() 里条件通路必须直接 return True。）
_CONDPFX = ("cond_field.", "cond_init.", "fft_bypass.")

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=FFN_HIDDEN, init_mode=INIT_MODE,
                       n_modules=N_MODULES, mod_gamma=MOD_GAMMA).to(device)
# ★ 断点续训（2026-10-19）：长训练动辄 2 小时，必须能从 ckpt 接。
#   RESUME=<路径> 载入权重（严格匹配才接，架构变了就当没给）。
#   ⚠ 优化器状态不恢复——AdamW 动量从零重启会短暂抖动，
#   但对过拟合设定无害，换来的是不用存两倍 ckpt。
RESUME = os.environ.get("RESUME", "").strip()
if RESUME:
    _rp = Path(RESUME) if Path(RESUME).is_absolute() else ROOT / RESUME
    if _rp.exists():
        _rsd = torch.load(_rp, map_location=device)
        _rsd = {k: v for k, v in _rsd.items() if not k.startswith("qwen.")}
        # ★★ shape 不匹配的层跳过（2026-10-21）：
        #   矩阵测试会改 COND_LOWRANK（R=16 → 64），此时 to_phase.0 /
        #   to_amp.0 的 shape 变了，直接 load_state_dict 会抛
        #   "size mismatch" 崩掉（fix_r2 就这么崩过）。
        #   正确做法：只过滤掉 shape 不同的键，其余权重照常继承，
        #   让新的低秩层用随机初始化从头学，物理栈/其他层不丢。
        _own = model.state_dict()
        _drop = [k for k, v in _rsd.items()
                 if k in _own and tuple(_own[k].shape) != tuple(v.shape)]
        # ★★ 低秩扩展继承（2026-10-21）：保证 R 变更组与继承组起点公平。
        #   COND_LOWRANK 16→64 时 to_phase.0 [16,512]→[64,512]、
        #   to_phase.1 [30720,16]→[30720,64]，直接随机初始化会让 B/D 组
        #   从「低秩层零起点」学起（实测第1 步 wcos=−0.2871，负的），
        #   而 A/C/E 组从 +0.5077 起步 —— 对照不公平，无法判断是
        #   「方向无效」还是「起点差」。
        #   修法：把旧低秩权重**复制**到新形状的前R_old 行
        #   （其余行补 0），相当于「用 R_old 的解扩展初始化更大秩」。
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
                    if _k.endswith(".0.weight"):      # [R, hidden] -> [R_new, hidden]
                        _new = torch.zeros_like(_tgt)
                        _new[:_v.shape[0]] = _v
                        _rsd[_k] = _new
                    elif _k.endswith(".1.weight"):    # [out, R] -> [out, R_new]
                        _new = torch.zeros_like(_tgt)
                        _new[:, :_v.shape[1]] = _v
                        _rsd[_k] = _new
                    else:                             # bias [out] 形状不变也跳过
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
            print(f"[RESUME] ★跳过 {len(_drop)} 个 shape 不匹配的层"
                  f"（低秩改变，随机初始化）：{_drop}")
            # ★ 2026-10-23 加固：`.0.weight` 是 RankLinear 低秩投影，随 COND_LOWRANK
            #   变动被跳过属设计意图；但 expander 层（.1.*）被跳过说明整个条件通路
            #   架构口径不对（典型：COND_INIT_PC / COND_PC_RANK 与 ckpt 不符），
            #   继续训练会让条件通路随机初始化 —— 几小时白跑，直接中止。
            _crit = [k for k in _drop
                     if k.startswith(("cond_init.", "cond_field.", "fft_bypass."))
                     and not k.endswith(".0.weight")]
            if _crit and os.environ.get("RESUME_ALLOW_DROP_COND", "0").lower() not in (
                    "1", "true", "yes", "on"):
                print(f"[RESUME][中止] 关键条件层 shape 不匹配 {len(_crit)} 个："
                      f"{_crit[:8]}", flush=True)
                print(f"[RESUME][中止] 当前 COND_INIT_PC={os.environ.get('COND_INIT_PC')}"
                      f" COND_PC_RANK={os.environ.get('COND_PC_RANK')}", flush=True)
                print(f"[RESUME][中止] 继续会让条件通路随机初始化，已中止。"
                      f" 确实要这么做请设 RESUME_ALLOW_DROP_COND=1", flush=True)
                raise SystemExit(3)
        else:
            print(f"[RESUME] ★ 低秩层已按 RESUME_R_OLD={_R_OLD} 扩展继承"
                  f"（起点公平）")
    elif os.environ.get("RESUME_STRICT", "1").lower() in ("0", "false", "no", "off"):
        print(f"[RESUME] ★ 路径不存在，忽略（RESUME_STRICT=0 已关闭严格模式）：{_rp}")
    else:
        # ★ 2026-10-23 加固：指定了续训却找不到文件，继续就是从随机初始化开始训练，
        #   与预期严重不符。直接中止，不静默降级。
        print(f"[RESUME][中止] 指定了续训但路径不存在：{_rp}", flush=True)
        print(f"[RESUME][中止] 当前 ROOT = {ROOT}", flush=True)
        print(f"[RESUME][中止] 如需旧行为（找不到就从头训），设 RESUME_STRICT=0", flush=True)
        raise SystemExit(2)
else:
    print("[RESUME] 未指定，从随机初始化开始")
_cond_params = [p for n, p in model.named_parameters()
                if n.startswith(_CONDPFX)]
_other_params = [p for n, p in model.named_parameters()
                 if not n.startswith(_CONDPFX)]
# ★ 只训最后 WARM_BLOCKS 个 block（其余 requires_grad=False → 不进优化器）
if WARM_BLOCKS > 0:
    _nblk = len(model.pmnncore.blocks)
    _keep = {f"pmnncore.blocks.{i}." for i in range(_nblk - WARM_BLOCKS, _nblk)}

    def _warm(n):
        # ★ 修：条件通路（cond_field./cond_init.）不受 WARM 影响，
        #   否则 _keep 只匹配 pmnncore.blocks.* 会把条件通路一起冻掉，
        #   表现为 |dθ|=0.000 的假阴性（2026-10-18 实测）。
        if n.startswith(_CONDPFX):
            return True
        return any(n.startswith(k) for k in _keep)

    _frozen = [n for n, _ in model.named_parameters() if not _warm(n)]
    for n, p in model.named_parameters():
        if not _warm(n):
            p.requires_grad_(False)
    _cond_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith(_CONDPFX)]
    _other_params = [p for n, p in model.named_parameters()
                     if p.requires_grad
                     and not n.startswith(_CONDPFX)]
opt = torch.optim.AdamW([
    {"params": _other_params, "lr": LR},
    {"params": _cond_params, "lr": COND_LR},
], lr=LR)
print(f"    优化器分组：其他={len(_other_params)} lr={LR}，条件通路={len(_cond_params)} lr={COND_LR}")
model.train()

# ★★ 第二十五步：cond_init 预热阶段（2026-10-18）
#   实测铁律：theta0 = theta0 + dtheta 让 loss_amp/loss_ph 的梯度**穿过整条
#   12 层物理栈反传到 dtheta**。物理栈此刻还不会用结构，于是「把 dtheta 压到 0」
#   是降低重建 loss 的最快捷径 —— 与「整体压低振幅捷径」同源，只是发生在相位上。
#   证据：perband5_150.log 里 |dθ|=0.006 配dθcos=+1.000 自相矛盾
#   （零向量余弦不可能为 1），而只训 cond_init 的秒级探针 22 秒就长到 0.77。
#   修法：前 COND_ONLY_STEPS 步**只训 cond_init**，物理栈全冻，
#   loss 只用 supervise（不经过物理栈 ⇒ 无压制梯度），先让通路长出结构再联合训。
# ★★ 第二十六步：对比判别损失 DISC_W（2026-10-18）
#   实测铁律（diag_cond_discriminability.py，训练后 ckpt）：
#     CLIP 三路原始向量两素材 cos = 0.33/0.19/0.27（区分度良好）
#       → build_cond_vector3 cos = 0.632
#       → **proj(hidden) cos = 0.9869（训练后反而恶化，未训练 0.9844）**
#       → dtheta 跨素材 cos = 0.99866，逐组 0.9996~0.99996，相对差异仅 13%
#   ⇒ **条件通路输出与素材无关的通用场**，这就是 dθcos 全程 +1.000 的真凶。
#   为什么supervise 修不了：它只要求「对齐某个 target 的方向场」，
#   而两素材 target 的主导方向高度相似（都是「一张图的大致布局」），
#   模型输出**一个折中通用场**就能降 loss ⇒ 区分素材从来不是必要的。
#   修法：显式加**只有分开素材才能降**的项——
#     用素材0 的 target 方向做判别：
#       L_disc = relu(cos(d0,dir0) − cos(d1,dir0) − margin)
#     d0=dtheta(cond of素材0)，d1=dtheta(cond of素材1)，dir0=素材0 的目标方向场。
#     模型若对两素材输出同一场，则两项相等 → L_disc=margin（顶格）→ 强惩罚。
DISC_W = float(os.environ.get("DISC_W", "0.0"))
# 目标余弦：要求两素材 dtheta 的跨素材余弦压到此处以下
DISC_TGT_COS = float(os.environ.get("DISC_TGT_COS", "0.60"))
# dθ 项的宽松目标（v4：避免把两素材推成反相关）
DISC_DTGT_COS = float(os.environ.get("DISC_DTGT_COS", "2.0"))
# ★★ 第三十一死穴（2026-10-20）：wcos margin 判别（旧判别双失效后的替代）
#   旧 h 项达标即死（cos 0.27<0.60 门限），旧 dθ 副项在 θ̂ 模式下
#   dtheta 被 wrap 随机化 cos≈0，目标 0.9 恒成立 ⇒ 两分支 3500 步恒 0。
#   新靶点直接挂「生成结果」：本步素材的相位对齐 wcos_self 必须高出
#   「另一素材条件生成的对齐 wcos_other」至少 MARGIN_DELTA。
#   MARGIN_W=0 关闭（对照用）。
MARGIN_W = float(os.environ.get("MARGIN_W", "0.0"))
MARGIN_DELTA = float(os.environ.get("MARGIN_DELTA", "0.05"))
# ★ wcos 绝对值目标（第三十一死穴配套）：margin 管判别，wcs 管「对齐强度」。
#   relu(WCS_TGT − wcos_self)：差多少罚多少，达标即停。
#   wcos 是「相位余弦」本身，比 loss_ph 的 MSE 更直接对齐「角度」。
WCS_W = float(os.environ.get("WCS_W", "0.0"))
WCS_TGT = float(os.environ.get("WCS_TGT", "0.30"))
# 方向判别的目标余量（辅助项，与 target 方向对齐度）
DISC_MARGIN = float(os.environ.get("DISC_MARGIN", "0.10"))
# ★ 第二十七死穴修复（2026-10-19）：绝对相位对齐权重
#   实测 |dθ|=0.77 而理想相位差≈π ⇒差4 倍搬不动随机初态
#   （随机 θ0 与 target 的 cos=0.00004；若 dθ=0.5×理想差则 cos=+0.4877）
#   abs_w>0 时 supervise 直接监督「θ0+dθ ≈ θ_tar」的相位差 MSE，
#   这是**唯一对 |dθ| 幅度产生真实压力**的项
#   （方向场loss 对幅度无要求 ⇒ dθ 大小不会自己长起来）。
PH_ABS_W = float(os.environ.get("PH_ABS_W", "0.0"))
# 第 4 路 FFT 场旁路的监督权重（形状+尺度双项已在模块内分开加权）
FFT_W = float(os.environ.get("FFT_W", "1.0"))



def loss_discrim(d0, d1, th_target, h0=None, h1=None):
    """对比判别损失（第二十六步，v3）。

    d0, d1  : [B,N,G] 两素材在同一条件组合下的 dtheta
    th_target: [B,N,D] 本步素材的目标相位
    h0, h1   : [B,hidden] **proj 的直接输出**（可选，但这是关键）

    ★ v2 教训（diag_proj_tail_act2.py，真实权重逐段实测）：
      跨素材 cos  cond_vec 0.632 → Linear1 0.719 → SiLU1 0.714
                   → **Linear2 0.967（跳跃点）** → SiLU2 h=0.987
                   → to_phase低秩 0.999 → dtheta 0.9987
      ⇒区分度死在 **proj 第二层 Linear**，**不是**激活函数
        （SiLU/Tanh/ReLU/LayerNorm/零均值归一 五种修法全测，
          h 的 cos 都在 0.97~0.99，**换激活一律无效**）。
      低秩段只再吃掉 3%，是无辜的。
    ★ v3 修法：判别项**直接加在 h（proj输出）上**，
      梯度直达 proj，绕过 R=16 低秩瓶颈。
      原来的只作用在 dtheta 上，梯度要穿 16 维子空间，
      proj 收到的压力被稀释了 32倍 → 压了 3000 步纹丝不动。
    ★ v4 平衡（实测 v3/v4 把 cos_dθ 压到 −1.0）：
      放宽 dθ 项目标到 0.90 **依然压不住** ⇒ cos_dθ=−1 是
      **h 分离后的必然结果**（同一套 to_phase 作用在已分开的表示上，
      必然给出反向场），不是 dθ 项推的。
      故默认 DISC_DTGT_COS=2.0（等价关闭），只留 h 项独立负责判别。
    """
    _loss = torch.zeros((), device=device)
    # 主项：h 的跨素材余弦（判别力的直接度量，梯度直达 proj）
    if h0 is not None and h1 is not None:
        _f0, _f1 = h0.reshape(-1), h1.reshape(-1)
        _n = _f0.norm() * _f1.norm()
        if float(_n) > 1e-9:
            _loss = _loss + torch.relu((_f0 @ _f1) / _n - DISC_TGT_COS)
    # 副项（默认关闭，见上）
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
    model.eval()          # 阻断 dropout / BN 更新，减少前向噪声
    for _s in range(1, COND_ONLY_STEPS + 1):
        _co_opt.zero_grad(set_to_none=True)
        _src = random.randrange(2)
        _has = COND_COMBOS[random.randrange(len(COND_COMBOS))][1]
        _cv = torch.from_numpy(build_one_cv(_src, _has)).to(device)
        _dt, _dl, _pp = model.cond_init(_cv, return_parts=True, n_pos=len(t))
        _tgt = target[_src:_src + 1]
        # ★ 第二十七死穴：绝对相位对齐需要 θ0。
        #   模型前向里 θ0 = rand(-π,π) 后加 dθ，这里必须**同一口径**重建。
        #   模型用 torch.manual_seed(1234) 在主循环前设定，这里同seed 采样。
        _th0 = (torch.rand(1, len(t), D, device=device) * 2 * math.pi
                - math.pi)
        _l = CondInitModulator.supervise(_dt, _dl, _tgt, d_model=D,
                                          w_ph=(a_tar[_src:_src + 1] ** 2).clamp(min=1e-8),
                                          guard_w=GUARD_W,
                                          theta0=_th0, abs_w=PH_ABS_W,
                                          theta_hat=_pp.get("theta_hat"))
        # ★ 第二十六步：对比判别项（预热期就要开始分开素材，
        #   否则联合期会在通用场上互相强化）
        _dc = torch.zeros((), device=device)
        if DISC_W > 0:
            _other = 1 - _src
            _cv2 = torch.from_numpy(build_one_cv(_other, _has)).to(device)
            _dt2, _, _ = model.cond_init(_cv2, return_parts=True, n_pos=len(t))
            # ★ v3：判别项直接加在 proj 输出 h 上（梯度直达 proj）
            _h0 = model.cond_init.proj(_cv)
            _h1 = model.cond_init.proj(_cv2)
            _dc = loss_discrim(_dt, _dt2, th_tar[_src:_src + 1],
                               h0=_h0, h1=_h1)
            _l = _l + DISC_W * _dc
        _l.backward()
        _co_opt.step()
        if _s % 50 == 0 or _s == 1:
            with torch.no_grad():
                # ★ _h0/_dt2 只在 DISC_W>0 分支里定义，
                #   打印在分支外 → DISC_W=0 时会 NameError。
                #   教训：诊断打印不能假设分支内变量一定存在。
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
    # 交回控制权给主循环
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith(_CONDPFX)
                         or (WARM_BLOCKS == 0
                             or any(n.startswith(f"pmnncore.blocks.{i}.")
                                    for i in range(len(model.pmnncore.blocks) - WARM_BLOCKS,
                                                    len(model.pmnncore.blocks)))))
    model.train()
    print("★ 预热结束，进入联合训练")

for step in range(1, STEPS + 1):
    #梯度清理由累积窗口末尾统一做（见下方 ACC_K 注释），此处不再 zero_grad
    src = random.randrange(2)
    combo_i = random.randrange(len(COND_COMBOS))
    combo_name, has = COND_COMBOS[combo_i]
    one_target = target[src:src + 1]
    # ★ 交叉训练：每步随机选条件组合，缺省路置零 + has=0
    one_cv = torch.from_numpy(build_one_cv(src, has)).to(device)
    dth = None
    a, th, diag = model(texts=None, target_field=one_target, t_coord=t, s_coord=s,
                        batch_size=1, cond_vec=one_cv, use_cond_field=True)
    pe = torch.remainder(th - th_tar[src:src + 1] + math.pi, 2 * math.pi) - math.pi
    la = torch.log1p(a.clamp(min=1e-8))
    lt = torch.log1p(a_tar[src:src + 1].clamp(min=1e-8))
    w = (a_tar[src:src + 1] / a_tar[src:src + 1].max().clamp(min=1e-8)).pow(AMP_P)
    loss_amp = (w * (la - lt).pow(2)).mean()
    w_ph = (a_tar[src:src + 1] ** 2).clamp(min=1e-8)
    loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
    # ★ wcos 带梯度版本（margin / wcs loss 用；行 630 的 no_grad 版仅打印）
    wcos_self_t = (((w_ph * torch.cos(pe)).sum() / w_ph.sum())
                   if (MARGIN_W > 0 or WCS_W > 0) else None)
    # ★ A/B 频谱监督（2026-10-18）：
    #   逐元素 log1p MSE 对「每点都小、叠加成灾」的高频噪声近乎无感，
    #   必须在**频带能量形状**层面加约束。B 项是占比（形状量），
    #   模型无法用整体压低振幅走捷径（第十八死穴）。
    at = a_tar[src:src + 1]
    loss_spec = loss_spec_shape(a, at, win=WIN, w_fft=W_FFT)
    loss_hf = loss_hf_guard(a, at, slack=HF_SLACK, win=WIN, w_fft=W_FFT)
    loss_lo = loss_lo_floor(a, at, floor=LO_FLOOR, win=WIN, w_fft=W_FFT)
    # ★ C 绝对幅度项（补形状项的分辨率盲区：spec 能报警但已饱和在 ln 量级，
    #   band_abs 给线性量级可当回归目标）+ D 相位专项（旧 loss_ph 把 97.82%
    #   梯度压在 DC，高频梯度被淹没 14~21 倍）
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
            loss_gen = CondInitModulator.supervise(
                dth, dln, one_target, d_model=D, w_ph=w_ph,
                guard_w=GUARD_W,
                theta0=diag.get("theta0"), abs_w=PH_ABS_W,
                theta_hat=diag.get("theta_hat"),
                a_hat=diag.get("a_hat"))
            # ★★ 第 4 路：FFT 旁路的双项监督（形状 + 尺度分开加权）
            #   两者缺一不可（实测）：只用 MSE → corr 0.85 但像素 std 只5（目标80）；
            #   只用能量比 → corr 掉到 0.36（丢逐位置形状）。
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
    # ★★ 第二十六步：对比判别项（联合期也要持续压通用场）
    loss_disc = torch.zeros((), device=device)
    if DISC_W > 0 and INIT_MODE == "init" and dth is not None:
        _oth = 1 - src
        _cv2 = torch.from_numpy(build_one_cv(_oth, has)).to(device)
        _dt2, _, _ = model.cond_init(_cv2, return_parts=True, n_pos=len(t))
        # ★ 形状对齐（2026-10-20）：θ̂+FFT 模式下 diag["dtheta"] 已展开为
        #   [B,N,D]（含 FFT 旁路修正），而 cond_init 直接返回 [B,N,G]。
        #   副项默认关闭 3500 步，这个形状 bug 从没被走到，今天 DISC_DTGT_COS
        #   一开就炸。统一到 dth 的口径再进 loss_discrim。
        if _dt2.shape != dth.shape:
            _dt2 = model.cond_init.expand_to_d(_dt2)
        _h0 = model.cond_init.proj(one_cv)
        _h1 = model.cond_init.proj(_cv2)
        loss_disc = loss_discrim(dth, _dt2, th_tar[src:src + 1],
                                 h0=_h0, h1=_h1)
    # ★★ 第三十一死穴修复：wcos margin 判别（靶点=生成结果）
    loss_margin = torch.zeros((), device=device)
    _mgo = float("nan")
    if MARGIN_W > 0 and wcos_self_t is not None:
        with torch.no_grad():
            _oth = 1 - src
            _cv_o = torch.from_numpy(build_one_cv(_oth, has)).to(device)
            a_o, th_o, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                                 batch_size=1, cond_vec=_cv_o, use_cond_field=True)
            pe_o = torch.remainder(th_o - th_tar[_oth:_oth + 1] + math.pi,
                                   2 * math.pi) - math.pi
            _ph_o = (a_tar[_oth:_oth + 1] ** 2).clamp(min=1e-8)
            _mgo = float((_ph_o * torch.cos(pe_o)).sum() / _ph_o.sum())
        loss_margin = torch.relu(_mgo + MARGIN_DELTA - wcos_self_t)
    # ★ wcos 绝对值目标（与 margin 配套：判别已达标时主推对齐强度）
    loss_wcs = torch.zeros((), device=device)
    if WCS_W > 0 and wcos_self_t is not None:
        loss_wcs = torch.relu(WCS_TGT - wcos_self_t)
    loss = (loss_amp + loss_ph + GEN_W * loss_gen
        + SPEC_W * loss_spec + HF_GUARD_W * loss_hf + LO_FLOOR_W * loss_lo
        + BAND_ABS_W * loss_ba + BAND_CAP_W * loss_bc
        + H_SPEC_W * loss_hs
        + PHASE_BAND_W * loss_pb + PHASE_FLOOR_W * loss_pf
        + DISC_W * loss_disc
        + MARGIN_W * loss_margin
        + WCS_W * loss_wcs)

    # ★★ 梯度累积（2026-10-18，第二十一死穴修复）：
    #   实测相邻步梯度余弦 ≈ 0（全参数 +0.053，固定单目标 −0.076），
    #   梯度**存在、通畅、分布正常**但**方向逐步随机** → 等效随机行走，
    #   这才是「相位 loss 完全不动」的根因（不是权重不够、不是 clamp 饱和）。
    #   S0 已验证**前向完全确定**（a/θ 三次逐位相同），所以噪声来自梯度本身，
    #   而非前向随机性 → **平均 K 步梯度必然有效**。
    #   ACC_K=1 即退化为原行为（每步都更新），便于 A/B 对照。
    (loss / ACC_K).backward()
    if (step % ACC_K) == 0:
        # 累积窗口末尾才更新：把 K 步的随机方向平均掉
        opt.step()
        # ★★★ 第三十死穴约束（2026-10-20）：step 后投影 FFN w2 权重范数。
        #   替代此前崩溃的梯度 hook（'tuple' object has no attribute 'mul_'）。
        #   直接缩放权重：数值口径不变、历史 ckpt 兼容，
        #   实测 corr(log a) 0.478 → 0.500（diag_clip_ffn_w2.py）。
        model.pmnncore.project_w2()
        opt.zero_grad()
    # ★ 中途 ckpt：每 100 步落盘（曾因 1455 页面文件错误中途崩溃，
    #   只在最后存一次 → 整轮白跑。训练成本 hours 级，必须断点可续）
    if step % 100 == 0:
        with torch.no_grad():
            _sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
            _safe_torch_save(_sd, OUT_DIR / "cond3_ckpt.pt")
            del _sd
    # ★ 多快照（2026-10-17，昨日教训：中途 ckpt 每百步覆盖，回看不了
    #   dθcos 演化轨迹、且 outputs_real/cond3 曾整体消失）：每 600 步独立留档。
    #   长训 3000 步 = 5 份 × 68MB ≈ 340MB（D 盘，充足）。
    if step % 600 == 0:
        with torch.no_grad():
            _sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
            _safe_torch_save(_sd, OUT_DIR / f"cond3_ckpt_step{step}.pt")
            del _sd
    if step % 50 == 0 or step == 1:
        with torch.no_grad():
            c = torch.cos(pe)
            wcos = (w_ph * c).sum().item() / w_ph.sum().item()
            dabs = dth.abs().mean().item() if dth is not None else float("nan")
            # ★ 绝对指标打点（2026-10-18 D 项）：hi/lo 是**相对素材**的频谱形状量，
            #   不像 margin 那样「全噪声也成立」。生成 hi/lo 应贴合素材值
            #   （素材 0.01~0.05）；当前模型是 0.23（23 倍）= 高频噪声。
            #   ratio_hi = 模型 hi/lo / target hi/lo，>1 即高频超标。
            with torch.no_grad():
                r_pred = hf_ratio(a, win=WIN, w_fft=W_FFT).mean().item()
                r_tar_ = hf_ratio(at, win=WIN, w_fft=W_FFT).mean().item()
            ratio_hi = r_pred / max(r_tar_, 1e-12)
            # ★ 两素材 dtheta 余弦轨迹（2026-10-17 机制实验矩阵）：
            #   +1=两素材锚同向（死锁）、0=无关、<0=分开（条件在起作用）。
            #   与 diag_cond3.py C 节同口径：三路全给、双素材 no_grad 前向。
            #   随训练演化打点是定位"何时开始趋同"的唯一手段。
            dcos = float("nan")
            if INIT_MODE == "init":
                _dts = []
                for _dsrc in range(2):
                    _dcv = torch.from_numpy(build_one_cv(_dsrc, (1, 1, 1))).to(device)
                    _, _, _dd = model(texts=None, target_field=target[_dsrc:_dsrc + 1],
                                      t_coord=t, s_coord=s, batch_size=1,
                                      cond_vec=_dcv, use_cond_field=True)
                    _dt = _dd.get("dtheta")
                    if _dt is not None:
                        _dts.append(_dt.reshape(-1).float())
                if len(_dts) == 2:
                    _n = float(_dts[0].norm() * _dts[1].norm())
                    if _n > 1e-8:
                        dcos = float(_dts[0] @ _dts[1] / _n)
        print(f"  [{step:4d}] src={src} {combo_name:5s} loss={loss.item():.4f} "
              f"amp={loss_amp.item():.4f} phase={loss_ph.item():.4f} "
              f"gen={loss_gen.item():.4f} spec={loss_spec.item():.3f} "
              f"hfg={loss_hf.item():.4f} lo={loss_lo.item():.4f} "
              f"ba={loss_ba.item():.4f} hs={loss_hs.item():.3f} pb={loss_pb.item():.4f} "
              f"disc={float(loss_disc):.4f} "
              f"mg={float(loss_margin):.3f}(o{_mgo:+.2f}) "
              f"wcs={float(loss_wcs):.3f} "
              f"wcos={wcos:+.4f} |dθ|={dabs:.3f} "
              f"dθcos={dcos:+.3f} hi/lo比={ratio_hi:.2f}", flush=True)

sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
_safe_torch_save(sd, OUT_DIR / "cond3_ckpt.pt")
# ★ 教训（outputs_real/cond3 曾整体消失）：训完立刻带后缀备份一份
_bak = OUT_DIR / f"cond3_ckpt_{STEPS}_{time.strftime('%H%M')}.pt"
_safe_torch_save(sd, _bak)
print(f"[BACKUP] ckpt 备份 -> {_bak}")

# ---------------------------------------------------------------- 条件生成评测
CONDS = [
    ("文生", cv_text),      # 只文本 => GIF
    ("图生", cv_first),     # 只首帧 => GIF
    ("图文生", cv_tf),      # 文本+首帧 => GIF
    ("GIF本身", cv_gif),    # 只 GIF本身
    ("三路全给", cv_full),  # 全给
]

print(f"\n[4] 五种条件 × 两素材 交叉生成")
model.eval()
results = {}
absm = {}
with torch.inference_mode():
    for cv_name, cv in CONDS:
        for src_i in range(2):
            one = torch.from_numpy(cv[src_i:src_i + 1]).to(device)
            a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=one, use_cond_field=True)
            key = f"{cv_name}|src{src_i}"
            to_gif(a, th, OUT_DIR / f"{key.replace('|', '_')}.gif", key)
            z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
            fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
            gen = np.stack(decode_field_to_frames(
                fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
            ps = [psnr(gen, raw_imgs[k]) for k in range(2)]
            results[key] = ps
            # ★ 绝对指标（对**原始素材** raw_imgs，不是 codec 天花板）
            absm[key] = dict(psnr_self=ps[src_i], ac1=ac1(gen),
                             sobel=sobel_mag(gen), hilo=hilo(gen))
            winner = "自己" if ps[src_i] > ps[1 - src_i] else "错了"
            print(f"    {key}: vs_A={ps[0]:6.2f} dB  vs_B={ps[1]:6.2f} dB  -> {winner}")

# ---------------------------------------------------------------- D 项绝对判据
print(f"\n[5] ★ 绝对质量判据（D 项，2026-10-18）")
print("  ★ margin/wcos 是相对指标，全噪声图上同样成立 —— 不能单独验收。")
print("  判决三条：① PSNR 必须显著高于同 std 高斯噪声基线；")
print("          ② ac1 接近素材（低频结构在）；③ hilo 不超过素材 3 倍（高频没爆）。")
print("-" * 104)
print(f"{'对象':<22}{'PSNR':>8}{'噪声基线':>10}{'ac1':>8}{'sobel':>8}{'hilo':>8}{'素材hilo':>9}{'判决':>8}")
print("-" * 104)
abs_ok = True
for i in range(len(GIF_SPECS)):
    ref = raw_imgs[i]
    nb = noise_baseline(ref)
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

# ---------------------------------------------------------------- 判据（相对）
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
if all_ok and not abs_ok:
    print("    ⚠️ margin 全 PASS 但绝对质量 FAIL —— 这正是「相对指标自欺」的典型：")
    print("       图看着像噪声，但 src0 确实比 src1 更接近 src0。绝对判据优先。")
print(f"    总判定: {'PASS ✅' if final_ok else 'FAIL ❌'}")
print(f"    GIF 输出目录: {OUT_DIR}")
print("=" * 74)