# -*- coding: utf-8 -*-
"""双素材过拟合测试 —— 验证「模型能否靠条件特征生成对应 GIF」。

【测什么】
把 2 张不同 GIF 混合训练（batch=2，每步都见到这两张），让模型学会
「条件向量 → 对应 GIF」的映射。然后在推理时用三种条件各生成一次：

    条件A图   -> 应生成 GIF_A   （图生 GIF）
    条件B图   -> 应生成 GIF_B
    条件A文本 -> 应生成 GIF_A   （文生 GIF）
    条件B文本 -> 应生成 GIF_B
    条件A图文 -> 应生成 GIF_A   （图文生 GIF）

【判据：交叉还原矩阵】
             生成目标
条件源GIF_A  A_psnr  B_psnr
条件源GIF_B  B_psnr  A_psnr
对角(自己条件→自己目标) 必须显著大于 非对角。

【为什么不直接用 jina-clip 的 1024 维】
2026-10-15 已换成 Qwen3-VL-Embedding-2B = 2048 维（能编码 16 帧时序）。
本脚本用离线预编码，避免每步都过 2.1B 编码器（那要 200s/条）。

【运行】
cd 到v4200 快照根目录后
    python scripts/overfit_cond2.py [STEPS] [MAX_FRAMES]
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ★ 必须先注入 D 盘依赖：C:\Python312 的 transformers 是残缺的
#   （4.38.2 与 4.44.2 互相覆盖，只剩 4 个文件，缺 utils/versions.py），
#   而 model/aether_gif_model.py 会 `from transformers import AutoModelForCausalLM`。
#   好版本在 D:/AetherGIF-V1/deps312/（用户铁律：新装东西一律放 D 盘，
#   C 盘只剩 0.3 GB 装不下）。
# ★ 用「向上探测」而不是固定层数 —— 快照目录深度会变，写死 parents[N] 会失效。
_PROJ = ROOT
for _ in range(8):                       # 最多向上找 8 层
    if (_PROJ / "deps312").is_dir():
        break
    if _PROJ.parent == _PROJ:
        break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))           # 项目根：torch_accel_shim.py / int8_qwen_encoder.py 都在这里
_DEPS = _PROJ / "deps312"
_VENV = _PROJ / ".venv" / "Lib" / "site-packages"
print(f"[PATH] 项目根 = {_PROJ}")
print(f"[PATH] deps312 = {_DEPS}  存在={_DEPS.is_dir()}")
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

# ★ torch.accelerator 垫片：transformers 5.x 硬依赖，torch 2.5.1 没有该属性
#   ★ 必须在 import transformers 之前；torch_accel_shim.py 在项目根 _PROJ 下
try:
    from torch_accel_shim import install_full
    install_full()
except Exception as ex:
    print(f"[WARN] torch_accel_shim 加载失败：{ex}")
# ★ CPU 上必须限线程，否则大模型前向段错误（exit 139）
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import random
import numpy as np
import torch
from PIL import Image

from codec.gif_codec import encode_gif_to_field, decode_field_to_frames, load_gif_frames
from physics.kuramoto_sync import build_spacetime_index
from physics.cond_field import CondFieldPredictor
from physics.cond_init import CondInitModulator
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import (CondProjector, CLIP_EMB_DIM, build_cond_vector)

# ---------------- 编解码口径（与 overfit.py 完全一致，保证可比） ----------------
# D=125：rfft2 真实网格 = WIN(5) x 25 = 125，全保留不截断。
# ★网格感的根因是 codec 里spec[:, :D] 的非对称硬截断，不是变换本身。
# 【完整配置 · 方案 A 恢复】D=125/grid=32/TS=6/img=128，N=6144。
# ★ 训练显存靠 batch=1（交接铁律：B=1 2.55GB，B=2 分配器碎裂爆 4GB 卡）。
# ★ D 必须 125：64 会触发 codec「spec[:, :D] 非对称硬截断」→ 网格感复发（铁律）。
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = (PATCH * PATCH * 3) // 2 + 1  # 25

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
MAX_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 8   # int8 GPU 编码，8 帧仅 1.8s
LR = 3e-4
# ★ cond_field/cond_init 分组 lr（fit1~fit4 定位，2026-10-15）：
#   纯拟合对照里，lr=3e-4 时 chan_amp 400 步只涨 1.65 倍（基本没学），
#   1e-3 涨 6 倍且下降曲线最干净（lr=3e-3 发散）。
#   主 loss 梯度会稀释/扰动条件通路，给它 3.3× 全局的学习率补偿。
COND_LR = 1e-3
AMP_P = 0.5
GEN_W = 2.0   # cond_field supervise 权重（交接铁律：>4.0 会 NaN，上限 2.0）
INIT_W = 2.0  # ★ A 方案：cond_init 监督权重（同上限铁律）
# ★★ A/B 方案开关（2026-10-16）
#   INIT_MODE='init'  A 方案：条件只调初始相位/幅度（Un-0 范式）
#   INIT_MODE='field' 旧版：条件预测完整吸引中心场（对照）
#   N_MODULES=0       B 方案关闭；>0 时按时空索引分 Z 个 module 稀疏耦合
INIT_MODE = os.environ.get("INIT_MODE", "init")
N_MODULES = int(os.environ.get("N_MODULES", "8"))
MOD_GAMMA = float(os.environ.get("MOD_GAMMA", "0.15"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# 两张素材（★描述必须与 GIF 实际内容逐条核对，喂错会直接毁掉条件向量语义）
GIF_SPECS = [
    # 透明背景的 8 号球矢量图标（不是台球桌实拍）：深灰黑球体，左上白色圆圈+黑色粗体"8"，
    # 左上高光 + 右下深色弧线阴影营造 3D 立体感，背景透明
    ("8-ball_1f3b1.gif", "一张透明背景的台球\"8号球\"矢量图标：深灰黑色球体，左上方有一个白色圆形区域，"
     "白圈正中是黑色粗体无衬线数字\"8\"，球体左上角有高光、右下方有深色弧线阴影，"
     "整体呈扁平图标风格但有 3D 立体感"),
    # 二次元 Q 版表情包：黑色长发少女 + 红色蝴蝶结 + 深色水手服 + "前辈前辈~" 手写体
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "一张二次元Q版动漫表情包：黑色长直发少女，齐刘海，"
     "左侧头发绑红色蝴蝶结发带，闭着倒V字型\"^ ^\"的眼睛，眼角有汗滴和情绪波动线，"
     "脸颊泛红带三条斜线，嘴巴微张成小U型，双手举在头部两侧握拳挥舞，"
     "身穿深色水手服领制服、白色条纹衣领、胸前红色领结、细银色十字架项链，"
     "图片顶部有手写风格粗体艺术字\"前辈前辈~\"，末尾波浪号拉长"),
]

OUT_DIR = ROOT / "outputs_real" / "cond2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 编解码
def _decodable(a_, th_):
    """纯 torch 解码到 [T*WIN, IMG, IMG, 3]，**保持梯度**。

    严格复刻 codec.decode_field_to_frames 的变换链，但全程 torch（可微）。
    """
    z = torch.complex(a_ * torch.cos(th_), a_ * torch.sin(th_))
    N = z.shape[0]
    z = z * 10.0                # encode 标定常数
    flat = torch.fft.irfft2(z.reshape(N, WIN, W_FFT), s=(WIN, PATCH * PATCH * 3),
                            dim=(-2, -1)).real
    flat = flat.reshape(N, WIN * PATCH * PATCH * 3)
    blocks = flat.reshape(TS, GRID, GRID, WIN, PATCH, PATCH, 3)
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)
    return xt.reshape(TS * WIN, IMG, IMG, 3)


def to_gif(a, th, path, tag):
    """模型输出场 -> GIF 文件（normalize=True 可视化口径）。"""
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


# ---------------------------------------------------------------- 素材加载
print("=" * 74)
print(f"[1] 加载 {len(GIF_SPECS)} 张素材 -> 频域场")
gif_dir = ROOT / "data" / "gif_raw" / "noto"
fields, a_tars, th_tars = [], [], []
for name, desc in GIF_SPECS:
    gp = gif_dir / name
    if not gp.exists():
        alt = ROOT.parent.parent / "data" / "gif_raw" / "noto" / name
        if alt.exists():
            gp = alt
        else:
            raise FileNotFoundError(name)
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    fields.append(f)
    a_tars.append(torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12))
    th_tars.append(torch.atan2(f[..., D:], f[..., :D]))
    print(f"    {name}: N={f.shape[1]} a_mean={a_tars[-1].mean():.5f}")

# target 场：[2, N, 2D]  批量
target = torch.stack(fields, 0)
a_tar = torch.stack(a_tars, 0)          # [2,N,D]
th_tar = torch.stack(th_tars, 0)        # [2,N,D]
t, s = build_spacetime_index(TS, GRID, GRID, device)
print(f"    target batch = {tuple(target.shape)}")

# 素材解码图（当PSNR 基准，与模型输出同口径）
ref_imgs = []
for i in range(len(GIF_SPECS)):
    z = (a_tar[i] * torch.exp(1j * th_tar[i])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    ref_imgs.append(np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32))
print(f"    素材解码基准帧数 = {len(ref_imgs[0])}")

# ---------------------------------------------------------------- 条件预编码
print(f"\n[2] 加载缓存条件（data/clip_cache/）")
CLIP_CACHE = ROOT / "data" / "clip_cache"
cv_text = np.load(CLIP_CACHE / "cv_text.npy")
cv_image = np.load(CLIP_CACHE / "cv_image.npy")
cv_both = np.load(CLIP_CACHE / "cv_both.npy")
print(f"    文本余弦={float(cv_text[0] @ cv_text[1]):.4f}  图像余弦={float(cv_image[0] @ cv_image[1]):.4f}")

# 三种 cond_vec（batch=2）
conds = {"图生": cv_image, "文生": cv_text, "图文生": cv_both}

# ---------------------------------------------------------------- 训练
print(f"\n[3] 交替训练 {STEPS} 步（batch=1，随机素材 × 随机条件，完整配置）")
print(f"    A/B 方案：init_mode={INIT_MODE}, n_modules={N_MODULES}, mod_gamma={MOD_GAMMA}")
torch.manual_seed(1234)
random.seed(1234)
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       init_mode=INIT_MODE, n_modules=N_MODULES,
                       mod_gamma=MOD_GAMMA).to(device)
# ★ 分组 lr：条件通路吃 COND_LR（1e-3，fit1~fit4 定位），其余物理核保持 LR。
_cond_params = [p for n, p in model.named_parameters()
                if n.startswith("cond_field.") or n.startswith("cond_init.")]
_other_params = [p for n, p in model.named_parameters()
                 if not (n.startswith("cond_field.") or n.startswith("cond_init."))]
opt = torch.optim.AdamW([
    {"params": _other_params, "lr": LR},
    {"params": _cond_params, "lr": COND_LR},
], lr=LR)
print(f"    优化器分组：其他={len(_other_params)} 参数 lr={LR}，"
      f"条件通路={len(_cond_params)} 参数 lr={COND_LR}")
model.train()

for step in range(1, STEPS + 1):
    opt.zero_grad()
    src = random.randrange(2)                              # 随机素材 A/B
    cv_name = random.choice(list(conds))                    # 随机条件类型
    one_target = target[src:src + 1]                        # [1,N,2D]
    one_cv = torch.from_numpy(conds[cv_name][src:src + 1]).to(device)
    # ★ use_cond_field=True：训练/评测走同一通路。旧版训练走重建、评测走
    #   cond_field 预测，两通路脱节 → cond_field 随机初始化去生成 = FAIL 主因。
    a, th, diag = model(texts=None, target_field=one_target, t_coord=t, s_coord=s,
                        batch_size=1, cond_vec=one_cv, use_cond_field=True)
    pe = torch.remainder(th - th_tar[src:src + 1] + math.pi, 2 * math.pi) - math.pi
    la = torch.log1p(a.clamp(min=1e-8))
    lt = torch.log1p(a_tar[src:src + 1].clamp(min=1e-8))
    w = (a_tar[src:src + 1] / a_tar[src:src + 1].max().clamp(min=1e-8)).pow(AMP_P)
    loss_amp = (w * (la - lt).pow(2)).mean()
    w_ph = (a_tar[src:src + 1] ** 2).clamp(min=1e-8)
    loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
    # ★ 条件通路直接监督（否则梯度穿 12 block 太弱）
    loss_gen = torch.zeros((), device=device)
    if INIT_MODE == "init":
        # A 方案：监督初始调制量（低维方向量，不是完整场）
        dth, dln = diag.get("dtheta"), diag.get("dlna")
        if dth is not None:
            loss_gen = CondInitModulator.supervise(
                dth, dln, one_target, d_model=D, w_ph=w_ph)
    else:
        # 旧版：监督预测场
        pred = diag.get("pred_field")
        loss_gen = (CondFieldPredictor.supervise(pred, one_target, d_model=D)
                    if pred is not None else torch.zeros((), device=device))
    loss = loss_amp + loss_ph + GEN_W * loss_gen
    loss.backward()
    opt.step()
    # ★ 中间 ckpt：每 100 步落盘一次。曾因 1455 页面文件错误中途崩溃，
    #   旧版只在最后存一次 → 整轮白跑。训练成本 4~6h，必须断点可续。
    if step % 100 == 0:
        with torch.no_grad():
            _sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
            torch.save(_sd, OUT_DIR / "cond2_ckpt.pt")
            del _sd
    if step % 50 == 0 or step == 1:
        with torch.no_grad():
            c = torch.cos(pe)
            wcos = (w_ph * c).sum().item() / w_ph.sum().item()
        print(f"  [{step:4d}] src={src} {cv_name} loss={loss.item():.4f} "
              f"amp={loss_amp.item():.4f} phase={loss_ph.item():.4f} "
              f"gen={loss_gen.item():.4f} wcos={wcos:+.4f}", flush=True)

# ---------------------------------------------------------------- 条件生成评测
print(f"\n[4] 三种条件 × 两素材 交叉生成")
model.eval()
results = {}
gif_cache = {}

with torch.inference_mode():
    for cv_name, cv in [("图生", cv_image), ("文生", cv_text), ("图文生", cv_both)]:
        for src_i in range(2):
            # 单条件：只喂第 src_i 条，生成 1 条
            one = torch.from_numpy(cv[src_i:src_i + 1]).to(device)
            a, th, _ = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                             batch_size=1, cond_vec=one,
                             use_cond_field=True)
            key = f"{cv_name}|src{src_i}"
            gif_cache[key] = to_gif(a, th, OUT_DIR / f"{key.replace('|', '_')}.gif", key)
            # 与两个素材都比PSNR
            z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
            fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
            gen = np.stack(decode_field_to_frames(
                fld, TS, GRID, PATCH, WIN, IMG, D, normalize=False)).astype(np.float32)
            ps = [psnr(gen, ref_imgs[k]) for k in range(2)]
            results[key] = ps
            winner = "自己" if ps[src_i] > ps[1 - src_i] else "错了"
            print(f"    {key}: vs_A={ps[0]:6.2f} dB  vs_B={ps[1]:6.2f} dB  -> {winner}")

# ---------------------------------------------------------------- 判据
print(f"\n[5] 判据")
all_ok = True
for cv_name in ["图生", "文生", "图文生"]:
    k0, k1 = f"{cv_name}|src0", f"{cv_name}|src1"
    if k0 not in results:
        continue
    d0 = results[k0][0] - results[k0][1]     # 条件A 自己 - 别人（应 >0）
    d1 = results[k1][1] - results[k1][0]
    ok = d0 > 0 and d1 > 0
    all_ok = all_ok and ok
    print(f"    [{'PASS' if ok else 'FAIL'}] {cv_name}: "
          f"margin_A={d0:+6.2f} dB  margin_B={d1:+6.2f} dB")

print(f"\n[6] 总判定: {'PASS ✅ 模型能靠条件特征区分并生成对应 GIF' if all_ok else 'FAIL ❌ 条件未能区分素材'}")
print(f"    GIF 输出目录: {OUT_DIR}")

sd = {k: v for k, v in model.state_dict().items() if not k.startswith("qwen.")}
torch.save(sd, OUT_DIR / "cond2_ckpt.pt")
print(f"[CKPT] {OUT_DIR / 'cond2_ckpt.pt'} ({sum(v.numel() for v in sd.values()):,} params)")
print("=" * 74)
