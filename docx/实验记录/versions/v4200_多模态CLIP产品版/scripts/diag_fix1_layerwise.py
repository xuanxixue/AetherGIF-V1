# -*- coding: utf-8 -*-
"""fix_r1 逐层相位诊断：条件通路初态对齐度 vs 12 层演化后对齐度。

回答核心问题（定根因的关键一刀）：
  Q1  条件通路（cond_init + FFT旁路）给出的初始相位 θ̂，与 target 相位对齐吗？
      → cos(θ_init, θ_tar)。若接近 0 → 条件通路根本没学会预测相位（通路侧问题）。
  Q2  12 层物理栈演化后，最终相位对齐度还剩多少？
      → cos(θ_final, θ_tar)。若初始高、演化后骤降 → 是物理栈在洗结构（栈侧问题）。
  Q3  逐层衰减发生在哪一层？哪一项（Kuramoto/FFN/漂移场）在最有问题的层。

加载 ckpt：outputs_real/fix_r1/cond3_ckpt_300_2008.pt（共 10.2M 参数）
口径：与 overfit_cond3.py 完全一致（COND_GROUPS=5, COND_LOWRANK=16,
      PHASE_ACT=arctan, RES_LIMIT=0.5, W2_NORM_MAX=3.0, KAPPA_MAX=6.0,
      FFT_FFTW=1, anchor_w=0.3）。
"""
import os, sys, math
from pathlib import Path
import numpy as np, torch

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
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# 与训练一致的环境口径
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")

# ---- 素材（与训练两张一致）----
gif_dir = ROOT / "data" / "gif_raw" / "noto"
GIFS = ["8-ball_1f3b1.gif",
        "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]
t, s = build_spacetime_index(TS, GRID, GRID, device)

def load_field(name):
    gp = gif_dir / name
    if not gp.exists():
        gp = ROOT / name
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)
    a_t = torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12)
    th_t = torch.atan2(f[..., D:], f[..., :D])
    return f, a_t, th_t

fi0, a_t0, th_t0 = load_field(GIFS[0])
fi1, a_t1, th_t1 = load_field(GIFS[1])
print(f"src0 N={fi0.shape[1]} a_mean={a_t0.mean():.5f} | src1 a_mean={a_t1.mean():.5f}")

# ---- ckpt ----
CKPT = ROOT / "outputs_real" / "fix_r1" / "cond3_ckpt_300_2008.pt"
print(f"[ckpt] {CKPT.name}")
sd = torch.load(CKPT, map_location=device)
sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}

m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=256, init_mode="init", n_modules=8,
                   mod_gamma=0.15).to(device)
m.load_state_dict(sd, strict=False)
m.eval()
print(f"参数量 = {sum(p.numel() for p in m.parameters()):,}; "
      f"cond_init.n_groups={m.cond_init.n_groups}, "
      f"n_osc={m.cond_init.n_osc}, "
      f"fft_bypass={'on' if m.fft_bypass is not None else 'off'}")
print()

# ---- 三路全给条件（src0）----
cc = ROOT / "data" / "clip_cache"
cv = torch.from_numpy(build_cond_vector3(
    np.load(cc / "text3.npy")[0:1], np.load(cc / "first3.npy")[0:1],
    np.load(cc / "gif3_video.npy")[0:1], (1, 1, 1))).to(device)

def cos_ph(a, th_f, th_t):
    """a² 加权的相位余弦（与训练 wcos 同口径）。"""
    w = (a.detach() ** 2).clamp(min=1e-8)
    pe = torch.remainder(th_f - th_t.detach() + math.pi, 2 * math.pi) - math.pi
    return float((w * torch.cos(pe)).sum() / w.sum())

torch.manual_seed(123)
with torch.no_grad():
    # 初始 a0/θ0（同一随机源）
    a0 = torch.exp(torch.randn(1, len(t), D, device=device) * 0.8 - 2.5)
    th0 = torch.rand(1, len(t), D, device=device) * 2 * math.pi - math.pi

    # ★ 手动复刻 forward 的条件通路（与 overfit_cond3.py 完全一致）
    dth, dln, iparts = m.cond_init(cv, return_parts=True, n_pos=len(t),
                                   theta0=th0, a0=a0)
    dth_D = m.cond_init.expand_to_d(dth)      # [B,N,D]
    dln_D = m.cond_init.expand_to_d(dln)
    a_init = a0 * torch.exp(dln_D)
    th_init = th0 + dth_D
    _fs = fi0.unsqueeze(0)
    _ffd = None
    if m.fft_bypass is not None:
        fdth, fdla, _ = m.fft_bypass(_fs, return_parts=True)
        fdth_D = m.fft_bypass.expand_to_d(fdth)
        fdla_D = m.fft_bypass.expand_to_d(fdla)
        a_init = a_init * torch.exp(fdla_D)
        th_init = th_init + fdth_D
        _ffd = (fdth_D, fdla_D)
    anchor = th_init
    print("=" * 92)
    print(f"  Q0 分路定位：CLIP 条件路 vs FFT 旁路，各自的初态相位对齐度")
    print("=" * 92)
    # 只有 CLIP 条件（cond_init），无 FFT
    _a_c = a0 * torch.exp(dln_D); _t_c = th0 + dth_D
    print(f"  仅 CLIP 条件(cond_init)   : cos={cos_ph(_a_c, _t_c, th_t0):+.4f}")
    # 只有 FFT 旁路，无 CLIP（从纯随机起点 + 旁路修正）
    if _ffd is not None:
        _a_f = a0 * torch.exp(_ffd[1]); _t_f = th0 + _ffd[0]
        print(f"  仅 FFT 旁路(逐位置解码)   : cos={cos_ph(_a_f, _t_f, th_t0):+.4f}")
    # 二者相加（训练实际路径）
    print(f"  二者相加(实际训练初态)     : cos={cos_ph(a_init, th_init, th_t0):+.4f}")
    print()
    print("=" * 92)
    print(f"  Q1 条件通路初态（θ̂ + FFT修正） vs target 相位")
    print("=" * 92)
    print(f"  cos(a_init, th_tar)  = {cos_ph(a_init, th_init, th_t0):+.4f}   (0=随机, 1=完美)")
    print(f"  cos(a_ref,  th_tar)  = {cos_ph(a_t0, th_t0, th_t0):+.4f}    (target 自比=1.00 基准)")
    print()

    om = m._init_omega(t, s)
    # 时间场
    tf_of = m.pmnncore.time_field
    print("=" * 92)
    print(f"  Q2/Q3 12 层物理栈逐层相位对齐度（初态=上面的 θ̂，anchor_w=0.3）")
    print("=" * 92)
    print(f"  {'层':>4} | {'cos(θ,tar)':>11} | {'σ_space':>8} | {'|resΔ|':>8} | 判定")
    print("  " + "-" * 54)
    a, th = a_init, th_init

    def rep(thx, tag):
        pe = torch.remainder(thx[0] - th_t0 + math.pi, 2 * math.pi) - math.pi
        cs = float((a_t0.detach() ** 2 * torch.cos(pe)).sum() /
                   (a_t0.detach() ** 2).sum())
        sp = float(thx.reshape(-1, D).std(dim=0).mean())
        v = "结构弱" if sp < 0.15 else ("中" if sp < 0.4 else "健康")
        print(f"  {tag:>4} | {cs:>+11.4f} | {sp:>8.4f} | {'':>8} | {v}")

    rep(th, "init")
    # 基线：同 ckpt 但**不跑 FFT 旁路**、不跑 cond_init（纯随机起点走栈）—— 栈的天花板
    for i, blk in enumerate(m.pmnncore.blocks):
        tf = tf_of(i, device)
        a, th, _d = blk(a, th, om, t, s,
                        kappa=(tf["kappa"] * torch.tensor([2.0], device=device)
                               ).clamp(max=6.0),
                        T=tf["T"] * torch.tensor([0.5], device=device),
                        omega_d=tf["omega_d"],
                        targets=None, protos=None, mod_id=None,
                        cond_anchor=anchor, anchor_weight=0.3,
                        )
        rep(th, f"{i+1}")
    print()

    # 最终场空间域质量（与素材比）
    fld_o = torch.cat([a * torch.cos(th), a * torch.sin(th)], -1)[0].cpu().float()
    from codec.gif_codec import decode_field_to_frames
    fo = np.stack(decode_field_to_frames(fld_o, TS, GRID, PATCH, WIN, IMG, D,
                                         normalize=True)).astype(np.float64)
    ft = np.stack(decode_field_to_frames(fi0.cpu().float(),
                                         TS, GRID, PATCH, WIN, IMG, D,
                                         normalize=True)).astype(np.float64)
    def hilo(x):
        g = x.mean(axis=-1)
        F = np.abs(np.fft.fftshift(np.fft.fft2(
            g - g.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
        H, W_ = F.shape[1], F.shape[2]
        yy, xx = np.mgrid[0:H, 0:W_]
        r = np.sqrt((yy - H // 2) ** 2 + (xx - W_ // 2) ** 2)
        return float(F[:, r > H * 0.35].sum() / max(F[:, r < H * 0.15].sum(), 1e-12))
    psnr = 10 * math.log10(255.0 ** 2 /
                           max(((fo - ft) ** 2).mean(), 1e-12))
    print(f"  最终空间域：PSNR={psnr:6.2f}dB  hilo(输出)={hilo(fo):.3f}  "
          f"hilo(素材)={hilo(ft):.3f}  → 高频倍率={hilo(fo)/max(hilo(ft),1e-9):.0f}x")
