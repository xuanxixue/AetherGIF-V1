# -*- coding: utf-8 -*-
"""theta-hat 条件通路健康探针：表达层(arctan 饱和) 与 监督层(theta0 口径 bug) 双重定位。

背景(2026-10-21 根因链, diag_ident / diag_layer_fidelity / diag_phase_decomp):
  初态 = theta0_rand + dtheta_ci + dtheta_fft = theta_hat_clip + theta_hat_fft
  (theta-hat 模式 dtheta=wrap(theta_hat-theta0), 故 force_init 的 theta0 被整体覆盖)。
  实测 fix_sep_500 的 cos(theta_init, theta_tar) ~ 0.165 (偏 ~1.35 rad),
  12 层物理演化总共只修 ~0.2 rad -> 初态方向是当前第一瓶颈。

本探针零训练回答两个问题(不猜, 全部实测):

  A) 表达层: to_phase 输出 _pre 是否饱和在 arctan 端点?
     |_pre|>>1 -> arctan 梯度 1/(1+x^2) 死亡 -> theta_hat 锁死 +-pi 附近。
     (proj 末尾 LayerNorm 是 2026-10-21 才加的, fix_sep_500 是否含需实测)

  B) 监督层: loss_ph_abs 的 theta0 口径是否有 bug?
     线上: diag["theta0"] = theta0_m = model L451 赋值,
           而 L451 在 L432(theta0+=dtheta) / L445(theta0+=fdth) **之后** ->
           diag["theta0"] 是**修正后初态** theta_hat_clip+theta_hat_fft,
           不是注释(L549-552)设计的**随机原始 theta0**。
     后果: supervise 的 ideal = th_ch - (theta0_pass + dtheta_pass)
           = th_ch - [(theta_hat_clip+theta_hat_fft) + wrap(theta_hat_clip-theta0_rand)]
           = 无意义复合量 -> 唯一对相位方向有真实压力的项(PH_ABS_W=1.0)监督错目标。

输出:
  1. h / _pre 分布 + arctan 梯度 + 饱和率
  2. theta_hat_clip / theta_hat_fft / theta_init 的 cos(., theta_tar)
     (未加权 / a^2 加权 / per-band)
  3. loss_ph_abs 错误口径 vs 正确口径: 值 + 对 to_phase[1].weight 的梯度范数
  4. 幅度侧对照: a_hat vs a_tar 的 corr (log1p 域)
判定:
  - |_pre| 中位 >> 3 且 arctan 梯度 < 1e-3 -> 表达层饱和, 先治 arctan
  - 两口径 loss/梯度显著不同 -> 口径 bug 实锤, 先修 diag["theta0"] 口径
  - 两者都正常 -> 才轮到「加强监督」(原方案 A)
"""
import math
import os
import sys
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
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass
# 架构环境变量必须与训练完全一致, 否则 to_phase 维度对不上
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("IDEAL_CLAMP", "3.141592653589793")   # fix_sep_500 用 pi
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = Path(os.environ.get("CKPT", str(ROOT / "outputs_real" / "fix_sep_500" / "cond3_ckpt.pt")))
SRC = int(os.environ.get("SRC", "0"))
GIF_NAMES = ["8-ball_1f3b1.gif", "3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"]

print("=" * 78)
print(f"[theta-hat health] CKPT={CKPT.name} SRC={SRC} device={device}")
print("=" * 78)

# ---------------------------------------------------------------- 素材
gif_dir = ROOT / "data" / "gif_raw" / "noto"
gp = gif_dir / GIF_NAMES[SRC]
if not gp.exists():
    alt = ROOT.parent.parent / "data" / "gif_raw" / "noto" / GIF_NAMES[SRC]
    gp = alt if alt.exists() else None
if gp is None or not gp.exists():
    raise FileNotFoundError(GIF_NAMES[SRC])
f_tar = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device)   # [N,2D]
if f_tar.dim() == 2:
    f_tar = f_tar.unsqueeze(0)                              # [1,N,2D]
a_tar = torch.sqrt(f_tar[..., :D] ** 2 + f_tar[..., D:] ** 2 + 1e-12)          # [1,N,D]
th_tar = torch.atan2(f_tar[..., D:], f_tar[..., :D])
Ntot = a_tar.shape[1]

CLIP_CACHE = ROOT / "data" / "clip_cache"
text3 = np.load(CLIP_CACHE / "text3.npy")
first3 = np.load(CLIP_CACHE / "first3.npy")
gif3v = np.load(CLIP_CACHE / "gif3_video.npy")
cv = build_cond_vector3(text3[SRC:SRC + 1], first3[SRC:SRC + 1], gif3v[SRC:SRC + 1], (1, 1, 1))
one_cv = torch.from_numpy(cv).to(device)

# ---------------------------------------------------------------- 模型
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(CKPT, map_location=device)
_r = model.load_state_dict(sd, strict=False)
print(f"[load] missing={len(_r.missing_keys)} unexpected={len(_r.unexpected_keys)}")
if _r.missing_keys:
    print(f"       missing(前8)={_r.missing_keys[:8]}")
model.eval()
t, s = build_spacetime_index(TS, GRID, GRID, device)
N = len(t)

# proj 结构 + LayerNorm 是否真正载入(新参数 missing 则是老 ckpt)
pj = model.cond_init.proj
print(f"[proj] len={len(pj)}  LayerNorm={type(pj[-1]).__name__}")
if isinstance(pj[-1], torch.nn.LayerNorm):
    ln = pj[-1]
    print(f"       LN weight mean={ln.weight.mean().item():.4f} "
          f"(=1.0 且 buf 缺失 -> 老 ckpt 无 LN, 归一化未生效)")

# ================================================================ A) 表达层
print("\n" + "-" * 78)
print("A) 表达层: to_phase 输出 _pre 的 arctan 饱和度")
print("-" * 78)
torch.manual_seed(0)
a0_rand = torch.exp(torch.randn(1, N, D, device=device) * 0.8 - 2.5)
th0_rand = torch.rand(1, N, D, device=device) * 2 * math.pi - math.pi

with torch.no_grad():
    h = model.cond_init.proj(one_cv)                        # [B,hidden]
    _pre = model.cond_init.to_phase(h)                      # [B, 30720]
    flat = _pre.reshape(-1)
    sat10 = (flat.abs() > 10).float().mean().item()
    sat50 = (flat.abs() > 50).float().mean().item()
    grad_arctan = (1.0 / (1.0 + flat ** 2))                 # d(arctan)/d_pre
    print(f"  h: std={h.std().item():.3f} mean_abs={h.abs().mean().item():.3f}")
    print(f"  _pre: min={flat.min():+.2f} max={flat.max():+.2f} "
          f"mean_abs={flat.abs().mean().item():.3f} std={flat.std().item():.3f}")
    print(f"  |_pre| 分位: q50={flat.abs().median().item():.2f} "
          f"q90={flat.abs().quantile(0.9).item():.2f} "
          f"q99={flat.abs().quantile(0.99).item():.2f}")
    print(f"  饱和率: |_pre|>10 = {sat10*100:.1f}%   |_pre|>50 = {sat50*100:.1f}%")
    print(f"  arctan 梯度 1/(1+x^2): mean={grad_arctan.mean().item():.3e} "
          f"median={grad_arctan.median().item():.3e}")

# ================================================================ 构建三路初态(手动重现 model L418-L445)
with torch.no_grad():
    dth_ci, dln_ci, parts_ci = model.cond_init(
        one_cv, return_parts=True, n_pos=N, theta0=th0_rand, a0=a0_rand)
    theta_hat = parts_ci["theta_hat"]                       # [B,N,G]
    a_hat = parts_ci.get("a_hat")                           # [B,N,G] or None
    # dth_ci 在 theta-hat 模式下已是 [B,N,D] = wrap(th_D - th0_rand)
    th_hat_clip_D = th0_rand + dth_ci                       # = theta_hat 展开到 D
    # fft 旁路(训练期 _fft_src = target_field)
    _fdth = _fdla = None
    if model.fft_bypass is not None:
        _fdth, _fdla, _fparts = model.fft_bypass(f_tar, return_parts=True)
        _fdth_D = model.fft_bypass.expand_to_d(_fdth)       # [B,N,D]
        _fdla_D = model.fft_bypass.expand_to_d(_fdla)
        th_init = th_hat_clip_D + _fdth_D                   # 实际初态(model L445)
        a_init = a0_rand * torch.exp(model.cond_init.expand_to_d(dln_ci)) \
            * torch.exp(_fdla_D)
    else:
        th_init = th_hat_clip_D
        a_init = a0_rand * torch.exp(model.cond_init.expand_to_d(dln_ci))
    print(f"\n[形状] dth_ci={tuple(dth_ci.shape)} theta_hat={tuple(theta_hat.shape)} "
          f"th_init={tuple(th_init.shape)}")
    print(f"       th_init - (th0_rand+dth_ci+fdth_D) 最大差 = "
          f"{(th_init - (th0_rand + dth_ci + (_fdth_D if _fdth is not None else 0))).abs().max().item():.2e}"
          f" (应为 0, 验证手动重现与模型一致)")

# ================================================================ B) cos 分解
print("\n" + "-" * 78)
print("B) 初态相位方向: cos(*, theta_tar) 分解")
print("-" * 78)

def cos_metrics(th_f, th_t, a_t):
    """th_f/th_t/a_t: [1,N,D] -> 未加权/加权/per-band cos"""
    pe = torch.remainder(th_f - th_t + math.pi, 2 * math.pi) - math.pi
    c_uw = torch.cos(pe).mean().item()
    w = (a_t ** 2).clamp(min=1e-8)
    c_w = float((w * torch.cos(pe)).sum() / w.sum())
    # per-band: c%25 线性分 5 箱(与 _group_index 一致)
    peb = pe.reshape(N, D)
    cb = torch.cos(peb)
    cidx = torch.arange(D, device=device)
    widx = cidx % 25
    edges = torch.linspace(0, 25, 6, device=device)
    gid = torch.bucketize(widx.float(), edges[1:-1].float()).clamp(0, 4)
    bands = []
    for g in range(5):
        m = (gid == g)
        if bool(m.any()):
            bands.append(cb[:, m].mean().item())
    # 去 DC 再算一次(监督口径去 DC)
    c_nodc = torch.cos(pe[:, :, 1:]).mean().item()
    return c_uw, c_w, c_nodc, bands

with torch.no_grad():
    for tag, th_x in [("theta_hat_clip(条件)", th_hat_clip_D),
                      ("theta_init(clip+fft)", th_init)]:
        c_uw, c_w, c_nodc, bands = cos_metrics(th_x, th_tar, a_tar)
        print(f"  {tag:24s}: cos 未加权={c_uw:+.4f} 加权={c_w:+.4f} "
              f"去DC={c_nodc:+.4f}")
        print(f"      per-band(w分箱)={[f'{b:+.3f}' for b in bands]}")

# ================================================================ C) loss_ph_abs 口径对比
print("\n" + "-" * 78)
print("C) loss_ph_abs 监督口径对比: 错误(线上) vs 正确(随机原始 theta0)")
print("-" * 78)

def loss_ph_abs_build(dtheta, theta0_pass, th_tar_s, a_tar_s, clamp):
    """重现 supervise L585-644 的绝对相位项(主前向口径, dtheta 已 [B,N,D]).
    theta0_pass: 传进 supervise 的 theta0(两种口径)。
    返回 (loss, ideal)。"""
    # _G = int(dtheta.shape[-1]) = 125 -> 每通道一组, _dD = dtheta 本身
    _th0_d = theta0_pass[0][:, 1:]                          # [N, D-1]
    _dth_d = dtheta[0][:, 1:]
    ideal = torch.remainder(
        th_tar_s[0][:, 1:] - (_th0_d + _dth_d) + math.pi,
        2 * math.pi) - math.pi
    w_ch = (a_tar_s[0].detach() ** 2).clamp(min=1e-8)[:, 1:]
    w_norm = w_ch / w_ch.mean(dim=0, keepdim=True).clamp(min=1e-8)
    return (w_norm * ideal.clamp(max=clamp) ** 2).mean(), ideal

th_tar_s = th_tar                                          # [1,N,D]
a_tar_s = a_tar
IC = float(os.environ.get("IDEAL_CLAMP", str(math.pi / 2)))

# 错误口径: theta0_pass = theta_init(修正后初态) <- 线上 diag["theta0"]
loss_err, ideal_err = loss_ph_abs_build(dth_ci, th_init, th_tar_s, a_tar_s, IC)
# 正确口径: theta0_pass = th0_rand(随机原始)
loss_ok, ideal_ok = loss_ph_abs_build(dth_ci, th0_rand, th_tar_s, a_tar_s, IC)

with torch.no_grad():
    print(f"  IDEAL_CLAMP = {IC:.4f}")
    print(f"  错误口径 loss = {loss_err.item():.4f}   ideal|: "
          f"mean={ideal_err.abs().mean().item():.3f} "
          f"median={ideal_err.abs().median().item():.3f}")
    print(f"  正确口径 loss = {loss_ok.item():.4f}   ideal|: "
          f"mean={ideal_ok.abs().mean().item():.3f} "
          f"median={ideal_ok.abs().median().item():.3f}")
    # 正确口径下 cos(被监督量 th0+dth = theta_hat_clip)
    _supervised_ok = th0_rand + dth_ci                      # = theta_hat_clip_D
    c_uw, c_w, c_nodc, _ = cos_metrics(_supervised_ok, th_tar, a_tar)
    print(f"  正确口径被监督量 th0+dth == theta_hat_clip: 去DC cos={c_nodc:+.4f} 加权={c_w:+.4f}")
    # 错误口径被监督量
    _supervised_err = th_init + dth_ci
    c_uw2, c_w2, c_nodc2, _ = cos_metrics(_supervised_err, th_tar, a_tar)
    print(f"  错误口径被监督量 th_init+dth        : 去DC cos={c_nodc2:+.4f} 加权={c_w2:+.4f}")

# 梯度范数对比(两种口径各自 backward, 看 to_phase[1].weight.grad)
# ★ 必须用可求导图重建: dth_g 含梯度路径, theta0_m_g 复现线上
#   diag["theta0"] = theta0_m = th0+dth+fdth(修正后初态, L451) 的自指结构。
_fdth_const = _fdth_D if _fdth is not None else 0.0
dth_g, _, _ = model.cond_init(one_cv, return_parts=True, n_pos=N,
                              theta0=th0_rand, a0=a0_rand)      # 可求导
theta0_m_g = th0_rand + dth_g + _fdth_const                     # 线上 diag["theta0"]
loss_err2, _ = loss_ph_abs_build(dth_g, theta0_m_g, th_tar_s, a_tar_s, IC)
loss_ok2, _ = loss_ph_abs_build(dth_g, th0_rand, th_tar_s, a_tar_s, IC)
_grads = {}
for tag, _l in [("错误口径", loss_err2), ("正确口径", loss_ok2)]:
    model.zero_grad(set_to_none=True)
    # 第一次 backward 会释放共享图, 用 retain_graph 保住
    _l.backward(retain_graph=(tag == "错误口径"))
    g = model.cond_init.to_phase[1].weight.grad
    _grads[tag] = g.detach().clone()
    gn = g.norm().item() if g is not None else float("nan")
    print(f"  [{tag}] to_phase[1].weight.grad norm = {gn:.4e}")
if _grads.get("错误口径") is not None and _grads.get("正确口径") is not None:
    ga, gb = _grads["错误口径"].reshape(-1), _grads["正确口径"].reshape(-1)
    cs = float((ga * gb).sum() / (ga.norm() * gb.norm() + 1e-12))
    print(f"  两口径梯度方向 cos = {cs:+.4f} (接近 0 => 梯度方向几乎正交/无关)")

# ================================================================ D) 微型训练对照
# 30 步纯 cond_init 优化, 三口径对比, 直接验证「修口径 theta_hat 就能学」
print("\n" + "-" * 78)
print("D) 微型训练对照: 30 步纯 cond_init 优化, 三口径对比")
print("-" * 78)
STEPS = int(os.environ.get("MICRO_STEPS", "30"))
opt_lr = float(os.environ.get("MICRO_LR", "3e-4"))
# ★ 对照公平性: 三 mode 共享同一 model, 必须每 mode 从同一初始参数出发
_ci_state0 = {k: v.detach().clone()
              for k, v in model.cond_init.state_dict().items()}


def micro_run(tag, mode):
    """mode: ok=正确口径 / err=错误口径 / none=无监督(对照)"""
    model.cond_init.load_state_dict(_ci_state0)                 # 恢复初始参数
    torch.manual_seed(1234)
    opt = torch.optim.AdamW(model.cond_init.parameters(), lr=opt_lr)
    for st in range(1, STEPS + 1):
        # 每步重采随机初态(与训练一致: theta-hat 模式依赖 theta0)
        _a0 = torch.exp(torch.randn(1, N, D, device=device) * 0.8 - 2.5)
        _th0 = torch.rand(1, N, D, device=device) * 2 * math.pi - math.pi
        _f = _fdth_const
        dth_g, _, _ = model.cond_init(one_cv, return_parts=True, n_pos=N,
                                      theta0=_th0, a0=_a0)
        if mode == "ok":
            _l, _ = loss_ph_abs_build(dth_g, _th0, th_tar_s, a_tar_s, IC)
        elif mode == "err":
            _tm = _th0 + dth_g + _f
            _l, _ = loss_ph_abs_build(dth_g, _tm, th_tar_s, a_tar_s, IC)
        else:
            _l = dth_g.sum() * 0.0                 # 零 loss(只走 fwd, 不监督方向)
        opt.zero_grad(set_to_none=True)
        _l.backward()
        opt.step()
        if st % 10 == 0 or st == 1:
            with torch.no_grad():
                # theta_hat_clip = th0 + dth(与 th0 无关), 测其 cos
                _thc = _th0 + model.cond_init(
                    one_cv, return_parts=True, n_pos=N, theta0=_th0, a0=_a0)[0]
                c_uw, c_w, c_nodc, _ = cos_metrics(_thc, th_tar, a_tar)
                print(f"  [{tag} {st:3d}] loss={_l.item():.4f} "
                      f"theta_hat_clip cos 去DC={c_nodc:+.4f} 加权={c_w:+.4f}")


for tag, mode in [("正确口径", "ok"), ("错误口径", "err"), ("无监督  ", "none")]:
    print(f"\n  --- {tag} (lr={opt_lr}, {STEPS} 步) ---")
    micro_run(tag, mode)

# ================================================================ D) 幅度侧对照
print("\n" + "-" * 78)
print("D) 幅度侧对照: a_hat vs a_tar (log1p 域 corr)")
print("-" * 78)
with torch.no_grad():
    if a_hat is not None:
        a_hat_D = model.cond_init.expand_to_d(a_hat)        # [B,N,D]
        la = torch.log1p(a_hat_D.clamp(min=1e-8))[0]
        lt = torch.log1p(a_tar[0].clamp(min=1e-8))
        la_c = la - la.mean()
        lt_c = lt - lt.mean()
        corr = float((la_c * lt_c).sum() / (la_c.norm() * lt_c.norm() + 1e-8))
        print(f"  a_hat vs a_tar log1p corr = {corr:+.4f}")
        print(f"  a_hat: mean={a_hat_D.mean().item():.4f} median={a_hat_D.median().item():.4f}")
        print(f"  a_tar: mean={a_tar.mean().item():.4f} median={a_tar.median().item():.4f}")
        print(f"  a_init(=a_hat*a_fft/a0...) mean={a_init.mean().item():.4f}")
    else:
        print("  a_hat=None (cond_init 未走 a_hat 模式, ckpt 可能用的是旧 dlna)")

print("\n" + "=" * 78)
print("判定汇总:")
print("  - 若 |_pre| q50>3 且 arctan 梯度<1e-2 -> 表达层饱和")
print("  - 若 错误/正确口径 loss 与梯度显著不同 -> 监督口径 bug 实锤")
print("  - 若均正常且 cos 仍低 -> 才轮到加强监督(原方案 A)")
print("=" * 78)
