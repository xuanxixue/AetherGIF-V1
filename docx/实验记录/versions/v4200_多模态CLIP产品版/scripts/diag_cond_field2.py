# -*- coding: utf-8 -*-
"""诊断 2：区分「protos 没分化」vs「演化抹平了 protos 差异」。

150 步（DC 基 + 分组 lr）训练后评测 FAIL：6 次生成结果几乎一模一样
（vs_A 全 13.37~13.40，差异 <0.05dB）。两个假设：
  H1: cond_field 输出的 protos 本身没差异（学得不够）→ 继续训练
  H2: protos 有差异，但 12 blocks 物理演化把它抹平了 → 吸引机制太弱

本脚本测演化链条的三级差异：
  L1: cond_field 输出 protos（src0 vs src1）的幅度/相位差异
  L2: 演化第 1 个 block 后 a/theta 的差异
  L3: 12 blocks 全部演化后输出 a/theta 的差异（= 评测时的输出）

若 L1 大、L3 小 → H2（演化抹平），且逐点看是哪一级衰减。
条件向量复用 fit_cache.pt（不依赖 int8 编码器）。
"""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
    sys.path.insert(0, str(_DEPS))          # 完整 transformers 5.19，压过 C 盘残缺版
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch
try:
    from torch_accel_shim import install_full
    install_full()                          # transformers 5.x 硬依赖 torch.accelerator
except Exception:
    pass
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, TS = 125, 32, 6
device = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = ROOT / "outputs_real" / "cond2"
blob = torch.load(ROOT / "outputs_real" / "fit_cache.pt", map_location="cpu",
                   weights_only=True)
cv = blob["cv"].to(device).float()      # [2, 4097] 图文生
tgt = blob["tgt"].to(device)            # [2, N, 2D]
print(f"[0] device={device}  cv={tuple(cv.shape)}  tgt={tuple(tgt.shape)}")

t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)

print("\n[1] 加载模型 + ckpt（150 步 DC 基版）")
model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12).to(device)
sd = torch.load(OUT_DIR / "cond2_ckpt.pt", map_location=device)
model.load_state_dict(sd, strict=False)
model.eval()

with torch.no_grad():
    # ---- L1: protos 差异 ----
    protos = model.cond_field(cv, t_coord, s_coord)     # [2, N, 2D]
    a_p = torch.sqrt(protos[..., :D] ** 2 + protos[..., D:] ** 2)
    th_p = torch.atan2(protos[..., D:], protos[..., :D])
    d_p = (protos[0] - protos[1]).abs().mean().item()
    a0_m, a1_m = a_p[0].mean().item(), a_p[1].mean().item()
    # protos vs target 的相位对齐（各自）
    th_t = torch.atan2(tgt[..., D:], tgt[..., :D])
    a_t = torch.sqrt(tgt[..., :D] ** 2 + tgt[..., D:] ** 2)
    w_ph = (a_t ** 2).clamp(min=1e-8)
    cos0 = (w_ph[0] * torch.cos(th_p[0] - th_t[0])).sum() / w_ph[0].sum()
    cos1 = (w_ph[1] * torch.cos(th_p[1] - th_t[1])).sum() / w_ph[1].sum()
    print(f"[L1] protos 场差(src0 vs src1) = {d_p:.4e}")
    print(f"[L1] protos 幅度均值: src0={a0_m:.4f} src1={a1_m:.4f}  target: "
          f"{a_t[0].mean():.4f} / {a_t[1].mean():.4f}")
    print(f"[L1] protos 相位对齐 wcos: src0={cos0:+.4f} src1={cos1:+.4f}")
    print(f"[L1] protos 相位差(src0 vs src1) sin 均值 = "
          f"{torch.sin(th_p[0]-th_p[1]).abs().mean().item():.4e}")

    # ---- L2/L3: 手工跑演化链条，逐 block 看差异 ----
    B = 2
    a0 = torch.exp(torch.randn(B, len(t_coord), D, device=device) * 0.8 - 2.5)
    th0 = torch.rand(B, len(t_coord), D, device=device) * 2 * math.pi - math.pi
    omega_base = model._init_omega(t_coord, s_coord)
    text_embs = model.cond_projector(cv[:, :2048], cv[:, 2048:4096], cv[:, -1])
    omega_bias = model.omega_bias_head(text_embs)
    kappa_scale = torch.sigmoid(model.kappa_scale_head(text_embs)).squeeze(-1)
    T_scale = torch.sigmoid(model.T_scale_head(text_embs)).squeeze(-1)
    omega = omega_base + omega_bias.unsqueeze(1)
    kappa = (2.0 * kappa_scale).view(-1, 1, 1)
    T = (0.5 * T_scale).view(-1, 1, 1)

    da = (a0[0] - a0[1]).abs().mean().item()
    dth = torch.sin(th0[0] - th0[1]).abs().mean().item()
    print(f"\n[演化0] 初始噪声差异: |da|={da:.4e}  sin|dth|={dth:.4e}")

    a, theta = a0, th0
    for i in range(model.n_blocks):
        tf = model.pmnncore.time_field.forward(i, device)
        a, theta, dd = model.pmnncore.blocks[i](
            a, theta, omega, t_coord, s_coord,
            kappa=tf["kappa"] * kappa,
            T=tf["T"] * T,
            omega_d=tf["omega_d"],
            targets=None, protos=protos,
        )
        if i in (0, 2, 5, 11):
            da = (a[0] - a[1]).abs().mean().item()
            dth = torch.sin(theta[0] - theta[1]).abs().mean().item()
            print(f"[演化{i+1:2d}block] |da|={da:.4e}  sin|dth|={dth:.4e}")

    # ---- 最终：输出 decode 成帧看差异 ----
    print("\n[2] 最终输出差异（12 blocks 后）")
    da = (a[0] - a[1]).abs().mean().item()
    dth = torch.sin(theta[0] - theta[1]).abs().mean().item()
    print(f"[L3] |da|={da:.4e}  sin|dth|={dth:.4e}")
    print(f"[L3] 输出幅度均值: src0={a[0].mean():.4f} src1={a[1].mean():.4f}  "
          f"(target {a_t[0].mean():.4f}/{a_t[1].mean():.4f})")
    # 各自与 target 的 wcos
    pe = torch.remainder(theta - th_t + math.pi, 2 * math.pi) - math.pi
    wcos = (w_ph * torch.cos(pe)).sum(dim=(1, 2)) / w_ph.sum(dim=(1, 2))
    print(f"[L3] 输出相位对齐: src0={wcos[0]:+.4f} src1={wcos[1]:+.4f}")

print("\n[done] L1 大 + L3 小 = 演化抹平；L1 小 = cond_field 学得不够")
