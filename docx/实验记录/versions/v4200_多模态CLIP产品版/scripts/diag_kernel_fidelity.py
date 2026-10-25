# -*- coding: utf-8 -*-
"""★★ 复核：物理核到底能不能保持 target（2026-10-18 第二版）

【上一轮的自我纠错】
  diag_block_ablation 的振幅 PSNR=200 dB 是**假信号**：
  未训练时 FFN 的 w2 输出恒为 0（log_res mean/std/min/max 全 0），
  于是 a*exp(0)=a，PSNR 必然等于自比的 200 dB。振幅那一路测不出任何东西。
  唯一有效信息是相位：θ cos=+0.9855、|Δθ|=0.114 rad → 物理核基本保持相位。

  而 diag_info_bottleneck 的 3b（PSNR=1.56 dB）**很可能是我脚本自己的
  构造错误**：cond_anchor 传了 th[..., :1]（只取 D 维第 0 通道当吸引中心），
  漂移场会把全场相位拉向「第 0 通道的相位」，人为注入错误吸引子。

【本脚本严格测「保持能力」，构造无歧义】
  基线：target 场输入 → 逐层输出，测振幅 PSNR 与相位 cos。
  ★ 关键：用 drift_target = target_field（官方训练/推理都支持的通路），
    而不是自己拼的 cond_anchor。若这条通路下结构能被保持，
    则「物理核摧毁结构」的说法不成立，问题回到信息通路。

  另外测 amp 那一路到底是不是恒等（直接打印 log_res 统计确认）。

【四种配置】
  C0 drift=target 全开        —— 官方通路，target 当吸引中心
  C1 drift=target 关FFN      —— 只测 drift+Kuramoto 的保持能力
  C2 drift=target 只FFN      —— 跳过 Kuramoto
  C3 drift=None  cond_anchor=None（无吸引子，纯 Kuramoto+FFN）
"""
import gc
import os
import random
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

gif_dir = ROOT / "data" / "gif_raw" / "noto"
f = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"),
                        TS, GRID, PATCH, WIN, IMG, D).to(device)
target = f.unsqueeze(0)
a_tar = target[..., :D]
th_tar = torch.atan2(target[..., D:], target[..., :D])
t, s = build_spacetime_index(TS, GRID, GRID, device)

cc = ROOT / "data" / "clip_cache"
cv = torch.from_numpy(build_cond_vector3(
    np.load(cc / "text3.npy")[0:1], np.load(cc / "first3.npy")[0:1],
    np.load(cc / "gif3_video.npy")[0:1], (1, 1, 1))).to(device)

torch.manual_seed(0); random.seed(0); np.random.seed(0)
m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                   ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                   init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
m.eval()


def psnr_t(mse):
    return float(-10.0 * torch.log10(mse.clamp(min=1e-20)))


with torch.no_grad():
    emb = m._fuse_three_way(cv)
    omega = m._init_omega(t, s) + m.omega_bias_head(emb).unsqueeze(1)
    gi = torch.arange(len(t), device=device) // m.n_modules
    mod_id = gi % m.n_modules
    kappa = (2.0 * torch.sigmoid(m.kappa_scale_head(emb)).squeeze(-1)).view(-1, 1, 1)
    T = (0.5 * torch.sigmoid(m.T_scale_head(emb)).squeeze(-1)).view(-1, 1, 1)

print("=" * 96)
print("★★ 复核：物理核的「保持能力」（无自拼anchor，用官方 targets 通路）")
print("=" * 96)


@torch.no_grad()
def run(mode, n_layers=12):
    a, th = a_tar.clone(), th_tar.clone()
    rows = []
    for i in range(n_layers):
        blk = m.pmnncore.blocks[i]
        tf = m.pmnncore.time_field.forward(i, device)
        if mode == "C3":
            g = blk.drift(a, th, None, None, mod_id=mod_id,
                          cond_anchor=None, anchor_weight=0.0)
        else:
            g = blk.drift(a, th, target, None, mod_id=mod_id,
                          cond_anchor=None, anchor_weight=0.0)
        if mode in ("C1", "C3"):
            th_new = th
        else:
            th_new, _ = blk.kuramoto(th, a, omega, g, tf["kappa"] * kappa,
                                     tf["T"] * T, t, s)
        if mode == "C1":
            a_new, th_new = a, th_new
        else:
            fo = blk.ffn(a, th_new, tf["omega_d"], t, s)
            log_res, ph_res = fo.chunk(2, dim=-1)
            a_new = a * torch.exp(log_res.clamp(-6, 3))
            th_new = th_new + ph_res.clamp(-1.5, 1.5)
        a, th = a_new, th_new
        if i + 1 in (1, 2, 4, 8, 12):
            rows.append((i + 1, psnr_t((a[0] - a_tar[0]).pow(2).mean()),
                         float(torch.cos(th - th_tar).mean())))
    return rows


for mode, desc in [("C0", "drift=target 全开（官方通路）"),
                   ("C1", "drift=target，关FFN（只测物理保持）"),
                   ("C2", "drift=target，跳过 Kuramoto"),
                   ("C3", "drift=None 纯 Kuramoto+FFN（无吸引子）")]:
    rows = run(mode)
    print(f"\n  {desc}")
    for L, p, c in rows:
        print(f"      layer{L:>2d}: 振幅PSNR={p:>7.2f} dB   θ cos={c:+.4f}")

print("\n" + "-" * 96)
print("[验证] FFN 的 log_res 是否恒为 0（若是，上一轮振幅测试全是假信号）")
print("-" * 96)
with torch.no_grad():
    tf = m.pmnncore.time_field.forward(0, device)
    fo = m.pmnncore.blocks[0].ffn(a_tar.clone(), th_tar.clone(),
                                   tf["omega_d"], t, s)
    lr, pr = fo.chunk(2, dim=-1)
    print(f"  log_res: absmax={float(lr.abs().max()):.3e}"
          f"   ph_res: absmax={float(pr.abs().max()):.3e}")
    if float(lr.abs().max()) < 1e-6:
        print("  ★ log_res 恒为 0（未训练的 w2 输出为零）→ "
              "振幅通路当前是恒等映射，**未训练模型对振幅毫无影响**")
        print("    ⇒ 上一轮「振幅 PSNR=200 dB」是构造性必然，不是发现")

print("\n" + "-" * 96)
print("[结论口径]")
print("-" * 96)
print("  · C1（纯 drift+Kuramoto）的 θ cos 是判断「物理核是否摧毁相位结构」")
print("    的唯一有效指标。")
print("  · 若 C1 的 θ cos 明显高于 C0 → Kuramoto 是破坏者。")
print("  · C3 无吸引子时的 θ cos 给出生模式的相位基线。")
print("=" * 96)