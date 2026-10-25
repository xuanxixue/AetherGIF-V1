# -*- coding: utf-8 -*-
"""★ 相位梯度归因：为什么给高频相位 14~21 倍梯度仍学不动？（2026-10-18）

【背景】隔离实验（exp_phase_isolated.py）三组对照：
  P0 旧 loss_ph  ：50 步 old_ph 2.7653→2.7531，band_cos ≈ +0.04（不动）
  P2 新分组监督  ：25 步 new_ph 0.9535→0.9524，band_cos ≈ +0.04（也不动）
  → **加大高频相位梯度无效**。问题不在 loss 权重，在架构。

【怀疑的三处架构卡点】
 ① block.py 相位更新是 `theta_new = theta_new + ph_res.clamp(-1.5, 1.5)`，
    **逐元素硬 clamp**。若 ph_res 长期饱和在 ±1.5，clamp 的次梯度为 0
    → 相位梯度**彻底断流**（这是最可疑的一条）。
 ② FFN 相位输入是 `z = [a·cosθ, a·sinθ]`。若 a 在高频极小（本项目实测
    a_pred/a_tar 高频 10^5 倍差），则 z 的高频信号量级差异巨大，
    w1/w2 的线性层会把梯度按同一比例缩放 → **高频梯度被 a 压掉**。
    ★ 这是「梯度放大 20 倍仍不够」的根因：我们在 loss 侧放大，
      却在 FFN 输入侧被 a 的动态范围再压一次。
 ③ Kuramoto 相位演化：若 κ 吸引子把θ 拉向特定值，梯度被吸引子抵消。

【本脚本测什么】
  对 dL/dθ 按频段统计，**并逐条链路断开测试**：
   (a) 全链路 baseline
   (b) 绕过 clamp（直接用未 clamp 的 ph_res）
   (c) 绕过 FFN（θ 只经 Kuramoto）
   (d) 把 a 换成常数（消除 ② 的动态范围影响）
  → 定位到底是哪一处在吃梯度。
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
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel
from data.clip_condition import CLIP_EMB_DIM
from train.spectral_loss import loss_phase_band

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
W_FFT = 25
BANDS = [(0, 6), (6, 12), (12, 18), (18, 22), (22, 25)]
device = "cuda" if torch.cuda.is_available() else "cpu"

gif_dir = ROOT / "data" / "gif_raw" / "noto"
f = encode_gif_to_field(str(gif_dir / "8-ball_1f3b1.gif"), TS, GRID, PATCH, WIN, IMG, D).to(device)
th_tar = torch.atan2(f[..., D:], f[..., :D]).unsqueeze(0)
a_tar = torch.sqrt(f[..., :D] ** 2 + f[..., D:] ** 2 + 1e-12).unsqueeze(0)
tgt = f.unsqueeze(0)

cc = ROOT / "data" / "clip_cache"
text3 = np.load(cc / "text3.npy")
first3 = np.load(cc / "first3.npy")
gif3v = np.load(cc / "gif3_video.npy")
cv = np.concatenate([text3[:1], first3[:1], gif3v[:1], np.ones((1, 3), np.float32)], 1)
cv_t = torch.from_numpy(cv).to(device)
t, s = build_spacetime_index(TS, GRID, GRID, device)

model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                       ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                       init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
ck = ROOT / "outputs_real" / (os.environ.get("DIAG_TAG") or "ab2_probe_150") / "cond3_ckpt.pt"
if ck.exists():
    model.load_state_dict(torch.load(ck, map_location=device), strict=False)
    print(f"[载入] {ck.name}")
model.train()


def band_grad(th, loss):
    """按频段返回 dL/dth 的 RMS。"""
    gr, = torch.autograd.grad(loss, th, retain_graph=True)
    gr = gr.reshape(-1, WIN, W_FFT)
    return [float(gr[:, :, lo:hi].pow(2).mean().sqrt()) for lo, hi in BANDS]


print("=" * 92)
print("[相位梯度归因] 逐链路断开测试")
print("=" * 92)

# ---------- (a) baseline ----------
a, th, diag = model(texts=None, target_field=tgt, t_coord=t, s_coord=s,
                    batch_size=1, cond_vec=cv_t, use_cond_field=True)
loss = loss_phase_band(th, th_tar, win=WIN, w_fft=W_FFT)
ga = band_grad(th, loss)
print(f"\n(a) 全链路 baseline loss={loss.item():.4f}")
print(f"    band梯度: " + "  ".join(f"b{i}={g:.3e}" for i, g in enumerate(ga)))

# ---------- (b) 探测 clamp饱和率 ----------
print("\n(b) ★ FFN 相位残差 ph_res 的 clamp 饱和率")
tot = 0
sat = 0
absmax = 0.0
with torch.no_grad():
    # 逐 block 跑，截 ph_res
    aa = a
    thh = th
    blk = model.cond_field.pmnn_core if hasattr(model.cond_field, "pmnn_core") else None
    # 直接 hook 每个 block 的 ffn 输出
    hooks = []
    stats = []

    def mk(idx):
        def h(mod, inp, out):
            stats.append((idx, out[..., D:].abs().max().item(),
                          (out[..., D:].abs() >= 1.5).float().mean().item()))
        return h

    core = None
    for nm, m in model.named_modules():
        if nm.endswith("blocks") and hasattr(m, "__len__") and len(m) == 12:
            core = m
    for i, b in enumerate(core):
        hooks.append(b.ffn.register_forward_hook(mk(i)))
    model(texts=None, target_field=tgt, t_coord=t, s_coord=s,
          batch_size=1, cond_vec=cv_t, use_cond_field=True)
    for hk in hooks:
        hk.remove()
for idx, mx, sr in stats:
    absmax = max(absmax, mx)
    tot += 1
    if sr > 0.01:
        sat += 1
    print(f"    block{idx:2d}: |ph_res|max={mx:8.4f}  饱和率(|x|>=1.5)={sr*100:6.2f}%")
print(f"    → 全局 |ph_res|max={absmax:.4f}"
      f"（clamp 阈值 1.5；{'★ 已饱和 → clamp 次梯度为 0，相位梯度断流' if absmax>=1.5 else '未饱和，clamp 不是瓶颈'}）")

# ---------- (c) FFN 输入 a 的动态范围 ----------
print("\n(c) ★ FFN 相位输入 z=[a·cosθ, a·sinθ] 的动态范围")
with torch.no_grad():
    A = a.reshape(-1, WIN, W_FFT)
    for i, (lo, hi) in enumerate(BANDS):
        bm = A[:, :, lo:hi].mean().item()
        print(f"    band{i} w[{lo},{hi:2d}): a均值={bm:.6e}")
    a_dc = A[:, :, 0].mean().item()
    a_hi = A[:, :, 6:].mean().item()
    print(f"    → a(高频)/a(DC) = {a_hi/max(a_dc,1e-30):.3e}")
    print(f"    ★ FFN 的 w1 是**线性层**，dL/dz 沿 a 的动态范围被同比例缩放 →")
    print(f"      高频相位梯度被 FFN 输入侧的 a 再压一次（在 loss 侧放大 20 倍也不够）")

# ---------- (d) 训练中真实梯度 ----------
print("\n(d) 训练态真实梯度（用模型当前输出，非随机初始化）")
print(f"    高频/低频梯度比 = {max(ga[1:])/max(ga[0],1e-30):.4f}")
print(f"    band0 占总梯度 = {ga[0]/sum(ga)*100:.2f}%")
print("=" * 92)
print("  结论导向：")
print("  · 若 (b) 饱和 → 修block.py 的 clamp（换 tanh 软饱和，同 arctan 的思路）")
print("  · 若 (c) 动态范围极端 → FFN 相位输入需归一化（除以 a 的 band 均值）")
print("=" * 92)
# ================= 追加：三种激活下的真实相位梯度 =================
print("\n" + "=" * 92)
print("[追加] 三种相位残差激活的**真实 dL/dθ 梯度**对比（同一 loss、同一输入）")
print("=" * 92)
print(f"{'band':<8}{'clamp':>14}{'soft(tanh)':>14}{'soft2(RMS归一)':>16}")
print("-" * 92)
res = {}
import gc
_sd = torch.load(ck, map_location=device) if ck.exists() else None
for act in ("clamp", "soft", "soft2"):
    os.environ["PHASE_RES_ACT"] = act
    import importlib
    import physics.block as _bm
    importlib.reload(_bm)
    import model.aether_gif_model as _am
    importlib.reload(_am)
    m2 = _am.AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                            ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                            init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
    if _sd is not None:
        m2.load_state_dict(_sd, strict=False)
    m2.train()
    a2, th2, _ = m2(texts=None, target_field=tgt, t_coord=t, s_coord=s,
                    batch_size=1, cond_vec=cv_t, use_cond_field=True)
    l2 = loss_phase_band(th2, th_tar, win=WIN, w_fft=W_FFT)
    gr, = torch.autograd.grad(l2, th2, retain_graph=False)
    gr = gr.reshape(-1, WIN, W_FFT)
    res[act] = [float(gr[:, :, lo:hi].pow(2).mean().sqrt()) for lo, hi in BANDS]
    print(f"  [{act}] done", flush=True)
    del m2, a2, th2, gr, l2
    gc.collect(); torch.cuda.empty_cache()
for i, (lo, hi) in enumerate(BANDS):
    print(f"  band{i} w[{lo},{hi:2d}){res['clamp'][i]:>14.3e}{res['soft'][i]:>14.3e}"
          f"{res['soft2'][i]:>16.3e}")
print("-" * 92)
for act in ("clamp", "soft", "soft2"):
    hi_ = max(res[act][1:])
    print(f"  {act:9s} 高频最大梯度={hi_:.3e}  band0占比={res[act][0]/max(sum(res[act]),1e-30)*100:.1f}%")
print("=" * 92)

# ================= 再追加：梯度到底到达了哪些参数？ =================
print("\n" + "=" * 92)
print("[追加2] dL/dθ 的梯度最终到达各参数的量级（谁在吃梯度？）")
print("=" * 92)
os.environ["PHASE_RES_ACT"] = "soft2"
import importlib
import physics.block as _bm
importlib.reload(_bm)
import model.aether_gif_model as _am
importlib.reload(_am)
m3 = _am.AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                        ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                        init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
if _sd is not None:
    m3.load_state_dict(_sd, strict=False)
m3.train()
a3, th3, _ = m3(texts=None, target_field=tgt, t_coord=t, s_coord=s,
                batch_size=1, cond_vec=cv_t, use_cond_field=True)
l3 = loss_phase_band(th3, th_tar, win=WIN, w_fft=W_FFT)
l3.backward()
rows = []
for nm, p in m3.named_parameters():
    if p.grad is None:
        continue
    rows.append((float(p.grad.norm()), nm, tuple(p.shape)))
rows.sort(reverse=True)
print(f"{'参数':<46}{'grad norm':>14}{'形状':>20}")
print("-" * 92)
for gn, nm, sh in rows[:14]:
    print(f"  {nm:<44}{gn:>14.3e}{str(sh):>20}")
print("-" * 92)
nz = sum(1 for gn, _, _ in rows if gn > 0)
zero = sum(1 for gn, _, _ in rows if gn <= 1e-20)
print(f"  总参数张量 {len(rows)}：非零梯度 {nz}，**零梯度 {zero}**")
if rows:
    print(f"  最大 grad norm = {rows[0][0]:.3e}（{rows[0][1]}）")
    print(f"  最小非零 = {min(g for g,_,_ in rows if g>0):.3e}")
print("=" * 92)

# ================= 追加3：梯度方向是否一致？（可学性的真正判据）=================
print("\n" + "=" * 92)
print("[追加3] ★ 梯度方向一致性 —— 有梯度 ≠ 能下降")
print("  若相邻两步梯度方向余弦 ≈ 0 或为负 → loss 曲面在相位上混乱，")
print("  梯度无法积累成有效下降（同随机行走）。若 ≈ +1 → 方向一致可学。")
print("=" * 92)
os.environ["PHASE_RES_ACT"] = "soft2"
importlib.reload(_bm); importlib.reload(_am)
m4 = _am.AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                        ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                        init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
if _sd is not None:
    m4.load_state_dict(_sd, strict=False)
m4.train()
opt4 = torch.optim.SGD(m4.parameters(), lr=0.0)   # lr=0：只测梯度，不更新参数
grs, ls = [], []
for k in range(4):
    opt4.zero_grad()
    a4, th4, _ = m4(texts=None, target_field=tgt, t_coord=t, s_coord=s,
                    batch_size=1, cond_vec=cv_t, use_cond_field=True)
    l4 = loss_phase_band(th4, th_tar, win=WIN, w_fft=W_FFT)
    l4.backward()
    gsq = torch.cat([p.grad.reshape(-1) for p in m4.parameters()
                     if p.grad is not None])
    grs.append(gsq.clone())
    ls.append(float(l4))
print(f"  loss 序列: {[f'{v:.6f}' for v in ls]}")
for i in range(1, 4):
    c = float(torch.dot(grs[i - 1], grs[i]) /
              (grs[i - 1].norm() * grs[i].norm()).clamp(min=1e-30))
    print(f"  梯度余弦 g{i-1}·g{i} = {c:+.4f}")
c_all = []
for i in range(1, 4):
    c_all.append(float(torch.dot(grs[i - 1], grs[i]) /
                       (grs[i - 1].norm() * grs[i].norm()).clamp(min=1e-30)))
print(f"  → 平均余弦 {sum(c_all)/len(c_all):+.4f}")
print(f"  → 判读：{'方向混乱（≈0），梯度无法积累成下降 → **相位不可学的根因在此**' if sum(c_all)/len(c_all) < 0.3 else '方向基本一致，相位可学'}")
print("=" * 92)

# ================= 追加4：交叉训练是梯度混乱的来源吗？ =================
print("\n" + "=" * 92)
print("[追加4] ★ 假设：随机素材×随机条件组合（7 组合）把loss 曲面搅乱 → 梯度方向混乱")
print("  对照：固定 (src=0, 三路全给) 单一目标，看梯度余弦是否变正。")
print("=" * 92)
os.environ["PHASE_RES_ACT"] = "soft2"
importlib.reload(_bm); importlib.reload(_am)


def cos_probe(fixed, label, n=5):
    m = _am.AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                           ffn_hidden=int(os.environ.get("FFN_HIDDEN", "512")),
                           init_mode="init", n_modules=8, mod_gamma=0.15).to(device)
    if _sd is not None:
        m.load_state_dict(_sd, strict=False)
    m.train()
    g_list = []
    for k in range(n):
        m.zero_grad()
        src, has = (0, (1, 1, 1)) if fixed else (random.randrange(2),
                                                  random.choice([(1,1,1),(1,0,0),(0,1,0),(0,0,1),
                                                                 (1,1,0),(1,0,1),(0,1,1)]))
        cvx = build_cv_local(src, has)
        tt = th_tar_all[src:src + 1] if hasattr(th_tar_all, "shape") else None
        am, thm, _ = m(texts=None, target_field=fields_all[src:src + 1], t_coord=t, s_coord=s,
                       batch_size=1, cond_vec=cvx, use_cond_field=True)
        lm = loss_phase_band(thm, th_tar_all[src:src + 1], win=WIN, w_fft=W_FFT)
        lm.backward()
        g_list.append(torch.cat([p.grad.reshape(-1) for p in m.parameters()
                                 if p.grad is not None]).clone())
    cs = [float(torch.dot(g_list[i - 1], g_list[i]) /
                (g_list[i - 1].norm() * g_list[i].norm()).clamp(min=1e-30))
          for i in range(1, n)]
    print(f"  {label:<28} 平均梯度余弦 = {sum(cs)/len(cs):+.4f}")
    del m
    gc.collect(); torch.cuda.empty_cache()
    return sum(cs) / len(cs)


import random as _r
from data.clip_condition import CLIP_EMB_DIM as _CD, build_cond_vector3 as _bv
_r.seed(0)
import numpy as _np
fields_all = torch.stack([f], 0)
th_tar_all = th_tar


def build_cv_local(src, has):
    z = _np.zeros((1, _CD), dtype=_np.float32)
    c = _ROOT_CC
    return _bv(_np.load(c / "text3.npy")[src:src + 1] if has[0] else z,
               _np.load(c / "first3.npy")[src:src + 1] if has[1] else z,
               _np.load(c / "gif3_video.npy")[src:src + 1] if has[2] else z, has)


_ROOT_CC = ROOT / "data" / "clip_cache"
a_fix = cos_probe(True,  "固定 src0+三路全给")
a_rnd = cos_probe(False, "随机素材+随机条件(现行)")
print("-" * 92)
print(f"  → 固定单目标时梯度余弦 {a_fix:+.4f} vs 现行 {a_rnd:+.4f}")
print(f"  → 判读：{'交叉训练确实是梯度混乱的来源；相位专项应先做**单目标定位**再放开交叉' if a_fix > 3*abs(a_rnd) + 0.1 else '交叉训练不是主因，需另找'}")
print("=" * 92)
