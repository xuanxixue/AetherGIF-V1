# -*- coding: utf-8 -*-
"""为「保住细节优先」方案定参数：精确测出 v2500 与 v3000 的能量骨架差。

已确认（diag_phase_strata）：
  v2500: 81.9% 能量在 1% 大系数（集中在低频），相位误差 48°，但骨架对、细节可见
  v3000: 61.0% 能量在 1-10% 那档（中低频铺开），相位误差 2.8°，但细节糊
  能量从"少数低频"挪到"一大片中高频" = 糊

需要测：本模型 c 的 w 索引分布（c = h*25 + w 才是真实 rfft2 网格）
以及 v2500/v3000 各频段的能量占比差异，作为"能量重分配"的目标谱。
"""
import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE); sys.path.insert(0, os.path.join(BASE, "src"))
import torch
from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.aether_gif_model import AetherGIFModel

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
GIF = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
device = "cuda"

# ---- 1. 本模型真实 rfft2 网格 ----
from codec.gif_codec import decode_field_to_frames
field = encode_gif_to_field(GIF, TS, GRID, PATCH, WIN, IMG, D)
h_fft, w_fft = TS, PATCH * PATCH * 3 // 2 + 1   # rfft2(win, patch^2*3) 的输出形状
print("rfft2 真实网格 = h_fft=%d x w_fft=%d = %d 系数（d_model=%d 只取前 %d 个）"
      % (h_fft, w_fft, h_fft * w_fft, D, D))
print("通道 c ↔ (h=c//%d, w=c%%%d)  => c=0..63 覆盖 h=0,1,2 三行 × w=0..24 前25列"
      % (w_fft, w_fft))
print()
print(">>> 重要：c=0,25,50 是 w=0 的列（h=0,1,2），即**每行空间 DC**，能量最大")
print("    c=1..24 / 26..49 / 51..63 是 w=1..24 的列（真实频率），能量随 w 衰减")
print()

tgt_a = torch.sqrt(field[:, :D] ** 2 + field[:, D:] ** 2 + 1e-12)
t, sc = build_spacetime_index(TS, GRID, GRID, device)
fc = field.unsqueeze(0).to(device)

BANDS = [("w0(空间DC)", [c for c in range(D) if c % w_fft == 0]),
         ("w1-2(极低频)", [c for c in range(D) if 1 <= c % w_fft <= 2]),
         ("w3-6(低频)", [c for c in range(D) if 3 <= c % w_fft <= 6]),
         ("w7-12(中低频)", [c for c in range(D) if 7 <= c % w_fft <= 12]),
         ("w13-18(中高频)", [c for c in range(D) if 13 <= c % w_fft <= 18]),
         ("w19-24(高频)", [c for c in range(D) if 19 <= c % w_fft <= 24])]


def run(ck):
    m = AetherGIFModel(device=device, use_qwen=False).to(device)
    m.load_state_dict(torch.load(ck, map_location=device), strict=False)
    m.eval()
    torch.manual_seed(0)
    with torch.inference_mode():
        a, th, _ = m(texts=["台球"], target_field=fc, t_coord=t, s_coord=sc, batch_size=1)
    return a[0].cpu()


a25 = run(r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real/overfit_ckpt.pt")
a30 = run(os.path.join(BASE, "outputs_real", "overfit_ckpt.pt"))

print("=" * 104)
print("频段能量占比（Σ a² 在各频段的分配比例，= 该频段对图像亮度的贡献）")
print("=" * 104)
print("%-16s %10s %10s %10s | %10s %10s" % ("频段", "target", "v2500", "v3000", "2500/target", "3000/target"))

e_tar = (tgt_a ** 2).sum(0)
e25 = (a25 ** 2).sum(0)
e30 = (a30 ** 2).sum(0)
tot = [e_tar.sum(), e25.sum(), e30.sum()]
for name, idxs in BANDS:
    vt = e_tar[idxs].sum().item() / tot[0].item()
    v25 = e25[idxs].sum().item() / tot[1].item()
    v30 = e30[idxs].sum().item() / tot[2].item()
    print("%-16s %9.2f%% %9.2f%% %9.2f%% | %9.2fx %9.2fx"
          % (name, vt * 100, v25 * 100, v30 * 100,
             v25 / max(vt, 1e-9), v30 / max(vt, 1e-9)))

print()
print("=" * 104)
print("逐通道能量比（模型/target，v2500 vs v3000）—— 找最严重的失真通道")
print("=" * 104)
r25 = (e25 / e_tar.clamp(min=1e-12))
r30 = (e30 / e_tar.clamp(min=1e-12))
print("通道:  " + " ".join("c%-2d(w%-2d)" % (c, c % w_fft) for c in range(0, D, 6)))
print("v2500:  " + " ".join("%5.2f " % r25[c] for c in range(0, D, 6)))
print("v3000:  " + " ".join("%5.2f " % r30[c] for c in range(0, D, 6)))
print()
print("能量比 log10 标准差（越小越均匀）： v2500=%.3f  v3000=%.3f"
      % (torch.log10(r25.clamp(min=1e-6)).std(), torch.log10(r30.clamp(min=1e-6)).std()))
print("=" * 104)

# ---- 3. μ 是否还分化 ----
print()
print("=" * 104)
print("终态 μ 与 target 谱包络对齐情况（v3000）")
print("=" * 104)
m30 = torch.load(os.path.join(BASE, "outputs_real", "overfit_ckpt.pt"), map_location="cpu")
mu = m30.get("pmnncore.a_mu")
if mu is not None:
    logm = torch.log(tgt_a.mean(0).clamp(min=1e-8) + 1e-3)
    print("μ span=%.3f (target span=%.3f)  corr=%.4f"
          % (mu.max() - mu.min(), logm.max() - logm.min(),
             np_corr := float(torch.corrcoef(torch.stack([mu, logm]))[0, 1])))
    print("  c=0(w0)  μ=%.3f target=%.3f" % (mu[0], logm[0]))
    for c in (1, 6, 12, 18, 24, 30, 45, 63):
        print("  c=%-3d(w%-2d) μ=%+.3f target=%+.3f diff=%+.2f"
              % (c, c % w_fft, mu[c], logm[c], mu[c] - logm[c]))
else:
    print("未找到 pmnncore.a_mu")
print("=" * 104)