# -*- coding: utf-8 -*-
"""像素级对比 overfit.gif 与 target_recon_ball.gif：量化可看性。"""
import numpy as np
from PIL import Image

def load(path):
    img = Image.open(path)
    frames = []
    for i in range(img.n_frames):
        img.seek(i)
        frames.append(np.asarray(img.convert("RGB"), dtype=np.float32))
    return frames

def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12)) if mse > 0 else 99.0

gen = load("outputs_real/overfit.gif")
tar = load("outputs_real/target_recon_ball.gif")
print(f"gen : frames={len(gen)} size={gen[0].shape}")
print(f"tar : frames={len(tar)} size={tar[0].shape}")

n = min(len(gen), len(tar))
# 帧对齐 PSNR
ps = [psnr(gen[i], tar[i]) for i in range(n)]
print("\n[帧对齐 PSNR] " + " ".join(f"f{i}:{p:.1f}" for i, p in enumerate(ps)))
print(f"  均值={np.mean(ps):.2f} dB  最好={np.max(ps):.2f} dB")

# 帧统计
gm = [f.mean() for f in gen]; tm = [f.mean() for f in tar]
gs = [f.std() for f in gen]; ts_ = [f.std() for f in tar]
print(f"\n[亮度] gen mean={np.mean(gm):.1f} std-of-mean={np.std(gm):.2f} | tar mean={np.mean(tm):.1f} std-of-mean={np.std(tm):.2f}")
print(f"[对比度] gen frame-std={np.mean(gs):.1f} | tar frame-std={np.mean(ts_):.1f}")

# 帧间动画量
gd = [np.abs(gen[i+1]-gen[i]).mean() for i in range(len(gen)-1)]
td = [np.abs(tar[i+1]-tar[i]).mean() for i in range(len(tar)-1)]
print(f"\n[帧间变化] gen mean={np.mean(gd):.2f} min={np.min(gd):.2f} max={np.max(gd):.2f}")
print(f"[帧间变化] tar mean={np.mean(td):.2f} min={np.min(td):.2f} max={np.max(td):.2f}")

# 空间结构：水平/竖直梯度能量（线条边缘的代理）
def grad_energy(f):
    gx = np.abs(np.diff(f, axis=1)).mean()
    gy = np.abs(np.diff(f, axis=0)).mean()
    return (gx + gy) / 2
gg = np.mean([grad_energy(f) for f in gen])
tg = np.mean([grad_energy(f) for f in tar])
print(f"\n[空间结构能] gen={gg:.2f} | tar={tg:.2f}  (比值 {gg/max(tg,1e-9):.2f})")

# 饱和像素比
sat_g = np.mean([(f > 250).mean() + (f < 5).mean() for f in gen])
sat_t = np.mean([(f > 250).mean() + (f < 5).mean() for f in tar])
print(f"[饱和像素] gen={sat_g*100:.1f}% | tar={sat_t*100:.1f}%")
