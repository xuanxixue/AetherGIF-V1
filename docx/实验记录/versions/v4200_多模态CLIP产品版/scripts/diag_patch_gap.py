# -*- coding: utf-8 -*-
"""v4200 patch 间不连续诊断（纯图像分析，不加载模型）

疑问：训练日志 hfg=0.0000（HF_GUARD 从未触发，训练频域 hi/lo 只超标 1.8 倍），
但评测图像空间 hilo=5.02（素材 0.01，超标 500 倍）。
两个口径的 300 倍差 → 假说：模型输出在 patch 内部频谱形状接近素材，
但 patch **之间**不连续，边界跳变在图像空间贡献了巨量高频。

口径（与 v3500 网格比诊断一致）：
  PATCH=4, GRID=32, IMG=128
  边界跳变 = 4 的倍数坐标处的相邻像素差分均值
  内部跳变 = 其余位置
  网格比 = 边界/内部（=1.0 说明无 patch 结构，>>1 说明 patch 间断裂）
同时测 hilo / ac1 / sobel 做交叉验证。

对比对象：
  - 原始素材（resample 到 30 帧 128x128）
  - 模型生成 GIF（mx_A 的 三路全给_src0.gif）
  - 同 std 高斯噪声（负对照，patch 结构应最弱）
"""
import numpy as np
from PIL import Image

V4 = r"D:/AetherGIF-V1/docx/实验记录/versions/v4200_多模态CLIP产品版"
PATCH, IMG = 4, 128


def load_frames(p, n_target=30):
    im = Image.open(p)
    out = []
    i = 0
    while True:
        try:
            im.seek(i)
        except EOFError:
            break
        out.append(np.asarray(im.convert("RGB").resize((IMG, IMG),
                                                       Image.BILINEAR),
                              dtype=np.float32))
        i += 1
    x = np.stack(out)
    if x.shape[0] > n_target:
        idx = np.linspace(0, x.shape[0] - 1, n_target).round().astype(int)
        x = x[idx]
    return x


def patch_metrics(x):
    g = x.mean(axis=-1)                                   # [T,H,W]
    dh = np.abs(g[:, :, 1:] - g[:, :, :-1])               # 水平差分
    dv = np.abs(g[:, 1:, :] - g[:, :-1, :])               # 垂直差分
    ch = np.arange(1, g.shape[2]) % PATCH == 0            # patch 边界列
    cv = np.arange(1, g.shape[1]) % PATCH == 0
    bnd = np.concatenate([dh[:, :, ch].ravel(), dv[:, cv, :].ravel()])
    inn = np.concatenate([dh[:, :, ~ch].ravel(), dv[:, ~cv, :].ravel()])
    return bnd.mean(), inn.mean(), bnd.mean() / max(inn.mean(), 1e-9)


def hilo(x):
    g = x.mean(axis=-1)
    F = np.abs(np.fft.fftshift(np.fft.fft2(
        g - g.mean(axis=(1, 2), keepdims=True)), axes=(1, 2))) ** 2
    H, W = F.shape[1], F.shape[2]
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
    return float(F[:, r > H * 0.35].sum() / max(F[:, r < H * 0.15].sum(), 1e-12))


def ac1(x):
    a = x - x.mean()
    return float((a[..., :-1] * a[..., 1:]).sum() / max((a * a).sum(), 1e-12))


def sobel(x):
    g = x.mean(axis=-1)
    gx = np.abs(np.roll(g, -1, axis=2) - np.roll(g, 1, axis=2)) / 2
    gy = np.abs(np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) / 2
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def report(name, x):
    b, iN, ratio = patch_metrics(x)
    print(f"  {name:26s} 边界={b:6.2f} 内部={iN:6.2f} 网格比={ratio:6.3f} "
          f"hilo={hilo(x):7.3f} ac1={ac1(x):+.3f} sobel={sobel(x):6.2f}")
    return ratio


print("=" * 96)
print("patch 间不连续诊断（PATCH=4, IMG=128, 30 帧）")
print("=" * 96)

# 原始素材（src0 = 8号球）
raw0 = load_frames(f"{V4}/data/gif_raw/noto/8-ball_1f3b1.gif")
report("素材0 原始", raw0)

# 模型生成（mx_A 200 步，三路全给 src0）
for tag in ["mx_A", "mx_B", "mx_E"]:
    p = f"{V4}/outputs_real/{tag}/三路全给_src0.gif"
    try:
        report(f"模型生成 {tag}", load_frames(p))
    except Exception as e:
        print(f"  {tag}: 读取失败 {e}")

# fix_r4 基线（相位监督较弱时的产物）
try:
    report("模型生成 fix_r4",
           load_frames(f"{V4}/outputs_real/fix_r4/三路全给_src0.gif"))
except Exception as e:
    print(f"  fix_r4: {e}")

# 噪声负对照（与素材同 std）
rng = np.random.default_rng(0)
n = np.clip(rng.normal(raw0.mean(), raw0.std(), raw0.shape), 0, 255)
report("同std高斯噪声", n.astype(np.float32))

# 素材自身做 patch 网格参照：素材的网格比应接近 1.0
print()
print("判读：素材网格比≈1.0（无 patch 结构）；若模型网格比>>1，")
print("      则图像高频主要来自 patch 边界跳变，训练频域守卫看不见它。")
