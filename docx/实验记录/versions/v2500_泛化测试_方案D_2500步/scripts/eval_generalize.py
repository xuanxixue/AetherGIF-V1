# -*- coding: utf-8 -*-
"""泛化测试对比：新素材 2500 步 vs v2500 台球基线。

口径与 v2500 基线日志（overfit_2500_rerun.log）完全一致：
  - 像素 std / 逐帧 std：空间域对比度
  - 边界跳变 / 内部跳变 / 网格比：patch 边界的 LED 网格感（PATCH=4 切边界）
  - Sobel 锐度：高频细节
  - PSNR：必须用 normalize=False（clamp 口径），与基线 24.57 dB 可比
"""
import sys
sys.path.insert(0, './src')
import numpy as np
from PIL import Image
from codec.gif_codec import load_gif_frames, decode_field_to_frames, encode_gif_to_field

PATCH = 4
GIF = 'data/gif_raw/noto/3D87B14FBB731ED3A948AC6DD9EA6EBC.gif'


def frames_from_gif(path):
    return load_gif_frames(path, (128, 128))


def frames_to_array(field):
    return np.stack(decode_field_to_frames(field, 6, 32, PATCH, 5, 128, 64, normalize=False)).astype(np.float32)


def boundary_metrics(arr):
    """按 PATCH 切边界，比较边界列差分 vs 内部列差分的幅度。"""
    b, i = [], []
    for f in arr:                                  # 每帧 [H,W,3]
        d = np.abs(np.diff(f, axis=1))              # 水平方向差分 [H,W-1,3]
        xs = np.arange(d.shape[1])
        mask = (xs % PATCH) == (PATCH - 1)          # patch 右边界列
        b.append(d[:, mask].mean())
        i.append(d[:, ~mask].mean())
    return float(np.mean(b)), float(np.mean(i))


def sobel(arr):
    k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    out = []
    for f in arr:
        g = f.mean(axis=2)
        gx = np.zeros_like(g)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                gx += k[dy + 1, dx + 1] * np.roll(np.roll(g, dy, 0), dx, 1)
        out.append(np.abs(gx).mean())
    return float(np.mean(out))


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def report(name, arr, tgt_clamp=None):
    std = float(arr.std())
    pstd = np.array([f.std() for f in arr])
    bb, ib = boundary_metrics(arr)
    sb = sobel(arr)
    fd = float(np.abs(np.diff(arr, axis=0)).mean())
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    line = (f"{name:<26} std={std:6.2f}  逐帧std min/med/max="
            f"{pstd.min():5.1f}/{np.median(pstd):5.1f}/{pstd.max():5.1f}  "
            f"边界={bb:5.2f} 内部={ib:5.2f} 网格比={bb/max(ib,1e-9):5.3f}  "
            f"Sobel={sb:5.2f}  p1-p99={p1:5.1f}-{p99:5.1f}  frame-diff={fd:5.2f}")
    print(line)
    if tgt_clamp is not None:
        print(f"{'':<26} PSNR(clamp)={psnr(arr, tgt_clamp):.2f} dB")
    return dict(std=std, boundary=bb, internal=ib, ratio=bb / max(ib, 1e-9),
                sobel=sb, frame_diff=fd, p1=p1, p99=p99)


if __name__ == '__main__':
    # 1) 素材（原分辨率 resize 128）
    src = frames_from_gif(GIF).astype(np.float32)
    # 2) codec 天花板（clamp 口径，与基线同）
    fld = encode_gif_to_field(GIF, 6, 32, PATCH, 5, 128, 64)
    codec = frames_to_array(fld)
    # 3) 模型输出
    model = None
    try:
        from codec.gif_codec import decode_field_to_frames as _
        import torch
        sd = torch.load('outputs_real/overfit_ckpt.pt', map_location='cpu')
        # 复现 overfit.py 末尾的推理路径
        sys.path.insert(0, '.')
        sys.path.insert(0, './src')
        from model.aether_gif_model import AetherGIFModel
        from physics.kuramoto_sync import build_spacetime_index
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        m = AetherGIFModel(device=dev, use_qwen=False).to(dev)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
        target = fld.unsqueeze(0).to(dev)
        t, s = build_spacetime_index(6, 32, 32, dev)
        m.eval()
        with torch.inference_mode():
            a, th, _ = m(texts=["头像"], target_field=target, t_coord=t, s_coord=s, batch_size=1)
        z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
        out = torch.cat([z.real, z.imag], dim=-1).float().cpu()
        model = frames_to_array(out)
    except Exception as e:
        print(f"[WARN] 无法加载模型产物: {e}")

    print("\n" + "=" * 100)
    print("泛化测试：新素材 3D87B1...（二次元头像）  vs  v2500 台球基线")
    print("=" * 100)
    print(f"{'对象':<26} {'对比度/网格/锐度指标（图像空间，PATCH=4 切边界）'}")
    print("-" * 100)
    r_src = report("① 原始素材", src)
    r_codec = report("② codec 天花板", codec)
    if model is not None:
        r_model = report("③ 模型 2500 步输出", model, tgt_clamp=codec)
    print("-" * 100)
    print("v2500 台球基线参照：素材 边界4.99/内部4.94/比1.010/std42.8/Sobel4.20"
          " | codec顶 6.81/5.59/1.218/35.9/4.89 | 模型 19.92/12.30/1.620/27.8/11.33")
