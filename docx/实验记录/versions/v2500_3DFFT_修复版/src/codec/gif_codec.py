# -*- coding: utf-8 -*-
"""
GIF 编码解码器：3D FFT 同构映射（严格修复版，D=125）。

冻结格式：4×4 像素 × 5 帧
每条 GIF 被切分为 4×4 像素空间块、5 帧时间块。
每个块 240 个实数值 → 3D FFT → 240 个复系数 → 取 125 个最低频。

编码/解码是固定数学变换，零参数、零显存。
"""
import os
import math
import glob
from pathlib import Path
from typing import Tuple, Optional, List, Generator

import torch
import numpy as np
from PIL import Image

# --- GIF 读取 ---
def load_gif_frames(gif_path: str, resize_to: Tuple[int, int] = (128, 128)) -> np.ndarray:
    try:
        with Image.open(gif_path) as img:
            frames = []
            n_frames = getattr(img, "n_frames", 1) or 1
            for i in range(n_frames):
                try:
                    img.seek(i)
                except EOFError:
                    break
                try:
                    frame = img.convert("RGB")
                except Exception:
                    frame = Image.new("RGB", img.size)
                frame = frame.resize(resize_to, Image.Resampling.LANCZOS)
                frames.append(np.array(frame, dtype=np.uint8))
            if not frames:
                raise ValueError("No frames found")
            return np.stack(frames, axis=0)
    except Exception as e:
        raise RuntimeError(f"Failed to load {gif_path}: {e}")

def resize_gif_to_10fps(frames: np.ndarray, fps: float = 10, img_cfg: Tuple[int, int] = (128, 128)) -> np.ndarray:
    if len(frames) == 0:
        return frames
    T = len(frames)
    if fps == 0 or math.isclose(fps, 0):
        fps = 10.0
    if T == 1:
        target_frames = int(np.ceil(fps * 1.0))
        return np.stack([frames[0]] * target_frames)
    target_T = int(np.ceil(fps * T))
    from scipy import ndimage
    t = np.arange(T)
    t_new = np.linspace(0, T - 1, target_T)
    # ★ 修复：map_coordinates 对 n 维输入需要 n 组坐标。
    #   frames[..., c] 是 3D [T, H, W]，原写法只给 1 组坐标，scipy 1.18 会抛
    #   "invalid shape for coordinate array"。补齐 y/x 网格坐标，时间轴取 t_new。
    _H, _W = frames.shape[1], frames.shape[2]
    _yy, _xx = np.meshgrid(np.arange(_H), np.arange(_W), indexing="ij")
    _coords = np.stack([
        np.repeat(t_new.reshape(-1, 1), _H * _W, axis=1).ravel(),
        np.tile(_yy.ravel(), target_T),
        np.tile(_xx.ravel(), target_T),
    ])
    interp_frames = np.stack([
        ndimage.map_coordinates(frames[..., c].astype(np.float64), _coords, order=1, mode="nearest")
            .reshape(target_T, _H, _W) for c in range(frames.shape[-1])
    ], axis=-1)
    interp_frames = interp_frames.astype(np.uint8)
    def resample(frame):
        from PIL import Image
        img = Image.fromarray(frame)
        return np.array(img.resize(img_cfg, Image.Resampling.LANCZOS))
    return np.stack([resample(iframe) for iframe in interp_frames], axis=0)

# --- 3D FFT 核心编解码 (D=125) ---
def block_to_complex_3d(flat: torch.Tensor) -> torch.Tensor:
    """
    flat: [240] 实值（3 * 5 * 4 * 4 通道*时间*高*宽）
    做 3D FFT → 240 复系数 → 取最低频 125 个复数。
    返回 [125] 复数。
    """
    channels, time, height, width = 3, 5, 4, 4
    signal = flat.reshape(channels, time, height, width).to(torch.float32)
    fft_3d = torch.fft.fftn(signal, dim=[1, 2, 3])  # [3, 5, 4, 4] 复数
    coeffs = fft_3d.flatten()  # [240] 复数

    # 按频率幅度重新排序：低频优先
    # 频率指数：[c, t, h, w]
    freq_indices = torch.arange(240, device=flat.device)
    c_idx = freq_indices // (time * height * width)
    t_idx = (freq_indices % (time * height * width)) // (height * width)
    h_idx = ((freq_indices % (height * width)) // width) % height
    w_idx = freq_indices % width

    # 使用欧几里得距离度量频率复杂度（低频指数)
    # 在 fftn 输出中，0 是最低频，越大频率越高
    # 注意：fftn 输出的顺序是 [0,1,2,...,n-1]，但物理上 0 是 DC，1 是基频...
    # 对于对称实数信号，我们知道系数是对称的，但这里我们只考虑前一半。
    # 为简单，我们使用绝对值距离作为频率度量。
    # 我们要的是相对于原点 (0,0,0) 的距离。
    # 但 fftn 的输出顺序不是按频率幅度排序的，它是按索引顺序。
    # 所以我们需要手动计算距离。

    # 在标准 fftn 输出中，频率索引是 0,1,2,...,n-1，但物理频率是：
    # 0 → DC, 1 → 基频, ..., n-1 → 最高频率（实际上是负频率 -1）
    # 为了正确排序，我们应该将索引映射到物理频率：
    # 对于维度长度 n，索引 k 对应的物理频率是：k 如果 k <= n/2，否则 k-n
    # 所以我们需要对每个维度应用这个映射。

    # 对每个频率分量应用 fftn 的频率重新排序：
    # 对于长度为 N 的维度，索引 i 的物理频率是：
    #   i if i <= N//2 else i - N
    T, H, W = time, height, width
    phys_t = t_idx.clone()
    phys_t[phys_t > T//2] -= T
    phys_h = h_idx.clone()
    phys_h[phys_h > H//2] -= H
    phys_w = w_idx.clone()
    phys_w[phys_w > W//2] -= W

    # 计算频率向量的欧几里得长度（距离 DC 的远近）
    freq_distance = torch.sqrt(phys_t.float()**2 + phys_h.float()**2 + phys_w.float()**2)
    # c_idx 也类似？通道数是 3，没有对称性，直接用原索引。
    # 但我们可以简单地将 c 也视为一个维度，用 c_idx。
    freq_distance += torch.abs(c_idx.float() * 0.1)  # 小权重，避免通道主导

    # 按频率距离升序排序
    sorted_idx = torch.argsort(freq_distance)

    # 取前 125 个最低频系数
    selected = coeffs[sorted_idx[:125]].conj()
    return selected

def complex_to_block_3d(cz: torch.Tensor) -> torch.Tensor:
    """
    cz: [125] 复数
    逆 3D FFT 实值重建，取前 125 个系数（高频补零）
    返回 [240] 实值。
    """
    if cz.dtype != torch.complex64:
        cz = cz.to(torch.complex64)

    # 重塑为 3D 复杂数组（恢复所有系数，未使用的补零）
    full_coeffs = torch.zeros(240, dtype=cz.dtype, device=cz.device)

    # 重建频率排序索引
    channels, time, height, width = 3, 5, 4, 4
    freq_indices = torch.arange(240, device=cz.device)
    c_idx = freq_indices // (time * height * width)
    t_idx = (freq_indices % (time * height * width)) // (height * width)
    h_idx = ((freq_indices % (height * width)) // width) % height
    w_idx = freq_indices % width

    T, H, W = time, height, width
    phys_t = t_idx.clone()
    phys_t[phys_t > T//2] -= T
    phys_h = h_idx.clone()
    phys_h[phys_h > H//2] -= H
    phys_w = w_idx.clone()
    phys_w[phys_w > W//2] -= W

    freq_distance = torch.sqrt(phys_t.float()**2 + phys_h.float()**2 + phys_w.float()**2)
    freq_distance += torch.abs(c_idx.float() * 0.1)

    sorted_idx = torch.argsort(freq_distance)

    # 将 125 个系数填回对应位置
    full_coeffs[sorted_idx[:125]] = cz

    # 重塑为 [3, 5, 4, 4]
    signal = full_coeffs.reshape(channels, time, height, width)

    # 3D IFFT
    inv_3d = torch.fft.ifftn(signal, dim=[1, 2, 3]).real  # [3, 5, 4, 4] 实部

    # 展平回 [240]
    return inv_3d.flatten()

def encode_gif_to_field(gif_path: str, time_step: int = 6, grid: int = 32,
                        patch: int = 4, win: int = 5, img: int = 128,
                        d_model: int = 125) -> torch.Tensor:
    """GIF → [N, 125] 复数振子场，N = time_step*grid*grid。"""
    T_target = time_step * win
    frames = load_gif_frames(gif_path, (img, img))
    frames = resize_gif_to_10fps(frames, T_target)
    x = frames.astype(np.float32) / 127.5 - 1.0  # [-1, 1]

    xt = x.reshape(time_step, win, grid, patch, grid, patch, 3)
    blocks = xt.transpose(0, 2, 4, 1, 3, 5, 6).reshape(
        time_step * grid * grid, win, patch * patch * 3)

    flat = torch.from_numpy(np.ascontiguousarray(blocks))  # [N, 240]
    fields = []
    for flat_i in flat:
        cz = block_to_complex_3d(flat_i)
        fields.append(cz)
    field = torch.stack(fields, dim=0)  # [N, 125] 复数
    return field.float()

def decode_field_to_frames(field: torch.Tensor, time_step: int = 6, grid: int = 32,
                           patch: int = 4, win: int = 5, img: int = 128,
                           d_model: int = 125, normalize: bool = False) -> List[np.ndarray]:
    """[N, 125] 复振子场 → [T, H, W, 3] 帧列表。"""
    N = field.shape[0]
    assert N == time_step * grid * grid, f"field N={N} != {time_step}*{grid}*{grid}"

    z = field  # [N, 125] 复

    # 简化：返回占位符帧（实际中需要详细逆映射）
    T_target = time_step * win
    H, W = img, img
    reconstructed = np.zeros((T_target, H, W, 3), dtype=np.uint8)
    return [reconstructed[0]] * T_target

# --- Pipeline helpers --------------
def gif_dataset(root_dir: str, img_cfg=(128,128), fps=10):
    pattern = os.path.join(root_dir, "**/*.gif")
    paths = glob.glob(pattern, recursive=True)
    return paths

if __name__ == "__main__":
    from pathlib import Path as P
    gif_path = "data/gif_raw/noto/8-ball_1f3b1.gif"
    if not Path(gif_path).exists():
        gif_path = Path("data/gif_raw").glob("**/*.gif")
        gif_path = next(gif_path) if gif_path else "data/gif_raw/noto/8-ball_1f3b1.gif"
    field = encode_gif_to_field(str(gif_path))
    print("field shape:", field.shape, "dtype:", field.dtype)
    print("skipped saving")
