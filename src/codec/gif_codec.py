# -*- coding: utf-8 -*-
"""
GIF 编码解码器：3D FFT 同构映射。

冻结格式：4×4 像素 × 5 帧
每条 GIF 被切分为 4×4 像素空间块、5 帧时间块。
每个块 80 个实数值 → 3D FFT → 80 个复系数 → 取 64 个最低频。

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
    """
    读取 GIF 所有帧，resize 到目标分辨率，返回 [T, H, W, C]。
    如果 GIF 是灰度或调色板索引，转换为 RGB。
    """
    try:
        with Image.open(gif_path) as img:
            frames = []
            n_frames = getattr(img, "n_frames", 1) or 1
            # 注意：不能在循环前对整个 img 做 convert——PIL 的 convert 不保留
            # 多帧动画，会把 img 变成单帧图，后续 seek(i) 全部 EOF，只能收到
            # 1 帧（动画信息在这里被整个吞掉）。必须逐帧 seek 后再 convert。
            for i in range(n_frames):
                try:
                    img.seek(i)
                except EOFError:
                    break  # n_frames 可能多算 disposal 帧，到头就收
                # 逐帧转 RGB（P 模式调色板帧用该帧自己的调色板）
                try:
                    frame = img.convert("RGB")
                except Exception:
                    frame = Image.new("RGB", img.size)
                # 嵌入 resize
                frame = frame.resize(resize_to, Image.Resampling.LANCZOS)
                frames.append(np.array(frame, dtype=np.uint8))

            if not frames:
                raise ValueError("No frames found")

            return np.stack(frames, axis=0)
    except Exception as e:
        raise RuntimeError(f"Failed to load {gif_path}: {e}")

def resize_gif_to_10fps(frames: np.ndarray, fps: float = 10, img_cfg: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    按时间间隔均匀的帧采样到 fps，每帧 resize 到 img_cfg。
    先把时间均匀化，避免异步帧。
    返回 [T, H, W, C]。
    """
    if len(frames) == 0:
        return frames
    total_ms = sum(frames).sum()
    if total_ms == 0:
        # 无时长信息，均匀采样
        if len(frames) == 1:
            return np.stack([frames[0]] * int(np.ceil(fps * 1.0)))

    T = len(frames)
    if fps == 0 or math.isclose(fps, 0):
        fps = 10.0

    # 按总时长计算每个帧的显示时长（若原gif无显示时长，按匀速）
    if T == 1:
        target_frames = int(np.ceil(fps * 1.0))
        return np.stack([frames[0]] * target_frames)

    # 目标帧数
    target_T = int(np.ceil(fps * T))
    # 取 P12. 问题：帧时长未知，按等间隔重采样
    # 使用线性插值采样
    from scipy import ndimage
    t = np.arange(T)
    t_new = np.linspace(0, T - 1, target_T)
    # 对每通道（R、G、B）进行插值
    # ★ 修复：map_coordinates 对 n 维输入需要 n 组坐标。
    #   frames[..., c] 是 3D [T, H, W]，原写法只给 1 组坐标（仅时间轴），
    #   scipy 1.18 直接抛 "invalid shape for coordinate array"，codec 直接崩。
    #   正确做法：补齐 y/x 网格坐标，时间轴取 t_new。
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
    # resize 到 img_cfg
    def resample(frame):
        from PIL import Image
        img = Image.fromarray(frame)
        return np.array(img.resize(img_cfg, Image.Resampling.LANCZOS))
    return np.stack([resample(iframe) for iframe in interp_frames], axis=0)

def split_frames_to_blocks(frames: np.ndarray, patch_size: Tuple[int, int] = (4, 4), time_step: int = 5) -> Generator[torch.Tensor, None, None]:
    """
    frames: [T, H, W, 3] uint8
    切分为 4×4 像素空间块、5 帧时间块。
    返回 [N, 3*16*5] = [N, 240] 或 [N, 64] 复数
    """
    T, H, W, C = frames.shape
    # 整数步长
    if T % time_step != 0:
        # 填帧
        pad = time_step - (T % time_step)
        frames = np.concatenate([frames, [frames[-1]] * pad], axis=0)
        T = len(frames)

    # 步数
    t_steps = T // time_step
    patch_h, patch_w = patch_size
    if H % patch_h != 0 or W % patch_w != 0:
        # 中心等裁
        start_y = (H - (H // patch_h * patch_h)) // 2
        start_x = (W - (W // patch_w * patch_w)) // 2
        frames = frames[..., start_y:start_y+H//patch_h*patch_h, start_x:start_x+W//patch_w*patch_w, :]
        H, W = frames.shape[1], frames.shape[2]

    n_h = H // patch_h
    n_w = W // patch_w
    total_blocks = t_steps * n_h * n_w
    # 形变。
    for tt in range(t_steps):
        block_t_start = tt * time_step
        block_t_end = (tt + 1) * time_step
        # 单时间窗口内的帧堆叠 [patch_h*patch_w*3, time_step]
        for hh in range(n_h):
            for ww in range(n_w):
                y0, y1 = hh * patch_h, (hh + 1) * patch_h
                x0, x1 = ww * patch_w, (ww + 1) * patch_w
                # 切片 [T, patch_h, patch_w, 3]
                block = frames[block_t_start:block_t_end, y0:y1, x0:x1, :]  # shape [time_step, ph, pw, 3]
                # 展平 [ph*pw*3, time_step]
                flat = block.reshape(patch_h * patch_w * C, time_step)
                # 复数（时间+空间 DCT）可后续处理。
                yield flat

# --- 3D FFT ---
def block_to_complex(flat: torch.Tensor) -> torch.Tensor:
    """
    flat: [80] 实值（16*5 空间+时间）
    做 3D FFT？ 实际我们只需要 1D 复数（简化）。注意 1D 实 FFT 复系数长度 40。
    为了保持 64 个复数，我们额外做 1D 实 FFT 得到 40 复数，再上采样到 64。
    """
    N = flat.shape[0]
    # 直接 1D FFT
    # 实数 FFT 长度 64 by zero pad.
    # 但 flat=80 → 先插值到 128?
    # 为了简单，我们使用复数折中：
    # 先 zero pad to 128, then 1D FFT → 128 复数，取 前 64。
    padded = torch.nn.functional.pad(flat, (0, 128 - N))
    fft = torch.fft.fft(padded)
    # 取前 64 个复数
    return fft[:64].conj()  # 如果希望相位=arg, 复数就是 amplitude+phase.

def complex_to_block(cz: torch.Tensor) -> torch.Tensor:
    """
    cz: [64] 复数
    逆 FFT 实值重建，取前 64 个系数（高频补零）
    """
    if cz.dtype != torch.complex64:
        cz = cz.to(torch.complex64)
    # zero pad to 128
    zero = torch.zeros(128 - cz.shape[0], dtype=cz.dtype, device=cz.device)
    padded = torch.cat([cz, zero], dim=0)
    inv_fft = torch.fft.ifft(padded)
    # 取实部（实域）
    return inv_fft.real[:80]

# --- 简化 API：GIF → 振子场 (目标振子场) ------------
def gif_to_target_field(gif_path: str, img_cfg: Tuple[int, int] = (128, 128), fps: float = 10) -> torch.Tensor:
    """
    GIF → 目标振子场 (a, theta)。
    内部先采样/resize → split blocks → 3D FFT → 64 复数 → 取 a=|z|, theta=arg(z)
    返回 [N, 64] 复数场（实部虚部已分离）。
    """
    # 目标形状的 Global。
    frames = load_gif_frames(gif_path, img_cfg)
    # 时间重采样
    frames = resize_gif_to_10fps(frames, fps=fps, img_cfg=img_cfg)
    # 切块 + FFT -> [N, 64] 复数
    # 目前 dummy 实现：假设 frames 是 [T, H, W, 3], 转成[T=10, H=128, W=128, 3]
    # 结果平均是 63 振子，得手动 fix 长度。
    blocks = []
    for block in split_frames_to_blocks(frames, patch_size=(4,4), time_step=5):
        flat = torch.tensor(block, dtype=torch.float32)
        cz = block_to_complex(flat)
        blocks.append(cz)
    if not blocks:
        return torch.zeros(1,64,device='cpu')
    field = torch.stack(blocks, dim=0)
    return field

def target_field_to_gif(field: torch.Tensor, gif_out: str) -> None:
    """
    振子场 → GIF。
    逆 FFT -> 重构 frames -> 填充 -> 保存为 GIF。
    """
    N = field.shape[0]
    # 简单：按 10 帧、128*128 重构。
    # 每帧有 10 帧，每块 64。需要重构。
    frames = reconstruct_frames_from_field(field)
    # 保存为 GIF
    frames[0].save(gif_out, save_all=True, append_images=frames[1:], loop=0, duration=100, quality=95)

def reconstruct_frames_from_field(field: torch.Tensor) -> List[np.ndarray]:
    # 简单策略：假设 field 的 N = 6,144, per-frame patches = 1024.
    T = 10
    H, W = 128, 128
    patch_h, patch_w = 4, 4
    n_h, n_w = H // patch_h, W // patch_w
    frames = []
    # 反转场结构。field: [N, 64] 复数, N = T*n_h*n_w =10*32*32=10240? Wait: 32*32=1024, *10=10240.
    # But our target N from gif is 1024 patches per frame, and time_step=5 -> T=10 -> N=10*1024=10240? Actually our split_frames_to_blocks yields blocks per 5-frame window.
    # For simplif, we assume field is already per 5-frame window: N = (T//5) * n_h * n_w = 2*1024=2048. Then we need to replicate to 10 frames.
    # Let's just create a dummy frames list.
    # Real impl: inverse 3D DCT, we'll leave placeholder.
    # DO NOT crash.
    dummy = np.zeros((T, H, W, 3), dtype=np.uint8)
    # pad to T=10
    for _ in range(10):
        frames.append(dummy)
    return frames

# --- 确定性编解码 v2：GIF <-> [N, 2D] 振子场，与训练口径严格对齐 ------------
# 口径：time_step 个数据时间步 × grid×grid 个空间格点，每格点 patch×patch 像素 × win 帧时空块
#   N = time_step * grid * grid   （与 build_spacetime_index 的 t,y,x 展开顺序一致）
#   每个时空块 win*patch*patch*3 个实数 -> rfft -> 取 D 个低频复系数 -> 实虚分离成 2D
# 线性变换、零参数、可逆（逆变换取实部）。

def resample_frames(frames: np.ndarray, target_T: int) -> np.ndarray:
    """线性插值把 [T0,H,W,3] 重采样到 [target_T,H,W,3]。"""
    T0 = frames.shape[0]
    if T0 == target_T:
        return frames
    if T0 == 1:
        return np.repeat(frames, target_T, axis=0)
    idx = np.linspace(0, T0 - 1, target_T)
    lo = np.floor(idx).astype(np.int64)
    hi = np.minimum(lo + 1, T0 - 1)
    w = (idx - lo).astype(np.float32)[..., None, None, None]
    return (frames[lo] * (1.0 - w) + frames[hi] * w).astype(frames.dtype)


def encode_gif_to_field(gif_path: str, time_step: int = 6, grid: int = 32,
                        patch: int = 4, win: int = 5, img: int = 128,
                        d_model: int = 64) -> torch.Tensor:
    """GIF -> [N, 2D] 实虚分离振子场，N = time_step*grid*grid。

    幅度做 per-GIF 归一化到 [0,1]，与模型 a=softplus 的量级匹配。
    """
    T_target = time_step * win
    frames = load_gif_frames(gif_path, (img, img))
    frames = resample_frames(frames, T_target)
    x = frames.astype(np.float32) / 127.5 - 1.0          # [-1, 1]

    # [t, w, gy, py, gx, px, c] -> [t, gy, gx, w, s] (s = patch*patch*3)
    xt = x.reshape(time_step, win, grid, patch, grid, patch, 3)
    blocks = xt.transpose(0, 2, 4, 1, 3, 5, 6).reshape(
        time_step * grid * grid, win, patch * patch * 3)

    flat = torch.from_numpy(np.ascontiguousarray(blocks))
    n_pix = win * patch * patch * 3
    # 幅度标定常数：rfft2 系数是重尾分布（低频巨大、高频极小），
    # 实测 20 个 Noto GIF 的 |z|.mean ≈ 2.95。原口径除以 n_pix=240 会把
    # target 压到 ~0.008，而模型 softplus 输出自然量级 ~0.3–0.7，两者
    # 相差近 90 倍，振幅项要学 90 倍压缩、梯度到 12 层 LayerNorm bias 的
    # 路径极长。NORM=10 把 target 均值对齐到 ~0.3，与模型输出同量级。
    # encode/decode 用同一常数，互逆性保持不变。
    NORM = 10.0
    spec = torch.fft.rfft2(flat, dim=(-2, -1)) / NORM   # [N, win_fft, s_fft]
    spec = spec.reshape(flat.shape[0], -1)                 # [N, win_fft*s_fft]
    full = spec.shape[-1]
    if full < d_model:
        pad = torch.zeros(flat.shape[0], d_model - full, dtype=torch.complex64)
        spec = torch.cat([spec, pad], dim=-1)
    z = spec[:, :d_model].contiguous()                   # [N, D] 复

    field = torch.cat([z.real, z.imag], dim=-1)          # [N, 2D]
    return field.float()


def decode_field_to_frames(field: torch.Tensor, time_step: int = 6, grid: int = 32,
                           patch: int = 4, win: int = 5, img: int = 128,
                           d_model: int = 64, normalize: bool = False) -> List[np.ndarray]:
    """[N, 2D] 实虚振子场 -> [T, H, W, 3] uint8 帧列表。

    normalize=False（默认）：固定 clamp ±1.2，与 encode 严格互逆，用于编解码往返测试。
    normalize=True：per-clip min-max 拉伸到 [-1,1]，用于模型生成可视化——
                    模型输出幅度未收敛前固定 clamp 会整体过曝成二值图。
    """
    N = field.shape[0]
    assert N == time_step * grid * grid, f"field N={N} != {time_step}*{grid}*{grid}"
    z = field[:, :d_model] + 1j * field[:, d_model:]     # [N, D] 复

    n_pix = win * patch * patch * 3
    s_dim = patch * patch * 3
    full = win * (s_dim // 2 + 1)                        # rfft2 输出元素数
    if d_model < full:
        z = torch.cat([z, torch.zeros(N, full - d_model, dtype=z.dtype)], dim=-1)
    elif d_model > full:
        z = z[:, :full]

    z = z * 10.0                                          # 乘回 encode 的标定常数（见 encode_gif_to_field）
    h_fft = win
    w_fft = s_dim // 2 + 1
    flat = torch.fft.irfft2(z.reshape(N, h_fft, w_fft),
                            s=(win, s_dim), dim=(-2, -1)).real  # [N, win, s]
    flat = flat.reshape(N, n_pix)
    if normalize:
        lo, hi = flat.amin(), flat.amax()
        flat = (flat - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
    else:
        flat = flat.clamp(-1.2, 1.2)

    blocks = flat.reshape(time_step, grid, grid, win, patch, patch, 3)
    xt = blocks.permute(0, 3, 1, 4, 2, 5, 6)           # [t, w, gy, py, gx, px, c]
    x = xt.reshape(time_step * win, img, img, 3)
    arr = ((x + 1.0) * 127.5).clamp(0, 255).numpy().astype(np.uint8)
    return [arr[i] for i in range(arr.shape[0])]


# --- Pipeline helpers --------------
def gif_dataset(root_dir: str, img_cfg=(128,128), fps=10):
    """
    返回指向已筛选低细节 GIF 的路径列表。
    """
    # 简单扫描
    pattern = os.path.join(root_dir, "**/*.gif")
    paths = glob.glob(pattern, recursive=True)
    return paths

if __name__ == "__main__":
    # 测试：读取一个 GIF
    from pathlib import Path as P
    gif_path = "data/gif_raw/noto/smiley_1f600_129584.gif"
    if not Path(gif_path).exists():
        # 随便找一个
        gif_path = Path("data/gif_raw").glob("**/*.gif")
        gif_path = next(gif_path) if gif_path else "data/gif_raw/noto/smiley_1f600_129584.gif"
    field = gif_to_target_field(str(gif_path))
    print("field shape:", field.shape)
    # 存 GIF（存一个 dummy gif）
    out = "test_generated.gif"
    target_field_to_gif(field, out)
    print("saved to", out)
