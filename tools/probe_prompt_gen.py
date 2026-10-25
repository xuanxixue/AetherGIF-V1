# -*- coding: utf-8 -*-
"""任意文本提示词 → v4200 条件生成探针（只读权重，产物落独立目录）。

背景：`scripts/infer.py` 是 cond2 时代旧脚本（cond 维度 2049、期望 ck["model"] 包装），
`model.generate()` 本身也标注为 B 方案遗留（A 方案下 cond_field 无梯度、protos 随机，
文档明确"不走本方法"）。所以这里走**训练评测同款**路径：

    forward(texts=None, target_field=None, use_cond_field=True, cond_vec=[1,6147])

即：文本经冻结的 Qwen3-VL-Embedding 编码 → build_cond_vector3(文本路开、其余路置零)
→ cond_init 调初始态 → 12 块物理演化 → decode_field_to_frames。

用法：
  C:/Python312/python.exe tools/probe_prompt_gen.py --text "提示词" --out 输出.gif [--seed 0]

  # 批量：传多个 --text，各自生成
"""
import os
import sys
import argparse
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
V4200 = PROJ / "docx" / "实验记录" / "versions" / "v4200_多模态CLIP产品版"

# ---- 三件套 + 与 train_guided.py 一致的环境默认 ----
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "deps312"))
sys.path.insert(0, str(PROJ / "int8_deps"))
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass

os.environ.setdefault("COND_INIT_PC", "1")
os.environ.setdefault("COND_PC_RANK", "8")
os.environ.setdefault("COND_GROUPS", "5")
os.environ.setdefault("COND_LOWRANK", "16")
os.environ.setdefault("PHASE_ACT", "arctan")
os.environ.setdefault("RES_LIMIT", "0.5")
os.environ.setdefault("W2_NORM_MAX", "3.0")
os.environ.setdefault("KAPPA_MAX", "6.0")
os.environ.setdefault("FFT_FFTW", "1")
os.environ.setdefault("IDEAL_CLAMP", str(math.pi))

sys.path.insert(0, str(V4200))
sys.path.insert(0, str(V4200 / "src"))

import numpy as np
import torch
from PIL import Image

from model.aether_gif_model import AetherGIFModel
from codec.gif_codec import decode_field_to_frames
from physics.kuramoto_sync import build_spacetime_index
from data.clip_condition import CLIP_EMB_DIM, N_COND3, build_cond_vector3

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6

DEFAULT_CKPT = V4200 / "outputs_real" / "color4_resume_1023" / "guided_ckpt.pt"


def to_gif(a, th, path):
    """与 train_guided.py 的 to_gif 完全同口径（normalize=True）。"""
    z = (a[0] * torch.exp(1j * th[0])).to(torch.complex64)
    fld = torch.cat([z.real, z.imag], dim=-1).float().cpu()
    frames = np.stack(decode_field_to_frames(
        fld, TS, GRID, PATCH, WIN, IMG, D, normalize=True)).astype(np.uint8)
    pil = [Image.fromarray(np.ascontiguousarray(f)) for f in frames]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pil[0].save(path, save_all=True, append_images=pil[1:], loop=0, duration=100)
    return len(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="append", default=None,
                    help="提示词，可重复传多个")
    ap.add_argument("--cache-text", action="append", default=None,
                    help="用 clip_cache/text3.npy 里的第 N 条离线文本条件（训练分布内，"
                         "对照用），可重复，如 --cache-text 0 --cache-text 1")
    ap.add_argument("--out", default=None, help="输出路径（单个时用；多个自动编号）")
    ap.add_argument("--outdir", default=str(PROJ / "outputs" / "_probe_tmp" / "prompt_gen"))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=int, default=3, help="秒（内部换算 T_step）")
    args = ap.parse_args()
    if not args.text and not args.cache_text:
        ap.error("至少要给 --text 或 --cache-text")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] device={device}  ckpt={args.ckpt}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 模型（与训练完全同参数） ----
    model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                           ffn_hidden=256, init_mode="init", n_modules=8,
                           mod_gamma=0.15).to(device)

    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    ck = {k: v for k, v in ck.items() if not k.startswith("qwen.")}
    ret = model.load_state_dict(ck, strict=False)
    if ret.unexpected_keys:
        raise RuntimeError(f"多余键：{ret.unexpected_keys[:5]}")
    if ret.missing_keys:
        raise RuntimeError(f"缺失键：{ret.missing_keys[:5]}")
    print(f"[CKPT] missing=0 unexpected=0，参数量={sum(p.numel() for p in model.parameters()):,}")
    model.eval()

    # ---- 文本编码（CPU，GPU 装不下 fp16 权重） ----
    from data.clip_condition import ClipConditionEncoder
    enc = None
    if args.text:
        enc = ClipConditionEncoder(device="cpu")

    t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)
    z = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)

    jobs = []
    for txt in (args.text or []):
        jobs.append((txt, None))
    if args.cache_text:
        # ★ 训练分布内对照：直接用 clip_cache/text3.npy 的第 N 条离线条件
        _t3 = np.load(V4200 / "data" / "clip_cache" / "text3.npy")
        for s_idx in args.cache_text:
            i = int(s_idx)
            jobs.append((f"[cache_text#{i}] src{i}", _t3[i:i + 1]))

    for i, (txt, pre_emb) in enumerate(jobs):
        t_e = pre_emb if pre_emb is not None else enc.encode_text([txt])
        cond = build_cond_vector3(t_e, z, z, (1, 0, 0))   # 只开文本路
        assert cond.shape == (1, N_COND3), cond.shape
        cv = torch.from_numpy(cond).to(device).float()

        with torch.inference_mode():
            a, th, _ = model(texts=None, target_field=None, t_coord=t_coord,
                             s_coord=s_coord, batch_size=1, cond_vec=cv,
                             use_cond_field=True)

        if args.out and len(jobs) == 1:
            out = Path(args.out)
        else:
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in txt)[:40]
            out = Path(args.outdir) / f"{i}_{safe}.gif"

        n = to_gif(a, th, out)
        a_rng = (float(a.min()), float(a.max()))
        print(f"[GEN] {txt}")
        print(f"      -> {out}  ({n} 帧, a_range={a_rng[0]:.3f}~{a_rng[1]:.3f})")


if __name__ == "__main__":
    main()
