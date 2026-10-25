# -*- coding: utf-8 -*-
"""Qwen3-VL-Embedding-2B 条件区分验证 —— 用官方封装，测两个真实 GIF。

要点（都是踩过的坑）：
1. 必须用官方 Qwen3VLEmbedder，不要自己 mean-pool。官方是 last_token_pooling
   （_pooling_last），用 mean pooling 得到的向量**语义不对**，检索会崩。
2. 每个样本要带 instruction（system prompt），它是检索质量的决定因素。
3. 输入格式是 dict 列表：{"text":..., "image":..., "video":..., "instruction":...}
4. 模型对 video 原生支持（GIF 可以直接当 video 喂），这比 jina-clip-v2 只吃首帧强。
"""
import importlib.util
import sys
from pathlib import Path

import torch

# ★ 不能用 sys.path + from src.models... —— 本项目自己就有 src 包（D:/AetherGIF-V1/src），
#   同名遮蔽会报 ModuleNotFoundError: No module named 'src.models'。
#   解法：按绝对文件路径 importlib 直读官方实现文件，绕开包解析。
_IMPL = Path(__file__).parent / "Qwen3-VL-Embedding" / "src" / "models" / "qwen3_vl_embedding.py"
_spec = importlib.util.spec_from_file_location("_qwen3vl_embedding_impl", _IMPL)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_qwen3vl_embedding_impl"] = _mod
_spec.loader.exec_module(_mod)
Qwen3VLEmbedder = _mod.Qwen3VLEmbedder

MODEL_PATH = r"D:/AetherGIF-V1/models/qwen3vl/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master"

GIF_A = r"D:/AetherGIF-V1/data/gif_raw/noto/8-ball_1f3b1.gif"
GIF_B = r"D:/AetherGIF-V1/data/gif_raw/noto/3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"

# 条件生成用的中文描述（本项目数据集是中文 group 标签）
DESC_A = "一个黑色的8号台球在绿色台球桌上滚动"
DESC_B = "一个彩色立方体在白色背景上旋转"


def main():
    print("=" * 70)
    print("加载 Qwen3-VL-Embedding-2B ...")
    embedder = Qwen3VLEmbedder(
        model_name_or_path=MODEL_PATH,
        max_length=8192,
        max_frames=16,
        fps=1,
        dtype=torch.float32,
    )
    print("加载完成")
    print("=" * 70)

    # ---------- 1. 文本条件 ----------
    print("\n[1] 文本条件编码")
    txt = embedder.process([
        {"text": DESC_A, "instruction": "Represent the animated GIF content."},
        {"text": DESC_B, "instruction": "Represent the animated GIF content."},
    ])
    print(f"    shape={tuple(txt.shape)} dtype={txt.dtype}")
    print(f"    L2 norm A={txt[0].norm():.4f}  B={txt[1].norm():.4f}")
    sim_txt = torch.dot(txt[0], txt[1]).item()
    print(f"    文本间余弦 = {sim_txt:.4f}")

    # ---------- 2. GIF 作为 video 输入（原生支持） ----------
    print("\n[2] GIF 作为 video 输入编码")
    vid = embedder.process([
        {"video": GIF_A, "instruction": "Represent the animated GIF content."},
        {"video": GIF_B, "instruction": "Represent the animated GIF content."},
    ])
    print(f"    shape={tuple(vid.shape)}")
    print(f"    L2 norm A={vid[0].norm():.4f}  B={vid[1].norm():.4f}")
    sim_vid = torch.dot(vid[0], vid[1]).item()
    print(f"    视频间余弦 = {sim_vid:.4f}")

    # ---------- 3. 交叉模态相似度矩阵（关键判据） ----------
    print("\n[3] 交叉模态相似度矩阵（行=文本, 列=GIF视频）")
    mat = txt @ vid.T
    for i, name in enumerate(("A(8ball)", "B(cube)")):
        print(f"    text_{name}: " + "  ".join(f"gif_{n}={mat[i, j]:.4f}"
                                             for j, n in enumerate(("A(8ball)", "B(cube)"))))

    # 对角线是否显著大于非对角线 = 能否区分
    diag = torch.diagonal(mat).mean().item()
    off = ((mat.sum() - torch.diagonal(mat).sum()) / 2).item()
    margin = diag - off
    print(f"    对角均值={diag:.4f}  非对角均值={off:.4f}  margin={margin:+.4f}")
    ok = margin > 0.05 and torch.diagonal(mat).argmax().item() == 0
    print(f"    区分判定: {'PASS ✅ 两个 GIF 的条件可区分' if margin > 0.05 else 'FAIL ❌ 条件不可区分'}")

    # ---------- 4. 单模态消融：文本能否单独区分 ----------
    print("\n[4] 单模态区分力")
    print(f"    文本 A vs B      = {sim_txt:.4f}  (越低越好，要<0.7)")
    print(f"    GIF视频 A vs B   = {sim_vid:.4f}")
    print(f"    文本A vs GIF_A   = {mat[0, 0]:.4f}  (配对，应最高)")
    print(f"    文本B vs GIF_B   = {mat[1, 1]:.4f}  (配对)")
    print(f"    文本A vs GIF_B   = {mat[0, 1]:.4f}  (错配)")
    print(f"    文本B vs GIF_A   = {mat[1, 0]:.4f}  (错配)")

    # ---------- 5. 显存 ----------
    print("\n[5] 资源占用")
    if torch.cuda.is_available():
        print(f"    CUDA 显存峰值 = {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    else:
        print("    CPU 推理（本 venv 装的是 torch CPU 版）")
    nparam = sum(p.numel() for p in embedder.model.parameters())
    print(f"    参数量 = {nparam / 1e9:.3f} B")
    print("=" * 70)


if __name__ == "__main__":
    main()