# -*- coding: utf-8 -*-
"""区分「记忆还原」与「条件组合生成」的三个硬指标。

指标1 同条件两次生成是否相同
    相同 → 模型在查表/记忆（把训练样本背下来）
    不同 → 起点随机 + 条件引导 = 生成（扩散模型同款机制）

指标2 条件空间连续性：两个条件插值，输出场是否平滑变化
    平滑 → 学到的是**映射函数** f(cond) → field，可以接受任意条件
    跳变/塌缩 → 学到的是有限查表，只能复现训练过的点

指标3 条件信息量 vs 场信息量
    算清楚这是不是本质上的有损压缩（决定生成是「精确」还是「典型化」）
"""
import os
import sys
import json
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from model.aether_gif_model import AetherGIFModel
from physics.kuramoto_sync import build_spacetime_index
from data.clip_condition import CLIP_EMB_DIM

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
CACHE = ROOT / "data" / "clip_cache"
CKPT = ROOT / "outputs_real" / "ckpt" / "best.pt"


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_emb = np.load(CACHE / "text_emb.npy", mmap_mode="r")
    groups = json.load(open(CACHE / "group_list.json", encoding="utf-8"))
    items = json.load(open(ROOT / "data" / "index.json", encoding="utf-8"))

    model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)
    ck = torch.load(CKPT, map_location=device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"[CKPT] step={ck.get('step')}（40 步短训，只验证架构性质，不看画质）\n")

    t_coord, s_coord = build_spacetime_index(TS, GRID, GRID, device)
    N = len(t_coord)

    def make_cond(gi):
        t_e = np.asarray(text_emb[gi], np.float32)[None, :]
        i_e = np.zeros((1, CLIP_EMB_DIM), np.float32)
        has = np.array([[0.0]], np.float32)
        return torch.from_numpy(np.concatenate([t_e, i_e, has], 1)).to(device)

    c_a = make_cond(14)      # Lucky-Star
    c_b = make_cond(36)      # K-On

    print("=" * 74)
    print("指标1  同条件两次生成是否相同（记忆 vs 生成的分水岭）")
    print("=" * 74)
    with torch.no_grad():
        o1 = model.generate(duration=3, cond_vec=c_a)
        o2 = model.generate(duration=3, cond_vec=c_a)
    # generate 返回 (a, theta)
    d_a = (o1[0] - o2[0]).abs().mean().item()
    d_t = (o1[1] - o2[1]).abs().mean().item()
    rel = d_a / max(o1[0].abs().mean().item(), 1e-9)
    print(f"  同条件两次输出：振幅平均差={d_a:.6f}  相位平均差={d_t:.6f}")
    print(f"  相对差异 = {rel*100:.2f}%")
    if rel > 0.02:
        print("  → 输出显著不同：模型在**生成**，不是在背训练样本 ✓")
    else:
        print("  → 输出几乎相同：模型在查表/记忆 ✗")

    print()
    print("=" * 74)
    print("指标2  条件空间连续性（能否接受训练集里没出现过的条件）")
    print("=" * 74)
    with torch.no_grad():
        fs = []
        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cv = (1 - w) * c_a + w * c_b
            f = model.cond_field(cv, t_coord, s_coord)
            fs.append(f)
            print(f"  w={w:.2f}  场均值={f.mean().item():+.5f}  absmax={f.abs().max().item():.5f}")
    print("  逐步差异（应随 w 平滑变化，不是突然跳变）:")
    for i in range(1, 5):
        d = (fs[i] - fs[i - 1]).abs().mean().item()
        print(f"    w={i*0.25:.2f} 与 w={(i-1)*0.25:.2f} 的差 = {d:.6f}")

    # 场空间的离散度作为参照：训练集里两个真实条件的场差多大
    with torch.no_grad():
        fa = model.cond_field(c_a, t_coord, s_coord)
        fb = model.cond_field(c_b, t_coord, s_coord)
    ab = (fb - fa).abs().mean().item()
    step_d = [(fs[i] - fs[i - 1]).abs().mean().item() for i in range(1, 5)]
    even = all(abs(step_d[i] - step_d[0]) < 0.6 * step_d[0] for i in range(1, 4))
    print(f"\n  两个真实条件（{groups[14]} vs {groups[36]}）的场差 = {ab:.6f}")
    if even and max(step_d) < ab:
        print("  → 插值路径平滑、且位于两个真实条件之间：")
        print("    学到的是**连续映射**，中间条件（训练集没有）也有意义 ✓")
        print("    这正是「在见过的元素里组合出没见过的」所需的能力")
    else:
        print("  → 插值不平滑，映射可能是有限查表 ✗")

    print()
    print("=" * 74)
    print("指标3  信息量体检（决定生成是「精确复现」还是「典型化」）")
    print("=" * 74)
    n_field = N * 2 * D
    n_cond = CLIP_EMB_DIM * 2 + 1
    print(f"  条件维度 = {n_cond}")
    print(f"  场维度   = {n_field}  ({N} 位置 × {2*D} 实虚)")
    print(f"  压缩比   = {n_field/n_cond:.0f} : 1")
    print(f"  → 单个条件不可能唯一确定一个场。{n_field/n_cond:.0f} 倍有损压缩意味着")
    print("    模型学到的必然是**该条件下的典型场（条件期望）**，不是精确复现。")
    print(f"  驻波基压缩：{model.cond_field.n_basis} 个基 → 每基 {2*D} 维，")
    print(f"    可表达上限 = {model.cond_field.n_basis * 2 * D} 维（相对场的 {n_field} 维）")
    print(f"    条件权重 {model.cond_field.n_basis} 维 → 每个基的权重是实数，")
    print(f"    所以**连续插值有无限多种组合** —— 这是组合能力的上限来源。")


if __name__ == "__main__":
    main()