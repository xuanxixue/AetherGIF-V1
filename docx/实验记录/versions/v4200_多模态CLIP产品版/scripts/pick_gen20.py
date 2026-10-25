# -*- coding: utf-8 -*-
"""从素材库挑 20 张，构造泛化验证集。

【验证设计 · 用户指定】
  A 组 10 张：动作相同、画面内容不同   → 测「结构：同动作能否泛化到新画面」
  B 组 10 张：画面内容相同、动作不同   → 测「结构：同画面能否泛化到新动作」

【实测分组，不靠目录名猜】
  对每张 GIF 算两个可测量：
    - 运动量 motion = 逐帧像素差均值（区分快/慢/静止）
    - 结构特征 struct = 首帧+末帧降采样灰度向量（区分画面内容）

  A 组：motion 聚在同一档（=动作幅度相同），struct 两两差异最大（=画面不同）
  B 组：严格限定「同一作品目录内」配对（同角色/同场景 = 画面内容相同），
        motion 比值 >=3x 且都动态（=动作不同），全局贪心不重复用图。

【为什么 B 组必须限定同一作品目录】
  跨作品用灰度缩略图算余弦相似度会误判——Yuru-Camp 和 Kannagi 是两部
  完全不同的动画，却因构图巧合被 sim>0.985 归为「同画面」。同一作品目录
  （同角色同画风）才是「画面内容相同」的可靠来源。
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageSequence

DATA = Path("D:/AetherGIF-V1/data/gif_raw")
OUT = ROOT / "data" / "gen20"
OUT.mkdir(parents=True, exist_ok=True)
PROJ = Path("D:/AetherGIF-V1")
ANIME = DATA / "anime"

SIZE = 64                    # 结构特征降采样分辨率
B_SIM_MIN = 0.93             # B 组「同画面」相似度下限
B_MOTION_MIN = 1.0           # B 组两个都须动（排除静止）
B_MOTION_RATIO = 3.0         # B 组动作差异：快慢比值


def load_gif_stats(path, size=SIZE):
    """读 GIF，返回 (motion, struct, bright, T)。失败返回 None。"""
    try:
        im = Image.open(path)
        frames = []
        for fr in ImageSequence.Iterator(im):
            frames.append(np.asarray(fr.convert("L").resize((size, size)),
                                     dtype=np.float32))
        if len(frames) < 3:
            return None
        arr = np.stack(frames)                      # [T,size,size]
        motion = float(np.abs(arr[1:] - arr[:-1]).mean())
        struct = (arr[0] + arr[-1]).reshape(-1) / 255.0
        bright = float(arr.mean())
        return motion, struct, bright, arr.shape[0]
    except Exception:
        return None


def struct_sim(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def scan_work(work_dir):
    """扫描单个作品目录，返回 [(path, motion, struct, frames), ...]"""
    ss = []
    for p in sorted(work_dir.rglob("*.gif")):
        r = load_gif_stats(p)
        if r is not None:
            motion, struct, bright, T = r
            ss.append(dict(path=p, motion=motion, struct=struct,
                           frames=T, bright=bright))
    return ss


def pick_diverse(pool, k=10):
    """贪心挑 k 张两两 struct 差异最大的。"""
    if not pool:
        return []
    picked = [pool[0]]
    while len(picked) < min(k, len(pool)):
        best, best_d = None, 1e9
        for s in pool:
            if any(s is p for p in picked):
                continue
            d = min(struct_sim(s["struct"], p["struct"]) for p in picked)
            if d < best_d:
                best_d, best = d, s
        if best is None:
            break
        picked.append(best)
    return picked


def main():
    # ============ 收集全部素材 ============
    # anime 作品子目录（同角色/同场景系列）+ animetv/gifs 子目录
    work_dirs = [d for d in ANIME.iterdir() if d.is_dir() and d.name != "animetv"]
    animetv = ANIME / "animetv" / "gifs"
    if animetv.is_dir():
        work_dirs += [d for d in animetv.iterdir() if d.is_dir()]

    print(f"[scan] 作品目录 {len(work_dirs)} 个，开始实测运动量与结构特征")
    all_stats = []
    per_work = {}
    for w in sorted(work_dirs):
        ss = scan_work(w)
        if len(ss) >= 2:
            per_work[w.name] = ss
        all_stats.extend(ss)

    print(f"[scan] 实测成功 {len(all_stats)} 张（{len(per_work)} 个作品有 >=2 张）")
    if len(all_stats) < 40:
        print("[ERR] 素材不足")
        return

    motions = np.array([s["motion"] for s in all_stats])
    q33, q66 = np.percentile(motions, [33, 66])
    print(f"[scan] 运动量 {motions.min():.2f}~{motions.max():.2f}，"
          f"中位 {np.median(motions):.2f}，档位切点 慢<{q33:.2f} 中<{q66:.2f}")

    # ============ A 组：同动作档、不同画面 ============
    mid_pool = [s for s in all_stats if q33 <= s["motion"] < q66]
    print(f"[A组] 中速档候选 {len(mid_pool)} 张，贪心挑画面差异最大的 10 张")
    groupA = pick_diverse(mid_pool, 10)

    simsA = [struct_sim(groupA[i]["struct"], groupA[j]["struct"])
             for i in range(len(groupA)) for j in range(i + 1, len(groupA))]
    print(f"[A组] 10 张，平均结构相似度={np.mean(simsA):.3f}（越低越好，<0.9=画面真的不同）")
    print(f"[A组] 运动量 {min(s['motion'] for s in groupA):.2f}~"
          f"{max(s['motion'] for s in groupA):.2f}（同档=动作幅度相同）")

    # ============ B 组：同画面（同作品目录）、不同动作 ============
    print("[B组] 同一作品目录内搜索「同画面不同动作」配对...")
    pairs = []
    for work_name, ss in per_work.items():
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                s = struct_sim(ss[i]["struct"], ss[j]["struct"])
                m1, m2 = ss[i]["motion"], ss[j]["motion"]
                if s < B_SIM_MIN:
                    continue
                if min(m1, m2) < B_MOTION_MIN:
                    continue
                if max(m1, m2) < B_MOTION_RATIO * min(m1, m2):
                    continue
                ratio = max(m1, m2) / min(m1, m2)
                pairs.append(dict(a=ss[i], b=ss[j], sim=s, ratio=ratio,
                                  work=work_name))

    pairs.sort(key=lambda x: (-x["sim"], -x["ratio"]))
    print(f"[B组] 找到 {len(pairs)} 对候选（sim>={B_SIM_MIN} & 都动 & 快慢比>={B_MOTION_RATIO}）")

    # 贪心：优先 sim 高、快慢比大的，且整组不重复用同一张图
    groupB, used = [], set()
    for p in pairs:
        if len(groupB) >= 10:
            break
        if p["a"]["path"] in used or p["b"]["path"] in used:
            continue
        groupB.append(p["a"])
        groupB.append(p["b"])
        used.add(p["a"]["path"])
        used.add(p["b"]["path"])
    groupB = groupB[:10]

    print(f"[B组] 选出 {len(groupB)} 张（{len(groupB)//2} 对）")
    if len(groupB) >= 2:
        mo = [s["motion"] for s in groupB]
        print(f"[B组] 运动量 {min(mo):.2f}~{max(mo):.2f}（跨档=动作不同）")

    # ============ 落盘清单 ============
    manifest = {"A_same_motion_diff_scene": [], "B_same_scene_diff_motion": []}
    for tag, grp in [("A_same_motion_diff_scene", groupA),
                     ("B_same_scene_diff_motion", groupB)]:
        for s in grp:
            try:
                rel = str(Path(s["path"]).relative_to(PROJ))
            except ValueError:
                rel = str(s["path"])
            manifest[tag].append(dict(
                path=rel,
                motion=round(s["motion"], 3),
                frames=s["frames"],
                struct_norm=round(float(np.linalg.norm(s["struct"])), 3),
            ))

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"[清单] {OUT / 'manifest.json'}")
    print(f"  A 组（同动作/不同画面）{len(groupA)} 张")
    for m in manifest["A_same_motion_diff_scene"][:10]:
        print(f"    motion={m['motion']:>6.2f}  {m['path']}")
    print(f"  B 组（同画面/不同动作）{len(groupB)} 张")
    for m in manifest["B_same_scene_diff_motion"][:10]:
        print(f"    motion={m['motion']:>6.2f}  {m['path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()