# -*- coding: utf-8 -*-
"""结构自编码器训练 + A/B 组解耦泛化验证（Step 1）

【测什么】
  20 张图（data/gen20/manifest.json）：
    A 组 10 张：动作相同、画面不同
    B 组 10 张：画面相同、动作不同（同一作品目录内 5 对）
  训练结构自编码器重建这 20 张，然后验证解耦：
    z_scene（画面因子）：  A 组内应分开、B 组配对应靠近
    z_motion（动作因子）： A 组内应靠近、B 组配对应分开

【判据（不追求重建 PSNR，追求解耦）】
  scene 解耦度  = mean(跨组 scene 距离) − mean(同组/配对 scene 距离)  > 0
  motion 解耦度 = mean(配对 motion 距离) − mean(A 组内 motion 距离)     > 0

【运行】
  cd 到 v4200 快照根目录后
    C:/Python312/python.exe scripts/train_structure_ae.py [STEPS]
  环境变量：DIM_Z / N_BLOCKS / LR / COND_LR / N_MODULES / MOD_GAMMA / STEPS
"""
import math
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir():
        break
    if _PROJ.parent == _PROJ:
        break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))
_DEPS = _PROJ / "deps312"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass
try:
    from torch_accel_shim import install_full
    install_full()
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

from codec.gif_codec import encode_gif_to_field
from physics.kuramoto_sync import build_spacetime_index
from model.structure_ae import StructureAutoEncoder

# ---------------- 编解码口径（与全项目一致） ----------------
D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
N = TS * GRID * GRID   # 6144

DIM_Z = int(os.environ.get("DIM_Z", "256"))
N_BLOCKS = int(os.environ.get("N_BLOCKS", "12"))
LR = float(os.environ.get("LR", "1e-4"))
N_MODULES = int(os.environ.get("N_MODULES", "8"))
MOD_GAMMA = float(os.environ.get("MOD_GAMMA", "0.15"))
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("STEPS", "300"))
W_SCENE = float(os.environ.get("W_SCENE", "0.2"))   # 解耦正则强度
W_MOT = float(os.environ.get("W_MOT", "0.2"))
AMP_P = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = ROOT / "outputs_real" / "structure_ae"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GIF_DIR = Path("D:/AetherGIF-V1")   # manifest.path 已相对项目根（data\gif_raw\...）


# ---------------------------------------------------------------------------
def load_manifest():
    m = json.loads((ROOT / "data" / "gen20" / "manifest.json").read_text(encoding="utf-8"))
    A = m["A_same_motion_diff_scene"]
    B = m["B_same_scene_diff_motion"]
    return A, B


def encode_one(rel):
    gp = GIF_DIR / Path(rel)
    f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D)
    return f


def split_field(field):
    """[N,2D] -> (a, theta)"""
    re, im = field[..., :D], field[..., D:]
    a = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
    th = torch.atan2(im, re)
    return a, th


# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("[1] 加载 20 张图 -> 频域场")
    A_specs, B_specs = load_manifest()
    print(f"    A 组 {len(A_specs)} 张 / B 组 {len(B_specs)} 张")

    A_rel = [s["path"] for s in A_specs]
    # B 组按 manifest 顺序两两配对：0-1, 2-3, 4-5, ...
    B_rel = [s["path"] for s in B_specs]
    B_pairs = [(B_rel[i], B_rel[i + 1]) for i in range(0, len(B_rel), 2)]

    all_rel = A_rel + B_rel
    fields = [encode_one(r) for r in all_rel]
    targets = torch.stack([f for f in fields], 0)      # [20, N, 2D]
    a_tars = torch.stack([split_field(f)[0] for f in fields], 0)
    th_tars = torch.stack([split_field(f)[1] for f in fields], 0)
    print(f"    target batch = {tuple(targets.shape)}")

    t, s = build_spacetime_index(TS, GRID, GRID, device)

    # 索引映射（nA 之后是 B 组的平铺下标）
    nA = len(A_rel)
    _flat_idx = {r: k for k, r in enumerate(all_rel)}
    B_pair_idx = [(_flat_idx[p0], _flat_idx[p1]) for p0, p1 in B_pairs]
    print("\n[2] 建模型")
    torch.manual_seed(1234)
    model = StructureAutoEncoder(d_model=D, dim_z=DIM_Z, n_blocks=N_BLOCKS,
                                 n_modules=N_MODULES, mod_gamma=MOD_GAMMA).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    参数量 = {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    print(f"\n[3] 训练 {STEPS} 步（batch=1，随机抽 20 张）")
    for step in range(1, STEPS + 1):
        opt.zero_grad()
        i = np.random.randint(0, len(all_rel))
        one = targets[i:i + 1].to(device)
        a, th, z_scene, z_motion, diag = model(one, t, s, TS, GRID)

        # 重建 loss（log1p 域，与全项目口径一致）
        pe = torch.remainder(th - th_tars[i:i + 1].to(device) + math.pi, 2 * math.pi) - math.pi
        la = torch.log1p(a.clamp(min=1e-8))
        lt = torch.log1p(a_tars[i:i + 1].to(device).clamp(min=1e-8))
        w = (a_tars[i:i + 1].to(device) / (a_tars[i:i + 1].to(device).max() + 1e-8)).pow(AMP_P)
        loss_amp = (w * (la - lt).pow(2)).mean()
        w_ph = (a_tars[i:i + 1].to(device) ** 2).clamp(min=1e-8)
        loss_ph = (w_ph * pe.pow(2)).sum() / w_ph.sum()
        loss = loss_amp + loss_ph

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == 1:
            print(f"  [{step:4d}] loss={loss.item():.4f} amp={loss_amp.item():.4f} "
                  f"phase={loss_ph.item():.4f}", flush=True)

    # ------------------------------------------------------------------ 解耦评测
    print("\n[4] 解耦评测：A/B 组结构向量距离")
    model.eval()
    with torch.inference_mode():
        Zs, Zm = [], []
        for k in range(len(all_rel)):
            one = targets[k:k + 1].to(device)
            zs, zm, _ = model.encoder(one, TS, GRID)
            Zs.append(zs[0].cpu())
            Zm.append(zm[0].cpu())
    Zs = torch.stack(Zs)   # [20, dim_z]
    Zm = torch.stack(Zm)

    def cos_dist(a, b):
        a = a / (a.norm() + 1e-8)
        b = b / (b.norm() + 1e-8)
        return float((1.0 - (a * b).sum()).item())

    # A 组内 scene 两两距离（应大：画面不同）
    A_scene = [cos_dist(Zs[i], Zs[j]) for i in range(nA) for j in range(i + 1, nA)]
    # B 组配对 scene 距离（应小：同画面）
    B_scene = [cos_dist(Zs[p0], Zs[p1]) for p0, p1 in B_pair_idx]

    # A 组内 motion 距离（应小：同动作）
    A_motion = [cos_dist(Zm[i], Zm[j]) for i in range(nA) for j in range(i + 1, nA)]
    # B 组配对 motion 距离（应大：不同动作）
    B_motion = [cos_dist(Zm[p0], Zm[p1]) for p0, p1 in B_pair_idx]

    scene_sep = np.mean(A_scene) - np.mean(B_scene)     # >0 = 场景解耦
    motion_sep = np.mean(B_motion) - np.mean(A_motion)  # >0 = 动作解耦

    print(f"    scene 因子：A组内两两距离={np.mean(A_scene):.4f}  B配对距离={np.mean(B_scene):.4f}")
    print(f"      -> scene 解耦度 = {np.mean(A_scene):.4f} − {np.mean(B_scene):.4f} = {scene_sep:+.4f}  {'✅' if scene_sep > 0 else '❌'}")
    print(f"    motion 因子：A组内两两距离={np.mean(A_motion):.4f}  B配对距离={np.mean(B_motion):.4f}")
    print(f"      -> motion 解耦度 = {np.mean(B_motion):.4f} − {np.mean(A_motion):.4f} = {motion_sep:+.4f}  {'✅' if motion_sep > 0 else '❌'}")
    print(f"    综合：{'✅ 结构模型解耦成功（动作与画面分离）' if (scene_sep > 0 and motion_sep > 0) else '❌ 未解耦'}")

    # 保存
    sd = {k: v for k, v in model.state_dict().items()}
    torch.save(sd, OUT_DIR / "structure_ae.pt")
    print(f"\n[CKPT] {OUT_DIR / 'structure_ae.pt'} ({sum(v.numel() for v in sd.values()):,} params)")
    print("=" * 74)


if __name__ == "__main__":
    main()