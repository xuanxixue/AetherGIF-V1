# -*- coding: utf-8 -*-
"""导出「模型内部逐步生成」的可视化数据（给 HTML 演示用）。

思路：给 PMNNCore 的 12 个 PhysicsBlock 各注册 forward hook，
抓每一层输出的 (a, theta) 快照，再逐层解码成画面。这样演示的是
**真实前向过程**，不是事后编的动画。

对每层导出：
  layer_XX.png    该层场解码出的 6 帧（时间均匀采样）横排拼图
  layer_XX_e.png  该层振幅能量热力图（6 帧 × 64px，sum over d_model）
meta.json 里含每层指标：与 target 的相位一致性 wcos、总能量、能量集中度。

用法（必须分进程跑，两个版本顶层包名相同）：
  C:/Python312/python.exe tools/export_viz.py --model v3600
  C:/Python312/python.exe tools/export_viz.py --model v4200
"""
import os
import sys
import json
import math
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
VERS = PROJ / "docx" / "实验记录" / "versions"
V3600 = VERS / "v3600_方案D_D125"
V4200 = VERS / "v4200_多模态CLIP产品版"

D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
SDIM = PATCH * PATCH * 3
WFFT = SDIM // 2 + 1


def build_env(model: str):
    for p in [str(PROJ), str(PROJ / "deps312"), str(PROJ / "int8_deps")]:
        sys.path.insert(0, p)
    try:
        from torch_accel_shim import install_full
        install_full()
    except Exception:
        pass
    if model == "v4200":
        for k, v in dict(COND_INIT_PC="1", COND_PC_RANK="8", COND_GROUPS="5",
                         COND_LOWRANK="16", PHASE_ACT="arctan", RES_LIMIT="0.5",
                         W2_NORM_MAX="3.0", KAPPA_MAX="6.0", FFT_FFTW="1",
                         IDEAL_CLAMP=str(math.pi)).items():
            os.environ.setdefault(k, v)
        root = V4200
    else:
        root = V3600
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    return root


def field_from(a, th):
    z = a * torch.exp(1j * th)
    return torch.cat([z.real, z.imag], dim=-1).float()


def frames_from(field, normalize=True):
    """[N,2D] -> np [T,H,W,3] uint8。复用项目 codec 口径。"""
    from codec.gif_codec import decode_field_to_frames
    return np.stack(decode_field_to_frames(
        field.cpu(), TS, GRID, PATCH, WIN, IMG, D, normalize=normalize)).astype(np.uint8)


def energy_map(a):
    """[N,D] 振幅 -> [TS,H,W] 能量热力图（对 d_model 求和）。"""
    a = a.reshape(TS, GRID, GRID, -1).sum(-1)      # [T,H,W]
    a = a - a.min()
    a = a / (a.max() + 1e-9)
    return (a * 255).astype(np.uint8)


def colorize(gray):
    """灰度 -> 伪彩热力图（蓝-青-绿-黄-红）。"""
    x = gray.astype(np.float32) / 255.0
    r = np.clip(1.5 * x - 0.35, 0, 1)
    g = np.clip(1.5 - abs(x - 0.55) * 3.0, 0, 1)
    b = np.clip(1.2 - 1.8 * x, 0, 1)
    out = np.stack([r, g, b], -1) * 255
    return out.astype(np.uint8)


def save_strip(frames, path, cell=128):
    """6 帧横排拼图。"""
    from PIL import Image
    n = len(frames)
    cv = Image.new("RGB", (cell * n, cell), (0, 0, 0))
    for i, f in enumerate(frames):
        im = Image.fromarray(f).resize((cell, cell), Image.LANCZOS)
        cv.paste(im, (i * cell, 0))
    cv.save(path)


def save_energy(emaps, path, cell=64):
    from PIL import Image
    n = len(emaps)
    cv = Image.new("RGB", (cell * n, cell), (0, 0, 0))
    for i, m in enumerate(emaps):
        im = Image.fromarray(colorize(m)).resize((cell, cell), Image.NEAREST)
        cv.paste(im, (i * cell, 0))
    cv.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v3600", "v4200"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--src", type=int, default=0, help="v4200 用第几个素材的条件")
    ap.add_argument("--text", default="台球", help="v3600 的文本输入（其文本通路为占位）")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    root = build_env(args.model)

    global torch, np
    import numpy as np
    import torch
    from model.aether_gif_model import AetherGIFModel
    from physics.kuramoto_sync import build_spacetime_index
    from codec.gif_codec import load_gif_frames, resample_frames, encode_gif_to_field

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = PROJ / "viz" / ("data_" + args.model)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- 模型 ----------------
    if args.model == "v4200":
        ck_path = Path(args.ckpt) if args.ckpt else (
            V4200 / "outputs_real" / "colband150_1024b" / "guided_ckpt.pt")
        model = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                               ffn_hidden=256, init_mode="init", n_modules=8,
                               mod_gamma=0.15).to(device)
    else:
        ck_path = Path(args.ckpt) if args.ckpt else (V3600 / "outputs_real" / "overfit_ckpt.pt")
        model = AetherGIFModel(d_model=D, device=device, use_qwen=False).to(device)

    sd = torch.load(str(ck_path), map_location=device, weights_only=True)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
    ret = model.load_state_dict(sd, strict=False)
    print(f"[CKPT] {ck_path.name}  missing={len(ret.missing_keys)} "
          f"unexpected={len(ret.unexpected_keys)}")
    model.eval()

    t, s = build_spacetime_index(TS, GRID, GRID, device)

    # ---------------- 注册逐层 hook ----------------
    snaps = []

    def mk_hook():
        def h(mod, inp, out):
            aa = out[0].detach()
            tt = out[1].detach()
            snaps.append((aa, tt))
        return h

    handles = [blk.register_forward_hook(mk_hook()) for blk in model.pmnncore.blocks]

    # ---------------- 输入条件 ----------------
    tf = None
    cond_info = {}
    if args.model == "v4200":
        VD = PROJ / "训练素材22" / "vectors"
        text = np.load(VD / "text22.npy")
        first = np.load(VD / "first22.npy")
        gifv = np.load(VD / "gif22.npy")
        from data.clip_condition import build_cond_vector3
        i = args.src
        cv = build_cond_vector3(text[i:i + 1], first[i:i + 1], gifv[i:i + 1], (1, 1, 1))
        cv_t = torch.from_numpy(cv).to(device).float()
        meta22 = json.loads((VD / "meta22.json").read_text(encoding="utf-8"))
        cond_info = dict(kind="三路条件向量", dim=int(cv.shape[1]),
                         item=meta22["items"][i]["file"],
                         text_norm=float(np.linalg.norm(text[i])),
                         first_norm=float(np.linalg.norm(first[i])),
                         gif_norm=float(np.linalg.norm(gifv[i])))
        with torch.inference_mode():
            a, th, diag = model(texts=None, target_field=None, t_coord=t, s_coord=s,
                                batch_size=1, cond_vec=cv_t, use_cond_field=True)
        # 目标（仅用于算 wcos 指标，不参与生成）
        gp = PROJ / "训练素材22" / meta22["items"][i]["file"]
        tf = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D).to(device).unsqueeze(0)
    else:
        with torch.inference_mode():
            a, th, diag = model(texts=[args.text], target_field=None, t_coord=t, s_coord=s,
                                batch_size=1)
        cond_info = dict(kind="无真实条件（占位文本向量）", dim=0, item=args.text,
                         text_norm=0.0, first_norm=0.0, gif_norm=0.0)

    for h in handles:
        h.remove()

    print(f"[HOOK] 抓到 {len(snaps)} 层快照")
    assert len(snaps) == 12, f"期望 12 层，实际 {len(snaps)}"

    # ---------------- 目标场（算指标 & 对照预览） ----------------
    a_tar = th_tar = None
    if tf is not None:
        a_tar = torch.sqrt(tf[..., :D] ** 2 + tf[..., D:] ** 2 + 1e-12)
        th_tar = torch.atan2(tf[..., D:], tf[..., :D])

    # ---------------- 逐层导出 ----------------
    layers = []
    for li, (aa, tt) in enumerate(snaps):
        a0, th0 = aa[0], tt[0]
        fld = field_from(a0, th0)
        # 时间均匀取 6 帧
        allf = frames_from(fld, normalize=True)          # [30,128,128,3]
        idx = [int(k * (len(allf) - 1) / 5) for k in range(6)]
        save_strip([allf[j] for j in idx], outdir / f"layer_{li:02d}.png")
        em = energy_map(a0)
        save_energy([em[j] for j in idx], outdir / f"layer_{li:02d}_e.png")

        rec = dict(layer=li, a_mean=float(a0.mean()), a_max=float(a0.max()),
                   energy=float(a0.pow(2).sum()))
        # 与 target 的功率加权相位一致性（target 存在时）
        if th_tar is not None:
            pe = torch.remainder(th0 - th_tar[0] + math.pi, 2 * math.pi) - math.pi
            w = a_tar[0] ** 2
            rec["wcos"] = float((w * torch.cos(pe)).sum() / w.sum().clamp(min=1e-12))
        # 同步度（Kuramoto order parameter 的近似：相位模长）
        rec["sync"] = float(torch.abs(torch.exp(1j * th0).mean()))
        layers.append(rec)
        print(f"  L{li:02d} a_mean={rec['a_mean']:.4f} sync={rec['sync']:.3f} "
              f"wcos={rec.get('wcos', float('nan')):+.4f}")

    # ---------------- 最终帧 + GIF ----------------
    a0, th0 = a[0].detach(), th[0].detach()
    fld = field_from(a0, th0)
    final_frames = frames_from(fld, normalize=True)

    from PIL import Image
    pil = [Image.fromarray(np.ascontiguousarray(f)) for f in final_frames]
    pil[0].save(outdir / "final.gif", save_all=True, append_images=pil[1:],
                loop=0, duration=100)
    idx = [int(k * (len(final_frames) - 1) / 5) for k in range(6)]
    save_strip([final_frames[j] for j in idx], outdir / "final_strip.png")
    save_energy([energy_map(a0)[j] for j in idx], outdir / "final_energy.png")

    # 目标对照（v4200）
    if tf is not None:
        tgt_frames = frames_from(tf[0], normalize=True)
        pil2 = [Image.fromarray(np.ascontiguousarray(f)) for f in tgt_frames]
        pil2[0].save(outdir / "target.gif", save_all=True, append_images=pil2[1:],
                     loop=0, duration=100)
        idx2 = [int(k * (len(tgt_frames) - 1) / 5) for k in range(6)]
        save_strip([tgt_frames[j] for j in idx2], outdir / "target_strip.png")

    meta = dict(model=args.model, ckpt=str(ck_path), seed=args.seed,
                n_blocks=len(layers), shape=dict(D=D, GRID=GRID, PATCH=PATCH,
                                                WIN=WIN, IMG=IMG, TS=TS),
                cond=cond_info, layers=layers,
                final=dict(a_mean=float(a0.mean()), sync=float(torch.abs(torch.exp(1j * th0).mean())),
                           energy=float(a0.pow(2).sum())))
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"[SAVED] {outdir}")
    print(f"[IMG] {len(list(outdir.glob('*.png')))} png, "
          f"{len(list(outdir.glob('*.gif')))} gif")


if __name__ == "__main__":
    main()
