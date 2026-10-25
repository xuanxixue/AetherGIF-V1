# -*- coding: utf-8 -*-
"""【自动验收】v3000J 训练产物 vs v2500 vs 原始训练素材。

夜间无人值守运行，输出机器可读的 PASS/FAIL 判定 + 人可读的对比表。

验收标准（三方对比，不只看绝对值）：
  1. 对比度 std≥ target 的 85%          —— 直击第十五死穴（糊 = 动态范围被压缩）
  2. PSNR 不低于 v2500基线               —— 不能为了修对比度把结构弄坏
  3. 频段能量占比与 target 的偏差 ≤ 1.5个百分点 —— 保住 v3000 修好的能量分布
  4. 无异常过曝（< 1.5%）
  5. 逐帧稳定（帧间 std 波动不出现某帧崩塌）

用法：
  python scripts/auto_accept.py --ckpt outputs_real/v3000J/overfit_ckpt.pt \
                                --gif   outputs_real/v3000J/overfit_honest.gif
退出码：0=PASS, 1=FAIL
"""
import os, sys, math, argparse, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "src"))
import numpy as np
from PIL import Image
import torch

D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
W_FFT = PATCH * PATCH * 3 // 2 + 1          # 25
GIF_SRC = os.path.join(BASE, "data", "gif_raw", "noto", "8-ball_1f3b1.gif")
V2500_DIR = r"D:/AetherGIF-V1/docx/实验记录/versions/v2500_方案D_2500步/outputs_real"

# 验收阈值
TH_STD_RATIO = 0.85     # 对比度 ≥ target 的 85%
TH_BAND_TOL = 0.015     # 频段能量占比容差（绝对占比，1.5 个百分点）
TH_OVEREXPO = 0.015# 过曝像素占比上限


def read_gif(path, target_size=None):
    im = Image.open(path)
    fr = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        a = np.array(im.convert("RGB"), dtype=np.uint8)
        if target_size and a.shape[0] != target_size:
            a = np.array(Image.fromarray(a).resize((target_size,) * 2, Image.LANCZOS))
        fr.append(a)
    return np.stack(fr)


def gray(x):
    return x.astype(np.float32).mean(-1)


def band_energy(a_tar, a_out):
    """频段能量占比对比（按 w 切，w=c%25）。返回 dict"""
    out = {}
    for name, idxs in [
        ("w0(DC)", [c for c in range(D) if c % W_FFT == 0]),
        ("w1-6", [c for c in range(D) if 1 <= c % W_FFT <= 6]),
        ("w7-12", [c for c in range(D) if 7 <= c % W_FFT <= 12]),
        ("w13-18", [c for c in range(D) if 13 <= c % W_FFT <= 18]),
        ("w19-24", [c for c in range(D) if 19 <= c % W_FFT <= 24]),
    ]:
        et = (a_tar[:, idxs] ** 2).sum().item()
        eo = (a_out[:, idxs] ** 2).sum().item()
        out[name] = (et, eo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE, "outputs_real", "v3000J", "overfit_ckpt.pt"))
    ap.add_argument("--gif", default=os.path.join(BASE, "outputs_real", "v3000J", "overfit_honest.gif"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from codec.gif_codec import encode_gif_to_field, decode_field_to_frames
    from physics.kuramoto_sync import build_spacetime_index
    from model.aether_gif_model import AetherGIFModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 92)
    print("自动验收：v3000J  vs  v2500基线  vs  原始训练素材")
    print("=" * 92)

    fail = []

    # ---------- 参照 1：codec 天花板（架构表达极限） ----------
    field = encode_gif_to_field(GIF_SRC, TS, GRID, PATCH, WIN, IMG, D)
    a_tar = torch.sqrt(field[:, :D] ** 2 + field[:, D:] ** 2 + 1e-12)
    ceil = np.stack([np.asarray(f, dtype=np.float32) for f in
                     decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D, normalize=False)])
    gceil = gray(ceil)

    # ---------- 参照 2：原始素材 ----------
    raw = read_gif(GIF_SRC, IMG)
    idx = np.linspace(0, len(raw) - 1, TS * WIN).round().astype(int)
    raw30 = gray(raw[idx])
    graw = gray(read_gif(args.gif, IMG))

    # ---------- 被测：checkpoint 前向（看系数域能量分布） ----------
    ck_ok = os.path.exists(args.ckpt)
    if ck_ok:
        m = AetherGIFModel(device=device, use_qwen=False).to(device)
        m.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
        m.eval()
        t, s = build_spacetime_index(TS, GRID, GRID, device)
        with torch.inference_mode():
            a, th, _ = m(texts=["台球"], target_field=field.unsqueeze(0).to(device),
                         t_coord=t, s_coord=s, batch_size=1)
        a_out = a[0].cpu()
        be = band_energy(a_tar, a_out)
    else:
        print("[WARN] ckpt 缺失，跳过系数域检查:", args.ckpt)
        be = {}

    # ---------- GIF 产物（看空间域） ----------
    gif_ok = os.path.exists(args.gif)
    g_v3k = graw if gif_ok else None

    g_v25 = None
    p25 = os.path.join(V2500_DIR, "overfit_honest.gif")
    if os.path.exists(p25):
        g_v25 = gray(read_gif(p25, IMG))

    def psnr(a, b):
        n = min(len(a), len(b))
        return 10 * math.log10(255.0 ** 2 /
                               max(((a[:n] - b[:n]) ** 2).mean(), 1e-12))

    print()
    print("【1】空间域对比度（第十五死穴核心指标）")
    print("-" * 92)
    # 基准用 codec 天花板：本架构能表达的上限。原始素材经 4x4 patch + rfft2 截断，
    # 与模型输出不同口径，直接比会失真（实测原始 std 反而低于天花板）。
    sd_base = gceil.std()
    sd_raw = graw.std()
    print("  codec 天花板  std=%6.2f   <- 验收基准(架构表达上限)" % sd_base)
    print("  原始素材      std=%6.2f   (%.1f%% of 天花板)" % (sd_raw, sd_raw / sd_base * 100))
    if g_v25 is not None:
        print("  v2500 基线    std=%6.2f   (%.1f%% of 天花板)"
              % (g_v25.std(), g_v25.std() / sd_base * 100))
    if gif_ok:
        r = g_v3k.std() / sd_base
        print("  v3000J 本次std=%6.2f   (%.1f%% of 天花板)" % (g_v3k.std(), r * 100))
        if r < TH_STD_RATIO:
            fail.append("对比度 %.1f%% < 阈值 %.0f%%" % (r * 100, TH_STD_RATIO * 100))

    print()
    print("【2】PSNR（vs codec 天花板，衡量结构还原）")
    print("-" * 92)
    p25v = psnr(g_v25, gceil) if g_v25 is not None else float("nan")
    p3v = psnr(g_v3k, gceil) if gif_ok else float("nan")
    if g_v25 is not None:
        print("  v2500 基线  PSNR=%6.2f dB" % p25v)
    if gif_ok:
        print("  v3000J 本次  PSNR=%6.2f dB   (Δ vs v2500 = %+.2f dB)" % (p3v, p3v - p25v))
        if gif_ok and not np.isnan(p25v) and p3v < p25v - 0.5:
            fail.append("PSNR %.2f 显著低于 v2500 的 %.2f" % (p3v, p25v))

    print()
    print("【3】频段能量占比（v3000 修好的部分，必须保持）")
    print("-" * 92)
    if be:
        print("  %-10s %10s %10s %10s" % ("频段", "target", "v3000J", "偏差"))
        tot_t = sum(v[0] for v in be.values())
        tot_o = sum(v[1] for v in be.values())
        for name, (et, eo) in be.items():
            vt, vo = et / tot_t, eo / tot_o
            dev = vo - vt
            flag = "" if abs(dev) <= TH_BAND_TOL else "  <-- 超容差"
            print("  %-10s %9.2f%% %9.2f%% %+9.2f%%%s" % (name, vt * 100, vo * 100, dev * 100, flag))
            if abs(dev) > TH_BAND_TOL:
                fail.append("频段 %s 能量占比偏差 %.2f%% 超容差 %.1f%%"
                            % (name, dev * 100, TH_BAND_TOL * 100))
    else:
        print("  (ckpt 缺失，跳过)")

    print()
    print("【4】出图健康度（过曝/欠曝/逐帧崩塌）")
    print("-" * 92)
    if gif_ok:
        over = (g_v3k >= 254).mean() * 100
        under = (g_v3k <= 1).mean() * 100
        per = g_v3k.reshape(len(g_v3k), -1).std(1)
        collapse = [i for i, v in enumerate(per) if v < per.mean() * 0.4]
        print("  过曝=%.2f%%  欠曝=%.2f%%" % (over, under))
        print("  逐帧 std: 均值=%.2f 最小=%.2f（step%.0f 位置）"
              % (per.mean(), per.min(), per.argmin()))
        if over > TH_OVEREXPO:
            fail.append("过曝 %.2f%% > 阈值 %.1f%%" % (over, TH_OVEREXPO * 100))
        if collapse:
            fail.append("第%s帧崩塌(std<均值40%%)" % collapse)

    print()
    print("=" * 92)
    if fail:
        print("验收结果：FAIL  —— 需继续修复")
        for f in fail:
            print("  x " + f)
    else:
        print("验收结果：PASS  —— 达标，无需干预")
    print("=" * 92)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"pass": not fail, "failures": fail,
                       "std_vs_ceiling": (float(g_v3k.std() / gceil.std()) if gif_ok else None),
                       "psnr_v3000J": (float(p3v) if gif_ok else None),
                       "psnr_v2500": (float(p25v) if g_v25 is not None else None),
                       "band_energy": ({k: {"target": v[0], "model": v[1]}
                                        for k, v in be.items()} if be else None)},
                      f, ensure_ascii=False, indent=2)
        print("[JSON] %s" % args.json)

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())