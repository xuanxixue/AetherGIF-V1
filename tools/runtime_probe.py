# -*- coding: utf-8 -*-
"""运行时深度探针 —— 只读，不修改/删除任何已有文件。

目的：GitHub 事故的文件还原可能不完整，此处对**训练全链路的实际执行路径**做
真实调用验证（不是语法检查、不是 import 检查），把只剩运行时才暴露的 bug 挖出来。

★ 为什么用子进程隔离：根 `src/` 与 `v4200/src/` 的顶层包名完全相同
（model / physics / codec / data / train），同进程内第二次 `import model`
会命中 sys.modules 缓存而拿到**另一个生态**的模块。日常脚本靠"只挂一个 src"
避免此问题，探针为了同时验两套必须分进程。

覆盖：
  A/B. 编解码闭环（encode -> decode -> PSNR、本次修的 resize 路径）
  C.   数据资产（index.json 抽样、GIF 可解码抽样）
  D.   根生态（方案 D）：模型构造 + ckpt 加载 + 物理核 fwd/bwd     [--phase=root]
  E.   v4200 主战场：模型 + ckpt + 数据 + 1 步 fwd/bwd（不落盘）  [--phase=v4200]
  F.   训练脚本静默失败 / 数据安全静态扫描

用法：
  C:/Python312/python.exe tools/runtime_probe.py            # 全部（自动起子进程）
  C:/Python312/python.exe tools/runtime_probe.py --fast     # 跳过耗时段
  C:/Python312/python.exe tools/runtime_probe.py --phase=root    # 仅根生态
  C:/Python312/python.exe tools/runtime_probe.py --phase=v4200   # 仅 v4200
"""
import os
import sys
import json
import time
import traceback
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PHASE = ""
for _a in sys.argv[1:]:
    if _a.startswith("--phase="):
        PHASE = _a.split("=", 1)[1]

# ---------------------------------------------------------------- 环境三件套
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deps312"))
sys.path.insert(0, str(ROOT / "int8_deps"))

try:
    from torch_accel_shim import install_full
    install_full()
    print("[ENV] torch_accel_shim.install_full() OK")
except Exception as ex:
    print(f"[ENV][WARN] torch_accel_shim 失败：{ex}")

import numpy as np
import torch

print(f"[ENV] torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
try:
    import transformers
    print(f"[ENV] transformers {transformers.__version__} <- {transformers.__file__}")
except Exception as ex:
    print(f"[ENV][FAIL] transformers 导入失败：{ex}")
    transformers = None

FAST = "--fast" in sys.argv
device = "cuda" if torch.cuda.is_available() else "cpu"

V4200 = ROOT / "docx" / "实验记录" / "versions" / "v4200_多模态CLIP产品版"

failures = []
warnings = []


def check(name, fn, fatal=False):
    t0 = time.time()
    try:
        r = fn()
        dt = time.time() - t0
        print(f"[PASS] {name}  ({dt:.2f}s)  {r if r is not None else ''}")
        return True
    except Exception as ex:
        dt = time.time() - t0
        print(f"[FAIL] {name}  ({dt:.2f}s)  {type(ex).__name__}: {ex}")
        traceback.print_exc()
        failures.append((name, f"{type(ex).__name__}: {ex}"))
        if fatal:
            print("\n[ABORT] 致命失败")
            sys.exit(1)
        return False


# ================================================================
# D. 根生态（独立阶段）—— sys.path 只挂根 src
# ================================================================
if PHASE == "root":
    print("\n" + "=" * 70)
    print("D. 根生态（方案 D, d_model=64）")
    print("=" * 70)
    sys.path.insert(0, str(ROOT / "src"))

    def _root_model_ckpt():
        from model.aether_gif_model import AetherGIFModel
        m = AetherGIFModel(d_model=64, ffn_hidden=256, n_blocks=12,
                           device="cpu", use_qwen=False)
        ck = ROOT / "outputs_real" / "overfit_ckpt.pt"
        if not ck.exists():
            raise FileNotFoundError(str(ck))
        m.load_checkpoint(str(ck))
        n = sum(p.numel() for p in m.parameters())
        return f"参数量={n:,} ckpt={ck.name}"

    check("根生态 模型+ckpt 加载", _root_model_ckpt)

    def _root_physics():
        from physics.block import PMNNCore
        from physics.kuramoto_sync import build_spacetime_index
        T, H, W = 6, 32, 32
        tc, sc = build_spacetime_index(T, H, W, "cpu")
        N = T * H * W
        a = torch.nn.functional.softplus(torch.randn(1, N, 64))
        th = torch.rand(1, N, 64) * 6.283 - 3.1415
        om = torch.randn(1, N, 64) * 0.5
        core = PMNNCore(64, 12)
        a2, th2 = core(a, th, om, tc, sc)
        (a2.mean() - th2.mean()).abs().backward()
        assert torch.isfinite(a2).all(), "a2 出现 NaN/Inf"
        return f"fwd+bwd OK, a_range=[{a2.min():.3f},{a2.max():.3f}]"

    if not FAST:
        check("根生态 物理核 fwd+bwd", _root_physics)

    def _root_generate():
        """推理入口真实跑通（含 save 路径）。写到独立 _probe_tmp 防覆盖已有产物。"""
        from model.aether_gif_model import AetherGIFModel
        m = AetherGIFModel(d_model=64, ffn_hidden=256, n_blocks=12,
                           device="cpu", use_qwen=False)
        m.load_checkpoint(str(ROOT / "outputs_real" / "overfit_ckpt.pt"))
        m.eval()
        outdir = ROOT / "outputs" / "_probe_tmp"
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / "probe_gen.gif"
        with torch.no_grad():
            m.generate(text="探针", duration=3, save_path=str(p))
        assert p.exists() and p.stat().st_size > 1000, "GIF 未生成或过小"
        return f"{p.relative_to(ROOT)} {p.stat().st_size}B"

    if not FAST:
        check("根生态 generate 出图", _root_generate)

    print("\n" + "-" * 70)
    print(f"[PHASE=root] 失败 {len(failures)} 项")
    for n, e in failures:
        print(f"   - {n}: {e}")
    sys.exit(1 if failures else 0)


# ================================================================
# E. v4200 主战场（独立阶段）—— sys.path 只挂 v4200/src
# ================================================================
if PHASE == "v4200":
    print("\n" + "=" * 70)
    print("E. v4200 主战场（23.04M, CondInitPC）")
    print("=" * 70)

    # ★ 与 train_guided.py 完全一致的环境默认（这些变量在本地模块导入时被读取）
    os.environ.setdefault("COND_INIT_PC", "1")
    os.environ.setdefault("COND_PC_RANK", "8")
    os.environ.setdefault("COND_GROUPS", "5")
    os.environ.setdefault("COND_LOWRANK", "16")
    os.environ.setdefault("PHASE_ACT", "arctan")
    os.environ.setdefault("RES_LIMIT", "0.5")
    os.environ.setdefault("W2_NORM_MAX", "3.0")
    os.environ.setdefault("KAPPA_MAX", "6.0")
    os.environ.setdefault("IDEAL_CLAMP", "3.141592653589793")
    os.environ.setdefault("FFT_FFTW", "1")

    sys.path.insert(0, str(V4200))
    sys.path.insert(0, str(V4200 / "src"))

    def _v4200_data():
        cc = V4200 / "data" / "clip_cache"
        need = ["text3.npy", "first3.npy", "gif3_video.npy",
                "cv3_full.npy", "cv3_text.npy", "cv3_first.npy",
                "cv3_gif.npy", "cv3_tf.npy"]
        miss = [n for n in need if not (cc / n).exists()]
        assert not miss, f"clip_cache 缺 {miss}"
        text3 = np.load(cc / "text3.npy")
        cv3 = np.load(cc / "cv3_full.npy")
        assert text3.shape[0] == 2, text3.shape
        assert cv3.ndim == 2 and cv3.shape[0] == 2, cv3.shape
        return f"8 个 npy 齐全, text3={text3.shape}, cv3_full={cv3.shape}"

    check("v4200 clip_cache 数据", _v4200_data)

    def _v4200_cond():
        from data.clip_condition import CLIP_EMB_DIM, N_COND3, build_cond_vector3
        z = np.zeros((1, CLIP_EMB_DIM), dtype=np.float32)
        v = build_cond_vector3(z, z, z, (1, 1, 0))
        assert v.shape == (1, N_COND3), v.shape
        return f"CLIP_EMB_DIM={CLIP_EMB_DIM} N_COND3={N_COND3}"

    check("v4200 条件向量构造", _v4200_cond)

    def _v4200_spec_loss():
        from train.spectral_loss import (band_profile, hf_ratio, lo_ratio,
                                         loss_hf_guard, loss_lo_floor,
                                         loss_band_abs, loss_spec_shape)
        a = torch.rand(1, 250, 125) * 0.3
        r = hf_ratio(a, win=5, w_fft=25)
        l = lo_ratio(a, win=5, w_fft=25)
        assert torch.isfinite(r).all() and torch.isfinite(l).all()
        return f"hf_ratio={float(r.mean()):.4f} lo_ratio={float(l.mean()):.4f}"

    check("v4200 频谱 loss 全家桶", _v4200_spec_loss)

    def _v4200_model():
        from model.aether_gif_model import AetherGIFModel
        m = AetherGIFModel(d_model=125, device="cpu", use_qwen=False, n_blocks=12,
                           ffn_hidden=256, init_mode="init", n_modules=8,
                           mod_gamma=0.15)
        n = sum(p.numel() for p in m.parameters())
        assert n == 23044534, f"参数量 {n:,} != 23044534（架构漂移！）"
        return f"参数量={n:,} 与记录一致"

    check("v4200 模型构造（参数量校验）", _v4200_model)

    def _v4200_ckpt():
        from model.aether_gif_model import AetherGIFModel
        m = AetherGIFModel(d_model=125, device="cpu", use_qwen=False, n_blocks=12,
                           ffn_hidden=256, init_mode="init", n_modules=8,
                           mod_gamma=0.15)
        ck = V4200 / "outputs_real" / "color2_long" / "guided_ckpt.pt"
        if not ck.exists():
            raise FileNotFoundError(str(ck))
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("qwen.")}
        ret = m.load_state_dict(sd, strict=False)
        assert len(ret.unexpected_keys) == 0, f"多余键 {ret.unexpected_keys[:3]}"
        assert len(ret.missing_keys) == 0, f"缺失键 {ret.missing_keys[:3]}"
        return f"{ck.parent.name}/{ck.name} missing=0 unexpected=0"

    check("v4200 ckpt 加载（color2 均衡点）", _v4200_ckpt)

    def _v4200_fwd_bwd():
        from model.aether_gif_model import AetherGIFModel
        from physics.kuramoto_sync import build_spacetime_index
        from codec.gif_codec import encode_gif_to_field
        D, GRID, PATCH, WIN, IMG, TS = 125, 32, 4, 5, 128, 6
        gp = V4200 / "data" / "gif_raw" / "noto" / "8-ball_1f3b1.gif"
        if not gp.exists():
            raise FileNotFoundError(str(gp))
        f = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D)
        t, s = build_spacetime_index(TS, GRID, GRID, device)
        target = f.unsqueeze(0).to(device)
        m = AetherGIFModel(d_model=D, device=device, use_qwen=False, n_blocks=12,
                           ffn_hidden=256, init_mode="init", n_modules=8,
                           mod_gamma=0.15).to(device)
        cv = torch.from_numpy(np.load(V4200 / "data" / "clip_cache" / "cv3_full.npy")[0:1]).to(device)
        a, th, diag = m(texts=None, target_field=target, t_coord=t, s_coord=s,
                        batch_size=1, cond_vec=cv, use_cond_field=True)
        loss = a.mean().abs() + th.mean().abs()
        loss.backward()
        assert torch.isfinite(a).all(), "a 含 NaN/Inf"
        vram = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else 0
        return f"a={tuple(a.shape)} finite, loss={float(loss):.4f}, peakVRAM={vram:.0f}MB"

    if not FAST:
        check("v4200 前向+反向 1 步（真实显存）", _v4200_fwd_bwd)

    def _v4200_bypass_module():
        import importlib
        for mod in ["physics.fft_bypass", "physics.cond_init_pc", "physics.cond_field"]:
            importlib.import_module(mod)
        return "fft_bypass / cond_init_pc / cond_field 均可导入"

    check("v4200 条件/旁路模块导入", _v4200_bypass_module)

    print("\n" + "-" * 70)
    print(f"[PHASE=v4200] 失败 {len(failures)} 项")
    for n, e in failures:
        print(f"   - {n}: {e}")
    sys.exit(1 if failures else 0)


# ================================================================
# A/B. 编解码闭环 —— 通用段（根 src 即可，v4200 codec 同源）
# ================================================================
print("\n" + "=" * 70)
print("A/B. 编解码闭环（src/codec/gif_codec）")
print("=" * 70)

sys.path.insert(0, str(ROOT / "src"))


def _codec_roundtrip():
    from codec.gif_codec import (encode_gif_to_field, decode_field_to_frames,
                                 load_gif_frames, resample_frames)
    D, GRID, PATCH, WIN, IMG, TS = 64, 32, 4, 5, 128, 6
    gp = ROOT / "data" / "gif_raw" / "noto" / "8-ball_1f3b1.gif"
    if not gp.exists():
        raise FileNotFoundError(str(gp))
    field = encode_gif_to_field(str(gp), TS, GRID, PATCH, WIN, IMG, D)
    rec = np.stack(decode_field_to_frames(field, TS, GRID, PATCH, WIN, IMG, D))
    raw = resample_frames(load_gif_frames(str(gp), (IMG, IMG)), TS * WIN)
    mse = ((raw.astype(np.float64) - rec.astype(np.float64)) ** 2).mean()
    psnr = 99.0 if mse < 1e-12 else 10 * np.log10(255.0 ** 2 / mse)
    assert field.shape == (TS * GRID * GRID, 2 * D), field.shape
    return f"field={tuple(field.shape)} PSNR={psnr:.2f}dB"


check("codec 往返（8-ball, D=64）", _codec_roundtrip)


def _codec_resize_path():
    """本次修复的 map_coordinates 坐标维度 bug —— 走真实 resize 路径。"""
    from codec.gif_codec import load_gif_frames, resize_gif_to_10fps
    gp = ROOT / "data" / "gif_raw" / "noto" / "8-ball_1f3b1.gif"
    fr = load_gif_frames(str(gp), (128, 128))
    out = resize_gif_to_10fps(fr)
    assert out.ndim == 4 and out.shape[-1] == 3, out.shape
    return f"in T={fr.shape[0]} -> out T={out.shape[0]} shape={out.shape}"


check("codec resize_gif_to_10fps（本次修复点）", _codec_resize_path)


# ================================================================
# C. 数据资产
# ================================================================
print("\n" + "=" * 70)
print("C. 数据资产完整性")
print("=" * 70)


def _index_json():
    idx_path = ROOT / "data" / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(str(idx_path))
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    entries = idx if isinstance(idx, list) else idx.get("files") or idx.get("items") or []
    total = len(entries)
    # ★ index.json 里的相对路径基准是 data/gif_raw（见 audit_project.py 同口径）
    gif_raw = ROOT / "data" / "gif_raw"
    step = max(1, total // 200)
    miss = []
    checked = 0
    for e in entries[::step][:200]:
        rel = e if isinstance(e, str) else (e.get("path") or e.get("file") or "")
        if not rel:
            continue
        checked += 1
        if not (gif_raw / rel).exists() and not (ROOT / "data" / rel).exists():
            miss.append(rel)
    assert not miss, f"抽样缺失 {len(miss)}/{checked} 条，例：{miss[:3]}"
    return f"条目={total} 抽样{checked}条全命中"


check("data/index.json 抽样（基准 data/gif_raw）", _index_json)


def _gif_decodable():
    from PIL import Image
    d = ROOT / "data" / "gif_raw" / "noto"
    gifs = sorted(d.glob("*.gif")) if d.exists() else []
    assert gifs, "noto 目录无 GIF"
    sample = gifs[:: max(1, len(gifs) // 40)][:40]
    bad = []
    for gp in sample:
        try:
            with Image.open(gp) as im:
                im.seek(0)
                im.convert("RGB")
                if getattr(im, "n_frames", 1) < 1:
                    bad.append((gp.name, "n_frames=0"))
        except Exception as ex:
            bad.append((gp.name, f"{type(ex).__name__}"))
    assert not bad, f"不可解码 {len(bad)} 个：{bad[:3]}"
    return f"抽样 {len(sample)}/{len(gifs)} 个全部可解码"


check("GIF 可解码抽样（noto）", _gif_decodable)


# ================================================================
# F. 静默失败模式静态扫描（在 main 进程做，无需特定生态）
# ================================================================
print("\n" + "=" * 70)
print("F. 训练脚本静默失败 / 数据安全静态扫描")
print("=" * 70)


def _scan_script(py_path: Path):
    txt = py_path.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    hits = []
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("#"):
            continue
        if "if _rp.exists():" in s:
            # RESUME 块最长约 40 行（含 RESUME_R_OLD 低秩扩展），窗口给足 60 行
            blk = "\n".join(lines[i:i + 60])
            if "else:" not in blk:
                hits.append((i, "RESUME 静默跳过（路径不存在时无任何提示，会悄悄从随机初始化训练）"))
        if s.startswith("torch.save(") and "OUT_DIR" in s:
            hits.append((i, "裸 torch.save 写 OUT_DIR（非原子：写入中途失败会留下损坏 ckpt）"))
    return hits


for name in ["train_guided.py", "overfit_cond3.py"]:
    p = V4200 / "scripts" / name
    if not p.exists():
        warnings.append(f"{name} 不存在")
        continue
    hits = _scan_script(p)
    if hits:
        print(f"[WARN] {name}:")
        for ln, desc in hits:
            print(f"         L{ln}: {desc}")
            warnings.append(f"{name} L{ln} {desc}")
    else:
        print(f"[PASS] {name}: 未发现静默失败模式")


# ================================================================
# 起子进程跑 D / E 两个隔离阶段
# ================================================================
print("\n" + "=" * 70)
print("D/E. 双生态隔离验证（子进程，避免包名缓存冲突）")
print("=" * 70)

py = sys.executable
for phase, label in [("root", "根生态"), ("v4200", "v4200 主战场")]:
    cmd = [py, str(HERE / "runtime_probe.py"), f"--phase={phase}"]
    if FAST:
        cmd.append("--fast")
    print(f"\n----- [subprocess] {label} -----")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    for line in r.stdout.splitlines():
        if line.startswith(("[PASS]", "[FAIL]", "[ENV]", "[INFO]", "[LOAD]")):
            print(line)
    if r.returncode != 0:
        failures.append((f"{label}（子进程）", f"退出码 {r.returncode}"))
        # 只打印失败附近的栈
        tail = "\n".join(r.stdout.splitlines()[-30:])
        print("[子进程输出尾部]")
        print(tail)
        if r.stderr.strip():
            print("[stderr]")
            print(r.stderr[-2000:])
    else:
        print(f"[PASS] {label} 子进程全部通过")


# ================================================================
# 汇总
# ================================================================
print("\n" + "=" * 70)
print("汇总")
print("=" * 70)
if failures:
    print(f"[FAIL] {len(failures)} 项运行时失败：")
    for n, e in failures:
        print(f"   - {n}: {e}")
else:
    print("[OK] 全部运行时检查通过")

if warnings:
    print(f"\n[WARN] {len(warnings)} 项需关注（静态扫描，非运行时失败）：")
    seen = set()
    for w in warnings:
        if w in seen:
            continue
        seen.add(w)
        print(f"   - {w}")

print("\n（本探针全程只读，未修改/删除任何已有文件）")
sys.exit(1 if failures else 0)
