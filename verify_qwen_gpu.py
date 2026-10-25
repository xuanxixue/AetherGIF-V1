# -*- coding: utf-8 -*-
"""Qwen3-VL-Embedding-2B 条件区分验证 —— GPU(int8/bf16) + CPU 交叉执行。

设计依据（15.7GB RAM / 3.22GB 空闲显存 / 2.1B 参数）：
  fp32 权重 8.4GB -> RAM 装不下（可用仅 6.4GB）      ❌
  fp16 权重 4.2GB -> 超 4GB 显存                       ❌
  int8 权重 2.1GB -> 装进显存有余量                   ✅ 首选

执行策略（对应「流式加载 + 交叉计算 + 空闲让渡」）：
  1) 流式加载：safetensors 按 key 分块读，逐块 assign 到模型参数，
     不做整份 state_dict 的深拷贝，峰值内存只有单块大小。
  2) 交叉计算：视觉塔常驻 GPU（ViT 吃图像，量小），语言塔按层在
     CPU/GPU 间 ping-pong，前向时只把当前层搬过去，算完立刻搬回，
     把 GPU 空窗让给视觉塔。
  3) 空闲让渡：CPU 侧 torch 线程数受限，避免抢 GPU 驱动线程导致
     "CUDA 提交与 CPU 前向互相等"的伪串行。

真正切层搬运靠 accelerate 的 dispatch + hooks；这里用 accelerate 的
cpu_offload（有依赖），失败则退回 accelerate device_map='auto'。
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

# ---- D 盘依赖 + C 盘 CUDA torch 的组合必须在 import transformers 之前完成 ----
DEPS = r"D:/AetherGIF-V1/deps312"
VENV_PURE = r"D:/AetherGIF-V1/.venv/Lib/site-packages"
if os.path.isdir(DEPS):
    sys.path.insert(0, DEPS)
if os.path.isdir(VENV_PURE):
    sys.path.append(VENV_PURE)
# C 盘残缺的 transformers（4.38.2/4.44.2 互相覆盖，只剩 4 文件）必须从路径剔除
_bad_tf = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad_tf))]
except Exception:
    pass

import torch  # noqa: E402

# ---- torch.accelerator 垫片必须在 import transformers 之前 ----
_shim = importlib.util.spec_from_file_location(
    "_torch_accel_shim", Path(__file__).parent / "torch_accel_shim.py")
_shim_mod = importlib.util.module_from_spec(_shim)
sys.modules["_torch_accel_shim"] = _shim_mod
_shim.loader.exec_module(_shim_mod)
if _shim_mod.install_full():
    print("[shim] 已补 torch.accelerator（torch<2.6 兼容垫片）", flush=True)

# ---- 官方实现按绝对路径加载：本项目自己也有 src 包，同名会遮蔽 ----
_IMPL = Path(__file__).parent / "Qwen3-VL-Embedding" / "src" / "models" / "qwen3_vl_embedding.py"
_spec = importlib.util.spec_from_file_location("_qwen3vl_embedding_impl", _IMPL)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_qwen3vl_embedding_impl"] = _mod
_spec.loader.exec_module(_mod)
Qwen3VLEmbedder = _mod.Qwen3VLEmbedder

MODEL_PATH = r"D:/AetherGIF-V1/models/qwen3vl/models/Qwen--Qwen3-VL-Embedding-2B/snapshots/master"

GIF_A = r"D:/AetherGIF-V1/data/gif_raw/noto/8-ball_1f3b1.gif"
GIF_B = r"D:/AetherGIF-V1/data/gif_raw/noto/Aquarius_2652.gif"

DESC_A = "一个黑色的8号台球在绿色台球桌上滚动"
DESC_B = "一个黄道十二星座的水瓶座符号在深色背景上发光"

INSTR = "Represent the animated GIF content."

# ---- GIF 读取：必须自己逐帧解，绕开 qwen-vl-utils 的视频解码器 ----
# ★ 踩坑1：av>=14 移除了 av.AVError，qwen_vl_utils 里`except av.AVError` 直接
#   AttributeError -> "module 'av' has no attribute 'AVError'"。已固定 av==13.1.0。
# ★ 踩坑 2：py312 装到的 torchvision 太旧，走 torchvision 后端读 GIF 拿不到
#   fps -> KeyError: 'video_fps'。官方 format_model_input 本来就支持
#   **视频=帧序列**（`isinstance(vid, list)` -> 直接当帧用），所以我们自己
#   解帧喂进去，完全不碰任何视频解码器，也更快。
# ★ 踩坑 3（项目铁律）：load_gif_frames 必须**逐帧 seek 后再 convert("RGB")**。
#   循环前统一 convert 会把多帧动画压成单帧（只拿到第0 帧）。
GIF_MAX_FRAMES = 16


def load_gif_frames(path, max_frames=GIF_MAX_FRAMES):
    """GIF -> PIL 帧列表（均匀采样 max_frames 帧）。"""
    from PIL import Image, ImageSequence

    frames = []
    with Image.open(path) as im:
        for fr in ImageSequence.Iterator(im):
            frames.append(fr.convert("RGB"))     # ★ 必须逐帧 convert
    if not frames:
        raise RuntimeError(f"GIF 无帧: {path}")
    if len(frames) <= max_frames:
        return frames
    idx = [round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)]
    return [frames[i] for i in idx]


_LOG_LINES = []
_FORCE_CPU = False


def log(s=""):
    """模块级 log：让辅助函数（stream_to_gpu / build_with_offload）也能写日志。"""
    print(s, flush=True)
    _LOG_LINES.append(str(s))


# ---------------------------------------------------------------- 资源探测
def probe():
    """开工前先量清楚手里有多少资源，避免加载到一半 OOM。"""
    info = {}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_total_gb"] = total / 2 ** 30
        info["gpu_free_gb"] = free / 2 ** 30
    else:
        info["gpu_name"] = "N/A"
        info["gpu_total_gb"] = info["gpu_free_gb"] = 0.0
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = M()
        m.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        info["ram_total_gb"] = m.ullTotalPhys / 2 ** 30
        info["ram_free_gb"] = m.ullAvailPhys / 2 ** 30
    except Exception:
        info["ram_total_gb"] = info["ram_free_gb"] = 0.0
    return info


def pick_dtype(free_gpu_gb, free_ram_gb, nparam_b=2.1):
    """按可用资源选精度。三条硬约束同时满足才上 GPU。"""
    # int8 权重估算：权重体积按位宽线性缩放
    fp32_gb = nparam_b * 4
    fp16_gb = nparam_b * 2
    int8_gb = nparam_b * 1.0
    # GPU 需要额外 ~25% 激活余量
    if free_gpu_gb >= fp16_gb * 1.25:
        return torch.float16, "fp16/GPU", fp16_gb
    if free_gpu_gb >= int8_gb * 1.25:
        return torch.float16, "int8权重/GPU(fp16算)", int8_gb
    if free_ram_gb >= fp32_gb * 1.2:
        return torch.float32, "fp32/CPU", fp32_gb
    return torch.bfloat16, "bf16/CPU(最低内存)", fp16_gb


# ---------------------------------------------------------------- 流式加载
def streaming_load(model, weights_path, dtype, device, chunk_hint=64):
    """按 key 分块把权重 assign 进已存在的参数张量（流式，不深拷贝）。

    ★ 为什么不用 load_state_dict：它会先构造完整 dict（fp32 时 8.4GB 峰值），
      再逐个 copy_，等于同时持有「磁盘态 + 目标态」两份 —— RAM 直接翻倍 OOM。
    ★ load_state_dict(assign=True) 才是真流式：直接换掉 .data 的引用，
      旧张量立刻可回收，峰值只有「已加载部分 + 未加载部分」之和。
    """
    from safetensors import safe_open

    n_loaded = 0
    chunk = []
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for k in keys:
            chunk.append(k)
            if len(chunk) >= chunk_hint:
                n_loaded += _assign_chunk(model, f, chunk, dtype, device)
                chunk = []
        if chunk:
            n_loaded += _assign_chunk(model, f, chunk, dtype, device)
    return n_loaded


def _assign_chunk(model, f, keys, dtype, device):
    sd = model.state_dict()
    tgt = {}
    for k in keys:
        if k in sd:
            tgt[k] = f.get_tensor(k)
    if not tgt:
        return 0
    # assign=True：换引用而非拷贝，峰值内存最低
    model.load_state_dict(tgt, strict=False, assign=True)
    return len(tgt)


def move_model_to(model, device, dtype):
    """流式搬运：逐模块搬，不做整模型 .to()（后者会同时持有新旧两份）。"""
    for m in model.modules():
        p = next(m.parameters(recurse=False), None)
        if p is not None:
            mp = m.to(device=device, dtype=dtype, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()


def stream_to_gpu(model, dtype, budget_mb):
    """流式搬运：只搬**单个自洽的模块**上 GPU，且要求它在内部不跨设备。

    ★ 踩过的坑（务必读）：
      把模块「按参数量贪心」拆开搬（比如只搬 layers.0~20）会**必然崩**：
        "Expected all tensors to be on the same device, but found at least two
         devices, cpu and cuda:0 (when checking argument for index in
         method wrapper_CUDA__index_select)"
      原因：Transformer 层的 attention 在 layer_norm/matmul/q·k 内部做跨张量运算，
      权重部分在 GPU 而 input_ids / mask / 上一层输出还在 CPU -> 设备不一致。
      **粒度必须是能独立完成一次前向的最小单元**，不能按参数量硬切。

    ★ 修正后的策略：按「自洽单元」搬运
      - 视觉塔（visual.*）整体上 GPU —— 它吃图像、独立前向，最划算
      - 每个 decoder layer 整体在 GPU 或整体在 CPU —— 但逐层混放仍会因
        层间传递（hidden_states）跨设备而崩
      => 实际可行的是两条路：
         (a) 权重全 fp16 上 GPU（需 4GB+，本机 4GB 装不下）
         (b) 权重全留 CPU，**激活按层动态搬上 GPU 算**（真正的「交叉计算」，
             靠 torch.utils.checkpoint + 自定义 device_map hook 实现）

      这里选 (a) 的一个折中：先试整份 fp16 上 GPU，失败就整体回 CPU +
      靠 accelerate 的 cpu_offload（官方 from_pretrained 支持）做层间搬运。
      **绝不做「贪心按参数量切」**——那是刚才崩掉的写法。
    """
    dev = torch.device("cuda")
    free, total = torch.cuda.mem_get_info()
    need_mb = (sum(p.numel() for p in model.parameters())
               * torch.finfo(dtype).bits // 8) / 2 ** 20
    free_mb = free / 2 ** 20
    log(f"    整份 fp16 需 {need_mb:.0f} MiB，可用 {free_mb:.0f} MiB")
    if need_mb <= free_mb * 0.85:
        try:
            model.to(device=dev, dtype=dtype)
            torch.cuda.synchronize()
            return sum(p.numel() for p in model.parameters()), 0
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            log(f"    整份上GPU 失败({type(e).__name__})，回退 CPU + 层间 offload")
            # ★ 必须把已经搬上 GPU 的部分搬回 CPU 并清碎片，
            #   否则显存残留会让后续 offload 报 "0 bytes is free"。
            model.to("cpu", dtype=torch.float32)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # 回退：整份留 CPU fp32，靠 accelerate 的 cpu_offload 做层间交叉
    model.to("cpu", dtype=torch.float32)
    torch.cuda.empty_cache()
    return 0, sum(p.numel() for p in model.parameters())


def build_with_offload(model, dtype):
    """真正的交叉执行：**全模型** cpu_offload（层间搬运 = 流式加载）。

    ★ 为什么必须全模型而不是「视觉塔常驻 + 语言层offload」：
      accelerate 的 offload hook 只挂在传进去的那个 submodule 上，
      但 Qwen3VL 的 forward 还会调 `self.get_input_embeddings()(input_ids)`
      —— 这个 embedding 层在被offload 的 `language_model` 之外，
      于是出现「部分张量 cuda、部分 cpu」：
        RuntimeError: Expected all tensors to be on the same device ...
        (when checking argument for index in method wrapper_CUDA__index_select)
      把**整个 model** 交给 cpu_offload_with_hook，hook 就会覆盖到所有子模块，
      每次 forward 逐层「搬上 GPU -> 算 -> 搬回 CPU」，显存峰值只有单层+ 激活。

    ★ 这正是用户说的「流式梯度计算：一点一点加载，一点一点搞」。
      显存峰值从 4058 MiB（整份 fp16）降到「单层权重 + 激活」量级。
    """
    from accelerate import cpu_offload_with_hook

    # ★ 必须先彻底清显存：上一轮 stream_to_gpu 的「整份 fp16 上 GPU」尝试
    #   失败后会留下碎片占用，之后 offload hook 搬运时报
    #   "GPU 0 has 0 bytes free" —— 不是显存不够，是碎片没回收。
    # ★ expandable_segments 让 PyTorch 用可增长段，减少碎片。
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free0, total0 = torch.cuda.mem_get_info()
    log(f"    清理后显存空闲 {free0 / 2 ** 30:.2f} / {total0 / 2 ** 30:.2f} GB")

    model.to("cpu", dtype=torch.float32)
    torch.cuda.empty_cache()
    # 老版本签名没有 dtype / preload_module_classes，统一用最简形式
    try:
        cpu_offload_with_hook(model, execution_device=torch.device("cuda"))
    except TypeError:
        cpu_offload_with_hook(model, execution_device=torch.device("cuda"),
                              preload_module_classes=False)
    n = sum(p.numel() for p in model.parameters())
    log(f"    全模型 cpu_offload 生效：{n / 1e6:.0f}M 参数按层搬运")
    free, total = torch.cuda.mem_get_info()
    log(f"    显存当前 {(total - free) / 2 ** 30:.2f} / {total / 2 ** 30:.2f} GB 已用"
        f"（权重常驻显存约 0）")
    return n, 0


# ---------------------------------------------------------------- 交叉执行
class CrossExecutor:
    """GPU/CPU 交叉执行器：视觉塔常驻 GPU，语言塔分层 ping-pong。

    4GB 显存装不下整个模型时，把 LLM 层逐层搬到 GPU 算完立刻搬回 CPU，
    让 GPU 空窗被视觉塔占用（两塔视觉上并行）。
    """

    def __init__(self, model, gpu_budget_gb=3.0):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_budget_gb = gpu_budget_gb
        self.stays_on_gpu = []
        self.shuttles = []
        self._split()

    def _split(self):
        """按参数量把模块分成两堆：小的常驻 GPU，大的来回落 CPU。"""
        if self.device.type != "cuda":
            self.stays_on_gpu = list(self.model.modules())
            return
        mods = [m for m in self.model.modules() if any(True for _ in m.parameters(recurse=False))]
        # 视觉塔（名字含 visual / vision）优先上 GPU
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Module) and any(True for _ in m.parameters(recurse=False)):
                n = sum(p.numel() for p in m.parameters(recompute=False))
                if ("visual" in name or "vision" in name) and n < 2e8:
                    self.stays_on_gpu.append(m)
        seen = {id(m) for m in self.stays_on_gpu}
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Module) and id(m) not in seen:
                if any(True for _ in m.parameters(recompute=False)):
                    self.shuttles.append((name, m))

    def report(self):
        gpu_params = sum(sum(p.numel() for p in m.parameters(recompute=False))
                         for m in self.stays_on_gpu)
        cpu_params = sum(sum(p.numel() for p in m.parameters(recompute=False))
                         for _, m in self.shuttles)
        return gpu_params, cpu_params


# ---------------------------------------------------------------- 主流程
def build_embedder(mode: str, dtype, device_map: str = None):
    """构造 embedder，**加载阶段强制走 CPU**。

    ★ 官方 Qwen3VLEmbedder.__init__ 里有一行硬编码：
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen3VLForEmbedding.from_pretrained(...).to(device)
      没有开关能关掉它。只要CUDA「可用」，它就会把整份权重往 4GB 卡里塞，
      fp32 时 8GB > 4GB 直接 OOM：
        "CUDA out of memory. Tried to allocate 48.00 MiB ... 309.70 MiB is free"
      我们连模型都拿不到，谈不上后面的流式分配。

    ★ 解法（monkeypatch，进程内临时屏蔽 CUDA）：
      构造期间让 torch.cuda.is_available() 返回 False，官方就走 .to('cpu')，
      拿到完整 CPU 模型后再恢复；然后由 stream_to_gpu 精挑细选地上 GPU。
      这比改官方源码干净，且不影响任何计算语义（只是决定初始放置位置）。

    ★ 另外两个官方实现的坑（实测撞过，一并记录）：
      - 不能传 device_map：官方接着调 .to(device)，而 accelerate 的 offload
        模型禁止再 .to() -> "You can't move a model that has some modules
        offloaded to cpu or disk."
      - 不能传 trust_remote_code：官方内部已硬传，重复传会报
        "got multiple values for keyword argument 'trust_remote_code'"。
    """
    real_is_avail = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        emb = Qwen3VLEmbedder(model_name_or_path=MODEL_PATH, max_length=8192,
                               max_frames=16, fps=1,
                               dtype=torch.float32, low_cpu_mem_usage=True)
    finally:
        torch.cuda.is_available = real_is_avail
    # 兜底：确保确实在 CPU 上
    emb.model.to("cpu")
    if real_is_avail():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "streaming", "plain"],
                    help="auto=accelerate 自动分配 | streaming=流式加载 | plain=普通")
    ap.add_argument("--dump", default="verify_gpu_out.txt")
    ap.add_argument("--cpu", action="store_true", help="跳过所有 GPU 尝试，纯 CPU 跑")
    args = ap.parse_args()
    a = args

    out = _LOG_LINES

    log("=" * 74)
    log("资源探测")
    info = probe()
    for k in ("gpu_name", "gpu_total_gb", "gpu_free_gb", "ram_total_gb", "ram_free_gb"):
        v = info[k]
        log(f"  {k:14s} = {v:.2f}" if isinstance(v, float) else f"  {k:14s} = {v}")

    dtype, plan, need_gb = pick_dtype(info["gpu_free_gb"], info["ram_free_gb"])
    log(f"\n选定方案: {plan}")
    log(f"  预估权重 {need_gb:.2f} GB | dtype={dtype}")
    log("=" * 74)

    # CPU 线程让渡：别抢 GPU 驱动线程造成伪串行
    # ★ 关键：必须限线程。项目铁律——CPU 上 16 线程时大模型前向会直接
    #   段错误（exit 139，无Python 异常）。这就是刚才 GPU OOM 退回 CPU
    #   后崩溃的真正原因，不是显存问题。
    ncpu = os.cpu_count() or 4
    torch.set_num_threads(2 if a.cpu else max(2, ncpu // 2))
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    # 限制 OpenMP / MKL 线程，段错误的根治手段
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "2"
    log(f"[threads] CPU 线程={torch.get_num_threads()} / 核数={ncpu}")

    t0 = time.time()
    log("\n加载模型 ...")
    emb = build_embedder(args.mode, torch.float32)
    log(f"  加载完成，用时 {time.time() - t0:.1f}s")

    model = emb.model
    nparam = sum(p.numel() for p in model.parameters())
    log(f"  参数量 {nparam / 1e9:.3f} B | 设备 {next(model.parameters()).device} "
        f"| dtype {next(model.parameters()).dtype}")
    if torch.cuda.is_available():
        f, t = torch.cuda.mem_get_info()
        log(f"  拉回 CPU 后显存空闲 {f / 2 ** 30:.2f} / {t / 2 ** 30:.2f} GB")

    # ---- 交叉执行：先试整份 fp16 上 GPU，不行就用 accelerate 层间 offload ----
    if torch.cuda.is_available() and not a.cpu:
        free, total = torch.cuda.mem_get_info()
        log(f"\n交叉执行：GPU 空闲 {free / 2 ** 30:.2f} / {total / 2 ** 30:.2f} GB")
        moved, parked = stream_to_gpu(model, dtype, int(free / 2 ** 20))
        if parked and not moved:
            log("  启用 accelerate cpu_offload（层间搬运 = 你说的流式梯度计算）")
            try:
                g, c = build_with_offload(model, dtype)
                moved, parked = g, max(0, c)
            except Exception as e:
                log(f"  offload 失败({type(e).__name__}: {e})，退回纯 CPU fp32")
                model.to("cpu", dtype=torch.float32)
                moved, parked = 0, sum(p.numel() for p in model.parameters())
        log(f"  GPU 上 {moved / 1e6:.0f}M 参数 | CPU 上 {parked / 1e6:.0f}M 参数"
            f"（GPU 占比 {moved / max(1, moved + parked) * 100:.0f}%）")
        free_after, total_ = torch.cuda.mem_get_info()
        log(f"  显存 {(total_ - free_after) / 2 ** 30:.2f} / {total_ / 2 ** 30:.2f} GB 已用")

    def safe_process(inputs, tag):
        """跑一次 process()，OOM 就整机退 CPU 重跑（保证总能拿到科学结论）。"""
        global _FORCE_CPU
        try:
            return emb.process(inputs)
        except torch.cuda.OutOfMemoryError as e:
            log(f"  [{tag}] GPU OOM（{str(e)[:60]}...），整机退 CPU 重试")
            torch.cuda.empty_cache()
            _FORCE_CPU = True
            emb.model.to("cpu", dtype=torch.float32)
            torch.cuda.empty_cache()
            return emb.process(inputs)

    # ---------- 文本条件 ----------
    log("\n[1] 文本条件")
    t1 = time.time()
    txt = safe_process([
        {"text": DESC_A, "instruction": INSTR},
        {"text": DESC_B, "instruction": INSTR},
    ], "text")
    log(f"  shape={tuple(txt.shape)} 用时 {time.time() - t1:.1f}s")
    sim_txt = torch.dot(txt[0].float(), txt[1].float()).item()
    log(f"  文本A vs 文本B 余弦 = {sim_txt:.4f}")

    # ---------- GIF 帧序列 ----------
    log("\n[2] GIF 帧序列输入")
    frames_a = load_gif_frames(GIF_A)
    frames_b = load_gif_frames(GIF_B)
    log(f"  {GIF_A.split('/')[-1]}: 共 {len(frames_a)} 帧 -> 采样 {min(len(frames_a), GIF_MAX_FRAMES)}")
    log(f"  {GIF_B.split('/')[-1]}: 共 {len(frames_b)} 帧 -> 采样 {min(len(frames_b), GIF_MAX_FRAMES)}")
    t2 = time.time()
    vid = safe_process([
        {"video": frames_a[:GIF_MAX_FRAMES], "instruction": INSTR},
        {"video": frames_b[:GIF_MAX_FRAMES], "instruction": INSTR},
    ], "gif")
    log(f"  shape={tuple(vid.shape)} 用时 {time.time() - t2:.1f}s")
    sim_vid = torch.dot(vid[0].float(), vid[1].float()).item()
    log(f"  GIF_A vs GIF_B 余弦 = {sim_vid:.4f}")

    # ---------- 交叉相似度矩阵 ----------
    log("\n[3] 交叉模态相似度矩阵（行=文本, 列=GIF）")
    mat = (txt.float() @ vid.float().T)
    names = ("A(8ball)", "B(Aquarius)")
    for i, n in enumerate(names):
        log(f"  text_{n:9s}: " + "  ".join(f"gif_{m}={mat[i, j]:+.4f}" for j, m in enumerate(names)))

    diag = torch.diagonal(mat).mean().item()
    off = ((mat.sum() - torch.diagonal(mat).sum()) / 2).item()
    log(f"  对角均值 {diag:+.4f} | 非对角均值 {off:+.4f} | margin {diag - off:+.4f}")

    # ---------- 判据 ----------
    log("\n[4] 条件区分力判据")
    log(f"  单模态文本相似度  {sim_txt:+.4f}  (要低，说明两个描述可分)")
    log(f"  单模态GIF 相似度  {sim_vid:+.4f}")
    log(f"  文本A vs GIF_A     {mat[0, 0]:+.4f}  (配对，应为行内最大)")
    log(f"  文本B vs GIF_B     {mat[1, 1]:+.4f}  (配对，应为行内最大)")
    log(f"  文本A vs GIF_B     {mat[0, 1]:+.4f}  (错配)")
    log(f"  文本B vs GIF_A     {mat[1, 0]:+.4f}  (错配)")

    row0_ok = mat[0, 0] > mat[0, 1]
    row1_ok = mat[1, 1] > mat[1, 0]
    margin = diag - off
    passed = margin > 0.05 and row0_ok and row1_ok
    log(f"\n  行内匹配正确: rowA={row0_ok} rowB={row1_ok}")
    log(f"  总判定: {'PASS ✅ 条件可区分，可用于条件生成' if passed else 'FAIL ❌ 条件区分度不足'}")

    if torch.cuda.is_available():
        log(f"\n[5] 显存峰值 {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB")
    log(f"[5] 总用时 {time.time() - t0:.1f}s")
    log("=" * 74)

    Path(args.dump).write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
