# -*- coding: utf-8 -*-
"""
Int8QwenEncoder —— Qwen3-VL-Embedding-2B 的 int8 GPU 编码器（正式模块）

★ 用途
    离线预编码：把文本 / GIF 帧序列批量编码成 2048 维 L2 归一化条件向量。
    相比 CPU fp32（文本 4.6s/条、16 帧 GIF 471.7s/条），int8 GPU 快到
    **文本 1.4s/条、8 帧 GIF 1.8s/条**（实测），全量 878 个 GIF 预编码从
    约 48 小时压到分钟级。

★ 显存
    int8 峰值 2444 MiB（vs fp16 的 4058 MiB），4GB 卡可装下，剩余显存交给
    物理核的完整配置（12 block / grid32 / TS6）。

★ 运行环境（唯一能跑的组合，缺一不可）
    - 解释器：C:\\Python312\\python.exe（唯一有 CUDA，torch 2.5.1+cu121）
    - transformers 5.19：D:\\AetherGIF-V1\\deps312
    - bitsandbytes 0.50.2：D:\\AetherGIF-V1\\int8_deps
    - qwen_vl_utils：D:\\AetherGIF-V1\\int8_deps\\qwen_vl_utils（纯 Python 副本）
    - 必须在 import transformers 之前内联 patch torch.accelerator（见下）

★ 三个硬坑（复现必读，全部实测撞过）
    1. 官方 Qwen3VLEmbedder.__init__ 硬编码 from_pretrained(**kwargs).to(device)，
       int8 模型（device_map='auto'）禁止再 .to() → 绕开该类，直接用
       Qwen3VLForEmbedding.from_pretrained + BitsAndBytesConfig(load_in_8bit=True)。
    2. int8 模型无单一 .device，官方 process() 里 {k:v.to(self.model.device)} 会炸
       → 本模块自己实现 process()，固定 cuda:0。
    3. 不能把 .venv/site-packages 整目录加进 sys.path —— 里面的 cpu torch 会
       遮蔽 cu121 torch。qwen_vl_utils 单独复制到 int8_deps 即可。

★ 第四个坑（2026-10-16 定位，决定性 bug）
    官方 video 分支两条路：list[PIL] 只受 total_pixels 约束（无时序重采样），
    str 路径才走真视频管线（真 fps + smart_nframes 时序重采样）。但本机
    torchvision 0.20.1 + 新版 PyAV 不兼容：read_video 的 info 丢失 video_fps
    -> fetch_video 双连环 KeyError -> 官方 _preprocess_inputs 的 try/except
    把整个输入兜底成 "NULL" 空文本 -> 两个完全不同的 GIF 产出**完全相同**的
    嵌入（余弦 +1.0030），表现就是「GIF 向量毫无区分度」。
    修复 = 注入 PyAV 后端（install_pyav_video_backend），GIF 路径直传走
    encode_video_files()。env 变量 FORCE_QWENVL_VIDEO_READER 是模块加载时的
    全局快照 + lru_cache，事后设无效，必须直接替换函数。

用法：
    from int8_qwen_encoder import Int8QwenEncoder
    enc = Int8QwenEncoder()
    te = enc.encode_texts(["一个黑色的8号台球在绿色台球桌上滚动"])     # [N,2048]
    ve = enc.encode_gif_paths([r"xxx.gif"], max_frames=8)             # [N,2048] 帧列表分支
    vv = enc.encode_video_files([r"xxx.gif"])                         # [N,2048] 真视频分支
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

INT8_DEPS = r"D:\AetherGIF-V1\int8_deps"
DEPS312 = r"D:\AetherGIF-V1\deps312"
for _p in (DEPS312, INT8_DEPS):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch

# ★ torch<2.6 无 torch.accelerator，transformers 5.x 硬依赖，必须先 patch
if not hasattr(torch, "accelerator"):
    class _AcceleratorShim:
        @staticmethod
        def is_available():
            return torch.cuda.is_available()
        @staticmethod
        def device_count():
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        @staticmethod
        def current_accelerator():
            return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None
        @staticmethod
        def current_device_index():
            return torch.cuda.current_device() if torch.cuda.is_available() else -1
        @staticmethod
        def is_bf16_supported(device=None):
            if not torch.cuda.is_available():
                return False
            major, _ = torch.cuda.get_device_capability(device)
            return major >= 8
    torch.accelerator = _AcceleratorShim

import importlib.util
import numpy as np
from PIL import Image, ImageSequence

MODEL_DIR = r"D:\AetherGIF-V1\models\qwen3vl\models\Qwen--Qwen3-VL-Embedding-2B\snapshots\master"
_IMPL_PATH = r"D:/AetherGIF-V1/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py"

INSTR_TEXT = "Represent the user's input."
INSTR_IMAGE = "Represent the still image content."
INSTR_VIDEO = "Represent the animated GIF content."


def _load_official_impl():
    spec = importlib.util.spec_from_file_location("_qwen3vl_emb", _IMPL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_qwen3vl_emb"] = mod
    spec.loader.exec_module(mod)
    return mod


def install_pyav_video_backend():
    """把 PyAV 视频解码后端注入 qwen_vl_utils，绕开本机坏掉的 torchvision。

    ★ 为什么必须注入（2026-10-16 定位的决定性 bug）
      torchvision 0.20.1 + 新版 PyAV 不兼容：read_video 返回的 info 丢失
      video_fps -> qwen_vl_utils.fetch_video 双连环 KeyError -> 官方
      _preprocess_inputs 的 try/except 把整个输入兜底成 "NULL" 空文本对话
      -> 两个完全不同的 GIF 产出**完全相同**的嵌入（余弦 +1.0030）。
      表现就是「GIF 向量没有区分度」，不专门断言根本看不出来。

    ★ 为什么不能设环境变量
      FORCE_QWENVL_VIDEO_READER 是 vision_process 模块加载时的全局变量
      快照，且 get_video_reader_backend 套了 lru_cache，事后 os.environ
      赋值无效（实测 [DBG] 仍输出 torchvision）。直接替换模块属性最彻底。

    ★ PyAV 路径
      av.open(GIF) 拿 stream.average_rate（真 fps）+ container.decode(video=0)
      解全部帧，再交回官方 smart_nframes 按 fps=1.0 做时序重采样，与官方
      video=str 分支语义完全一致。GIF 容器在 PyAV 里是能解出视频流的。

    幂等：重复调用只覆盖同名后端，无副作用。
    """
    import time
    import av
    from qwen_vl_utils import vision_process as vp
    from qwen_vl_utils.vision_process import smart_nframes

    def _read_video_pyav(ele):
        video_path = ele["video"]
        if video_path.startswith("file://"):
            video_path = video_path[7:]
        st = time.time()
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps_num = stream.average_rate.numerator
            fps_den = stream.average_rate.denominator
            video_fps = float(fps_num) / max(fps_den, 1)
            frames = [f.to_ndarray(format="rgb24")
                      for f in container.decode(video=0)]
        if len(frames) == 0:
            raise ValueError(f"PyAV 解码出 0 帧: {video_path}")
        arr = np.stack(frames)                                  # [T,H,W,3] uint8
        total_frames = arr.shape[0]
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps
        video = torch.from_numpy(arr[idx]).permute(0, 3, 1, 2)  # [T,3,H,W] uint8
        vp.logger.info(
            f"pyav: {video_path=}, {total_frames=}, fps={video_fps:.2f}, "
            f"sample={nframes}, time={time.time() - st:.3f}s")
        return video, dict(fps=video_fps, frames_indices=idx,
                           total_num_frames=total_frames,
                           video_backend="pyav"), sample_fps

    vp.VIDEO_READER_BACKENDS["pyav"] = _read_video_pyav
    vp.get_video_reader_backend = lambda: "pyav"
    print("[int8] PyAV 视频后端已注入（绕开坏掉的 torchvision read_video）")


class Int8QwenEncoder:
    """int8 GPU 编码器。懒加载，构造时不碰模型，首次 encode 时才加载。"""

    def __init__(self, model_dir: str = MODEL_DIR, max_frames: int = 8):
        self.model_dir = model_dir
        self.max_frames = max_frames
        self._emb = None
        self._impl = None
        self.device = None

    def _ensure(self):
        if self._emb is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("int8 编码器必须有 CUDA（用 C:\\Python312\\python.exe 运行）")

        # ★ 先注入 PyAV 视频后端，再加载官方实现。注入只改 qwen_vl_utils 的
        #   模块属性（后端 dict + get_video_reader_backend 函数），官方实现
        #   import 时复用同一份模块缓存，先后顺序都生效，放前面语义最清晰。
        install_pyav_video_backend()

        impl = _load_official_impl()
        self._impl = impl
        Qwen3VLForEmbedding = impl.Qwen3VLForEmbedding
        Qwen3VLEmbedder = impl.Qwen3VLEmbedder

        from transformers import BitsAndBytesConfig
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        print("[int8] 加载 Qwen3VLForEmbedding (int8)...")
        model = Qwen3VLForEmbedding.from_pretrained(
            self.model_dir, trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto", low_cpu_mem_usage=True,
        )
        model.eval()
        processor = Qwen3VLProcessor.from_pretrained(self.model_dir, padding_side='right')

        # 绕过官方 __init__ 的 .to(device)，直接 object.__new__ + 手工填字段
        emb = object.__new__(Qwen3VLEmbedder)
        emb.model = model
        emb.processor = processor
        emb.max_length = 8192
        emb.min_pixels = impl.MIN_PIXELS
        emb.max_pixels = impl.MAX_PIXELS
        emb.total_pixels = impl.MAX_TOTAL_PIXELS
        emb.fps = 1.0
        emb.max_frames = self.max_frames
        emb.default_instruction = INSTR_TEXT
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._emb = emb
        print(f"[int8] 加载完成, 显存 alloc={torch.cuda.memory_allocated()/1024**2:.0f}MiB")

    def _process(self, items, normalize=True):
        """自实现 process：固定 cuda:0，绕开 int8 模型无单一 .device 的坑。"""
        emb = self._emb
        convs = [emb.format_model_input(
            text=e.get('text'), image=e.get('image'),
            video=e.get('video'), instruction=e.get('instruction'),
            fps=e.get('fps'), max_frames=e.get('max_frames')) for e in items]
        pi = emb._preprocess_inputs(convs)
        pi = {k: v.to(self.device) for k, v in pi.items()}
        out = emb.forward(pi)
        hidden = out['last_hidden_state']
        am = out['attention_mask']
        last = am.shape[1] - am.flip(dims=[1]).argmax(dim=1) - 1
        row = torch.arange(hidden.shape[0], device=hidden.device)
        e = hidden[row, last]
        if normalize:
            e = torch.nn.functional.normalize(e, dim=-1)
        return e

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=4):
        self._ensure()
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            outs.append(self._process(
                [{"text": b, "instruction": INSTR_TEXT} for b in batch]).float().cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)

    @torch.no_grad()
    def encode_gif_paths(self, gif_paths, batch_size=1):
        """GIF 路径列表 → [N,2048]。逐帧 seek 后 convert('RGB')（项目铁律）。"""
        self._ensure()
        outs = []
        for i in range(0, len(gif_paths), batch_size):
            batch = gif_paths[i:i + batch_size]
            frame_lists = [self._load_frames(p) for p in batch]
            outs.append(self._process(
                [{"video": fl, "instruction": INSTR_VIDEO} for fl in frame_lists]).float().cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)

    @torch.no_grad()
    def encode_video_files(self, video_paths, batch_size=1, fps=1.0):
        """视频/GIF **文件路径** → [N,2048]。走 format_model_input 的 str 视频
        分支（真 fps + smart_nframes 时序重采样），是真正的「视频向量」。

        ★ 与 encode_gif_paths 的区别（两者的语义不可混用）
          - encode_gif_paths：PIL 解帧成 list[PIL] -> 只受 total_pixels 约束，
            **没有时序重采样**，等价于「一堆图」，丢动速信息。
          - encode_video_files：路径直传 -> PyAV 解出真 fps 与全部帧 ->
            smart_nframes(fps=1.0) 按时间轴均匀采样 max_frames 帧，
            「一段动图」的语义。

        ★ PyAV 后端由 _ensure() 自动注入，调用方无需关心。若注入失败
          （torchvision 兜底链路仍生效），结果会静默退化成 NULL 空文本
          向量（所有视频余弦 ≈ +1.0），务必核对两素材余弦。
        """
        self._ensure()
        outs = []
        for i in range(0, len(video_paths), batch_size):
            batch = video_paths[i:i + batch_size]
            outs.append(self._process(
                [{"video": p, "instruction": INSTR_VIDEO,
                  "fps": fps, "max_frames": self.max_frames} for p in batch])
                .float().cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)

    @torch.no_grad()
    def encode_images(self, images, batch_size=1):
        """静态单图列表 → [N,2048]。走 format_model_input 的 image 分支（带
        min/max_pixels），与 GIF 帧序列的 video 分支严格区分 —— 首帧图单独
        编码时语义是「一张静图」，不是「一段动图的第 0 帧」。"""
        self._ensure()
        outs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            outs.append(self._process(
                [{"image": im, "instruction": INSTR_IMAGE} for im in batch]).float().cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)

    @torch.no_grad()
    def encode_frames(self, frame_lists, batch_size=1):
        """帧列表（每个是 list[PIL.Image]）→ [N,2048]。"""
        self._ensure()
        outs = []
        for i in range(0, len(frame_lists), batch_size):
            batch = frame_lists[i:i + batch_size]
            outs.append(self._process(
                [{"video": self._sample(fl), "instruction": INSTR_VIDEO} for fl in batch]).float().cpu().numpy())
        return np.concatenate(outs, 0).astype(np.float32)

    def _load_frames(self, path):
        frames = []
        with Image.open(path) as im:
            for fr in ImageSequence.Iterator(im):
                frames.append(fr.convert("RGB"))
        return self._sample(frames)

    def _sample(self, frames):
        if len(frames) <= self.max_frames:
            return frames
        idx = [round(i * (len(frames) - 1) / (self.max_frames - 1))
               for i in range(self.max_frames)]
        return [frames[i] for i in idx]


if __name__ == "__main__":
    enc = Int8QwenEncoder()
    te = enc.encode_texts(["一个黑色的8号台球在绿色台球桌上滚动"])
    print("文本:", te.shape, "范数", np.linalg.norm(te, axis=-1))
    g = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto\8-ball_1f3b1.gif"
    ve = enc.encode_gif_paths([g])
    print("GIF :", ve.shape, "范数", np.linalg.norm(ve, axis=-1))
    g2 = r"D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版\data\gif_raw\noto\3D87B14FBB731ED3A948AC6DD9EA6EBC.gif"
    ve2 = enc.encode_gif_paths([g2])
    print("两素材余弦:", float((ve[0] @ ve2[0]).item()), " 峰值显存",
          torch.cuda.max_memory_reserved()/1024**2, "MiB")