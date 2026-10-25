# -*- coding: utf-8 -*-
"""环境探针：直接用 Python 查版本/依赖/GPU，不做任何目录猜测。"""
import importlib
import sys


def probe(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed")
    except Exception as e:
        return "MISSING(%s)" % type(e).__name__


print("python      ", sys.version.split()[0])
print("executable  ", sys.executable)

mods = [
    "torch", "torchvision", "torchaudio", "transformers",
    "numpy", "einops", "scipy", "PIL", "cv2",
    "datasets", "accelerate", "safetensors", "tokenizers",
]
print("--- packages ---")
for m in mods:
    print("%-12s %s" % (m, probe(m)))

try:
    import torch
    print("--- cuda ---")
    print("avail       ", torch.cuda.is_available())
    print("torch.cuda  ", torch.version.cuda)
    print("cudnn       ", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu0        ", torch.cuda.get_device_name(0))
        print("vram GB     ", round(p.total_memory / 1024**3, 2))
        free, total = torch.cuda.mem_get_info()
        print("vram free GB", round(free / 1024**3, 2))
except Exception as e:
    print("torch probe failed:", repr(e))
