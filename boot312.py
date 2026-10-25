# -*- coding: utf-8 -*-
"""Python 3.12 (CUDA) 启动器 —— 用 D 盘依赖，屏蔽 C 盘残缺的 transformers。

★ 为什么需要这个：
  C:/Python312 的 site-packages/transformers 是**残缺的**（4.38.2 与 4.44.2
  两个版本互相覆盖，只剩 4 个文件，缺 utils/versions.py）。而 Qwen3-VL-Embedding
  需要 transformers >= 5.0（才有 qwen3_vl 模块）。

★ 为什么不能直接 pip install：
  C 盘只剩 0.3 GB，装不下。且 pip install 到 site-packages 会再次覆盖。

★ 解法（三个动作）：
  1. sitecustomize：把 D:/AetherGIF-V1/deps312 插到 sys.path 最前面，
     让好版本压过 C 盘残缺版。
  2. C 盘残缺 transformers 重命名隔离（D:/AetherGIF-V1/.venv 里的 cp313
     二进制扩展对 py312 不可用，所以只隔离 C 盘那份）。
  3. 保留 C 盘 torch（唯一带 CUDA 的），复用 D 盘纯 Python 依赖。

用法：
  C:/Python312/python.exe boot312.py verify_qwen_gpu.py --mode auto
"""
import os
import sys

DEPS = r"D:/AetherGIF-V1/deps312"
VENV_PURE = r"D:/AetherGIF-V1/.venv/Lib/site-packages"

# 必须最优先：好版本 transformers
if os.path.isdir(DEPS):
    sys.path.insert(0, DEPS)
# venv 里只有纯 Python 包可复用，二进制扩展（tokenizers/regex/torch）
# 是 cp313 编译的，py312 加载会 ImportError，所以放最后兜底。
if os.path.isdir(VENV_PURE):
    sys.path.append(VENV_PURE)

# 把 C 盘残缺的 transformers 目录从路径中剔除
_bad = r"C:\Python312\Lib\site-packages\transformers"
sys.path = [p for p in sys.path
            if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]

# python312_scripts：保证不是 py313 的 bin 在前面
sys.argv = sys.argv[1:] if sys.argv and sys.argv[0].endswith("boot312.py") else sys.argv

if __name__ == "__main__":
    target = sys.argv[0]
    print(f"[boot312] sys.path[0] = {sys.path[0]}", flush=True)
    import torch
    # ★ 补（2026-10-23）：transformers 5.x 硬依赖 torch.accelerator，torch 2.5.1 缺。
    #   必须先垫片再 import transformers，否则报 "Could not import module 'AutoModelForCausalLM'"。
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    try:
        from torch_accel_shim import install_full
        install_full()
    except Exception as ex:
        print(f"[boot312][WARN] torch_accel_shim 加载失败：{ex}", flush=True)
    import transformers
    print(f"[boot312] transformers {transformers.__version__} <- {transformers.__file__}", flush=True)
    print(f"[boot312] torch {torch.__version__} CUDA={torch.cuda.is_available()}", flush=True)
    print("-" * 60, flush=True)
    # 交棒给目标脚本
    _g = {"__name__": "__main__", "__file__": target}
    exec(compile(open(target, encoding="utf-8").read(), target, "exec"), _g)
