# -*- coding: utf-8 -*-
"""torch.accelerator 垫片 —— 让 torch 2.5.1 能跑 transformers 5.x。

★ 背景：
  transformers 5.19 的 `utils/import_utils.py` 会调`torch.accelerator.current_accelerator()`。
  该属性是 **torch 2.6 才引入**的抽象设备接口，2.5.1 没有 -> 直接 AttributeError。

★ 为什么用垫片而不是升级 torch：
  - 升级到 2.6+ 需下 2.5 GB wheel，而 pip 临时目录落在 C 盘（仅剩 0.3 GB）
    -> `No space left on device`。用户铁律：新装东西一律放 D 盘。
  - 实测除 `torch.accelerator` 外，transformers 5.19 + Qwen3VL 在 2.5.1 上
    **没有任何其他版本障碍**（已验证 qwen3_vl 全模块导入成功）。
  - 垫片只补 4 个只读访问器，语义与 2.6 官方实现一致，不改变任何计算行为。

★ 用法：必须在 `import transformers` **之前**执行本文件。
"""
import torch


class _AcceleratorShim:
    """torch 2.6 `torch.accelerator` 的最小兼容实现。"""

    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def device_count() -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    @staticmethod
    def current_accelerator():
        if not torch.cuda.is_available():
            return None
        return torch.device("cuda", torch.cuda.current_device())

    @staticmethod
    def current_device_index() -> int:
        return torch.cuda.current_device() if torch.cuda.is_available() else -1


def install():
    """幂等安装垫片。返回 True 表示确实打了补丁。"""
    if hasattr(torch, "accelerator"):
        return False
    torch.accelerator = _AcceleratorShim
    return True


# 有些代码路径还查 torch.accelerator.is_bf16_supported / device_properties
def _bf16_supported(device=None):
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8

def install_full():
    """装垫片 + 常见扩展属性。"""
    patched = install()
    if patched:
        acc = torch.accelerator
        acc.is_bf16_supported = staticmethod(_bf16_supported)
        acc.get_device_capability = staticmethod(
            lambda device=None: torch.cuda.get_device_capability(device))
        acc.get_device_properties = staticmethod(
            lambda device=None: torch.cuda.get_device_properties(device))
        acc.set_device_index = staticmethod(
            lambda idx: torch.cuda.set_device(idx))
    return patched


if __name__ == "__main__":
    print("patched:", install_full())
    print("torch", torch.__version__, "CUDA", torch.cuda.is_available())
    print("accelerator:", torch.accelerator.current_accelerator())
