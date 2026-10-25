import sys
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.device_count() if torch.cuda.is_available() else None)
try:
    import bitsandbytes
    print("bitsandbytes: available")
except Exception as e:
    print("bitsandbytes: NOT available -", e)
