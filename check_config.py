import json
import os

CFG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "Qwen3-VL-Embedding-2B", "models",
    "Qwen--Qwen3-VL-Embedding-2B", "snapshots", "master", "config.json",
)
config = json.load(open(CFG, encoding="utf-8"))
print(json.dumps(config, indent=2, ensure_ascii=False))
