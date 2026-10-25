"""定位本地已下载的模型快照目录（跨平台，兼容 Linux 缓存与 Windows 项目内缓存）。"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

CANDIDATE_ROOTS = [
    HERE / "models" / "Qwen3-VL-Embedding-2B" / "models",
    HERE / "models",
    pathlib.Path("/tmp/modelscope_cache/models"),
    pathlib.Path.home() / ".cache" / "modelscope" / "models",
]

for root in CANDIDATE_ROOTS:
    if not root.exists():
        continue
    print(f"[ROOT] {root}")
    for x in sorted(root.iterdir()):
        if not x.is_dir():
            continue
        print(f"  Dir {x.name}:")
        for sub in sorted(x.iterdir()):
            print(f"    {sub.name}")
            if sub.is_dir() and sub.name == "snapshots":
                for snap in sorted(sub.iterdir()):
                    print(f"      [snapshot] {snap}")
                    for f in sorted(snap.iterdir()):
                        if f.is_file():
                            size_mb = f.stat().st_size / 1024 ** 2
                            print(f"        {f.name}  ({size_mb:.1f} MB)")
