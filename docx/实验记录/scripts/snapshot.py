# -*- coding: utf-8 -*-
"""AetherGIF-V1 版本快照工具。

用法（在项目根目录运行）：
    C:\\Python312\\python.exe docx/实验记录/scripts/snapshot.py \
        --name "v3100_方案X_3100步" --src src --script scripts/overfit.py \
        --gif data/gif_raw/noto/8-ball_1f3b1.gif

效果：
    在 docx/实验记录/versions/<name>/ 下创建自包含快照：
        src/          完整源码（自动把 aether_gif_model.py 的 from src.* 改成无前缀）
        scripts/overfit.py
        data/gif_raw/noto/<gif>
        版本说明.md    骨架（自动填：参数量指纹占位、loss 占位、超参占位）

设计目标：杜绝「只有一份文档、没有代码快照、版本回退不了」的历史问题。
每训一个新版本，跑一次本工具即可留档。
"""
import argparse
import shutil
import sys
from pathlib import Path

MODEL_IMPORT_FIX = {
    "from src.physics.block": "from physics.block",
    "from src.physics.time_field": "from physics.time_field",
    "from src.physics.kuramoto_sync": "from physics.kuramoto_sync",
    "from src.physics.standing_wave_ffn": "from physics.standing_wave_ffn",
    "from src.physics.drift_field": "from physics.drift_field",
    "from src.physics.thermodynamics": "from physics.thermodynamics",
    "from src.codec.gif_codec": "from codec.gif_codec",
    "from src.qwen_encoder.qwen_encoder": "from qwen_encoder.qwen_encoder",
}


def fix_model_imports(model_path: Path):
    """把 aether_gif_model.py 的 from src.* 改成无前缀，并删 sys.path.insert('..')。"""
    text = model_path.read_text(encoding="utf-8")
    changed = False
    # 删除把外层塞进 path 的那行（快照必须自洽，不依赖外层项目）
    lines = text.splitlines()
    new_lines = []
    for ln in lines:
        if ln.strip().startswith("sys.path.insert(0, '.' if __name__"):
            changed = True
            continue
        new_lines.append(ln)
    text = "\n".join(new_lines)
    for old, new in MODEL_IMPORT_FIX.items():
        if old in text:
            text = text.replace(old, new)
            changed = True
    model_path.write_text(text, encoding="utf-8")
    return changed


def main():
    ap = argparse.ArgumentParser(description="AetherGIF-V1 版本快照")
    ap.add_argument("--name", required=True, help="版本目录名，如 v3100_方案X_3100步")
    ap.add_argument("--src", default="src")
    ap.add_argument("--script", default="scripts/overfit.py")
    ap.add_argument("--gif", default="data/gif_raw/noto/8-ball_1f3b1.gif")
    ap.add_argument("--root", default=None, help="项目根目录，默认取当前工作目录")
    ap.add_argument("--extra", nargs="*", default=[], help="额外要备份的文件/目录（相对项目根）")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path.cwd()
    base = root / "docx" / "实验记录" / "versions" / args.name
    base.mkdir(parents=True, exist_ok=True)

    # 1. 复制 src（清理 __pycache__）
    src_src = root / args.src
    shutil.copytree(src_src, base / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 2. 导入自洽化
    model = base / "src" / "model" / "aether_gif_model.py"
    if model.exists():
        fixed = fix_model_imports(model)
        print(f"[snapshot] model 导入自洽化{'已修复' if fixed else '（无需修改）'}: {model}")

    # 3. 复制训练脚本
    (base / "scripts").mkdir(exist_ok=True)
    shutil.copy(root / args.script, base / "scripts" / Path(args.script).name)

    # 4. 复制训练数据
    gif_src = root / args.gif
    gif_dst = base / "data" / "gif_raw" / "noto" / gif_src.name
    gif_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(gif_src, gif_dst)

    # 5. 额外文件
    for p in args.extra:
        sp = root / p
        dp = base / p
        dp.parent.mkdir(parents=True, exist_ok=True)
        if sp.is_dir():
            shutil.copytree(sp, dp, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy(sp, dp)

    # 6. 生成版本说明骨架
    readme = base / "版本说明.md"
    readme.write_text(
        f"""# 版本说明 —— {args.name}

> **一句话定位**：（请填写这个版本在做什么、相比上个版本改了什么）
> **身份指纹**：参数量 = **（请运行冒烟测试后填入，例如 1,135,658）**

---

## 1. 版本标识

| 项 | 值 |
|---|---|
| 方案代号 | （填写） |
| 训练步数 | （填写） |
| 代码快照目录 | `versions/{args.name}/` |
| 对应日志 | （填写 overfit_out*.txt） |
| 结果 | （填写 PSNR / profile_cos / 用户评价） |

---

## 2. 代码结构

```
{args.name}/
├── src/
│   ├── model/aether_gif_model.py
│   ├── physics/           # block.py + 五算子
│   ├── codec/gif_codec.py
│   └── qwen_encoder/
├── scripts/overfit.py
├── data/gif_raw/noto/{Path(args.gif).name}
└── outputs_real/
```

---

## 3. 参数量指纹（回退核对依据）

- 总参数量 = **（冒烟验证后填）**
- 验证命令：`C:\\Python312\\python.exe scripts/overfit.py 3`

---

## 4. loss 口径（精确公式，复现必须一致）

（从 overfit.py 抄出本版本的 loss 精确公式，含归一化/权重/截断项）

---

## 5. 训练超参（复现必须一致）

```
STEPS / LR / 分组lr / warmup / SEED_MODE / use_qwen / AMP_P 等
```

---

## 6. 产物映射与现状

| 产物 | 路径 | 状态 |
|---|---|---|
| 训练日志 | | |
| 权重 checkpoint | outputs_real/overfit_ckpt.pt | |
| 生成 GIF | outputs_real/overfit.gif | |

---

## 7. 已知缺陷

（记录本版本已知问题，供派生版本参考）

---

*生成时间：自动快照。请补齐上方「（……）」占位内容。*
""",
        encoding="utf-8",
    )

    (base / "outputs_real").mkdir(exist_ok=True)
    print(f"[snapshot] 完成 → {base}")
    print(f"[snapshot] 下一步：cd 到快照目录，跑 `C:\\Python312\\python.exe scripts/overfit.py 3` 冒烟验证参数量指纹")


if __name__ == "__main__":
    main()