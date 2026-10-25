# AetherGIF-V1

> 物理矩阵神经网络 GIF 生成最小验证环境。当前以 v4200（23.04M 参数）为工作副本，根 `src/`（方案 D）为轻量验证。

## 一句话开跑

**启动训练**（v4200）：
```bash
cd "D:\AetherGIF-V1\docx\实验记录\versions\v4200_多模态CLIP产品版"
C:\Python312\python.exe scripts/train_guided.py
```

**检查环境**：
```bash
C:\Python312\python.exe scripts/env_probe.py
```

**生成 GIF**（根生态，轻量验证）：
```bash
C:\Python312\python.exe boot312.py scripts/inference.py "测试生成" --device cpu --output_dir outputs/inferenced
```

## 核心约束

- **唯一解释器**：`C:\Python312\python.exe`（3.12.0）
- **必须挂死的三件套**：
  1. `D:/AetherGIF-V1`（`torch_accel_shim.py`）
  2. `D:/AetherGIF-V1/deps312`（完整 transformers 5.19.0）
  3. `D:/AetherGIF-V1/int8_deps`（`qwen_vl_utils`）
- **激活顺序**：先插 sys.path → `from torch_accel_shim import install_full; install_full()` → 然后 `import transformers`
- **绝对禁止**：`rm` / `git clean` / `git reset --hard` / 文件移动 / 覆盖已有文件
- **`.workbuddy/`**：项目记忆目录，不是缓存，任何时候不得删除

## 双生态边界

|  | 根 `src/`（方案 D） | `v4200/src/`（产品版） |
|---|---|---|
| 参数量 | 1.13M / 2.02M | **23.04M** |
| `a_gamma_lvl` | 0.5 固定 | 可学习 γ=0.3 |
| `a_mu` | `log1p(0.05)` | `log(0.2)` |
| 条件初始化 | `CondInitModulator` | **`CondInitPC`**（`COND_INIT_PC=1 COND_PC_RANK=8`） |

**权重不兼容，禁止交叉覆盖**。

## 随仓库打包的关键文档

- `docs/仓库结构索引.md` —— 目录总览、双生态边界、常用入口
- `docs/环境与依赖说明.md` —— 三挂路径 + `install_full()` 时序
- `docs/已知缺失清单.md` —— 无法重建项目的如实登记

## 验证探针

- `audit_project.py` —— 只读完整性体检（语法/import/索引/ckpt）
- `scripts/test_codec.py` —— GIF ↔ 振子场往返保真度
- `scripts/env_probe.py` —— 依赖与环境探针

## 历史修复记录摘要

- 20 个 `gif_codec.py` 的 `map_coordinates` 坐标维度 bug（scipy 1.18 的 3D 输入必须 3 组坐标）
- `fix_h3_merge.py` / `verify_h3_merge_diff.py` 归档兜底（ComfyUI 工作流已丢失）
- 硬编码 Linux 路径全部改为 Windows（`models/Qwen3-VL-Embedding-2B/models/.../snapshots/master/`）

## 铁律

任何操作**默认只增不删**。破坏性命令必须先问。
