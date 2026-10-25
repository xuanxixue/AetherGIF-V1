# tools/ —— 工具工作区（新增，2026-10-23）

> **只增不删**整理的产物。此目录下全部是**新增**文件与工作区，
> **没有任何原件被移动到这里**。根目录的原始脚本一律保留原位。

## 子目录

| 目录 | 用途 |
|---|---|
| `analysis/` | 分析类工具工作区（当前为空，预留给后续新增分析脚本） |
| `verify/` | 验证类工具工作区（当前为空） |
| `comfyui_workflow/` | ComfyUI 工作流 JSON 归档（2 个文件，从 Downloads 复制） |

## 归档件清单

| 文件 | 说明 |
|---|---|
| `comfyui_workflow/deepseek_json_20261015_8e1698_fixed.json` | MiniMax H3 合并工作流的**修复产物**（含 `widget_ue_connectable` 补丁 + link 233 修复），已通过结构自检 PASS |
| `comfyui_workflow/deepseek_json_20261015_ff1c5c.json` | 同批次另一份工作流快照 |

## 关联的根目录脚本

- `fix_h3_merge.py` —— 修复 + 自检，幂等，源缺失时用归档副本
- `verify_h3_merge_diff.py` —— 差异审计；原始工作流已不在，转为对归档产物做结构自检

## 完整文档

见 `docs/仓库结构索引.md` 与 `docs/根目录脚本归类索引.md`。
