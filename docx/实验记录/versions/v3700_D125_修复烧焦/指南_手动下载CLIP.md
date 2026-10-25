# 手动下载 CLIP 模型到 Windows

## 方案 A: 从 HuggingFace 官网下载（需要 VPN）

### 步骤:
1. 访问 https://huggingface.co/openai/clip-vit-base-patch32
2. 点击 "Files and versions" 标签
3. 下载以下文件到 `D:\AetherGIF-V1\models\clip\` 目录:
   - `config.json`
   - `pytorch_model.bin` (或 `model.safetensors`)
   - `preprocessor_config.json`
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `vocab.json` (如果有)

## 方案 B: 从阿里云 Ask 下载 (推荐)

### 步骤:
1. 访问 https://github.com/aliyun/adl-db2ds-vit/blob/main/clip-vit-base-patch32
2. 查找镜像或下载链接

## 方案 C: 使用 `pip` 直接安装

### 步骤:
1. 关闭防火墙（临时）
2. 运行:
   ```powershell
   mkdir D:\models_clip_temp
   cd D:\models_clip_temp
   pip download openai/clip-vit-base-patch32 --no-deps --out .
   ```
3. 点击 `.whl` 文件的安装按钮

## 方案 D: 使用 `git` 克隆

### 步骤:
1. 在 Git Bash 或 PowerShell 中:
   ```bash
   git clone https://huggingface.co/openai/clip-vit-base-patch32
   ```
2. 将克隆的文件夹重命名为 `clip` 并移动到 `D:\AetherGIF-V1\models\`

## 快速实用命令

```powershell
# 一键下载并提取 (需要管理员权限和 pip 支持)
New-Item -ItemType Directory -Force -Path "D:\models_clip_temp" | Out-Null
cd D:\models_clip_temp
pip download openai/clip-vit-base-patch32 --no-deps
Expand-Archive *.whl -DestinationPath extracted
Copy-Item extracted\*\* -Destination D:\AetherGIF-V1\models\clip\ -Force
```

## 验证安装

```powershell
python -c "from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('D:/AetherGIF-V1/models/clip')"
```

如果没有报错，说明模型已成功加载。
