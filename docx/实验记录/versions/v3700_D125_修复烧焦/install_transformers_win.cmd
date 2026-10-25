@echo off
echo ================== 修复 transformers 依赖 ==================
echo [1/5] 升级 pip (使用 Python 3.12)
C:\Python312\python.exe -m pip install --upgrade pip
echo [2/5] 清理旧的 transformer 依赖
C:\Python312\python.exe -m pip uninstall -y transformers tokenizers huggingface-hub
echo [3/5] 安装预编译的 tokenizers (有轮子包)
C:\Python312\python.exe -m pip install --only-binary=:all: tokenizers==0.15.2
echo [4/5] 安装 transformers 4.44.2
C:\Python312\python.exe -m pip install --only-binary=:all: transformers==4.44.2 huggingface-hub==0.36.2
echo [5/5] 验证安装
C:\Python312\python.exe -c "import transformers; print('transformers version:', transformers.__version__)"

echo [完成] 请重新运行训练脚本：
echo   C:\Python312\python.exe -u "D:\AetherGIF-V1\docx\实验记录\versions\v3700_D125_修复烧焦\scripts\test_cond_gen_v4.py" 250 "8-ball_1f3b1.gif" 0.0003
