@echo off
REM 强制安装预编译包，避免 Windows 上编译 Rust 依赖

echo [安装] 仅使用预编译 wheel...
C:\Python312\python.exe -m pip install --only-binary=:all: ^
    transformers==4.44.2 ^
    huggingface-hub==0.36.2 ^
    tokenizers==0.15.2 ^
    protobuf==5.28.3

echo [验证] 检查 transformers 版本...
C:\Python312\python.exe -c "import transformers; print('transformers:', transformers.__version__)"

echo.
echo 安装完成！现在运行训练脚本:
echo C:\Python312\python.exe -u "D:\AetherGIF-V1\docx\实验记录\versions\v3700_D125_修复烧焦\scripts\test_cond_gen_v4.py" 250 "8-ball_1f3b1.gif" 0.0003
