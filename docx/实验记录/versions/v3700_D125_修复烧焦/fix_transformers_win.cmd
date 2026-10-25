@echo off
echo ================== 兼容性修复：transformers ==================
echo 检测到 Windows 环境缺少 MSVC 编译器，使用预编译 wheel 安装
echo.

echo [1/3] 升级 pip
C:\Python312\python.exe -m pip install --upgrade pip -q
echo [2/3] 卸载冲突包
C:\Python312\python.exe -m pip uninstall -y transformers tokenizers huggingface-hub -q
echo [3/3] 从官方源安装预编译包
C:\Python312\python.exe -m pip install --only-binary=:all: ^
    transformers==4.44.2 ^
    huggingface-hub==0.36.2 ^
    tokenizers==0.15.2 ^
    protobuf==5.28.3

echo.
echo 验证安装...
C:\Python312\python.exe -c "import transformers; print(transformers.__version__)"

echo.
echo 修复完成！请运行训练脚本：
echo C:\Python312\python.exe -u "D:\AetherGIF-V1\docx\实验记录\versions\v3700_D125_修复烧焦\scripts\test_cond_gen_v4.py" 250 "8-ball_1f3b1.gif" 0.0003
