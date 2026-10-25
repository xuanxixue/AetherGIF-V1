@echo off
echo ========================================
echo 自动修复: 安装 transformers 相关依赖
echo ========================================

:: 清理已损坏的安装
echo [1/3] 清理损坏包...
C:\Python312\python.exe -m pip uninstall -y transformers tokenizers huggingface-hub
C:\Python312\python.exe -m pip uninstall -y ~transformers~ ~tokenizers~ ~huggingface-hub~

:: 安装官方预编译的 Windows 版本 (最近稳定版)
echo [2/3] 安装转换依赖...
C:\Python312\python.exe -m pip install --user transformers==5.1.1 tokenizers==0.20.4 huggingface-hub==0.36.2

:: 验证组件
echo [3/3] 验证安装...
C:\Python312\python.exe -c "import transformers; from transformers import CLIPModel, CLIPProcessor; print('SUCCESS!')"

pause
