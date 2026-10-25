# -*- coding: utf-8 -*-
"""
自动安装并修复所有依赖，无视已损坏的包。
"""

import subprocess
import sys

def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode

def main():
    print("清除损坏包...")
    run(f'{sys.executable} -m pip uninstall -y transformers tokenizers huggingface-hub')
    run(f'{sys.executable} -m pip uninstall -y ~transformers* ~tokenizers* ~huggingface-hub*')

    print("\n安装纯净依赖...")
    run(f'{sys.executable} -m pip install --user transformers==4.38.2 tokenizers==0.15.2 huggingface-hub==0.36.2')

    print("\n验证...")
    probe = 'import transformers; print(transformers.__version__); from transformers import CLIPModel, CLIPProcessor; print("[SUCCESS] Transformers reproduced. CLIPModel imported OK")'
    ret = subprocess.run([sys.executable, "-c", probe], shell=False, capture_output=False)
    return ret.returncode

if __name__ == "__main__":
    exit(main())
