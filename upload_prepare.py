#!/usr/bin/env python3
import os
import subprocess

print("开始准备上传文件到 GitHub...")

# 核心目录和文件
core_items = [
    # 核心代码
    "src/",
    "configs/",
    "scripts/*.py",
    "accept_clip_condition.py",
    "boot312.py",
    "check_config.py",
    "check_imports.py",
    "download_quantized_qwen.py",
    "download_qwen_vl.py",
    "find_model_path.py",
    "fix_h3_merge.py",
    "install_int8.py",
    "int8_qwen_encoder.py",
    "load_int8_qwen.py",
    "mini_preencode.py",
    "mini_setup.py",
    "mini_verify.py",
    "mini_verify_domestic.py",
    "quantize_qwen_int8.py",
    "show_config.py",
    "space_map.txt",
    "torch_accel_shim.py",
    "test_complete.py",
    "test_core.py",
    "test_encode_video_files.py",
    "test_minimal.py",
    "test_qwen_2.py",
    "test_qwen_3.py",
    "test_qwen_4.py",
    "test_qwen_embeddings.py",
    "test_qwen_final.py",
    "test_qwen_load.py",
    "test_qwen_model.py",
    "test_qwen_simple.py",
    "test_video_branch.py",
    "test_video_branch2.py",
    "test_video_branch3.py",
    "test_video_branch4.py",
    "test_video_pyav.py",
    "verify_h3_merge_diff.py",
    "verify_int8_load.py",
    "verify_int8_process.py",
    "verify_qwen_cond.py",
    "verify_qwen_gpu.py",
    # 文档
    "Docs/",
    # 数据管道
    "data_pipeline/",
]

# 改变工作目录到项目根目录
os.chdir("D:/AetherGIF-V1")

# 执行 git add
try:
    # 添加单个文件
    for item in core_items:
        if item.endswith("/*.py"):
            # 匹配目录下的 .py 文件
            dir_part = item.replace("/*.py", "")
            py_files = []
            for root, dirs, files in os.walk(dir_part):
                # 跳过 _libs 和 __pycache__
                if "_libs" in root or "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith(".py"):
                        py_files.append(os.path.join(root, f))
            for f in sorted(py_files):
                subprocess.run(["git", "add", f], cwd="D:/AetherGIF-V1", check=True)
        else:
            # 直接添加文件
            subprocess.run(["git", "add", item], cwd="D:/AetherGIF-V1", check=True)

    print(f"成功添加了 {len(core_items)} 个项目")

    # 查看状态
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd="D:/AetherGIF-V1",
        capture_output=True,
        text=True,
    )
    staged_files = [line for line in result.stdout.splitlines() if line.strip()]
    print(f"\n已暂存 {len(staged_files)} 个文件:")
    for file_line in staged_files[:20]:
        print(file_line)
    if len(staged_files) > 20:
        print(f"... 还有 {len(staged_files) - 20} 个文件")

except subprocess.CalledProcessError as e:
    print(f"错误: {e}")
    print("请检查 Git 状态")
except Exception as e:
    print(f"发生异常: {e}")

print("\n准备完成，可以执行 git commit 和 git push")
