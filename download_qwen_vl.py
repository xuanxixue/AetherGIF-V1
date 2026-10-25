import os
import requests
import pathlib

# 创建项目目录
project_dir = 'Qwen_src'
os.makedirs(project_dir, exist_ok=True)

# 从 Hugging Face 下载 Qwen3VL 相关文件
files_to_download = [
    ('https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/resolve/main/modeling_qwen3_vl_embedding.py', 'modeling_qwen3_vl_embedding.py'),
    ('https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/resolve/main/configuration_qwen3_vl_embedding.py', 'configuration_qwen3_vl_embedding.py'),
    ('https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/resolve/main/processing_qwen3_vl.py', 'processing_qwen3_vl.py'),
]

for url, filename in files_to_download:
    filepath = pathlib.Path(project_dir) / filename
    if not filepath.exists():
        print(f'Downloading {url} -> {filepath}...')
        r = requests.get(url)
        if r.status_code == 200:
            filepath.write_text(r.text)
            print(f'✅ {filename}')
        else:
            print(f'❌ Failed to download {filename}')

print(f'\nDownloaded files in {project_dir}')
