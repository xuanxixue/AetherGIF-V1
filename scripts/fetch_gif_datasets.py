# -*- coding: utf-8 -*-
"""GIF 数据集批量下载器（国内可用源：hf-mirror / modelscope）。

注意：hf-mirror 的 API 需要浏览器 UA，否则 403。
"""
import os
import sys
import ssl
import json
import urllib.request

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

ROOT = 'D:/AetherGIF-V1/data/gif_raw'
os.makedirs(ROOT, exist_ok=True)


def fetch(url, dest, desc=''):
    """断点续传下载。返回 True=成功。"""
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f'[SKIP] {desc} 已存在 ({os.path.getsize(dest)/1e6:.1f} MB)')
        return True
    tmp = dest + '.part'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': UA})
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            total = int(r.headers.get('Content-Length', 0))
            got = 0
            with open(tmp, 'wb') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total and got % (10 << 20) < (1 << 20):
                        print(f'  {desc}: {got/1e6:.0f}/{total/1e6:.0f} MB', flush=True)
        os.replace(tmp, dest)
        print(f'[OK] {desc}  {os.path.getsize(dest)/1e6:.1f} MB')
        return True
    except Exception as e:
        print(f'[FAIL] {desc}: {type(e).__name__} {str(e)[:90]}')
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def hf_url(ds, path):
    return f'https://hf-mirror.com/datasets/{ds}/resolve/main/{path}'


def list_files(ds):
    try:
        req = urllib.request.Request(f'https://hf-mirror.com/api/datasets/{ds}',
                                     headers={'User-Agent': UA, 'Accept': 'application/json'})
        d = json.load(urllib.request.urlopen(req, timeout=30, context=CTX))
        return [s['rfilename'] for s in d.get('siblings', [])]
    except Exception as e:
        print(f'[LIST-FAIL] {ds}: {str(e)[:70]}')
        return []


# 目标包：(数据集ID, 文件名, 本地名, 说明)
TARGETS = [
    ('4eJIoBek/Old-GIFs-22k', '22000gifs.7z', 'bulk_22000.7z', '主源 22000 GIF / 352MB'),
    ('to-be/animated-gifs-pixel-art-captions-1k', 'data/train-00000-of-00001.parquet',
     'pixelart_1k.parquet', '像素动画 1k / 15MB'),
]

if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else None
    os.makedirs(f'{ROOT}/_packs', exist_ok=True)
    for ds, path, local, desc in TARGETS:
        if only and only not in desc:
            continue
        dest = f'{ROOT}/_packs/{local}'
        print(f'\n=== {desc} ===')
        print(f'    {ds}/{path}')
        fetch(hf_url(ds, path), dest, desc)
