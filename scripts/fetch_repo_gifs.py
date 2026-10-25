# -*- coding: utf-8 -*-
"""通用 GitHub 仓库批量下载 GIF（用 git trees 一次拿清单 + git blobs 拿文件）。

比逐目录遍历快得多，且天然去重（同一路径只下一个）。
用法：python scripts/fetch_repo_gifs.py <owner/repo> <输出子目录名>
"""
import os
import sys
import ssl
import json
import time
import base64
import urllib.request

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
TOK = open('D:/AetherGIF-V1/scripts/github_token.txt', encoding='utf-8').read().strip()
H = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/vnd.github+json',
     'Authorization': 'Bearer ' + TOK, 'Accept-Encoding': 'identity'}

ROOT = 'D:/AetherGIF-V1/data/gif_raw/anime'
MIN_FRAMES, MAX_FRAMES = 6, 120
MIN_SHORT, MAX_SHORT = 64, 800


def api(url, timeout=120, retries=4):
    last = None
    for k in range(retries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=H),
                                       timeout=timeout, context=CTX)
            return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 403 and b'rate limit' in (e.read() or b'').lower():
                print('  [限额用尽]'); sys.exit(0)
            last = e
        except Exception as e:
            last = e
        time.sleep(1.2 * (k + 1))
    raise last


def safe(name):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)


def main():
    repo = sys.argv[1]
    sub = sys.argv[2] if len(sys.argv) > 2 else safe(repo.split('/')[-1])
    outroot = os.path.join(ROOT, sub)
    os.makedirs(outroot, exist_ok=True)
    api_base = 'https://api.github.com/repos/%s' % repo

    print('[1/3] 取全量文件清单…')
    br = api(api_base)['default_branch']
    tree = api('%s/git/trees/%s?recursive=1' % (api_base, br))
    if tree.get('truncated'):
        print('  [warn] tree 截断，清单不完整')
    blobs = [x for x in tree.get('tree', [])
             if x['type'] == 'blob' and x['path'].lower().endswith('.gif')]
    print(f'  {len(blobs)} 个 gif')

    print('[2/3] 下载…')
    from PIL import Image
    ok = bad = 0
    for i, b in enumerate(blobs):
        if i % 50 == 0:
            print(f'  [{i+1}/{len(blobs)}] ok={ok}', flush=True)
        # 保留原始子目录结构（分类信息）
        dest = os.path.join(outroot, *b['path'].split('/'))
        if os.path.exists(dest):
            ok += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            j = api('%s/git/blobs/%s' % (api_base, b['sha']), timeout=120, retries=3)
            if j.get('encoding') != 'base64' or not j.get('content'):
                bad += 1
                continue
            data = base64.b64decode(j['content'])
            if data[:3] != b'GIF':
                bad += 1
                continue
            with open(dest, 'wb') as f:
                f.write(data)
        except Exception:
            bad += 1
            continue
        try:
            with Image.open(dest) as im:
                nf, (w, h) = getattr(im, 'n_frames', 1), im.size
                s = min(w, h)
            if nf < MIN_FRAMES or nf > MAX_FRAMES or s < MIN_SHORT or s > MAX_SHORT:
                os.remove(dest)
                bad += 1
                continue
        except Exception:
            if os.path.exists(dest):
                os.remove(dest)
            bad += 1
            continue
        ok += 1

    print('[3/3] 完成')
    kept = sum(len([f for f in fs if f.lower().endswith('.gif')])
               for _, _, fs in os.walk(outroot))
    print(f'  入库 {kept} 个（下载成功 {ok}，剔除/失败 {bad}）-> {outroot}')


if __name__ == '__main__':
    main()
