# -*- coding: utf-8 -*-
"""从 LTLA/acceptable-anime-gifs 批量下载二次元 GIF（断点续跑版）。

★ 硬约束：GitHub contents API 免登录限额 **60 次/小时**，且
  - 每次目录列举 = 1 次请求
  - 每个文件下载 = 1 次请求
  - 每个文件元数据 = 1 次请求
  → 全量 112 部约需数千次请求 = 数十小时。必须断点续跑 + 限额感知。

用法：
  python scripts/fetch_anime_gifs.py            # 跑到限额用完为止
  python scripts/fetch_anime_gifs.py --status   # 只看进度
"""
import os
import sys
import json
import time
import base64
import ssl
import urllib.request

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')
H = {'User-Agent': UA, 'Accept': 'application/vnd.github+json',
     'Accept-Encoding': 'identity'}

REPO = 'LTLA/acceptable-anime-gifs'
API = 'https://api.github.com/repos/%s' % REPO
ROOT = 'D:/AetherGIF-V1/data/gif_raw/anime'
META = os.path.join(ROOT, 'metadata.json')

MIN_FRAMES, MAX_FRAMES = 6, 120
MIN_SHORT, MAX_SHORT = 64, 800
MAX_BYTES = 20 * 1024 * 1024      # contents API 对超大文件返回空 content
RETRIES = 5

# GitHub token：优先环境变量 GITHUB_TOKEN，其次同目录的 token.txt
# 免登录限额仅 60 次/小时；带 token 是 5000 次/小时，差 80 倍
TOKEN = os.environ.get('GITHUB_TOKEN', '')
_tokfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'github_token.txt')
if not TOKEN and os.path.exists(_tokfile):
    with open(_tokfile, encoding='utf-8') as f:
        TOKEN = f.read().strip()
if TOKEN:
    H['Authorization'] = 'Bearer ' + TOKEN


class RateLimited(Exception):
    pass


def api(url, timeout=60, retries=RETRIES):
    last = None
    for k in range(retries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=H),
                                       timeout=timeout, context=CTX)
            return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 403 and b'rate limit' in (e.read() or b'').lower():
                raise RateLimited()
            last = e
        except Exception as e:
            last = e
        time.sleep(1.2 * (k + 1))
    raise last


def fetch_blob(sha, dest, timeout=120):
    """用 git/blobs 接口拿大文件。

    contents API 对 >1MB 的文件返回 encoding='none'（空 content），
    但 git/blobs/{sha} 没有这个限制——实测 1.7MB 文件能完整取回。
    """
    j = api(f'{API}/git/blobs/{sha}', timeout=timeout, retries=3)
    if j.get('encoding') != 'base64' or not j.get('content'):
        return 0
    data = base64.b64decode(j['content'])
    if data[:3] != b'GIF':
        return 0
    with open(dest, 'wb') as f:
        f.write(data)
    return len(data)


def quota():
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            'https://api.github.com/rate_limit', headers=H), timeout=20, context=CTX)
        d = json.loads(r.read())['resources']['core']
        return d['remaining'], d['reset']
    except Exception:
        return -1, 0


def load_meta():
    if os.path.exists(META):
        try:
            with open(META, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return []


def main():
    os.makedirs(ROOT, exist_ok=True)
    records = load_meta()
    done = {r['file'] for r in records}

    if '--status' in sys.argv:
        shows = len({r['show'] for r in records})
        rem, reset = quota()
        print(f'已下载 {len(done)} 个 GIF，覆盖 {shows} 部番剧')
        print(f'API 限额剩余 {rem}/60' + (f'，{max(0,(reset-time.time())/60):.0f} 分钟后重置'
                                        if reset else ''))
        return

    rem, reset = quota()
    print(f'[起始] 已下载 {len(done)} | API 剩余额度 {rem}/60')
    if rem <= 0:
        print(f'限额已用完，{max(0,(reset-time.time())/60):.0f} 分钟后再跑。')
        print('（建议：登录 GitHub 后把 token 写进环境变量 GITHUB_TOKEN，限额升到 5000/小时）')
        return

    from PIL import Image
    try:
        items = api(f'{API}/contents/registry')
    except RateLimited:
        print('限额用尽（拉目录就耗尽），稍后重跑。')
        return
    shows = sorted(x['name'] for x in items
                   if x['type'] == 'dir' and not x['name'].startswith('.'))
    print(f'共 {len(shows)} 部番剧，已完成 {len({r["show"] for r in records})} 部\n')

    stats = {'ok': 0, 'skip_big': 0, 'bad_frame': 0, 'bad_size': 0,
             'done_shows': 0, 'blob': 0}
    for si, show in enumerate(shows):
        if si % 5 == 0:
            r_, _ = quota()
            print(f'  [{si+1}/{len(shows)}] {show} | 本轮+{stats["ok"]} | 额度剩 {r_}',
                  flush=True)
        outdir = os.path.join(ROOT, show)
        has_done = any(r['show'] == show for r in records)
        if has_done and os.path.isdir(outdir):
            # 该番剧已处理过，只补缺（简单起见：整部跳过，靠 --force 重来）
            pass
        try:
            files = api(f'{API}/contents/registry/{show}')
        except RateLimited:
            print(f'\n  限额用尽于第 {si+1} 部，已下载部分全部保留。')
            print(f'  {max(0,(reset-time.time())/60):.0f} 分钟后重跑本脚本即可续传。')
            break
        except Exception:
            continue

        gif_names = [x['name'] for x in files
                     if x['type'] == 'file' and x['name'].lower().endswith('.gif')]
        # 番剧级元数据文件名是 <ID>_<NAME>.json（如 10165_Nichijou.json），
        # GIF 级元数据是纯数字名（如 0001.json）。按"去掉 .json 后是否为纯数字"区分。
        meta_map = {}
        for x in files:
            if x['type'] != 'file' or not x['name'].lower().endswith('.json'):
                continue
            stem = x['name'][:-5]
            if stem.isdigit():
                meta_map[stem] = x
        os.makedirs(outdir, exist_ok=True)

        for gname in gif_names:
            rel = f'{show}/{gname}'
            if rel in done:
                continue
            # ★ ent 必须是 **GIF 自己的目录条目**（size/sha 都在这里）。
            #   之前误用 meta_map（那是元数据 .json，size=224 左右），
            #   导致 1.7MB 的 gif 被当成 224B，contents API 返回空 content 后
            #   被误判为「>1MB 跳过」，blob 通道根本没被触发。
            ent = next((x for x in files if x['name'] == gname), {})
            meta_ent = meta_map.get(gname[:-4])
            size = ent.get('size', 0)
            sha = ent.get('sha')
            # contents API 对 >1MB 返回空 content，这类走 git/blobs（无大小限制）
            if size and size > MAX_BYTES:
                stats['skip_big'] += 1
                continue
            dest = os.path.join(outdir, gname)
            if not os.path.exists(dest):
                use_blob = size > 1048576 and sha
                try:
                    if use_blob:
                        n = fetch_blob(sha, dest)
                        if n == 0:
                            stats['skip_big'] += 1
                            continue
                        stats['blob'] += 1
                    else:
                        j = api(f'{API}/contents/registry/{show}/{gname}',
                                timeout=90, retries=2)
                        if j.get('encoding') != 'base64' or not j.get('content'):
                            # contents 拿不到就退回 blob
                            if not sha:
                                stats['skip_big'] += 1
                                continue
                            n = fetch_blob(sha, dest)
                            if n == 0:
                                stats['skip_big'] += 1
                                continue
                            stats['blob'] += 1
                        else:
                            data = base64.b64decode(j['content'])
                            if data[:3] != b'GIF':
                                continue
                            with open(dest, 'wb') as f:
                                f.write(data)
                except RateLimited:
                    print('\n  限额用尽（下载途中），已保存部分，重跑续传。')
                    with open(META, 'w', encoding='utf-8') as f:
                        json.dump(records, f, ensure_ascii=False, indent=1)
                    print(f'  本轮新增 {stats["ok"]} 个，累计 {len(records)} 个。')
                    return
                except Exception:
                    continue
            try:
                with Image.open(dest) as im:
                    nf, (w, h) = getattr(im, 'n_frames', 1), im.size
                    s = min(w, h)
            except Exception:
                stats['bad_frame'] += 1
                continue
            if nf < MIN_FRAMES or nf > MAX_FRAMES:
                stats['bad_frame'] += 1
                os.remove(dest)
                continue
            if s < MIN_SHORT or s > MAX_SHORT:
                stats['bad_size'] += 1
                os.remove(dest)
                continue

            md = {}
            try:
                c = api(f'{API}/contents/registry/{show}/{gname[:-4]}.json',
                        timeout=25, retries=1)
                md = json.loads(base64.b64decode(c['content']).decode('utf-8', 'replace'))
            except Exception:
                pass
            records.append({'show': show, 'file': rel, 'frames': nf,
                            'size': f'{w}x{h}', 'characters': md.get('characters', []),
                            'sentiments': md.get('sentiments', []),
                            'source': md.get('url', '')})
            done.add(rel)
            stats['ok'] += 1
            with open(META, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=1)

        stats['done_shows'] += 1

    with open(META, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    chars = {c for r in records for c in r['characters']}
    sents = {s for r in records for s in r['sentiments']}
    print(f'\n本轮新增 {stats["ok"]} | 累计 {len(records)} 个 | '
          f'覆盖 {len({r["show"] for r in records})} 部番剧')
    print(f'角色 {len(chars)} 个 | 情感标签 {len(sents)} 种')
    print(f'>1MB 跳过 {stats["skip_big"]} | blob 通道下载 {stats["blob"]} | '
          f'帧数不符 {stats["bad_frame"]} | 尺寸不符 {stats["bad_size"]}')


if __name__ == '__main__':
    main()
