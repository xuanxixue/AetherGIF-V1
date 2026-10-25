# -*- coding: utf-8 -*-
"""系统性搜索 GitHub 上所有二次元/表情包 GIF 源并统计真实 GIF 数量。

用 git/trees/{branch}?recursive=1 一次拿全量文件清单（比逐目录遍历快百倍），
按扩展名统计 gif/webp 数量，并按目录前缀聚类看分类粒度。
"""
import os
import ssl
import json
import time
import urllib.request
from collections import Counter

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
TOK = open('D:/AetherGIF-V1/scripts/github_token.txt', encoding='utf-8').read().strip()
H = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/vnd.github+json',
     'Authorization': 'Bearer ' + TOK, 'Accept-Encoding': 'identity'}

KEYWORDS = [
    'anime+gif', 'anime+gifs', 'anime+meme', 'anime+reaction', 'anime+emote',
    'chibi+meme', 'waifu+meme', 'anime+sticker', 'kawaii+meme', 'moe+meme',
    'anime+discord+emote', 'tenor+anime', 'giphy+anime', 'anime+emotes',
    'manga+meme', 'anime+animated+images', 'nijigasana', '表情包',
    'anime+booru+gif', 'donghua', 'anime+loop+gif', 'reaction+images',
    'anime+avatar+gif', 'senpai+meme', 'kawaii+animated', 'otaku+gif',
]


def api(url, timeout=90):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(url, headers=H), timeout=timeout, context=CTX).read())


def search_repos(q, per=10):
    try:
        r = api('https://api.github.com/search/repositories?q=%s&sort=stars&per_page=%d'
                % (q, per))
        return r.get('items', [])
    except Exception:
        return []


def tree_stats(repo):
    """返回 (gif数, webp数, gif路径列表, 总字节) 或 None。"""
    try:
        info = api('https://api.github.com/repos/%s' % repo)
        br = info['default_branch']
        t = api('https://api.github.com/repos/%s/git/trees/%s?recursive=1' % (repo, br))
        if t.get('truncated'):
            print('    [warn] tree 被截断，数量不准')
        tree = t.get('tree', [])
        gifs = [x for x in tree if x['type'] == 'blob' and x['path'].lower().endswith('.gif')]
        webps = [x for x in tree if x['type'] == 'blob' and x['path'].lower().endswith('.webp')]
        size = sum(x.get('size', 0) for x in gifs + webps)
        return len(gifs), len(webps), [x['path'] for x in gifs], size
    except Exception as e:
        return None


def main():
    seen = {}
    for q in KEYWORDS:
        for x in search_repos(q):
            seen[x['full_name']] = (x['stargazers_count'], x['size'],
                                    (x.get('license') or {}).get('spdx_id'))
        time.sleep(1.1)
    print('搜索到 %d 个候选仓库\n' % len(seen))

    # 按 star 过滤明显无关的（bot/dotfiles/GUI 等）
    skip_kw = ('gui', 'bot', 'dotfiles', 'extension', 'waifu2x', 'selfbot',
               'music', 'player', 'streaming', 'desktop', 'wallpaper', 'live2d')
    cands = []
    for repo, (st, sz, lic) in sorted(seen.items(), key=lambda x: -x[1][0]):
        low = repo.lower()
        if any(k in low for k in skip_kw):
            continue
        cands.append((repo, st, sz, lic))
    print('过滤后候选 %d 个，逐个查真实 GIF 数量：\n' % len(cands))

    results = []
    for i, (repo, st, sz, lic) in enumerate(cands[:40]):
        s = tree_stats(repo)
        if s is None:
            continue
        ng, nw, paths, size = s
        if ng + nw < 30:
            continue
        results.append((repo, st, ng, nw, size, lic))
        print('  %-44s star=%-5s gif=%-5d webp=%-4d %.0fMB  %s'
              % (repo, st, ng, nw, size / 1e6, lic or ''), flush=True)

    print('\n' + '=' * 88)
    print('GIF 数量排名')
    print('=' * 88)
    for repo, st, ng, nw, size, lic in sorted(results, key=lambda x: -(x[2] + x[3])):
        print('  %-46s gif+webp = %d' % (repo, ng + nw))
    print('\n合计（去重前）: %d 个 gif/webp' % sum(r[2] + r[3] for r in results))
    out = 'D:/AetherGIF-V1/scripts/_libs/_anime_repo_candidates.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump([{'repo': r[0], 'stars': r[1], 'gif': r[2], 'webp': r[3],
                    'bytes': r[4], 'license': r[5]} for r in results],
                  f, ensure_ascii=False, indent=1)
    print('候选清单 ->', out)


if __name__ == '__main__':
    main()
