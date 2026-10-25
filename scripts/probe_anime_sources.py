# -*- coding: utf-8 -*-
"""二次元/动漫表情包获取途径连通性实测报告。

结论（2026-10-13 实测，本机网络环境）：
  所有二次元专门站（booru 系 / pixiv / niconico / kemono / e-hentai）全部不可达。
  可达的国内站（微博/B站/新浪图床）需要 Referer 才能下图，且无公开 GIF 检索接口。
  → 不需要 API 的自动化批量获取，目前没有可行路径。
"""
import urllib.request
import ssl
import json
import time

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/120.0 Safari/537.36')


def probe(name, url, referer=None, want_img=False):
    h = {'User-Agent': UA, 'Accept-Encoding': 'identity',
         'Accept': 'image/avif,image/webp,image/*,*/*;q=0.8'}
    if referer:
        h['Referer'] = referer
    t = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=h),
                                   timeout=12, context=CTX)
        d = r.read()
        ok = ''
        if want_img:
            ok = 'IMG_OK' if (d[:3] == b'GIF' or d[:2] == b'\xff\xd8'
                              or d[:4] == b'RIFF' or d[:8] == b'\x89PNG\r\n\x1a\n') else 'not_img'
        print(f'  {name:<24} {r.status} {len(d):>7}B {ok:<8} {time.time()-t:.1f}s')
        return True
    except Exception as e:
        print(f'  {name:<24} ERR {str(e)[:44]}  {time.time()-t:.1f}s')
        return False


print('=' * 78)
print('一、二次元专门站（booru / 动漫社区）—— 全部不可达')
print('=' * 78)
for n, u in [
    ('danbooru.donmai.us', 'https://danbooru.donmai.us'),
    ('safebooru.donmai.us', 'https://safebooru.donmai.us'),
    ('nekos.best', 'https://nekos.best'),
    ('gelbooru.com', 'https://gelbooru.com'),
    ('rule34.xxx', 'https://rule34.xxx'),
    ('konachan.com', 'https://konachan.com'),
    ('yande.re', 'https://yande.re'),
    ('sankakucomplex', 'https://idol.sankakucomplex.com'),
    ('pixiv.net', 'https://www.pixiv.net'),
    ('nicovideo.jp', 'https://www.nicovideo.jp'),
    ('kemono.su', 'https://kemono.su'),
    ('e-hentai.org', 'https://e-hentai.org'),
    ('mihiro.pages.dev', 'https://mihiro.pages.dev'),
    ('moeimg.xyz', 'https://moeimg.xyz'),
]:
    probe(n, u)

print()
print('=' * 78)
print('二、国内站（微博/B站/新浪）—— 站点通，但图床需 Referer')
print('=' * 78)
for n, u, ref, img in [
    ('weibo 首页', 'https://weibo.com', None, False),
    ('bilibili 首页', 'https://www.bilibili.com', None, False),
    ('sinaimg 无referer', 'https://wx1.sinaimg.cn/large/006Jy0KBgy1gx2z4y1n0j.jpg', None, True),
    ('sinaimg +referer', 'https://wx1.sinaimg.cn/large/006Jy0KBgy1gx2z4y1n0j.jpg',
     'https://weibo.com/', True),
    ('b站搜索API', 'https://api.bilibili.com/x/web-interface/search/type'
     '?search_type=video&keyword=test', None, False),
    ('微博热搜API', 'https://weibo.com/ajax/side/hotSearch', 'https://weibo.com/', False),
]:
    probe(n, u, ref, img)

print()
print('=' * 78)
print('三、HuggingFace 镜像（可用，需 UA）—— 二次元动图依然没有')
print('=' * 78)
for ds in ['deepghs/safebooru-webp-4Mpixel', 'nyanko-devs/danbooru2026',
           '4eJIoBek/Old-GIFs-22k']:
    try:
        r = urllib.request.urlopen(urllib.request.Request(
            f'https://hf-mirror.com/api/datasets/{ds}',
            headers={'User-Agent': UA, 'Accept': 'application/json'}),
            timeout=25, context=CTX)
        sib = [s['rfilename'] for s in json.loads(r.read()).get('siblings', [])]
        gifs = [s for s in sib if s.lower().endswith('.gif')]
        webp = [s for s in sib if s.lower().endswith('.webp')]
        print(f'  {ds:<36} 总{len(sib):>5} gif={len(gifs):>5} webp={len(webp):>5}')
    except Exception as e:
        print(f'  {ds:<36} ERR {str(e)[:40]}')

print()
print('=' * 78)
print('结论')
print('=' * 78)
print('''
不需要 API 的批量获取二次元动图，在当前网络环境下无可行路径：
  1. booru/pixiv/niconico 等二次元站全部被阻断（403 或隧道超时）
  2. 微博/B站站点可达，但图床需 Referer，且没有公开的 GIF 检索接口
     （微博热搜 API 只返回文字词条 + 3 个 png 图标，无 GIF）
  3. HuggingFace 上的 booru 系数据集只有静态插画和视频，没有 GIF

可行的三条路（需要你选）：
  A. 给我 Giphy/Tenor API key —— 官方接口，可按关键词抓二次元表情包
  B. 你手动打包 —— 把收藏的 GIF 丢进 data/gif_raw/ 任意子目录即可
  C. 用现成 9004 个先训 —— 老素材但数量够，先验证多素材训练是否有效
''')
