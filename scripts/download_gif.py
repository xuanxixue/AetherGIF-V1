# -*- coding: utf-8 -*-
"""
下载 Noto Emoji 动画 GIF（主数据源，CC BY 4.0，最贴合"线条表情包"定位）。
国内可直连 fonts.gstatic.com；若不通则走镜像 mirror.kekse.biz。
输出到 data/gif_raw/noto/
"""
import os, sys, json, time, urllib.request, urllib.error, pathlib

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "gif_raw" / "noto"
RAW_DIR.mkdir(parents=True, exist_ok=True)

API_URLS = [
    "https://googlefonts.github.io/noto-emoji-animation/data/api.json",
    "https://mirror.kekse.biz/noto-emoji-animation/api.json",
]

def fetch_api():
    for url in API_URLS:
        try:
            print(f"[api] 尝试 {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
            print(f"[api] 成功，共 {len(data.get('icons', []))} 个表情")
            return data
        except Exception as e:
            print(f"[api] 失败: {e}")
    return None

def download_one(codepoint, name, out_dir, size="512"):
    """下载单个 GIF。size 可选 512 / 256 / 128。"""
    fn = out_dir / f"{name}_{codepoint}.gif"
    if fn.exists() and fn.stat().st_size > 0:
        return "skip"
    url = f"https://fonts.gstatic.com/s/e/notoemoji/latest/{codepoint}/{size}.gif"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            blob = r.read()
        if len(blob) < 100:
            return "tiny"
        fn.write_bytes(blob)
        return "ok"
    except urllib.error.HTTPError as e:
        return f"http{e.code}"
    except Exception as e:
        return f"err:{type(e).__name__}"

def main():
    api = fetch_api()
    if api is None:
        print("[fatal] API 全部不可达，退出")
        sys.exit(1)

    icons = api.get("icons", [])
    results = {"ok": 0, "skip": 0, "fail": 0}
    for i, icon in enumerate(icons):
        cp = icon.get("codepoint", "")
        tags = icon.get("tags", [])
        name = tags[0].strip(":") if tags else cp
        # 文件名安全化
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        st = download_one(cp, safe, RAW_DIR)
        if st == "ok":
            results["ok"] += 1
        elif st == "skip":
            results["skip"] += 1
        else:
            results["fail"] += 1
        if (i + 1) % 50 == 0:
            print(f"[progress] {i+1}/{len(icons)}  ok={results['ok']} skip={results['skip']} fail={results['fail']}")
            time.sleep(0.3)

    print(f"[done] 总计 {len(icons)}，成功 {results['ok']}，跳过 {results['skip']}，失败 {results['fail']}")
    print(f"[done] 输出目录: {RAW_DIR}")

if __name__ == "__main__":
    main()
