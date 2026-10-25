# -*- coding: utf-8 -*-
"""从 parquet 提取内嵌 GIF（修正版：image 是嵌套 struct）。

该数据集自带元数据：decoded_frames / total_duration_ms / color_count / tags，
正好用来做精细筛选 + 分类打标。
"""
import os
import io
import sys

sys.path.insert(0, r'D:\AetherGIF-V1\scripts\_libs')
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

PARQUET = 'D:/AetherGIF-V1/data/gif_raw/_packs/pixelart_1k.parquet'
OUT = 'D:/AetherGIF-V1/data/gif_raw/bulk_pixelart'

MIN_FRAMES, MAX_FRAMES = 6, 120
MIN_SHORT, MAX_SHORT = 96, 800


def tag_slug(tags):
    if not tags:
        return 'misc'
    t = [str(x).lower() for x in tags]
    for kw, name in [('pixel', 'pixel'), ('sprite', 'sprite'),
                     ('char', 'character'), ('ani', 'anime'),
                     ('emoji', 'emoji'), ('icon', 'icon'),
                     ('game', 'game'), ('animal', 'animal')]:
        if any(kw in x for x in t):
            return name
    return 'misc'


def main():
    os.makedirs(OUT, exist_ok=True)
    pf = pq.ParquetFile(PARQUET)
    print(f'[PARQUET] 行数={pf.metadata.num_rows}')
    total = kept = 0
    tagcount = {}
    for batch in pf.iter_batches(batch_size=64):
        d = batch.to_pydict()
        imgs = d.get('image') or []
        tags = d.get('tags') or []
        for i, item in enumerate(imgs):
            raw = (item or {}).get('bytes') if isinstance(item, dict) else None
            if not raw:
                continue
            total += 1
            try:
                im = Image.open(io.BytesIO(raw))
                n = getattr(im, 'n_frames', 1)
                if n < MIN_FRAMES or n > MAX_FRAMES:
                    continue
                s = min(im.size)
                if s < MIN_SHORT or s > MAX_SHORT:
                    continue
                tg = tag_slug(tags[i] if i < len(tags) else None)
                tagcount[tg] = tagcount.get(tg, 0) + 1
                sub = os.path.join(OUT, tg)
                os.makedirs(sub, exist_ok=True)
                with open(os.path.join(sub, f'pa_{total:05d}.gif'), 'wb') as f:
                    f.write(raw)
                kept += 1
            except Exception:
                continue
    print(f'[PARQUET] GIF {total} 个，合格 {kept} 个 -> {OUT}')
    print('[分类] ' + '  '.join(f'{k}={v}' for k, v in sorted(tagcount.items(), key=lambda x: -x[1])))


if __name__ == '__main__':
    main()
