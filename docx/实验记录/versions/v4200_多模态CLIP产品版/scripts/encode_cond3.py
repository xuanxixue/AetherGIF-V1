# -*- coding: utf-8 -*-
"""三向量条件缓存 —— 文本描述 / GIF 首帧静图 / GIF 本身（真视频）三路对齐向量。

★ 用户方案（2026-10-16）
  用 int8 量化版 Qwen3-VL-Embedding-2B 对同一个 GIF 提取三路向量：
    ① 文本描述向量（详细描述）
    ② 首帧图像向量（静态单图）
    ③ GIF 本身向量（文件路径直传，真视频分支：真 fps + 时序重采样）
  三个向量统一喂给模型训练（交叉训练，见 overfit_cond3.py），让三条对齐
  向量耦合同一个 GIF。

★ ★ 2026-10-16 修正：GIF 一路必须走真视频分支（encode_video_files）
    旧的 encode_gif_paths 把 GIF 解成帧列表走 list 分支，**没有时序重采样**，
    丢动速信息；更早的路径直传则因 torchvision 坏掉被兜底成 NULL 空文本
    （两素材向量完全相同）。现在 PyAV 后端已注入 int8_qwen_encoder，
    encode_video_files = 路径直传 + PyAV 真视频管线。

本脚本产出 data/clip_cache/ 下的三路独立向量 + 预拼好的 cond_vec3：
  - text3.npy / first3.npy / gif3_video.npy  ：各 [2, 2048]（L2 归一原始向量）
        ※ gif3_video.npy = 真视频分支向量（PyAV 真 fps + 时序重采样）
        ※ gif3.npy 是历史的**帧序列**向量（encode_gif_paths 产物），
          作为旧口径保留不被本脚本覆盖；cv3_* 缓存一律用 gif3_video 重建
          （见 rebuild_cv3_video.py）
  - cv3_full.npy / cv3_text.npy / cv3_first.npy / cv3_gif.npy / cv3_tf.npy
        ：预拼好的不同缺省组合，[2, 6147]（gif 段=真视频向量）
  - three_way_sim.json                        ：三路区分度体检

运行：
  cd 到 v4200 根目录后
  C:/Python312/python.exe -u scripts/encode_cond3.py
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_PROJ = ROOT
for _ in range(8):
    if (_PROJ / "deps312").is_dir():
        break
    if _PROJ.parent == _PROJ:
        break
    _PROJ = _PROJ.parent
sys.path.insert(0, str(_PROJ))
_DEPS = _PROJ / "deps312"
if _DEPS.is_dir():
    sys.path.insert(0, str(_DEPS))
_bad = r"C:\Python312\Lib\site-packages\transformers"
try:
    sys.path = [p for p in sys.path
                if os.path.normcase(os.path.abspath(p)) != os.path.normcase(os.path.abspath(_bad))]
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from PIL import Image, ImageSequence
from int8_qwen_encoder import Int8QwenEncoder
from data.clip_condition import build_cond_vector3, CLIP_EMB_DIM

# 文本描述必须与 overfit_cond3.py 完全一致
GIF_SPECS = [
    ("8-ball_1f3b1.gif", "一张透明背景的台球\"8号球\"矢量图标：深灰黑色球体，左上方有一个白色圆形区域，"
     "白圈正中是黑色粗体无衬线数字\"8\"，球体左上角有高光、右下方有深色弧线阴影，"
     "整体呈扁平图标风格但有 3D 立体感"),
    ("3D87B14FBB731ED3A948AC6DD9EA6EBC.gif", "一张二次元Q版动漫表情包：黑色长直发少女，齐刘海，"
     "左侧头发绑红色蝴蝶结发带，闭着倒V字型\"^ ^\"的眼睛，眼角有汗滴和情绪波动线，"
     "脸颊泛红带三条斜线，嘴巴微张成小U型，双手举在头部两侧握拳挥舞，"
     "身穿深色水手服领制服、白色条纹衣领、胸前红色领结、细银色十字架项链，"
     "图片顶部有手写风格粗体艺术字\"前辈前辈~\"，末尾波浪号拉长"),
]

gif_dir = ROOT / "data" / "gif_raw" / "noto"
OUT = ROOT / "data" / "clip_cache"
OUT.mkdir(parents=True, exist_ok=True)


def first_frame(path):
    with Image.open(path) as im:
        for fr in ImageSequence.Iterator(im):
            return fr.convert("RGB")


def cos(a, b):
    a = a / np.clip(np.linalg.norm(a), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b), 1e-8, None)
    return float(a @ b)


def main():
    enc = Int8QwenEncoder(max_frames=8)
    names = [n for n, _ in GIF_SPECS]
    descs = [d for _, d in GIF_SPECS]
    paths = [str(gif_dir / n) for n in names]

    print("[1/3] 文本描述向量")
    text3 = enc.encode_texts(descs, batch_size=2)

    print("[2/3] 首帧静图向量")
    first3 = enc.encode_images([first_frame(p) for p in paths], batch_size=1)

    print("[3/3] GIF 本身（真视频分支：PyAV 真 fps + 时序重采样）")
    gif3 = enc.encode_video_files(paths, batch_size=1)

    print("\n=== 三路区分度体检（src0 vs src1 余弦，越小越可分）===")
    tabs = {"文本": text3, "首帧": first3, "GIF本身": gif3}
    sim_report = {}
    for k, v in tabs.items():
        c = cos(v[0], v[1])
        sim_report[f"{k}_cos"] = c
        print(f"  {k}: cos={c:+.4f}")

    # 交叉对齐体检：文本→首帧→GIF 是否同一素材对齐
    for a_name, a in [("text", text3), ("first", first3), ("gif", gif3)]:
        for b_name, b in [("text", text3), ("first", first3), ("gif", gif3)]:
            if a_name == b_name:
                continue
            diag = cos(a[0], b[0]) + cos(a[1], b[1])
            off = cos(a[0], b[1]) + cos(a[1], b[0])
            sim_report[f"{a_name}~{b_name}_diag_minus_off"] = float(diag - off)

    # 预拼 cond_vec3 各缺省组合
    zero = np.zeros_like(text3)
    one = np.ones((len(names), 1), dtype=np.float32)
    zz = np.zeros((len(names), 1), dtype=np.float32)
    cv3_full = build_cond_vector3(text3, first3, gif3, (one, one, one))
    cv3_text = build_cond_vector3(text3, zero, zero, (one, zz, zz))
    cv3_first = build_cond_vector3(zero, first3, zero, (zz, one, zz))
    cv3_gif = build_cond_vector3(zero, zero, gif3, (zz, zz, one))
    cv3_tf = build_cond_vector3(text3, first3, zero, (one, one, zz))

    np.save(OUT / "text3.npy", text3)
    np.save(OUT / "first3.npy", first3)
    np.save(OUT / "gif3_video.npy", gif3)
    np.save(OUT / "cv3_full.npy", cv3_full)
    np.save(OUT / "cv3_text.npy", cv3_text)
    np.save(OUT / "cv3_first.npy", cv3_first)
    np.save(OUT / "cv3_gif.npy", cv3_gif)
    np.save(OUT / "cv3_tf.npy", cv3_tf)
    with open(OUT / "three_way_sim.json", "w", encoding="utf-8") as f:
        json.dump({
            "names": names,
            "n_cond3": CLIP_EMB_DIM * 3 + 3,
            "max_frames": 8,
            **sim_report,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] 三路向量已缓存到 {OUT}")
    print(f"  cond_vec3 维度 = {cv3_full.shape[1]} (= 2048*3 + 3 flags)")
    print("  评测缺省组合：cv3_full / cv3_text / cv3_first / cv3_gif / cv3_tf")


if __name__ == "__main__":
    main()