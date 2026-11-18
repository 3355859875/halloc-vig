"""按需下载 HalLoc 用到的 Visual Genome 图片。

只下指定切分（默认 val + test）涉及的图片，URL 来自 VG 官方 image_data.json。
已下载且校验通过的图片会跳过，可以反复运行续传。

用法（在作业目录下）：
    ~/.local/bin/uv run --no-project --with pillow --with certifi python src/download_vg_images.py
    ~/.local/bin/uv run --no-project --with pillow --with certifi python src/download_vg_images.py --splits val test --workers 16
"""

import argparse
import csv
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ssl

import certifi
from PIL import Image

# uv 自带的独立 Python 不读 macOS 系统证书，用 certifi 的根证书包做 HTTPS 校验
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUBSETS = ["halloc_caption.json", "halloc_instruct.json", "halloc_vqa.json"]


def needed_image_ids(splits):
    ids = set()
    for name in SUBSETS:
        for x in json.load(open(DATA / name, encoding="utf-8")):
            if x["split"] in splits:
                ids.add(int(x["image_id"]))
    return ids


def check_image(path, width, height):
    """返回 None 表示通过，否则返回问题描述。"""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
    except Exception as e:  # 损坏或不是图片
        return f"无法解码: {e}"
    if (w, h) != (width, height):
        return f"尺寸不符: {w}x{h}，元数据 {width}x{height}"
    return None


def download_one(meta, out_dir, retries=3):
    path = out_dir / f"{meta['image_id']}.jpg"
    if path.exists() and check_image(path, meta["width"], meta["height"]) is None:
        return meta["image_id"], "已存在", None
    err = None
    for i in range(retries):
        try:
            req = urllib.request.Request(meta["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=SSL_CTX) as r:
                data = r.read()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            err = check_image(tmp, meta["width"], meta["height"])
            if err is None:
                tmp.rename(path)
                return meta["image_id"], "下载", None
            tmp.unlink(missing_ok=True)
        except Exception as e:
            err = str(e)
        time.sleep(1 + i)
    return meta["image_id"], "失败", err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(DATA / "vg_images"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_all = {m["image_id"]: m for m in json.load(open(DATA / "vg_meta" / "image_data.json"))}
    ids = needed_image_ids(set(args.splits))
    missing_meta = sorted(i for i in ids if i not in meta_all)
    todo = [meta_all[i] for i in sorted(ids) if i in meta_all]
    print(f"需要 {len(ids)} 张，元数据缺失 {len(missing_meta)} 张")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(download_one, m, out_dir) for m in todo]
        for k, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if k % 500 == 0:
                print(f"  {k}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)

    status = {}
    for _, s, _ in results:
        status[s] = status.get(s, 0) + 1
    print("结果:", status, f"用时 {time.time() - t0:.0f}s")

    with open(DATA / "vg_images_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "status", "error"])
        for row in sorted(results):
            w.writerow(row)
        for i in missing_meta:
            w.writerow([i, "无元数据", ""])


if __name__ == "__main__":
    main()
