"""用 CLIP 图像向量给每张图找“长得像”的近邻图，作为 vig_score.py 的 near 对照。

近邻只在同一切分的图片池里找（val 找 val、test 找 test），只用图片本身，不看任何标注。
相似度高于 --max-sim 的视为近似重复图，跳过。

输出 <out>.json：{image_id: [[近邻 id, 余弦相似度], ...]}，按相似度从高到低；
同时保存图像向量 <out>.npy 和 id 顺序，方便以后复用（例如做演示里的检索）。

用法：
    .venv/bin/python src/clip_neighbors.py --split val --clip ~/models/clip-vit-large-patch14 --out results/clip_nn_val
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SUBSETS = ["halloc_caption.json", "halloc_instruct.json", "halloc_vqa.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-sim", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids = sorted({x["image_id"] for f in SUBSETS for x in json.load(open(DATA / f, encoding="utf-8"))
                  if x["split"] == args.split}, key=int)
    model = CLIPModel.from_pretrained(args.clip, dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(args.clip)

    embs = []
    with torch.no_grad():
        for i in range(0, len(ids), args.batch):
            ims = [Image.open(DATA / "vg_images" / f"{j}.jpg").convert("RGB") for j in ids[i : i + args.batch]]
            px = proc(images=ims, return_tensors="pt")["pixel_values"].cuda().half()
            e = model.get_image_features(pixel_values=px)
            if not torch.is_tensor(e):  # 新版 transformers 可能返回输出对象
                e = e.pooler_output
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
    E = torch.cat(embs).numpy()
    S = E @ E.T
    np.fill_diagonal(S, -1)

    nn, n_dup = {}, 0
    for a, row in zip(ids, S):
        order = np.argsort(-row)
        keep = []
        for j in order:
            if row[j] > args.max_sim:
                n_dup += 1
                continue
            keep.append([ids[j], round(float(row[j]), 4)])
            if len(keep) == args.k:
                break
        nn[a] = keep

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(nn))
    np.save(out.with_suffix(".npy"), E)
    out.with_name(out.name + "_ids.json").write_text(json.dumps(ids))
    top1 = [v[0][1] for v in nn.values()]
    topk = [v[-1][1] for v in nn.values()]
    rand = S[np.triu_indices(len(ids), 1)]
    print(f"{args.split}: {len(ids)} 张；跳过近似重复 {n_dup} 次；第 1 近邻相似度中位数 {np.median(top1):.3f}，"
          f"第 {args.k} 近邻 {np.median(topk):.3f}，随机两张图 {np.median(rand):.3f}")


if __name__ == "__main__":
    main()
