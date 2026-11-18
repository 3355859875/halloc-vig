"""图 5：近邻图为什么不如随机无关图。

自动挑幻觉词（核心词口径、实词、单个 token 最好）：随机无关图对照下很可疑、近邻图对照下不可疑，差得最多的几个，
每个画一行：原图 | 3 张 CLIP 近邻图 | 3 张随机无关图，每张图下标出模型在该图条件下给这个词的对数概率。
直观看到：近邻图和原图一样“没有”这个东西，概率同样低，对比被抵消；随机图里反而偶尔会有。

用法：
    uv run --no-project --with numpy --with pandas --with matplotlib --with pillow python src/viz_near.py \
        results/20261117_dev_multi_qwen25vl3b_val500 --rows 2 --out report/图/近邻图失败示例.pdf
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"pdf.fonttype": 42, "font.family": ["Arial Unicode MS", "PingFang SC", "sans-serif"]})
TYPE_ZH = {"object": "物体", "attribute": "属性", "relationship": "关系", "scene": "场景"}


def thumb(path, size=360):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))  # 中心裁成正方形
    return im.resize((size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    w = pd.read_csv(run / "words_core.csv")
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(run / "scores.jsonl", encoding="utf-8")}

    pos = w[(w.label == 1) & (w.content == 1)].copy()
    pos["gap"] = pos["vig_other4@max"] - pos["vig_near4@max"]
    picks, seen = [], set()
    for r in pos.sort_values("gap", ascending=False).itertuples():  # 每类型最多 1 个、每样本 1 个
        t = r.type.split("|")[0]
        if r.id in seen or t in {p[3] for p in picks}:
            continue
        picks.append((r, recs[r.id], TYPE_ZH[t], t))
        seen.add(r.id)
        if len(picks) == args.rows:
            break

    fig, axes = plt.subplots(len(picks), 7, figsize=(7.6, 1.6 * len(picks) + 0.5))
    for row, (r, rec, tzh, _) in enumerate(picks):
        offs = rec["offsets"]
        tix = [i for i, (s, e) in enumerate(offs) if s < r.start + len(r.word) and e > r.start]
        lp = lambda cond: max(-1e9, min(rec["logp"][cond][i] for i in tix))  # 取词内最低（与 @max 口径一致）
        panels = [("原图", rec["image_id"], "real")] + \
                 [(f"近邻 {j + 1}", rec["near_image_ids"][j], f"near_{j}") for j in range(3)] + \
                 [(f"随机 {j + 1}", rec["other_image_ids"][j], f"other_{j}") for j in range(3)]
        for col, (name, iid, cond) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(thumb(ROOT / "data" / "vg_images" / f"{iid}.jpg"))
            ax.set_xticks([])
            ax.set_yticks([])
            color = "#1c3faa" if cond == "real" else ("#e8590c" if cond.startswith("near") else "#3b5bdb")
            for s in ax.spines.values():
                s.set_edgecolor(color)
                s.set_linewidth(1.6 if cond == "real" else 0.8)
            ax.set_xlabel(f"{name}\nlog p = {lp(cond):.1f}", fontsize=7, color=color, labelpad=2)
        # 句子上下文：幻觉词前后各取一些字符
        text = rec["text"]
        a, b = max(0, r.start - 45), min(len(text), r.start + len(r.word) + 30)
        ctx = ("…" if a else "") + text[a:r.start] + "【" + r.word + "】" + text[r.start + len(r.word):b] + ("…" if b < len(text) else "")
        ctx = re.sub(r"\s+", " ", ctx)
        axes[row, 0].set_title(f"{tzh}幻觉「{r.word}」：{ctx}", fontsize=7.4, loc="left", pad=4)
    fig.text(0.5, 0.005, "log p = 模型在该图条件下给幻觉词的对数概率（越高 = 越“相信”图里有）。"
             "近邻图和原图一样低 → 对比被抵消；随机图有时更高 → 对比能揭露幻觉。",
             ha="center", va="bottom", fontsize=7.2, color="#495057")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.12, wspace=0.08, hspace=0.75)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print("示例：", [(p[0].id, p[0].word, p[2], round(p[0].gap, 2)) for p in picks])
    print(f"图已保存 {args.out}")


if __name__ == "__main__":
    main()
