"""图 1：方法示意图。用 val 里的一个真实例子，数字从结果文件读出（不手填）。

    原图 + 描述 ──┐
                  ├─> 同一个 VLM（不训练），teacher forcing 逐词取 log p
    K 张无关图 ───┘
    VIG(w) = 平均_k log p(w | 无关图_k) − log p(w | 原图)，越高越可疑

数字用演示网页同一个核心模块（src/demo/core.py）算：每个词取 VIG 最大的 token，
表里的“原图 / 无关图平均”就是这个 token 的值，所以 VIG = 两者之差；判断用 val 上冻结的阈值。

例子按规则自动挑（不手选）：按样本 id 排序，取第一条同时满足
    ① 样本内 VIG 最高的实词是标注的幻觉词，且超过阈值；
    ② 有一个真实的罕见词：原图下 log p < −6，但 VIG < −1.5（只看概率会误报，VIG 能排除）；
    ③ 描述不超过 70 个词。
第三个词取原图下 log p 最高的真实实词（普通词）。

用法：
    uv run --no-project --with numpy --with pandas --with scikit-learn --with matplotlib --with pillow \
        python src/viz_method.py results/20261117_dev_multi_qwen25vl3b_val500 --out report/图/方法示意图.pdf
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "demo"))
import core  # noqa: E402
plt.rcParams.update({"pdf.fonttype": 42, "font.family": ["Arial Unicode MS", "PingFang SC", "sans-serif"]})
BLUE, ORANGE, GRAY, RED = "#343a40", "#3b5bdb", "#495057", "#c92a2a"  # 原图 = 深灰，随机无关图 = 蓝（与其他图一致）


def thumb(iid, size=300):
    im = Image.open(ROOT / "data" / "vg_images" / f"{iid}.jpg").convert("RGB")
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size))


def pick(ref, want_id=None):
    for key, (name, rec) in sorted(ref.examples.items()):
        if want_id and rec["id"] != want_id:
            continue
        ex = ref.analyze(rec, name)
        cw = [x for x in ex["words"] if x["content"]]
        if not cw:
            continue
        top = max(cw, key=lambda x: x["vig"])
        rare = [x for x in cw if not x["label"] and x["logp_real"] < -6 and x["vig"] < -1.5]
        if want_id or (top["label"] and top["flag"] and rare and len(rec["text"].split()) <= 70):
            common = max((x for x in cw if not x["label"]), key=lambda x: x["logp_real"])
            return rec, [top, min(rare, key=lambda x: x["vig"]) if rare else common, common]
    raise SystemExit("没有满足条件的样本")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--id", help="指定样本（默认按规则自动挑）")
    ap.add_argument("--protocol", default=str(ROOT / "results/final/protocol_20261118.json"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    ref = core.Reference(run, [("val", run)], args.protocol)
    rec, picked = pick(ref, args.id)
    words = [x["word"] for x in picked]
    print("例子：", rec["id"], [(x["word"], x["logp_real"], x["logp_other"], x["vig"], x["flag"], x["label"]) for x in picked])

    W, H = 7.6, 3.9  # 英寸；画布坐标 x∈[0,100]、y∈[0,52]
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def img(iid, x, y, w, edge):
        h = w * (W / H) * (52 / 100)  # 正方形在画布坐标里的高度
        a = fig.add_axes([x / 100, y / 52, w / 100, h / 52])
        a.imshow(thumb(iid))
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_edgecolor(edge)
            sp.set_linewidth(1.5)
        return h

    def box(x, y, w, h, text, fs=7.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#f1f3f5", ec=GRAY, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, color=GRAY):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color=color, lw=1.1))

    # 左上：原图 + 描述片段
    h = img(rec["image_id"], 2, 27, 12, BLUE)
    ax.text(8, 27 + h + 1, "原图", ha="center", fontsize=8.5, color=BLUE, fontweight="bold")
    m = re.search(r"\b" + words[0] + r"\b", rec["text"])
    a = rec["text"].rfind(" ", 0, max(0, m.start() - 40)) + 1  # 从完整单词开始
    b = rec["text"].find(" ", m.end() + 25)
    snip = "…" + rec["text"][a:b if b > 0 else None].strip() + "…"
    ax.text(16, 27 + h + 1, "描述（待检查的文字）", fontsize=7.8, color=GRAY)
    ax.text(16, 27 + h - 1, textwrap.fill(snip, 30), fontsize=7.4, va="top", fontfamily="Arial", linespacing=1.3)
    # 左下：K 张无关图
    for k, oid in enumerate(rec["other_image_ids"][: args.k]):
        hh = img(oid, 2 + k * 8, 4, 7, ORANGE)
    ax.text(2, 4 + hh + 1.2, f"{args.k} 张随机无关图（同一切分里随机抽）", fontsize=8, color=ORANGE, fontweight="bold")

    # 中：同一个 VLM
    box(39, 29, 15, 10, "同一个 VLM\n（Qwen2.5-VL，不训练）\n逐词取 log p")
    box(39, 5, 15, 10, "同一个 VLM\n同一段文字，只换图\n逐词取 log p")
    arrow(36, 34, 38.4, 34, BLUE)
    arrow(34, 10, 38.4, 10, ORANGE)
    ax.text(46.5, 25.5, "log p(w | 原图)", ha="center", fontsize=7.6, color=BLUE)
    ax.text(46.5, 1.6, "平均 log p(w | 无关图)", ha="center", fontsize=7.6, color=ORANGE)

    # 右：数字表
    arrow(55, 33, 58.5, 31, BLUE)
    arrow(55, 11, 58.5, 20, ORANGE)
    x0, y0 = 59.5, 45
    for name, dx in [("词", 0), ("log p\n原图", 9), ("log p\n无关图平均", 16.5), ("VIG", 27), ("判断", 33.5)]:
        ax.text(x0 + dx, y0, name, fontsize=7.4, fontweight="bold", va="bottom", ha="left")
    ax.plot([x0, 99], [y0 - 0.6] * 2, color=GRAY, lw=0.6)
    for i, x in enumerate(picked):
        y = y0 - 4.2 - i * 5.6
        col = RED if x["flag"] else "#2b8a3e"
        ax.text(x0, y, x["word"], fontsize=8, fontfamily="Arial", fontweight="bold", color=RED if x["label"] else "black")
        ax.text(x0, y - 2.2, "标注：幻觉" if x["label"] else "标注：真实", fontsize=6.4, color=GRAY)
        ax.text(x0 + 9, y, f"{x['logp_real']:.2f}", fontsize=7.8, color=BLUE)
        ax.text(x0 + 16.5, y, f"{x['logp_other']:.2f}", fontsize=7.8, color=ORANGE)
        ax.text(x0 + 27, y, f"{x['vig']:+.2f}", fontsize=7.8, fontweight="bold", color=col)
        ax.text(x0 + 33.5, y, "可疑" if x["flag"] else "不可疑", fontsize=7.4, color=col)
    ax.text(x0, 23, "VIG(w) = 平均 log p(w | 无关图) − log p(w | 原图)", fontsize=7.6, fontweight="bold")
    ax.text(x0, 20, f"越高越可疑；VIG ≥ {ref.threshold:.2f}（val 上定的阈值）判为可疑", fontsize=7.2, color=GRAY)
    h_, r_ = picked[0], picked[1]
    note = [f"例：{r_['word']} 在原图下 log p 只有 {r_['logp_real']:.1f}，只看概率会被当成可疑；",
            f"但它在无关图下更不可能（{r_['logp_other']:.1f}），说明原图支持它，VIG 把它排除了。",
            f"{h_['word']} 正相反：原图下（{h_['logp_real']:.1f}）比无关图下（{h_['logp_other']:.1f}）",
            "更不可能出现，所以被判为可疑——它确实是标注的幻觉。",
            "（表中数字取每个词里 VIG 最大的那个 token）"]
    ax.text(x0, 15.5, "\n".join(note), fontsize=6.9, color=GRAY, va="top", linespacing=1.4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"图已保存 {args.out}")


if __name__ == "__main__":
    main()
