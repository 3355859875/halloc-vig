"""案例图：原图 + 描述逐词上色（越红越可疑），标注的幻觉词加粗并画下划线。

自动、可复现地挑 4 类案例（只按规则挑，不手选）：
    (a) 命中·关系类：含关系类幻觉，且样本内得分最高的实词就是幻觉词
    (b) 命中·物体或属性类：同上
    (c) 干净样本：没有幻觉，取“最高分”处于中位数的那条，看典型的误报程度
    (d) 漏检：含幻觉，但幻觉词在样本内排名最靠后的那条
颜色 = 该词分数在整次运行全部实词里的分位数，只给实词上色。

用法：
    uv run --no-project --with numpy --with pandas --with matplotlib --with pillow python src/viz_cases.py \
        results/<目录> --score "nll+vig_other4@max" --out report/图/案例_val.pdf
"""

import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from PIL import Image

matplotlib.colorbar.Colorbar.n_rasterize = 10**9  # 色条保持矢量（见 资料/笔记/技术/文档处理/matplotlib论文图保持纯矢量.md）
_orig_add_solids = matplotlib.colorbar.Colorbar._add_solids


def _vec_solids(self, *a, **k):  # 消除矢量色条的细缝
    _orig_add_solids(self, *a, **k)
    if getattr(self, "solids", None) is not None:
        self.solids.set_edgecolor("face")
        self.solids.set_linewidth(0.3)


matplotlib.colorbar.Colorbar._add_solids = _vec_solids

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams.update({"pdf.fonttype": 42, "font.family": ["Arial Unicode MS", "PingFang SC", "sans-serif"]})
TYPE_ZH = {"object": "物体", "attribute": "属性", "relationship": "关系", "scene": "场景"}
# 前 70% 基本不上色，越靠前越红，让真正突出的词跳出来
CMAP = colors.LinearSegmentedColormap.from_list(
    "sus", [(0, "#ffffff"), (0.7, "#fff4e6"), (0.85, "#ffc078"), (0.95, "#fa5252"), (1, "#c92a2a")])
TEXT_FONT = "Arial"  # 英文正文用 Arial（有粗体）


def pick_cases(w, max_words):
    c = w[w.content == 1].copy()
    c["rank"] = c.groupby("id")["score"].rank(ascending=False, method="first")
    n_words = w.groupby("id").size()
    ok_len = set(n_words[n_words <= max_words].index)
    pos = c[c.label == 1]
    best_pos_rank = pos.groupby("id")["rank"].min()
    types = pos.groupby("id")["type"].apply(lambda s: set("|".join(s).split("|")))
    cases = []
    # 命中：样本内得分第 1 的实词就是幻觉词，按这个词本身的类型分组
    top1 = c[c["rank"] == 1].set_index("id")
    hit = [i for i in best_pos_rank.index if best_pos_rank[i] == 1 and i in ok_len]
    rel = [i for i in hit if "relationship" in top1.loc[i, "type"]]
    oa = [i for i in hit if set(top1.loc[i, "type"].split("|")) & {"object", "attribute"}]
    if rel:
        cases.append(("(a) 命中：关系类幻觉", sorted(rel)[0]))
    if oa:
        cases.append(("(b) 命中：物体 / 属性类幻觉", sorted(oa)[0]))
    clean_ids = [i for i in c["id"].unique() if i not in best_pos_rank.index and i in ok_len]
    if clean_ids:
        mx = c[c.id.isin(clean_ids)].groupby("id")["score"].max().sort_values()
        cases.append(("(c) 无幻觉样本：看误报", mx.index[len(mx) // 2]))
    miss = best_pos_rank[[i for i in best_pos_rank.index if i in ok_len]].sort_values()
    if len(miss):
        cases.append(("(d) 漏检：幻觉词得分偏低", miss.index[-1]))
    return cases, c


def draw_text(ax, text, word_scores, word_labels, fig):
    """逐个空格分隔的词画出来，自动换行。word_scores/labels: {字符起点: 值}。"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    r = fig.canvas.get_renderer()
    x, y, lh = 0.0, 0.98, 0.085
    for m in re.finditer(r"\S+", text):
        s, e = m.span()
        inside = [k for k in word_scores if s <= k < e]
        q = max((word_scores[k] for k in inside), default=None)
        lab = any(word_labels.get(k, 0) for k in inside)
        kw = dict(fontsize=8.2, va="top", ha="left", transform=ax.transAxes, fontfamily=TEXT_FONT,
                  color="#212529" if q is not None else "#868e96", fontweight="bold" if lab else "normal")
        if q is not None:
            kw["bbox"] = dict(boxstyle="square,pad=0.12", fc=CMAP(q), ec="none")
        t = ax.text(x, y, m.group(), **kw)
        bb = t.get_window_extent(r).transformed(ax.transAxes.inverted())
        if x > 0 and x + bb.width > 1.0:
            t.remove()
            x, y = 0.0, y - lh
            t = ax.text(x, y, m.group(), **kw)
            bb = t.get_window_extent(r).transformed(ax.transAxes.inverted())
        if lab:
            ax.plot([x, x + bb.width], [y - bb.height - 0.012] * 2, color="#1c3faa", lw=1.8,
                    transform=ax.transAxes, solid_capstyle="butt", zorder=10, clip_on=False)
        x += bb.width + 0.012
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--score", default="nll+vig_other4@max")
    ap.add_argument("--labels", default="core")
    ap.add_argument("--max-words", type=int, default=75)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)

    w = pd.read_csv(run / f"words_{args.labels}.csv")
    w["score"] = w[args.score]
    cw = w[w.content == 1]
    allv = np.sort(cw["score"].values)
    w["q"] = np.searchsorted(allv, w["score"].values, side="right") / len(allv)  # 全局分位数
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(run / "scores.jsonl", encoding="utf-8")}
    cases, c = pick_cases(w, args.max_words)

    fig = plt.figure(figsize=(7.4, 2.35 * len(cases) + 0.6))
    gs = fig.add_gridspec(len(cases) + 1, 2, width_ratios=[1, 2.6], height_ratios=[1] * len(cases) + [0.18],
                          hspace=0.45, wspace=0.04)
    titles = []
    for row, (title, sid) in enumerate(cases):
        rec = recs[sid]
        ws = w[w.id == sid]
        axi = fig.add_subplot(gs[row, 0])
        axi.imshow(Image.open(ROOT / "data" / "vg_images" / f"{rec['image_id']}.jpg"))
        axi.axis("off")
        axt = fig.add_subplot(gs[row, 1])
        cws = ws[ws.content == 1]
        draw_text(axt, rec["text"], dict(zip(cws.start, cws.q)), dict(zip(ws.start, ws.label)), fig)
        gold = ws[ws.label == 1]
        top = cws.sort_values("score", ascending=False).head(3)
        gold_s = "、".join(f"{r.word}（{'/'.join(TYPE_ZH[t] for t in r.type.split('|'))}）" for r in gold.itertuples()) or "无"
        note = f"标注幻觉词：{gold_s}\n得分最高的 3 个词：" + "、".join(
            f"{r.word}{'✓' if r.label else ''}" for r in top.itertuples())
        note = "\n".join(textwrap.fill(line, 80) for line in note.split("\n"))
        axt.text(0, -0.04, note, fontsize=7.4, va="top", ha="left", color="#495057",
                 transform=axt.transAxes)
        axt.set_title(f"{sid}（图 {rec['image_id']}）", fontsize=7.5, loc="right", color="#868e96")
        titles.append((axt, title))

    cax = fig.add_subplot(gs[-1, 1])
    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=CMAP), cax=cax, orientation="horizontal")
    cax.set_xlabel(f"可疑程度（分数在全部实词中的分位数，分数 = {args.score.replace('@max', '')}）；"
                   "加粗 + 蓝色下划线 = 标注的幻觉词；灰字 = 停用词，不打分", fontsize=7.5)
    for axt, title in titles:  # 小标题统一左对齐到图的左边
        fig.text(0.125, axt.get_position().y1 + 0.004, title, fontsize=9, fontweight="bold", va="bottom")
    cax.tick_params(labelsize=7)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("案例：", cases)
    print(f"图已保存 {args.out}")


if __name__ == "__main__":
    main()
