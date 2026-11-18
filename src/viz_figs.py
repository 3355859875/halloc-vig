"""报告用的汇总图（矢量 PDF）：
    图 3  对照方式对比：灰图 / 不给图 / 1 张无关图 / 4 张无关图 / 1 张近邻图 / 8 张近邻图（实词，两种口径）
    图 6  分数分布：幻觉词 vs 正常词，分类型（核心词口径、实词）
    图 7  PR 曲线（核心词口径、实词）

用法：
    uv run --no-project --with numpy --with pandas --with scikit-learn --with matplotlib python src/viz_figs.py \
        --single results/20261117_dev_qwen25vl3b_val500 --multi results/20261117_dev_multi_qwen25vl3b_val500 --outdir report/图
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

plt.rcParams.update({"pdf.fonttype": 42, "font.family": ["Arial Unicode MS", "PingFang SC", "sans-serif"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 8.5})
TYPE_ZH = {"object": "物体", "attribute": "属性", "relationship": "关系", "scene": "场景"}


def fig_contrast(single, multi, out):
    items = [("nll@max", "不对照\n(nll)", single, "#adb5bd"),
             ("vig_blank@max", "灰图", single, "#ced4da"),
             ("vig_noimg@max", "不给图", single, "#ced4da"),
             ("vig_other1@max", "无关图\n×1", multi, "#91a7ff"),
             ("vig_other4@max", "无关图\n×4", multi, "#3b5bdb"),
             ("vig_near1@max", "近邻图\n×1", multi, "#ffc078"),
             ("vig_near8@max", "近邻图\n×8", multi, "#e8590c")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, L, zh in zip(axes, ["core", "official"], ["核心词口径", "官方口径"]):
        vals = []
        for key, _, run, _ in items:
            vals.append(json.load(open(Path(run) / f"metrics_{L}.json"))["content_words"][key]["pr_auc"])
        pos = json.load(open(Path(multi) / f"metrics_{L}.json"))["content_words"]["nll@max"]["pos_rate"]
        x = np.arange(len(items))
        ax.bar(x, vals, color=[c for *_, c in items], width=0.7)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.axhline(pos, color="black", lw=0.7, ls=":")
        ax.text(-0.45, pos + 0.002, f"随机 {pos:.3f}", ha="left", va="bottom", fontsize=7)
        ax.set_xticks(x, [n for _, n, _, _ in items], fontsize=7.5)
        ax.set_title(zh, fontsize=9)
        ax.set_ylabel("实词 PR-AUC")
    fig.tight_layout()
    fig.savefig(out)


def load_words(run, L="core"):
    w = pd.read_csv(Path(run) / f"words_{L}.csv")
    return w[w.content == 1].copy()


def fig_dist(multi, out):
    w = load_words(multi)
    neg = w[w.label == 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, col, zh in zip(axes, ["nll@max", "vig_other4@max"],
                           ["不对照 nll（越高越可疑）", "VIG 无关图×4（越高越可疑）"]):
        groups = [("正常词", neg[col].values, "#adb5bd")]
        for t in TYPE_ZH:
            v = w[(w.label == 1) & w.type.fillna("").str.contains(t)][col].values
            groups.append((f"{TYPE_ZH[t]}幻觉词\nn={len(v)}", v, "#e8590c" if t != "relationship" else "#c92a2a"))
        bp = ax.boxplot([g[1] for g in groups], showfliers=False, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", lw=1))
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(g[2])
            patch.set_alpha(0.75)
            patch.set_edgecolor("none")
        ax.axhline(np.median(neg[col]), color="#495057", lw=0.6, ls="--")
        ax.set_xticks(range(1, len(groups) + 1), [g[0] for g in groups], fontsize=7.2)
        ax.set_title(zh, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)


def fig_pr(multi, out, with_combo=True):
    w = load_words(multi)
    y = w.label.values
    curves = [("nll@max", "不对照 nll", "#868e96", "-"), ("entropy@max", "熵", "#ced4da", "-"),
              ("vig_other1@max", "VIG 无关图×1", "#91a7ff", "-"), ("vig_other4@max", "VIG 无关图×4", "#3b5bdb", "-"),
              ("nll+vig_other4@max", "nll + VIG×4", "#e8590c", "--")]
    if not with_combo:  # test 上的组合分数要用 val 冻结的常数（见 final_eval.py），这里不画，避免口径不一
        curves = curves[:-1]
    m = json.load(open(Path(multi) / "metrics_core.json"))["content_words"]
    fig, ax = plt.subplots(figsize=(3.8, 3.1))
    for col, zh, c, ls in curves:
        p, r, _ = precision_recall_curve(y, w[col].values)
        ax.plot(r, p, color=c, ls=ls, lw=1.3, label=f"{zh}（{m[col]['pr_auc']:.3f}）")
    ax.axhline(y.mean(), color="black", lw=0.7, ls=":", label=f"随机（{y.mean():.3f}）")
    ax.set_xlabel("召回率")
    ax.set_ylabel("精确率")
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=7, frameon=False, title="括号内为 PR-AUC", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", required=True, help="含灰图 / 不给图对照的那次运行")
    ap.add_argument("--multi", required=True, help="多张无关图 / 近邻图那次运行")
    ap.add_argument("--outdir", default="report/图")
    ap.add_argument("--test-run", help="给了就另出 test 版：PR曲线_test.pdf、分数分布_分类型_test.pdf（不画组合分数）")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig_contrast(args.single, args.multi, out / "对照方式对比.pdf")
    fig_dist(args.multi, out / "分数分布_分类型.pdf")
    fig_pr(args.multi, out / "PR曲线.pdf")
    if args.test_run:
        fig_dist(args.test_run, out / "分数分布_分类型_test.pdf")
        fig_pr(args.test_run, out / "PR曲线_test.pdf", with_combo=False)
        print("已保存 test 版：分数分布_分类型_test.pdf、PR曲线_test.pdf")
    print("已保存：对照方式对比.pdf、分数分布_分类型.pdf、PR曲线.pdf")


if __name__ == "__main__":
    main()
