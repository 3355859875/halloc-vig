"""比较不同模型在同一批样本上的结果：主表、分类型、配对检查，并画分类型对比图（矢量 PDF）。

用法：
    uv run --no-project --with numpy --with pandas --with scikit-learn --with matplotlib python src/compare_models.py \
        3B=results/20261117_dev_multi_qwen25vl3b_val500 7B=results/<7B 目录> --fig report/图/模型规模对比.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

plt.rcParams.update({"pdf.fonttype": 42, "font.family": ["Arial Unicode MS", "PingFang SC", "sans-serif"],
                     "axes.spines.top": False, "axes.spines.right": False})

SCORES = [("nll@max", "不对照 nll"), ("vig_other1@max", "VIG 无关图×1"), ("vig_other4@max", "VIG 无关图×4"),
          ("nll+vig_other4@max", "nll + VIG×4")]
TYPES = ["object", "attribute", "relationship", "scene"]
TYPE_ZH = {"object": "物体", "attribute": "属性", "relationship": "关系", "scene": "场景"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="名称=结果目录")
    ap.add_argument("--fig")
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    M = {n: {L: json.load(open(Path(d) / f"metrics_{L}.json")) for L in ["official", "core"]} for n, d in runs.items()}

    for L in ["core", "official"]:
        print(f"\n== 口径 {L}（实词）")
        print(f"{'分数':<22}" + "".join(f"{n + ' PR':>10}{n + ' ROC':>10}" for n in runs))
        for c, zh in SCORES:
            print(f"{zh:<20}" + "".join(f"{M[n][L]['content_words'][c]['pr_auc']:>10}{M[n][L]['content_words'][c]['roc_auc']:>10}"
                                         for n in runs))
        print("样本级 ROC：" + "  ".join(f"{n}: nll {M[n][L]['sample_level']['nll@max']['roc_auc']} / VIG×4 "
                                          f"{M[n][L]['sample_level']['vig_other4@max']['roc_auc']}" for n in runs))

    # 分类型（核心词口径，全部词）
    print("\n== 分类型 ROC-AUC（核心词口径，全部词）")
    tab = {}
    for n in runs:
        for c, zh in SCORES:
            tab[(n, zh)] = [M[n]["core"]["by_type"][t][c]["roc_auc"] for t in TYPES]
    df = pd.DataFrame(tab, index=[TYPE_ZH[t] for t in TYPES]).T
    print(df.to_string())

    # 配对检查：两次运行是否是同一批样本、同一批词
    W = {n: pd.read_csv(Path(d) / "words_core.csv") for n, d in runs.items()}
    names = list(runs)
    a, b = W[names[0]], W[names[-1]]
    same = len(a) == len(b) and (a[["id", "start", "label"]].values == b[["id", "start", "label"]].values).all()
    print("（以下为实词、核心词口径；方括号为按样本 bootstrap 1000 次的 95% 区间）")
    print(f"\n配对检查：{names[0]} 与 {names[-1]} 的词表{'一致' if same else '不一致！'}（{len(a)} / {len(b)} 个词）")
    if same:
        c = a[a.content == 1].index
        y = a.loc[c, "label"].values
        rng = np.random.default_rng(0)
        ids = a.loc[c, "id"].values
        uid = np.unique(ids)
        for col, zh in SCORES:
            d0, d1 = a.loc[c, col].values, b.loc[c, col].values
            # 按样本整块重抽（bootstrap 1000 次），看 ROC-AUC 差的 95% 区间
            diffs, pdiffs = [], []
            groups = {u: np.where(ids == u)[0] for u in uid}
            for _ in range(1000):
                sel = np.concatenate([groups[u] for u in rng.choice(uid, len(uid))])
                if y[sel].min() == y[sel].max():
                    continue
                diffs.append(roc_auc_score(y[sel], d1[sel]) - roc_auc_score(y[sel], d0[sel]))
                pdiffs.append(average_precision_score(y[sel], d1[sel]) - average_precision_score(y[sel], d0[sel]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            plo, phi = np.percentile(pdiffs, [2.5, 97.5])
            print(f"  {zh:<16} ROC 差（{names[-1]} − {names[0]}）= {roc_auc_score(y, d1) - roc_auc_score(y, d0):+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]；PR-AUC 差 = "
                  f"{average_precision_score(y, d1) - average_precision_score(y, d0):+.4f} [{plo:+.4f}, {phi:+.4f}]")

    if args.fig:
        fig, axes = plt.subplots(1, len(runs), figsize=(3.6 * len(runs), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        colors = ["#868e96", "#91a7ff", "#3b5bdb", "#e8590c"]
        x = np.arange(len(TYPES))
        w = 0.2
        for ax, n in zip(axes, runs):
            for i, (c, zh) in enumerate(SCORES):
                ax.bar(x + (i - 1.5) * w, df.loc[(n, zh)].values, w, color=colors[i], label=zh)
            ax.axhline(0.5, color="black", lw=0.6, ls=":")
            ax.set_xticks(x, [TYPE_ZH[t] for t in TYPES])
            ax.set_title(n, fontsize=10)
            ax.set_ylim(0.4, 0.95)
        axes[0].set_ylabel("ROC-AUC（核心词口径）")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, frameon=False, ncol=4, loc="lower center")
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig)
        print(f"\n图已保存 {args.fig}")


if __name__ == "__main__":
    main()
