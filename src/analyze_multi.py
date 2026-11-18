"""汇总“多张无关图 vs CLIP 近邻图”实验：K 曲线、分类型、样本级。

读 eval_words.py 输出的 metrics_official.json / metrics_core.json，打印表格，并画 K 曲线（矢量 PDF）。

用法：
    uv run --no-project --with numpy --with matplotlib python src/analyze_multi.py results/<目录> --fig report/图/对照张数曲线.pdf
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "font.family": ["Arial Unicode MS", "PingFang SC", "sans-serif"],
                     "axes.spines.top": False, "axes.spines.right": False})

GROUPS = {"other": "随机无关图", "near": "CLIP 近邻图"}
COLORS = {"other": "#3b5bdb", "near": "#e8590c"}


def k_values(metrics, group):
    ks = sorted({int(m.group(1)) for c in metrics["content_words"]
                 if (m := re.fullmatch(rf"vig_{group}(\d+)@max", c))})
    return ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--fig")
    args = ap.parse_args()
    run = Path(args.run_dir)
    M = {L: json.load(open(run / f"metrics_{L}.json")) for L in ["official", "core"]}

    for L, m in M.items():
        cw = m["content_words"]
        print(f"\n== 口径 {L}，实词正例率 {cw['nll@max']['pos_rate']}")
        print(f"{'分数':<26}{'PR-AUC':>8}{'ROC-AUC':>9}{'样本级ROC':>10}")
        rows = ["nll@max", "entropy@max"] + [f"vig_{g}{k}@max" for g in GROUPS for k in k_values(m, g)] \
               + [f"nll+vig_{g}{k_values(m, g)[-1]}@max" for g in GROUPS]
        for c in rows:
            print(f"{c:<26}{cw[c]['pr_auc']:>8}{cw[c]['roc_auc']:>9}{m['sample_level'][c]['roc_auc']:>10}")
        print("分类型 ROC-AUC（全部词）：")
        for t, d in m["by_type"].items():
            line = "  ".join(f"{c.replace('@max', '')}={d[c]['roc_auc']}" for c in rows if d.get(c))
            print(f"  {t:<13}{line}")

    if args.fig:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
        for ax, metric, name in zip(axes, ["pr_auc", "roc_auc"], ["PR-AUC", "ROC-AUC"]):
            for L, ls in [("core", "-"), ("official", "--")]:
                cw = M[L]["content_words"]
                for g, label in GROUPS.items():
                    ks = k_values(M[L], g)
                    ax.plot(ks, [cw[f"vig_{g}{k}@max"][metric] for k in ks], ls, marker="o", ms=3.5,
                            color=COLORS[g], label=f"{label}（{'核心词' if L == 'core' else '官方'}口径）")
                ax.axhline(cw["nll@max"][metric], ls=ls, color="#868e96", lw=1,
                           label=f"不对照 nll（{'核心词' if L == 'core' else '官方'}口径）")
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
            ax.set_xlabel("对照图张数 K")
            ax.set_ylabel(f"实词 {name}")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7.5, frameon=False, ncol=3, loc="lower center")
        fig.tight_layout(rect=(0, 0.17, 1, 1))
        Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig)
        print(f"\n图已保存 {args.fig}")


if __name__ == "__main__":
    main()
