"""最终评测：先在 val 上冻结参数（fit），再原样用到 test 上（apply）。test 只跑一次，不回头调。

fit：在 val 的实词上
    - 组合分数的标准化常数（均值、标准差）只从 val 算；
    - 阈值取 val 上 F1 最高的那个；
    写出协议文件（JSON），跑 test 之前就定下来。
apply：读协议文件，在 test 上算 PR-AUC、ROC-AUC、阈值下的 P/R/F1、分类型 ROC、样本级 ROC，
    以及按样本 bootstrap 1000 次的 95% 区间。

分数写法：单列如 "vig_other4@max"；组合如 "z:nll@max+vig_other4@max"（各列用 val 的均值、标准差标准化后相加）。

用法：
    python src/final_eval.py fit --method M1 --run results/<val 目录> --score vig_other4@max --protocol results/final/protocol.json
    python src/final_eval.py apply --method M1 --run results/<test 目录> --protocol results/final/protocol.json --out results/final
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

TYPES = ["object", "attribute", "relationship", "scene"]


def load(run, labels):
    w = pd.read_csv(Path(run) / f"words_{labels}.csv")
    return w


def score_of(w, spec, consts=None):
    if not spec.startswith("z:"):
        return w[spec].values.astype(float), None
    cols = spec[2:].split("+")
    if consts is None:
        cw = w[w.content == 1]
        consts = {c: [float(cw[c].mean()), float(cw[c].std())] for c in cols}
    s = sum((w[c].values - consts[c][0]) / consts[c][1] for c in cols)
    return s, consts


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i]), float(f1[i])


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4),
            "n_flagged": tp + fp}


def fit(args):
    proto = json.load(open(args.protocol)) if Path(args.protocol).exists() else {}
    entry = {"run": args.run, "score": args.score, "fitted_on": "val", "labels": {}}
    for L in ["core", "official"]:
        w = load(args.run, L)
        s, consts = score_of(w, args.score)
        cw = w.content.values == 1
        thr, f1 = best_threshold(w.label.values[cw], s[cw])
        entry["labels"][L] = {"consts": consts, "threshold": thr, "val_f1": round(f1, 4)}
        print(f"{args.method} {L}: 阈值 {thr:.4f}（val F1 {f1:.4f}）" + (f"，标准化常数 {consts}" if consts else ""))
    proto[args.method] = entry
    Path(args.protocol).parent.mkdir(parents=True, exist_ok=True)
    Path(args.protocol).write_text(json.dumps(proto, ensure_ascii=False, indent=1))


def boot(ids, y, s, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uid = np.unique(ids)
    groups = {u: np.where(ids == u)[0] for u in uid}
    vals = []
    for _ in range(n):
        sel = np.concatenate([groups[u] for u in rng.choice(uid, len(uid))])
        if y[sel].min() != y[sel].max():
            vals.append(fn(y[sel], s[sel]))
    return [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])]


def apply(args):
    proto = json.load(open(args.protocol))[args.method]
    out = {"method": args.method, "test_run": args.run, "score": proto["score"], "labels": {}}
    for L in ["core", "official"]:
        P = proto["labels"][L]
        w = load(args.run, L)
        s, _ = score_of(w, proto["score"], P["consts"])
        w["s"] = s
        cw = w[w.content == 1]
        y, sc, ids = cw.label.values, cw.s.values, cw.id.values
        res = {"n_words": int(len(cw)), "pos_rate": round(float(y.mean()), 4),
               "pr_auc": round(float(average_precision_score(y, sc)), 4),
               "pr_auc_ci": boot(ids, y, sc, average_precision_score),
               "roc_auc": round(float(roc_auc_score(y, sc)), 4),
               "roc_auc_ci": boot(ids, y, sc, roc_auc_score),
               "threshold_from_val": P["threshold"], "at_threshold": prf(y, (sc >= P["threshold"]).astype(int))}
        neg = cw[cw.label == 0]
        res["by_type_roc"] = {}
        for t in TYPES:
            pos = cw[(cw.label == 1) & cw.type.fillna("").str.contains(t)]
            if len(pos):
                sub = pd.concat([pos, neg])
                res["by_type_roc"][t] = {"roc_auc": round(float(roc_auc_score(sub.label, sub.s)), 4), "n_pos": int(len(pos))}
        g = w.groupby("id")
        ys = g["label"].max().values
        ms = w[w.content == 1].groupby("id")["s"].max().reindex(g.size().index).fillna(-1e9).values
        res["sample_roc_auc"] = round(float(roc_auc_score(ys, ms)), 4)
        out["labels"][L] = res
        a = res["at_threshold"]
        print(f"{args.method} {L}: PR-AUC {res['pr_auc']} {res['pr_auc_ci']}  ROC {res['roc_auc']} {res['roc_auc_ci']}  "
              f"P/R/F1 {a['precision']}/{a['recall']}/{a['f1']}  样本级 ROC {res['sample_roc_auc']}  随机 {res['pos_rate']}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"test_{args.method}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "apply"])
    ap.add_argument("--method", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--score")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", default="results/final")
    args = ap.parse_args()
    fit(args) if args.mode == "fit" else apply(args)


if __name__ == "__main__":
    main()
