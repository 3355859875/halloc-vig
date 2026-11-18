"""把 vig_score.py 的逐 token 结果聚合到词，按 HalLoc 标注评测。

标签口径沿用官方后处理（scripts/postprocess/postprocess_*.py）：一条标注里的所有区间都算幻觉，
例如属性类的属性词和物体词、关系类的主语/谓语/宾语都算。我们在“词”上评测（官方在各模型自己的 token 上），
词按字符区间对齐，因此换模型时可比。

分数越高越像幻觉：
    nll       -log p(词 | 原图)                 不用对照的基线
    entropy   原图条件下预测分布的熵           不确定性基线
    vig_<c>   log p(词 | 对照 c) - log p(词 | 原图)   即负的视觉信息增益
    vig_<组>K  同一组对照（other_0…、near_0…）取前 K 张的平均对数概率再减原图，K = 1, 2, 4, 8…
    nll+<x>   nll 与 x 各自按全部词标准化后相加（不用标签的固定组合）

用法：
    uv run python src/eval_words.py results/<目录名> [--labels official|core]
    official = 官方口径（默认）；core = 核心词口径，每条幻觉只标被改动的那个词（关系类按问答库 detailed_type 判断）
输出 metrics_<口径>.json、words_<口径>.csv
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
STOP = set("""a an the and or but of to in on at by for with from as is are was were be been being it its this that these
those there here he she they them his her their we you i me my your our which who whom what where when while than then
so very also has have had do does did not no can could would should will just into onto over under some any each all one
two other another more most such up down out off""".split())
TYPES = ["object", "attribute", "relationship", "scene"]


def spans_of(annotations):
    """返回 [(start, end, 类型)]。官方口径：一条标注里的全部区间。"""
    return spans_core(annotations, None)


# 核心词口径：每条幻觉只标真正被改动的部分
CORE_KEY = {"object": {"obj"}, "attribute": {"attribute"}, "scene": {"scene"}}


def spans_core(annotations, qa_type):
    """qa_type 为 None 时退化为官方口径；否则是 question_id -> detailed_type 的映射，
    用来判断关系类幻觉改的是 object 还是 predicate（问答库 qa_database.json 的字段）。"""
    res = []
    for t in TYPES:
        for ann in annotations.get(t, []):
            keep = None
            if qa_type is not None:
                if t in CORE_KEY:
                    keep = CORE_KEY[t]
                else:
                    qids = {p.get("question_id") for p in ann.values() if isinstance(p, dict)}
                    dts = {qa_type.get(q) for q in qids} & {"object", "predicate", "subject"}
                    keep = dts if len(dts) == 1 else None  # 判断不了就保留全部区间
            for key, part in ann.items():
                if keep is not None and key not in keep:
                    continue
                if isinstance(part, dict) and "char_index" in part:
                    a = list(map(int, str(part["char_index"]).split(":")))
                    s, e = (a[0], a[0] + 1) if len(a) == 1 else (a[0], a[1])
                    res.append((s, e, t))
    return res


def word_rows(rec, qa_type=None):
    spans = spans_core(rec["annotations"], qa_type)
    offs = rec["offsets"]
    singles, groups = [], defaultdict(list)
    for c in rec["logp"]:
        m = re.fullmatch(r"([a-z]+)_(\d+)", c)
        if m:
            groups[m.group(1)].append(c)
        elif c != "real":
            singles.append(c)
    rows = []
    for m in WORD_RE.finditer(rec["text"]):
        ws, we = m.span()
        tix = [i for i, (s, e) in enumerate(offs) if s < we and e > ws]
        if not tix:
            continue
        types = sorted({t for s, e, t in spans if s < we and e > ws})
        row = {"id": rec["id"], "word": m.group(), "start": ws, "label": int(bool(types)),
               "type": "|".join(types), "content": int(m.group().lower() not in STOP)}
        real = np.array([rec["logp"]["real"][i] for i in tix])
        ent = np.array([rec["entropy"][i] for i in tix])
        feats = {"nll": -real, "entropy": ent}
        for c in singles:
            feats[f"vig_{c}"] = np.array([rec["logp"][c][i] for i in tix]) - real
        for g, members in groups.items():
            members = sorted(members, key=lambda c: int(c.split("_")[1]))
            lp = np.array([[rec["logp"][c][i] for i in tix] for c in members])  # (张数, token)
            K = 1
            while K <= len(members):
                feats[f"vig_{g}{K}"] = lp[:K].mean(0) - real
                K *= 2
        for name, v in feats.items():
            row[f"{name}@max"] = float(v.max())   # 词里最可疑的 token
            row[f"{name}@mean"] = float(v.mean())
        rows.append(row)
    return rows


def metrics(y, s):
    y, s = np.asarray(y), np.asarray(s)
    if y.min() == y.max():
        return None
    return {"pr_auc": round(float(average_precision_score(y, s)), 4),
            "roc_auc": round(float(roc_auc_score(y, s)), 4),
            "pos_rate": round(float(y.mean()), 4), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--labels", default="official", choices=["official", "core"])
    args = ap.parse_args()
    run = Path(args.run_dir)
    qa_type = None
    if args.labels == "core":
        data = Path(__file__).resolve().parent.parent / "data"
        qa_type = {q["question_id"]: q["detailed_type"] for q in json.load(open(data / "qa_database.json", encoding="utf-8"))}
    recs = [json.loads(l) for l in open(run / "scores.jsonl", encoding="utf-8")]
    rows = [r for rec in recs for r in word_rows(rec, qa_type)]
    # 固定组合：nll 与每个 VIG 分数（@max）各自标准化后相加
    def z(col):
        v = np.array([r[col] for r in rows])
        return (v - v.mean()) / v.std()
    zn = z("nll@max")
    for col in [k for k in rows[0] if k.startswith("vig_") and k.endswith("@max")]:
        for r, a, b in zip(rows, zn, z(col)):
            r[f"nll+{col}"] = float(a + b)
    score_cols = [k for k in rows[0] if "@" in k]

    res = {"labels": args.labels, "n_samples": len(recs), "n_words": len(rows), "all_words": {}, "content_words": {}, "by_type": {}, "sample_level": {}}
    y = [r["label"] for r in rows]
    cw = [r for r in rows if r["content"]]
    for c in score_cols:
        res["all_words"][c] = metrics(y, [r[c] for r in rows])
        res["content_words"][c] = metrics([r["label"] for r in cw], [r[c] for r in cw])

    # 分类型：该类型的幻觉词 vs 所有非幻觉词
    for t in TYPES:
        sub = [r for r in rows if (t in r["type"].split("|")) or r["label"] == 0]
        res["by_type"][t] = {c: metrics([r["label"] for r in sub], [r[c] for r in sub]) for c in score_cols}

    # 样本级：整段有没有幻觉，用最可疑词的分数
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    ys = [int(any(r["label"] for r in v)) for v in by_id.values()]
    for c in score_cols:
        res["sample_level"][c] = metrics(ys, [max(r[c] for r in v) for v in by_id.values()])

    (run / f"metrics_{args.labels}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    with open(run / f"words_{args.labels}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"口径 {args.labels}：样本 {len(recs)}，词 {len(rows)}（幻觉词 {sum(y)}），实词 {len(cw)}（幻觉 {sum(r['label'] for r in cw)}）")
    print(f"{'分数':<22}{'全部词 PR-AUC':>14}{'ROC-AUC':>9}{'实词 PR-AUC':>13}{'ROC-AUC':>9}{'样本级 ROC':>11}")
    for c in score_cols:
        a, b, s = res["all_words"][c], res["content_words"][c], res["sample_level"][c]
        g = lambda m, k: (m or {}).get(k, "-")  # 没有正例或负例时指标为空
        print(f"{c:<22}{g(a, 'pr_auc'):>14}{g(a, 'roc_auc'):>9}{g(b, 'pr_auc'):>13}{g(b, 'roc_auc'):>9}{g(s, 'roc_auc'):>11}")


if __name__ == "__main__":
    main()
