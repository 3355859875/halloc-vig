"""把报告要用的数字从结果文件里汇总成一份底稿（results/final/报告数字.md），报告里的数字都从这里抄、可追溯。

用法：
    uv run --no-project --with numpy --with pandas python src/collect_numbers.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
RUNS = {
    "可行性 val50": "20261117_feasibility_qwen25vl3b_val50",
    "3B val500 单图对照": "20261117_dev_qwen25vl3b_val500",
    "3B val500 多图/近邻": "20261117_dev_multi_qwen25vl3b_val500",
    "7B val500": "20261118_dev_model_qwen25vl7b_val500",
    "3B test 全量": "20261118_test_qwen25vl3b_caption_all",
}
out = []
P = out.append


def m(run, L):
    f = R / run / f"metrics_{L}.json"
    if not f.exists():
        f = R / run / "metrics.json"
    return json.load(open(f))


P("# 报告数字底稿（自动生成，勿手改）\n")
P("由 `src/collect_numbers.py` 从 `results/` 下的结果文件生成。\n")

# 数据
P("## 数据\n")
d = json.load(open(ROOT / "data" / "halloc_caption.json"))
df = pd.DataFrame([{"split": x["split"], "img": x["image_id"],
                    "h": any(x["annotations"][k] for k in x["annotations"])} for x in d])
P("| 切分 | 样本 | 含幻觉 | 图片 |\n| --- | --- | --- | --- |")
for s in ["train", "val", "test"]:
    g = df[df.split == s]
    P(f"| {s} | {len(g)} | {int(g.h.sum())} | {g.img.nunique()} |")
P(f"\ncaption 合计 {len(df)} 条\n")

# 运行
P("## 运行耗时与规模\n")
P("| 运行 | 目录 | 样本 | 跳过 | 秒 | 词 | 实词 | 核心词正例 | 官方正例 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for name, run in RUNS.items():
    s = json.load(open(R / run / "run_summary.json"))
    try:
        wc = pd.read_csv(R / run / "words_core.csv")
        wo = pd.read_csv(R / run / "words_official.csv")
        c1 = wc[wc.content == 1]
        extra = f"{len(wc)} | {len(c1)} | {int(c1.label.sum())} | {int(wo[wo.content == 1].label.sum())}"
    except FileNotFoundError:
        extra = "— | — | — | —"
    P(f"| {name} | `{run}` | {s['n_total']} | {s['n_skipped_tokenize_mismatch']} | {s['seconds']} | {extra} |")

# val 指标
P("\n## val 指标（实词，@max）\n")
for name in ["3B val500 单图对照", "3B val500 多图/近邻", "7B val500"]:
    run = RUNS[name]
    for L in ["core", "official"]:
        mm = m(run, L)
        cw, sl = mm["content_words"], mm["sample_level"]
        P(f"\n### {name} · {L}（正例率 {cw['nll@max']['pos_rate']}）\n")
        P("| 分数 | PR-AUC | ROC-AUC | 样本级 ROC |\n| --- | --- | --- | --- |")
        for k, v in cw.items():
            if k.endswith("@max") and v:
                P(f"| {k} | {v['pr_auc']} | {v['roc_auc']} | {(sl.get(k) or {}).get('roc_auc', '—')} |")
    bt = m(run, "core")["by_type"]
    keys = [k for k in ["nll@max", "entropy@max", "vig_blank@max", "vig_other@max", "vig_other1@max", "vig_other4@max",
                        "vig_other8@max", "vig_near1@max", "vig_near4@max", "vig_near8@max", "nll+vig_other4@max"] if k in bt["object"]]
    P(f"\n{name} 分类型 ROC（核心词，全部词）\n\n| 分数 | " + " | ".join(bt) + " |\n| --- |" + " --- |" * len(bt))
    for k in keys:
        P(f"| {k} | " + " | ".join(str(bt[t][k]["roc_auc"]) if bt[t][k] else "—" for t in bt) + " |")

# 冻结协议与 test
P("\n## 冻结协议（val 上定）\n")
proto = json.load(open(R / "final" / "protocol_20261118.json"))
P("| 方法 | 分数 | 核心词阈值 | val F1 | 官方阈值 | val F1 |\n| --- | --- | --- | --- | --- | --- |")
for k, v in proto.items():
    c, o = v["labels"]["core"], v["labels"]["official"]
    P(f"| {k} | `{v['score']}` | {c['threshold']:.4f} | {c['val_f1']} | {o['threshold']:.4f} | {o['val_f1']} |")
P("\n## test（3B，按冻结协议）\n")
for f in sorted((R / "final").glob("test_*.json")):
    t = json.load(open(f))
    for L in ["core", "official"]:
        x = t["labels"][L]
        a = x["at_threshold"]
        P(f"- {t['method']} · {L}：PR-AUC {x['pr_auc']} {x['pr_auc_ci']}，ROC {x['roc_auc']} {x['roc_auc_ci']}，"
          f"P/R/F1 {a['precision']}/{a['recall']}/{a['f1']}（标出 {a['n_flagged']} 个），样本级 ROC {x['sample_roc_auc']}，"
          f"正例率 {x['pos_rate']}，实词 {x['n_words']}；分类型 " +
          "，".join(f"{k} {v['roc_auc']}（n={v['n_pos']}）" for k, v in x["by_type_roc"].items()))

# 其他
P("\n## 其他\n")
cmp = R / "final" / "compare_3B_7B_val.txt"
if cmp.exists():
    P("3B vs 7B（val，配对 bootstrap）：\n\n```\n" + cmp.read_text(encoding="utf-8").strip() + "\n```\n")
for p in sorted((R / "demo_presets").glob("*.json")):
    x = json.load(open(p, encoding="utf-8"))
    top = sorted([w for w in x["words"] if w["content"]], key=lambda w: -w["vig"])[:3]
    P(f"- 演示预设 {p.stem}（{x['title']}）：用时 {x['seconds']} 秒（{x['device']}），可疑 {x['summary']['n_flagged']} 个；最高 " +
      "，".join(f"{w['word']} VIG {w['vig']}（log p 原图 {w['logp_real']} / 无关 {w['logp_other']}）" for w in top))

(R / "final" / "报告数字.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
