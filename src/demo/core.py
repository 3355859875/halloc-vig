"""演示网页的打分核心（不依赖任何网页框架）。

和实验口径完全一致：
    - 打分复用 src/vig_score.py 的 build_inputs / token_logprobs；
    - 词的切分和停用词复用 src/eval_words.py 的 WORD_RE / STOP；
    - 每个词取 VIG 最大的那个 token（= eval_words 里的 vig_other4@max），
      同时给出这个 token 在原图下和无关图下的 log p，所以 VIG = 无关图平均 − 原图，正好可以拆开解释；
    - “可疑”阈值用 val 上冻结的阈值（results/final/protocol_20261118.json 里的 3B_vig4、核心词口径）；
    - 颜色 = 该词 VIG 在 val 全部实词中的分位数。
"""

import json
import random
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
from eval_words import STOP, WORD_RE  # noqa: E402

ROOT = SRC.parent
DATA = ROOT / "data"
K = 4
SCORE_COL = "vig_other4@max"
TYPE_ZH = {"object": "物体", "attribute": "属性", "relationship": "关系", "scene": "场景"}
# 前 70% 基本不上色，越靠前越红（和报告里的案例图同一套配色）
STOPS = [(0.0, (255, 255, 255)), (0.7, (255, 244, 230)), (0.85, (255, 192, 120)), (0.95, (250, 82, 82)), (1.0, (201, 42, 42))]


def color_of(q):
    for (q0, c0), (q1, c1) in zip(STOPS, STOPS[1:]):
        if q <= q1:
            t = (q - q0) / (q1 - q0) if q1 > q0 else 0
            return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))
    return "#%02x%02x%02x" % STOPS[-1][1]


class Reference:
    """val 上的分数分布、冻结阈值、示例样本。"""

    def __init__(self, val_run, example_runs, protocol):
        w = pd.read_csv(Path(val_run) / "words_core.csv")
        self.val_sorted = np.sort(w.loc[w.content == 1, SCORE_COL].values)
        self.threshold = float(json.load(open(protocol))["3B_vig4"]["labels"]["core"]["threshold"])
        self.examples = {}  # "来源:样本 id" -> (来源名, rec)
        self._labels = {}  # (来源名, 样本 id, 词起点) -> (标签, 类型)，核心词口径
        for name, run in example_runs:
            run = Path(run)
            if not (run / "words_core.csv").exists():
                continue
            words = pd.read_csv(run / "words_core.csv")
            for i, s, l, t in zip(words.id, words.start, words.label, words.type):
                self._labels[(name, i, int(s))] = (int(l), t if isinstance(t, str) else "")
            for line in open(run / "scores.jsonl", encoding="utf-8"):
                rec = json.loads(line)
                self.examples[f"{name}:{rec['id']}"] = (name, rec)
        self.index = []
        for key, (name, rec) in self.examples.items():
            res = self.analyze(rec, name)
            gold = [x for x in res["words"] if x["label"]]
            types = sorted({t for x in gold for t in x["type"].split("|") if t})
            top = max((x for x in res["words"] if x["content"]), key=lambda x: x["vig"], default=None)
            self.index.append({
                "key": key, "source": name, "id": rec["id"],
                "has_halluc": bool(gold),
                "tag": "无幻觉" if not gold else "含幻觉（" + "/".join(TYPE_ZH[t] for t in types) + "）",
                "hit": bool(gold) and top is not None and top["label"] == 1 and top["flag"],
            })
        self.index.sort(key=lambda x: x["key"])

    def quantile(self, v):
        return float(np.searchsorted(self.val_sorted, v, side="right") / len(self.val_sorted))

    def analyze(self, rec, source=None):
        """rec: vig_score.py 输出格式（text、offsets、logp[real / other_0…]）。返回逐词结果和摘要。"""
        text, offs = rec["text"], rec["offsets"]
        real = np.array(rec["logp"]["real"])
        others = np.array([rec["logp"][f"other_{k}"] for k in range(K)]).mean(0)
        words = []
        for m in WORD_RE.finditer(text):
            ws, we = m.span()
            tix = [i for i, (s, e) in enumerate(offs) if s < we and e > ws]
            if not tix:
                continue
            vig_t = others[tix] - real[tix]
            j = tix[int(np.argmax(vig_t))]
            vig = float(others[j] - real[j])
            q = self.quantile(vig)
            content = m.group().lower() not in STOP
            lab, typ = (0, "")
            if source is not None:
                lab, typ = self._labels.get((source, rec["id"], ws), (0, ""))
            words.append({"start": ws, "end": we, "word": m.group(), "content": content,
                          "logp_real": round(float(real[j]), 3), "logp_other": round(float(others[j]), 3),
                          "vig": round(vig, 3), "q": round(q, 4), "color": color_of(q) if content else None,
                          "flag": bool(content and vig >= self.threshold), "label": lab, "type": typ})
        cw = [x for x in words if x["content"]]
        flagged = [x for x in cw if x["flag"]]
        summary = {"n_content": len(cw), "n_flagged": len(flagged)}
        if source is not None:
            gold = [x for x in cw if x["label"]]
            summary.update({"n_gold": len(gold), "n_hit": sum(x["label"] for x in flagged),
                            "n_false": sum(1 - x["label"] for x in flagged)})
        return {"text": text, "words": words, "summary": summary,
                "image_id": rec.get("image_id"), "other_image_ids": rec.get("other_image_ids", [])[:K]}


class Scorer:
    """在线打分（懒加载模型；一次只处理一个请求）。"""

    def __init__(self, model_path, device="auto", pool_split="val"):
        self.model_path, self.device = model_path, device
        self.model = self.processor = None
        self.dev = None
        self.lock = threading.Lock()
        ids = sorted({x["image_id"] for x in json.load(open(DATA / "halloc_caption.json")) if x["split"] == pool_split})
        self.pool = [i for i in ids if (DATA / "vg_images" / f"{i}.jpg").exists()]

    def load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            dev = self.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.processor = AutoProcessor.from_pretrained(self.model_path, max_pixels=512 * 28 * 28)
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, dtype=torch.bfloat16).to(dev).eval()
            self.dev = dev
        return self.model, self.processor

    def caption(self, image, prompt="Describe this image in detail."):
        import torch
        with self.lock:
            model, proc = self.load()
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = proc(text=[text], images=[image], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            cap = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            cut = cap.rfind(".")  # 截到最后一个完整句子，避免半句话
            return " ".join((cap[: cut + 1] if cut > 0 else cap).split())

    def score(self, image, text, prompt="Describe this image in detail.", seed=0):
        from vig_score import build_inputs, token_logprobs
        with self.lock:
            model, proc = self.load()
            enc = proc.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            ref_ids, offsets = enc["input_ids"], enc["offset_mapping"]
            oids = random.Random(seed).sample(self.pool, K)
            conds = {"real": image}
            conds.update({f"other_{k}": Image.open(DATA / "vg_images" / f"{o}.jpg").convert("RGB") for k, o in enumerate(oids)})
            rec = {"id": "live", "text": text, "offsets": offsets, "logp": {}, "other_image_ids": oids}
            for name, im in conds.items():
                inputs, start = build_inputs(proc, prompt, text, im, len(ref_ids))
                ids, lp, _ = token_logprobs(model, inputs, start)
                if ids != ref_ids:
                    raise ValueError("切词对不齐：这段文字在上下文里的切分和单独切分不同，换个写法再试")
                rec["logp"][name] = lp
            return rec
