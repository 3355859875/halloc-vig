"""逐 token 打分：同一段文字分别配“原图”和“对照条件”，用 teacher forcing 取每个 token 的对数概率。

视觉信息增益 VIG(token) = log p(token | 原图, 前文) - log p(token | 对照, 前文)
幻觉词预期 VIG 偏低甚至为负（图里看不到，或和图矛盾）。

对照条件（--controls，逗号分隔）：
    blank    同尺寸的纯灰图（保留图像 token 的位置和数量，只去掉内容）
    noimg    完全不给图（纯文本语言先验）
    other    换成同切分里另一张随机图（1 张）
    other:K  K 张随机无关图，记为 other_0 … other_{K-1}
    near:K   K 张 CLIP 近邻图（“长得像但不是这张”），记为 near_0 …，需要 --nn-file（clip_neighbors.py 生成）
无关图和近邻图都只从同一切分的图片池里取（三个子集合起来）。

输出：每条样本一行 JSON，含文字、每个 token 的字符区间、各条件下的对数概率和原图条件下的熵。

用法（在作业目录下）：
    uv run python src/vig_score.py --subset caption --split val --n 50 --out results/<目录名>
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_samples(subset, split, n, seed):
    xs = [x for x in json.load(open(DATA / f"halloc_{subset}.json", encoding="utf-8")) if x["split"] == split]
    xs.sort(key=lambda x: x["id"])
    rng = random.Random(seed)
    if n and n < len(xs):
        xs = rng.sample(xs, n)
    return xs


def build_inputs(processor, prompt, text, image, n_text_tokens):
    """返回 (inputs, 回答起点下标)。image=None 表示不给图。
    回答接在提示末尾，起点 = 总长度 - 回答的 token 数（之后还会逐个核对 token，不一致就跳过该样本）。
    （20261118 起改成这样，省掉单独处理一遍前缀；旧写法是再跑一次 processor 求前缀长度，结果相同。）"""
    content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    prefix = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    imgs = [image] if image is not None else None
    full = processor(text=[prefix + text], images=imgs, return_tensors="pt")
    return full, full["input_ids"].shape[1] - n_text_tokens


@torch.no_grad()
def token_logprobs(model, inputs, start):
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0].float()  # (L, V)
    ids = inputs["input_ids"][0]
    logp = torch.log_softmax(logits[start - 1 : -1], dim=-1)  # 预测位置 start..L-1
    tgt = ids[start:]
    lp = logp.gather(1, tgt[:, None])[:, 0]
    ent = -(logp.exp() * logp).sum(-1)
    return tgt.tolist(), lp.tolist(), ent.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--subset", default="caption", choices=["caption", "instruct", "vqa"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=50, help="抽样条数，0 表示全部")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="抽样不变，只处理前 L 条（用于核对 / 冒烟测试）")
    ap.add_argument("--controls", default="blank,noimg,other")
    ap.add_argument("--nn-file", help="clip_neighbors.py 输出的近邻 JSON")
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28, help="限制图像 token 数（Qwen-VL 每 28x28 像素 1 个）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=1))

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    tok = processor.tokenizer

    xs = load_samples(args.subset, args.split, args.n, args.seed)
    if args.limit:
        xs = xs[: args.limit]
    controls = [c for c in args.controls.split(",") if c]
    pool = sorted({x["image_id"] for sub in ["caption", "instruct", "vqa"] for x in load_samples(sub, args.split, 0, 0)})
    nn = json.load(open(args.nn_file)) if args.nn_file else None
    open_img = lambda i: Image.open(DATA / "vg_images" / f"{i}.jpg").convert("RGB")
    rng = random.Random(args.seed + 1)

    t0 = time.time()
    n_bad = 0
    with open(out / "scores.jsonl", "w", encoding="utf-8") as f:
        for k, x in enumerate(xs, 1):
            prompt = x["prompt"].replace("<image>", "").strip()
            text = x["hallucinated_text"]
            img = Image.open(DATA / "vg_images" / f"{x['image_id']}.jpg").convert("RGB")

            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ref_ids, offsets = enc["input_ids"], enc["offset_mapping"]

            rec = {"id": x["id"], "image_id": x["image_id"], "text": text,
                   "annotations": x["annotations"], "offsets": offsets, "logp": {}}
            conds = {"real": img}
            for c in controls:
                if c == "blank":
                    conds[c] = Image.new("RGB", img.size, (128, 128, 128))
                elif c == "noimg":
                    conds[c] = None
                elif c == "other":
                    oid = rng.choice([i for i in pool if i != x["image_id"]])
                    conds[c] = open_img(oid)
                    rec["other_image_id"] = oid
                elif c.startswith("other:"):
                    oids = rng.sample([i for i in pool if i != x["image_id"]], int(c.split(":")[1]))
                    for j, oid in enumerate(oids):
                        conds[f"other_{j}"] = open_img(oid)
                    rec["other_image_ids"] = oids
                elif c.startswith("near:"):
                    nids = [i for i, _ in nn[x["image_id"]][: int(c.split(":")[1])]]
                    for j, nid in enumerate(nids):
                        conds[f"near_{j}"] = open_img(nid)
                    rec["near_image_ids"] = nids
                else:
                    raise ValueError(f"未知对照条件 {c}")
            ok = True
            for name, im in conds.items():
                inputs, start = build_inputs(processor, prompt, text, im, len(ref_ids))
                ids, lp, ent = token_logprobs(model, inputs, start)
                if ids != ref_ids:  # 上下文里的切分和单独切分不一致，就没法按字符区间对齐
                    ok = False
                    break
                rec["logp"][name] = lp
                if name == "real":
                    rec["entropy"] = ent
            if not ok:
                n_bad += 1
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 10 == 0:
                print(f"{k}/{len(xs)}  {time.time() - t0:.0f}s", flush=True)

    summary = {"n_total": len(xs), "n_skipped_tokenize_mismatch": n_bad, "seconds": round(time.time() - t0, 1),
               "gpu": torch.cuda.get_device_name(0)}
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(summary)


if __name__ == "__main__":
    main()
