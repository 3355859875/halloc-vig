"""演示网页后端（FastAPI）。前端是同目录 static/ 下的 HTML + CSS + JS。

接口：
    GET  /                     前端页面
    GET  /api/meta             阈值、模型、示例数量等
    GET  /api/examples         示例列表（可按 has_halluc / hit 过滤）
    GET  /api/example/{key}    一条示例的逐词结果
    GET  /api/image/{image_id} Visual Genome 图片（只接受数字 id）
    POST /api/caption          上传图片 → 模型写描述
    POST /api/detect           上传图片（或给示例图 id）+ 描述 → 逐词结果；带 save_as 时存成演示预设
    GET  /api/presets          演示预设列表（存好的在线检测结果，不用模型、秒出）
    GET  /api/preset/{name}    一个演示预设

只监听 127.0.0.1，不对外开放。模型在第一次在线请求时才加载；同一时间只处理一个在线请求。

用法（本机，在作业目录下）：
    ~/.local/bin/uv run --no-project --with fastapi --with uvicorn --with python-multipart \\
        --with torch --with torchvision --with "transformers>=4.57" --with accelerate \\
        --with pillow --with numpy --with pandas --with scikit-learn \\
        python src/demo/server.py --model ~/models/Qwen2.5-VL-3B-Instruct
然后浏览器打开 http://127.0.0.1:8000
"""

import argparse
import io
import json
import re
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import core

HERE = Path(__file__).resolve().parent
MAX_UPLOAD = 15 * 1024 * 1024
PRESETS = core.ROOT / "results" / "demo_presets"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def create_app(ref, scorer, model_name):
    app = FastAPI(title="图文幻觉词高亮器")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/api/meta")
    def meta():
        return {"threshold": ref.threshold, "k": core.K, "model": model_name,
                "model_loaded": scorer.model is not None, "device": scorer.dev,
                "n_examples": len(ref.index), "n_hit": sum(x["hit"] for x in ref.index),
                "default": next((x["key"] for x in ref.index if x["hit"]), ref.index[0]["key"] if ref.index else None)}

    @app.get("/api/examples")
    def examples(only: str = "all"):
        items = ref.index
        if only == "halluc":
            items = [x for x in items if x["has_halluc"]]
        elif only == "hit":
            items = [x for x in items if x["hit"]]
        elif only == "clean":
            items = [x for x in items if not x["has_halluc"]]
        return items

    @app.get("/api/example/{key}")
    def example(key: str):
        if key not in ref.examples:
            raise HTTPException(404, "没有这条示例")
        name, rec = ref.examples[key]
        out = ref.analyze(rec, name)
        out.update({"key": key, "id": rec["id"], "source": name, "with_labels": True})
        return out

    @app.get("/api/image/{image_id}")
    def image(image_id: str):
        if not image_id.isdigit():
            raise HTTPException(400, "图片 id 只能是数字")
        p = core.DATA / "vg_images" / f"{image_id}.jpg"
        if not p.exists():
            raise HTTPException(404, "没有这张图")
        return FileResponse(p, media_type="image/jpeg")

    async def read_image(file, image_id):
        if file is not None and file.filename:
            data = await file.read()
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "图片太大（上限 15 MB）")
            try:
                return Image.open(io.BytesIO(data)).convert("RGB")
            except UnidentifiedImageError:
                raise HTTPException(400, "读不了这张图片")
        if image_id and image_id.isdigit() and (core.DATA / "vg_images" / f"{image_id}.jpg").exists():
            return Image.open(core.DATA / "vg_images" / f"{image_id}.jpg").convert("RGB")
        raise HTTPException(400, "需要上传图片")

    @app.post("/api/caption")
    async def caption(file: UploadFile | None = File(None), image_id: str = Form("")):
        img = await read_image(file, image_id)
        t0 = time.time()
        cap = await _run(scorer.caption, img)
        return {"caption": cap, "seconds": round(time.time() - t0, 1), "device": scorer.dev}

    @app.get("/api/presets")
    def presets():
        if not PRESETS.exists():
            return []
        out = []
        for p in sorted(PRESETS.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({"name": p.stem, "title": d.get("title", p.stem)})
        return out

    @app.get("/api/preset/{name}")
    def preset(name: str):
        if not NAME_RE.match(name) or not (PRESETS / f"{name}.json").exists():
            raise HTTPException(404, "没有这个预设")
        return json.loads((PRESETS / f"{name}.json").read_text(encoding="utf-8"))

    @app.post("/api/detect")
    async def detect(text: str = Form(...), file: UploadFile | None = File(None), image_id: str = Form(""),
                     save_as: str = Form(""), title: str = Form("")):
        text = text.strip()
        if not text:
            raise HTTPException(400, "描述不能为空")
        if len(text) > 2000:
            raise HTTPException(400, "描述太长（上限 2000 字符）")
        img = await read_image(file, image_id)
        t0 = time.time()
        try:
            rec = await _run(scorer.score, img, text)
        except ValueError as e:
            raise HTTPException(422, str(e))
        out = ref.analyze(rec)
        out.update({"with_labels": False, "seconds": round(time.time() - t0, 1), "device": scorer.dev})
        if save_as:  # 只允许用示例图（数字 id）存预设，不保存用户上传的图片
            if not NAME_RE.match(save_as) or not image_id.isdigit():
                raise HTTPException(400, "预设名只能是字母数字下划线，且只能用示例图")
            PRESETS.mkdir(parents=True, exist_ok=True)
            out.update({"image_id": image_id, "title": title or save_as, "model": model_name})
            (PRESETS / f"{save_as}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out

    return app


async def _run(fn, *a):
    """模型推理放到线程池，不卡住事件循环。"""
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(fn, *a)


def main():
    root = core.ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "models" / "Qwen2.5-VL-3B-Instruct"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-run", default=str(root / "results/20261117_dev_multi_qwen25vl3b_val500"))
    ap.add_argument("--examples", nargs="*", default=["val=" + str(root / "results/20261117_dev_multi_qwen25vl3b_val500")],
                    help="来源名=结果目录，可多个（测试集跑完后可加 test=…）")
    ap.add_argument("--protocol", default=str(root / "results/final/protocol_20261118.json"))
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    ref = core.Reference(args.val_run, [e.split("=", 1) for e in args.examples], args.protocol)
    scorer = core.Scorer(args.model, args.device)
    print(f"示例 {len(ref.index)} 条（命中 {sum(x['hit'] for x in ref.index)} 条）；冻结阈值 {ref.threshold:.4f}")
    uvicorn.run(create_app(ref, scorer, Path(args.model).name), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
