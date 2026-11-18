#!/bin/bash
# 可行性小实验：val caption 抽 50 条，Qwen2.5-VL-3B，三种对照。在服务器作业目录下运行：
#   CUDA_VISIBLE_DEVICES=1 nohup bash src/run_feasibility.sh > logs/feasibility.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export OMP_NUM_THREADS=4
# 服务器上 hf-mirror 较慢，模型改从 ModelScope 下到 ~/models/ 再按本地路径加载
MODEL=${MODEL:-$HOME/models/Qwen2.5-VL-3B-Instruct}
OUT=results/20261117_feasibility_qwen25vl3b_val50
echo "开始 $(date '+%F %T')  GPU=${CUDA_VISIBLE_DEVICES:-未指定}  MODEL=$MODEL"
.venv/bin/python src/vig_score.py --model "$MODEL" --subset caption --split val --n 50 --seed 0 \
    --controls blank,noimg,other --out "$OUT"
.venv/bin/python src/eval_words.py "$OUT"
echo "结束 $(date '+%F %T')"
