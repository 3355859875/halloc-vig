#!/bin/bash
# 开发实验：多张无关图 vs CLIP 近邻图做对照，val caption 500 条（与 20261117_dev_qwen25vl3b_val500 同一批样本，seed 0）。
# 在服务器作业目录下运行：
#   CUDA_VISIBLE_DEVICES=1 nohup bash src/run_dev_multi.sh > logs/dev_multi_val500.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
MODEL=${MODEL:-$HOME/models/Qwen2.5-VL-3B-Instruct}
N=${N:-500}
OUT=${OUT:-results/20261117_dev_multi_qwen25vl3b_val${N}}
echo "开始 $(date '+%F %T')  GPU=${CUDA_VISIBLE_DEVICES:-未指定}  MODEL=$MODEL  N=$N"
.venv/bin/python src/vig_score.py --model "$MODEL" --subset caption --split val --n "$N" --seed 0 \
    --controls other:8,near:8 --nn-file results/clip_nn/clip_nn_val.json --out "$OUT"
for L in official core; do .venv/bin/python src/eval_words.py "$OUT" --labels $L; done
echo "结束 $(date '+%F %T')"
