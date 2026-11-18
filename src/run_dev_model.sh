#!/bin/bash
# 模型规模对比：换一个模型，在和 20261117_dev_multi_qwen25vl3b_val500 完全相同的 val 500 条、相同的 8 张随机无关图上打分。
# （vig_score.py 抽无关图时，近邻对照不消耗随机数，所以 --controls other:8 和 3B 那次 other:8,near:8 抽到的图逐条相同。）
# 在服务器作业目录下运行：
#   CUDA_VISIBLE_DEVICES=1 MODEL=$HOME/models/Qwen2.5-VL-7B-Instruct TAG=qwen25vl7b \
#       nohup bash src/run_dev_model.sh > logs/dev_model_qwen25vl7b.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
: "${MODEL:?要指定 MODEL}" "${TAG:?要指定 TAG}"
N=${N:-500}
OUT=${OUT:-results/$(date +%Y%m%d)_dev_model_${TAG}_val${N}}
echo "开始 $(date '+%F %T')  GPU=${CUDA_VISIBLE_DEVICES:-未指定}  MODEL=$MODEL  N=$N  OUT=$OUT"
.venv/bin/python src/vig_score.py --model "$MODEL" --subset caption --split val --n "$N" --seed 0 \
    --controls other:8 --out "$OUT"
for L in official core; do .venv/bin/python src/eval_words.py "$OUT" --labels $L; done
echo "结束 $(date '+%F %T')"
