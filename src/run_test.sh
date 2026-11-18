#!/bin/bash
# 最终测试：HalLoc-Caption test 全部 6,546 条，3B、7B 依次跑，对照 = 4 张随机无关图（seed 0）。只跑这一次。
# 方案和阈值已在 val 上冻结：results/final/protocol_20261118.json（README「最终方案」节记了 SHA-256）。
# 在服务器作业目录下运行：
#   CUDA_VISIBLE_DEVICES=1 nohup bash src/run_test.sh > logs/test_final.log 2>&1 < /dev/null &
#   只补跑 7B：CUDA_VISIBLE_DEVICES=<空卡> TAGS=qwen25vl7b nohup bash src/run_test.sh > logs/test_final_7B.log 2>&1 < /dev/null &
set -eu
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
for TAG in ${TAGS:-qwen25vl3b qwen25vl7b}; do  # 只补跑某个模型：TAGS=qwen25vl7b
    case $TAG in
        qwen25vl3b) MODEL=$HOME/models/Qwen2.5-VL-3B-Instruct ;;
        qwen25vl7b) MODEL=$HOME/models/Qwen2.5-VL-7B-Instruct ;;
    esac
    OUT=results/20261118_test_${TAG}_caption_all
    echo "开始 $TAG $(date '+%F %T')"
    .venv/bin/python src/vig_score.py --model "$MODEL" --subset caption --split test --n 0 --seed 0 \
        --controls other:4 --out "$OUT"
    for L in official core; do .venv/bin/python src/eval_words.py "$OUT" --labels $L > "$OUT/eval_$L.txt"; done
    echo "完成 $TAG $(date '+%F %T')"
done
echo "结束 $(date '+%F %T')"
