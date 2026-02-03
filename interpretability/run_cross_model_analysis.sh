#!/usr/bin/env bash
set -euo pipefail

BASE=../saved_models/full_cve_ids_3.1_header/42
# METRICS=(av ac pr ui s c i a)
METRICS=(pr)

K=1000
BS=8
MAXLEN=512
# SAMPLE=stratified
FOCUS=mixed

start_time=$(date +%s)

for m in "${METRICS[@]}"; do
  echo "=== ${m} ==="
  metric_start=$(date +%s)
  python cross_model_anecdotal_mixed_correctness.py \
    --run_dirs \
      "${BASE}/xlnet/${m}" \
      "${BASE}/lrp-bert/${m}" \
      "${BASE}/lrp-distilbert/${m}" \
    -K "${K}" \
    --batch_size "${BS}" \
    --max_length "${MAXLEN}" \
    --focus "${FOCUS}" \
    --export_jsonl_full \
    # --score_mode abs
    # --export_jsonl "topk_tokens_none.jsonl" \
    # --topk 10 \
    # --sample_mode "${SAMPLE}"


  metric_end=$(date +%s)
  metric_runtime=$((metric_end - metric_start))
  printf "Runtime for %s: %d min %d sec\n\n" "$m" $((metric_runtime/60)) $((metric_runtime%60))
done


end_time=$(date +%s)
total_runtime=$((end_time - start_time))
printf "Total runtime: %d min %d sec\n" $((total_runtime/60)) $((total_runtime%60))