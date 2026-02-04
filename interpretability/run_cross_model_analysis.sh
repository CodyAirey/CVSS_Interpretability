# #!/usr/bin/env bash
# set -euo pipefail

# BASE=../saved_models/full_cve_ids_3.1_header/42
# # METRICS=(av ac pr ui s c i a)
# METRICS=(pr)

# K=1000
# BS=8
# MAXLEN=512
# # SAMPLE=stratified
# FOCUS=mixed

# start_time=$(date +%s)

# for m in "${METRICS[@]}"; do
#   echo "=== ${m} ==="
#   metric_start=$(date +%s)
#   python cross_model_anecdotal_mixed_correctness.py \
#     --run_dirs \
#       "${BASE}/xlnet/${m}" \
#       "${BASE}/lrp-bert/${m}" \
#       "${BASE}/lrp-distilbert/${m}" \
#     -K "${K}" \
#     --batch_size "${BS}" \
#     --max_length "${MAXLEN}" \
#     --focus "${FOCUS}" \
#     --export_jsonl_full \
#     # --score_mode abs
#     # --export_jsonl "topk_tokens_none.jsonl" \
#     # --topk 10 \
#     # --sample_mode "${SAMPLE}"


#   metric_end=$(date +%s)
#   metric_runtime=$((metric_end - metric_start))
#   printf "Runtime for %s: %d min %d sec\n\n" "$m" $((metric_runtime/60)) $((metric_runtime%60))
# done


# end_time=$(date +%s)
# total_runtime=$((end_time - start_time))
# printf "Total runtime: %d min %d sec\n" $((total_runtime/60)) $((total_runtime%60))


#!/usr/bin/env bash
# set -euo pipefail

python3 cross_model_analysis.py \
  --run_dirs \
    ../saved_models/small_cve_ids_3.1_header/123/lrp-distilbert/pr \
    ../saved_models/small_cve_ids_3.1_header/123/lrp-bert/pr \
    ../saved_models/small_cve_ids_3.1_header/123/xlnet/pr \
  --out_dir cross_model_out/pr_seed42 \
  --per_bucket 12 \
  --seed 42 \
  --ig_steps 128 \
  --ig_internal_bs 8 \
  --ig_use_amp \
  --with_lime \
  --with_shap