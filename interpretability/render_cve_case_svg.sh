#!/usr/bin/env bash
set -euo pipefail

python render_cve_case_svg.py \
  --cve_id CVE-2023-22311 \
  --run_dirs \
    ../saved_models/small_cve_ids_3.1_header/123/lrp-distilbert/pr \
  --out_svg figures/CVE-2023-22311_pr.svg \
  --with_lime --with_shap \
  # --cve_id CVE-2021-32665 \
    # ../saved_models/full_cve_ids_3.1_header/42/xlnet/pr \
    # ../saved_models/full_cve_ids_3.1_header/42/lrp-bert/pr \