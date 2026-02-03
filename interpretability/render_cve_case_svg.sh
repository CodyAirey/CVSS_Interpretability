#!/usr/bin/env bash
set -euo pipefail

python render_cve_case_svg_no_desc.py \
  --cve_id CVE-2021-32665 \
  --run_dirs \
    ../saved_models/full_cve_ids_3.1_header/42/xlnet/pr \
    ../saved_models/full_cve_ids_3.1_header/42/lrp-bert/pr \
    ../saved_models/full_cve_ids_3.1_header/42/lrp-distilbert/pr \
  --out_svg figures/cve_2021_32665_pr_no_desc.svg \
  --with_lime --with_shap \