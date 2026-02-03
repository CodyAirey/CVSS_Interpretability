#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---- repo-root for local imports ----
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from utils.model_loader import load_model_and_tokenizer
from utils.interpretability_compute import compute_case, infer_metric_from_run_dir
from utils.interpretability_render import render_case_svg


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render one big SVG model card for a single CVE: predictions + attributions for all models and methods."
    )
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="One or more run_dirs (…/saved_models/<dataset>/<seed>/<model>/<metric>)")
    ap.add_argument("--cve_id", required=True,
                    help="CVE ID to render (must exist in the model CSV).")
    ap.add_argument("--out_svg", type=str, default="cve_card.svg",
                    help="Output SVG path.")
    ap.add_argument("--with_lime", action="store_true",
                    help="Include LIME rows (slow).")
    ap.add_argument("--with_shap", action="store_true",
                    help="Include SHAP rows (slow).")
    ap.add_argument("--lime_samples", type=int, default=1000,
                    help="LIME num_samples.")
    ap.add_argument("--shap_evals", type=int, default=500,
                    help="SHAP max_evals.")
    ap.add_argument("--svg_width", type=int, default=1800,
                    help="SVG width in px.")
    ap.add_argument("--svg_font", type=int, default=18,
                    help="Base font size in px.")

    # IG knobs (were env vars before)
    ap.add_argument("--ig_steps", type=int, default=int(os.getenv("IG_STEPS", "128")))
    ap.add_argument("--ig_internal_bs", type=int, default=int(os.getenv("IG_INTERNAL_BS", "8")))
    ap.add_argument("--ig_use_amp", action="store_true", default=(os.getenv("IG_USE_AMP", "1") == "1"))

    args = ap.parse_args()

    out_svg = Path(args.out_svg).resolve()
    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    if not run_dirs:
        raise SystemExit("No --run_dirs provided")

    models: List[Dict] = []
    for rd in run_dirs:
        cfg, tok, model, mtype = load_model_and_tokenizer(rd)
        max_len = int(cfg.get("max_length", 512))
        label = f"{rd.parent.name}/{rd.name}"
        models.append(dict(
            rd=rd,
            cfg=cfg,
            tok=tok,
            model=model,
            mtype=mtype,
            label=label,
            max_len=max_len,
        ))

    case = compute_case(
        models=models,
        cve_id=str(args.cve_id).strip(),
        repo_root=REPO_ROOT,
        with_lime=bool(args.with_lime),
        with_shap=bool(args.with_shap),
        lime_samples=int(args.lime_samples),
        shap_evals=int(args.shap_evals),
        ig_steps=int(args.ig_steps),
        ig_internal_bs=int(args.ig_internal_bs),
        ig_use_amp=bool(args.ig_use_amp),
    )

    render_case_svg(
        out_svg,
        cve_id=case["cve_id"],
        metric=case["metric"],
        gt_idx=case["gt_idx"],
        gt_label=case["gt_label"],
        description=case["description"],
        model_infos=case["model_infos"],
        methods=case["methods"],
        attribs=case["attribs"],
        width_px=int(args.svg_width),
        font_px=int(args.svg_font),
    )

    print(f"Wrote SVG: {out_svg}")


if __name__ == "__main__":
    main()
