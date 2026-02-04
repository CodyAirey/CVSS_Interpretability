#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---- repo-root for local imports ----
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from utils.model_loader import load_model_and_tokenizer
from utils.interpretability_compute import (
    compute_case,
    probs_from_texts,
)
from utils.interpretability_render import render_case_svg
from utils.cvss_data_engine_joana.CVSSDataset_modified import read_cvss_csv_with_ids
from utils.text_normalise import normalise_text


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class LoadedModel:
    run_dir: Path
    label: str
    cfg: Dict
    tok: object
    model: object
    model_type: str
    max_len: int


# ----------------------------
# Base components
# ----------------------------
def load_models(run_dirs: List[Path]) -> List[LoadedModel]:
    models: List[LoadedModel] = []
    for rd in run_dirs:
        cfg, tok, model, mtype = load_model_and_tokenizer(rd)
        max_len = int(cfg.get("max_length", 512))
        label = f"{rd.parent.name}/{rd.name}"  # eg: distilbert/av
        models.append(
            LoadedModel(
                run_dir=rd,
                label=label,
                cfg=cfg,
                tok=tok,
                model=model,
                model_type=mtype,
                max_len=max_len,
            )
        )
    if not models:
        raise SystemExit("No run_dirs loaded.")
    return models


def load_dataset_for_models(models: List[LoadedModel]) -> Tuple[List[str], List[str], np.ndarray, Dict]:
    """
    Uses the first model's config.json to locate the dataset CSV and class mapping.
    Returns (ids, texts_for_model, labels_np, base_cfg)
    """
    base_cfg = models[0].cfg
    data_file = base_cfg.get("data_file")
    if not data_file:
        raise SystemExit("config.json missing 'data_file' in the run_dir.")

    csv_path = REPO_ROOT / "cve_data" / "extended_analysis" / str(data_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset CSV: {csv_path}")

    ids, texts, labels = read_cvss_csv_with_ids(
        str(csv_path),
        int(base_cfg["label_position"]),
        list(base_cfg["classes"]),
    )

    use_norm = bool(base_cfg.get("use_normalised_tokens", False))
    texts_norm = [normalise_text(t, enabled=use_norm) for t in texts]

    labels_np = np.asarray(labels, dtype=int)
    ids_str = [str(x) for x in ids]
    return ids_str, texts_norm, labels_np, base_cfg


def batch_predict_all(models: List[LoadedModel], texts: List[str]) -> Dict[str, np.ndarray]:
    """
    Returns dict: model_label -> pred_idx array [N]
    """
    preds: Dict[str, np.ndarray] = {}
    for m in models:
        probs = probs_from_texts(m.model, m.tok, texts, max_length=m.max_len, batch_size=8)
        pred = probs.argmax(axis=1).astype(int)
        preds[m.label] = pred
    return preds


def bucket_indices(preds_by_model: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Buckets:
      - all_correct: every model matches GT
      - all_wrong: every model mismatches GT
      - mixed: at least one correct and at least one wrong
    """
    labels = list(preds_by_model.keys())
    if not labels:
        raise ValueError("No model predictions provided.")

    pred_mat = np.stack([preds_by_model[k] for k in labels], axis=1)  # [N, M]
    correct_mat = (pred_mat == y_true[:, None])

    all_correct = np.where(np.all(correct_mat, axis=1))[0]
    all_wrong = np.where(~np.any(correct_mat, axis=1))[0]
    mixed = np.where(np.any(correct_mat, axis=1) & ~np.all(correct_mat, axis=1))[0]

    return {
        "all_correct": all_correct,
        "mixed": mixed,
        "all_wrong": all_wrong,
    }


def sample_bucket(rng: np.random.Generator, idxs: np.ndarray, k: int) -> np.ndarray:
    if idxs.size == 0:
        return idxs
    if idxs.size <= k:
        return idxs
    return rng.choice(idxs, size=k, replace=False)


def write_index_html(out_dir: Path, manifest: Dict) -> None:
    """
    manifest structure:
      {
        "meta": {...},
        "buckets": {
          bucket_name: [{"cve_id": "...", "svg": "relative/path.svg", ...}, ...]
        }
      }
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "index.html"

    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #111827; }
    h1 { margin: 0 0 8px 0; font-size: 22px; }
    .meta { margin: 0 0 18px 0; color: #374151; }
    h2 { margin-top: 26px; font-size: 18px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
    .card { border: 1px solid #E5E7EB; border-radius: 12px; padding: 12px; }
    .card .title { font-weight: 700; margin-bottom: 8px; }
    .card object { width: 100%; height: auto; }
    .hint { color: #6B7280; font-size: 13px; margin-top: 6px; }
    """

    with p.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>")
        f.write("<title>Cross-model analysis</title>")
        f.write(f"<style>{css}</style></head><body>")

        meta = manifest.get("meta", {})
        f.write("<h1>Cross-model analysis</h1>")
        f.write("<div class='meta'>")
        f.write(f"Dataset: {meta.get('dataset','?')}<br>")
        f.write(f"Seed: {meta.get('seed','?')}<br>")
        f.write(f"Metric: {meta.get('metric','?')}<br>")
        f.write(f"Models: {meta.get('models','?')}")
        f.write("</div>")

        f.write("<div class='hint'>Each card is a single SVG combining predictions + attribution rows across models.</div>")

        buckets = manifest.get("buckets", {})
        for bname, items in buckets.items():
            f.write(f"<h2>{bname} ({len(items)})</h2>")
            f.write("<div class='grid'>")
            for it in items:
                title = f"{it['cve_id']} | GT={it['gt_label']} | {it['bucket']}"
                svg_rel = it["svg"]
                f.write("<div class='card'>")
                f.write(f"<div class='title'>{title}</div>")
                f.write(f"<object data='{svg_rel}' type='image/svg+xml'></object>")
                f.write("</div>")
            f.write("</div>")

        f.write("</body></html>")


# ----------------------------
# Orchestration
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-model anecdotal selection + rendering, using shared utils.")
    ap.add_argument("--run_dirs", nargs="+", required=True, help="Run dirs (…/saved_models/<dataset>/<seed>/<model>/<metric>)")
    ap.add_argument("--out_dir", type=str, default="cross_model_out", help="Output directory.")
    ap.add_argument("--per_bucket", type=int, default=12, help="How many examples to render per bucket.")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    ap.add_argument("--with_lime", action="store_true")
    ap.add_argument("--with_shap", action="store_true")
    ap.add_argument("--lime_samples", type=int, default=1000)
    ap.add_argument("--shap_evals", type=int, default=500)
    ap.add_argument("--ig_steps", type=int, default=128)
    ap.add_argument("--ig_internal_bs", type=int, default=8)
    ap.add_argument("--ig_use_amp", action="store_true", default=True)
    ap.add_argument("--svg_width", type=int, default=1800)
    ap.add_argument("--svg_font", type=int, default=18)
    args = ap.parse_args()

    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))

    models = load_models(run_dirs)
    ids, texts_norm, y_true, base_cfg = load_dataset_for_models(models)

    preds_by_model = batch_predict_all(models, texts_norm)
    buckets = bucket_indices(preds_by_model, y_true)

    # meta for the report
    metric = models[0].run_dir.name
    seed_str = models[0].run_dir.parent.parent.name
    dataset = models[0].run_dir.parent.parent.parent.name
    model_list = ", ".join(m.label for m in models)

    manifest = {
        "meta": {
            "dataset": dataset,
            "seed": seed_str,
            "metric": metric,
            "models": model_list,
        },
        "counts": {k: int(v.size) for k, v in buckets.items()},
        "buckets": {},
    }

    # render selected cases
    for bucket_name, idxs in buckets.items():
        chosen = sample_bucket(rng, idxs, int(args.per_bucket))
        bucket_dir = out_dir / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)

        items = []
        for j, idx in enumerate(chosen, start=1):
            cve_id = ids[int(idx)]
            gt_idx = int(y_true[int(idx)])
            try:
                gt_label = base_cfg["classes"][gt_idx]
            except Exception:
                gt_label = str(gt_idx)

            svg_name = f"{j:03d}_{cve_id}.svg"
            svg_path = bucket_dir / svg_name

            case = compute_case(
                models=[dict(
                    rd=m.run_dir,
                    cfg=m.cfg,
                    tok=m.tok,
                    model=m.model,
                    mtype=m.model_type,
                    label=m.label,
                    max_len=m.max_len,
                ) for m in models],
                cve_id=str(cve_id).strip(),
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
                svg_path,
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

            items.append({
                "bucket": bucket_name,
                "cve_id": case["cve_id"],
                "gt_label": case["gt_label"],
                "svg": f"{bucket_name}/{svg_name}",
            })

        manifest["buckets"][bucket_name] = items

    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_index_html(out_dir, manifest)
    print(f"Wrote: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
