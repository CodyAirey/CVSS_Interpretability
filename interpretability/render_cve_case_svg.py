#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Optional deps (LIME/SHAP)
try:
    import shap  # type: ignore
except Exception:
    shap = None
try:
    from lime.lime_text import LimeTextExplainer  # type: ignore
except Exception:
    LimeTextExplainer = None

# SVG writing
try:
    import svgwrite  # type: ignore
except Exception:
    svgwrite = None

# ---- repo-root for local imports ----
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Your utilities
from utils.cvss_data_engine_joana.CVSSDataset_modified import read_cvss_csv_with_ids
from utils.text_normalise import normalise_text
from utils.cvss_mappings import METRIC_TO_CLASSES
from utils.model_loader import load_model_and_tokenizer

from captum.attr import IntegratedGradients


# ---------------- device ----------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# ---------------- token utilities ----------------
_WORDPIECE_RE = re.compile(r"^##")
_SENTPIECE_PREFIX = "▁"


def _sanitize_ids_for_model(input_ids: torch.Tensor, tok, model) -> torch.Tensor:
    """
    Map any token id >= model's embedding size to UNK to prevent index errors.
    """
    if not torch.is_tensor(input_ids):
        return input_ids
    emb = model.get_input_embeddings()
    if emb is None:
        return input_ids
    vocab_n = int(emb.num_embeddings)
    if vocab_n <= 0:
        return input_ids
    unk_id = tok.unk_token_id if tok.unk_token_id is not None else 0
    if torch.any(input_ids >= vocab_n):
        ids = input_ids.clone()
        ids[ids >= vocab_n] = unk_id
        return ids
    return input_ids



def infer_metric_from_run_dir(run_dir: Path) -> Optional[str]:
    last = run_dir.name.lower()
    if last in METRIC_TO_CLASSES:
        return last
    parent = run_dir.parent.name.lower()
    if parent in METRIC_TO_CLASSES:
        return parent
    return None




# ---------------- predictions ----------------
@torch.inference_mode()
def probs_from_texts(model, tok, texts: List[str], max_length=512, batch_size=8) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        ids = enc["input_ids"].long()
        mask = enc["attention_mask"]
        ids = _sanitize_ids_for_model(ids, tok, model)
        ids = ids.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)
        logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits, -1).detach().cpu().numpy()
        out.append(probs)
        del enc, ids, mask, logits
    if not out:
        return np.zeros((0, getattr(getattr(model, "config", None), "num_labels", 1)))
    return np.vstack(out)


def predict_class_and_prob(model, tok, text: str, max_length=512) -> Tuple[int, float, np.ndarray]:
    probs = probs_from_texts(model, tok, [text], max_length=max_length, batch_size=1)
    if probs.size == 0:
        return 0, 0.0, probs
    pred = int(probs.argmax(axis=1)[0])
    p = float(probs[0, pred])
    return pred, p, probs


# ---------------- attributions ----------------
def _strip_specials(tok, ids_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
    pad_id = getattr(tok, "pad_token_id", None)
    cls_id = getattr(tok, "cls_token_id", None)
    sep_id = getattr(tok, "sep_token_id", None)

    special_mask = np.zeros_like(ids_np, dtype=bool)
    if pad_id is not None:
        special_mask |= (ids_np == pad_id)
    if cls_id is not None:
        special_mask |= (ids_np == cls_id)
    if sep_id is not None:
        special_mask |= (ids_np == sep_id)

    keep = (~special_mask) & (mask_np > 0)
    return keep


def ig_attrib(model, tok, text: str, target: int, max_length: int = 512) -> Tuple[List[str], np.ndarray]:
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    ids = enc["input_ids"].long()
    mask = enc["attention_mask"]

    ids = _sanitize_ids_for_model(ids, tok, model)
    ids = ids.to(DEVICE, non_blocking=True)
    mask = mask.to(DEVICE, non_blocking=True)

    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        embeds = emb_layer(ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        pad_vec = emb_layer.weight[pad_id : pad_id + 1]
        baseline = pad_vec.unsqueeze(1).expand_as(embeds).clone()

    embeds.requires_grad_(True)
    mask_for_model = mask

    def _forward(e):
        use_amp = (DEVICE.type == "cuda") and (os.getenv("IG_USE_AMP", "1") == "1")
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(inputs_embeds=e, attention_mask=mask_for_model).logits
        else:
            logits = model(inputs_embeds=e, attention_mask=mask_for_model).logits
        return logits[:, target]

    ig = IntegratedGradients(_forward)
    n_steps = int(os.getenv("IG_STEPS", "128"))
    ibs = int(os.getenv("IG_INTERNAL_BS", "8"))

    attr = ig.attribute(
        embeds,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=ibs,
        return_convergence_delta=False,
    )
    per_tok = attr.sum(-1).squeeze(0).detach().cpu().numpy().astype(np.float32)

    ids_np = ids[0].detach().cpu().numpy()
    mask_np = mask[0].detach().cpu().numpy().astype(np.float32)
    per_tok = per_tok * mask_np

    keep = _strip_specials(tok, ids_np, mask_np)

    toks_all = tok.convert_ids_to_tokens(ids_np.tolist())
    toks = [t for t, k in zip(toks_all, keep) if k]
    per_tok = per_tok[keep]

    return toks, per_tok


def lrp_attrib_pred_only(model, tok, text: str, pred: int, max_length=512) -> Optional[Tuple[List[str], np.ndarray]]:
    if not hasattr(model, "relprop"):
        return None

    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    ids = _sanitize_ids_for_model(enc["input_ids"].long(), tok, model)
    mask = enc["attention_mask"]

    ids = ids.to(DEVICE, non_blocking=True)
    mask = mask.to(DEVICE, non_blocking=True)

    model.zero_grad(set_to_none=True)
    logits = model(input_ids=ids, attention_mask=mask).logits

    seed = torch.zeros_like(logits)
    seed[0, pred] = 1.0
    try:
        R_all = model.relprop(seed, alpha=1.0, attention_mask=mask)
    except TypeError:
        R_all = model.relprop(seed, alpha=1.0)

    if R_all.dim() == 3:
        R_tok = R_all.sum(-1).squeeze(0)
    elif R_all.dim() == 2:
        R_tok = R_all.squeeze(0)
    else:
        raise RuntimeError(f"Unexpected LRP tensor shape: {tuple(R_all.shape)}")

    R = R_tok.detach().cpu().numpy().astype(np.float32)
    ids_np = ids[0].detach().cpu().numpy()
    mask_np = mask[0].detach().cpu().numpy().astype(np.float32)

    R *= mask_np
    keep = _strip_specials(tok, ids_np, mask_np)

    toks_all = tok.convert_ids_to_tokens(ids_np.tolist())
    toks = [t for t, k in zip(toks_all, keep) if k]
    R = R[keep]
    R = np.clip(R, 0.0, None)

    return toks, R


def lime_attrib(
    model,
    tok,
    text: str,
    target: int,
    max_length: int = 512,
    batch_size: int = 8,
    num_samples: int = 1000,
) -> Tuple[List[str], np.ndarray]:
    if LimeTextExplainer is None:
        raise RuntimeError("LIME not installed (pip install lime)")
    try:
        C = int(getattr(getattr(model, "config", None), "num_labels", 2))
    except Exception:
        C = 2
    class_names = [str(i) for i in range(C)]
    explainer = LimeTextExplainer(class_names=class_names, bow=False)

    def f(X: List[str]) -> np.ndarray:
        return probs_from_texts(model, tok, X, max_length=max_length, batch_size=batch_size)

    ex = explainer.explain_instance(text, f, labels=[int(target)], num_samples=int(num_samples))
    contribs = dict(ex.as_map()[int(target)])
    words = re.findall(r"\S+", text)
    weights = np.zeros(len(words), dtype=float)
    for i, w in contribs.items():
        if 0 <= i < len(words):
            weights[i] = float(w)
    return words, weights


def shap_attrib(
    model,
    tok,
    text: str,
    target: int,
    max_length: int = 512,
    batch_size: int = 8,
    nsamples: int = 500,
) -> Tuple[List[str], np.ndarray]:
    if shap is None:
        raise RuntimeError("SHAP not installed (pip install shap)")

    def f(X) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = X.ravel().tolist()
        elif isinstance(X, str):
            X = [X]
        X = [str(x) for x in X]
        return probs_from_texts(model, tok, X, max_length=max_length, batch_size=batch_size)

    masker = shap.maskers.Text()
    explainer = shap.Explainer(f, masker)
    explanation = explainer(np.array([text], dtype=object), max_evals=int(nsamples))

    vals = np.array(explanation.values)[0]
    words = list(explanation.data[0])

    if vals.ndim == 2 and target < vals.shape[1]:
        weights = vals[:, target]
    else:
        weights = vals if vals.ndim == 1 else np.zeros(len(words), dtype=float)

    return words, np.asarray(weights, dtype=float)


# ---------------- SVG rendering ----------------
def _approx_text_w_px(s: str, font_px: int) -> int:
    return int(math.ceil(len(s) * font_px * 0.56))


def _wrap_text_to_lines(text: str, max_width_px: int, font_px: int) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return [""]

    words = text.split(" ")
    lines: List[str] = []
    cur = ""

    for w in words:
        candidate = w if not cur else (cur + " " + w)
        if _approx_text_w_px(candidate, font_px) <= max_width_px:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _measure_wrapped_paragraph_height(text: str, max_width_px: int, font_px: int, line_h_px: Optional[int] = None) -> int:
    if line_h_px is None:
        line_h_px = int(font_px * 1.35)
    lines = _wrap_text_to_lines(text, max_width_px, font_px)
    return max(1, len(lines)) * line_h_px


def _draw_wrapped_paragraph(
    dwg,
    x: int,
    y: int,
    text: str,
    max_width_px: int,
    *,
    font_px: int,
    line_h_px: Optional[int] = None,
    fill: str = "#111827",
    font_family: str = "system-ui, -apple-system, Segoe UI, sans-serif",
) -> int:
    if line_h_px is None:
        line_h_px = int(font_px * 1.35)

    lines = _wrap_text_to_lines(text, max_width_px, font_px)
    for i, ln in enumerate(lines):
        dwg.add(
            dwg.text(
                ln,
                insert=(x, y + i * line_h_px),
                font_size=font_px,
                fill=fill,
                font_family=font_family,
            )
        )
    return len(lines) * line_h_px


def _norm_signed(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    m = float(np.max(np.abs(w))) if w.size else 0.0
    return w / m if m > 0 else w


def _norm_nonneg(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    m = float(np.max(w)) if w.size else 0.0
    return w / m if m > 0 else w


def _colour_signed(w: float):
    w = max(min(w, 1.0), -1.0)
    opacity = 0.15 + 0.75 * abs(w)
    if w >= 0:
        return "rgb(255,0,0)", opacity
    return "rgb(0,0,255)", opacity


def _colour_orange(w: float):
    w = max(min(w, 1.0), 0.0)
    opacity = 0.15 + 0.75 * w
    return "rgb(255,165,0)", opacity


def _measure_tokens_height(
    tokens: List[str],
    font_px: int,
    cell_w_px: int,
    *,
    pad_px: int = 10,
    line_gap_px: int = 8,
) -> int:
    if not tokens:
        return pad_px * 2 + int(font_px * 1.35) + 8

    toks = [t.replace("▁", " ") for t in tokens]
    x = pad_px
    max_x = cell_w_px - pad_px
    token_h = int(font_px * 1.35)
    space_px = int(font_px * 0.55)

    lines = 1
    for t in toks:
        s = t
        if not s.strip():
            continue
        tw = _approx_text_w_px(s, font_px)
        box_w = tw + int(font_px * 0.9)
        if x + box_w > max_x:
            lines += 1
            x = pad_px
        x += box_w + space_px

    return pad_px * 2 + lines * token_h + (lines - 1) * line_gap_px + 8


def _draw_wrapped_tokens(
    dwg,
    x0: int,
    y0: int,
    w_px: int,
    h_px: int,
    tokens: List[str],
    weights: np.ndarray,
    *,
    signed: bool,
    font_px: int,
    pad_px: int = 10,
    line_gap_px: int = 8,
) -> None:
    dwg.add(
        dwg.rect(
            insert=(x0, y0),
            size=(w_px, h_px),
            fill="white",
            stroke="#E5E7EB",
            stroke_width=1,
            rx=8,
            ry=8,
        )
    )

    if not tokens or weights is None or len(tokens) == 0:
        dwg.add(
            dwg.text(
                "(no tokens)",
                insert=(x0 + pad_px, y0 + pad_px + font_px),
                font_size=font_px,
                fill="#6B7280",
                font_family="system-ui, -apple-system, Segoe UI, sans-serif",
            )
        )
        return

    toks = [t.replace("▁", " ") for t in tokens]
    w = np.asarray(weights, dtype=float)

    if signed:
        wn = _norm_signed(w)
        colour = _colour_signed
    else:
        wn = _norm_nonneg(w)
        colour = _colour_orange

    x = x0 + pad_px
    y = y0 + pad_px + font_px + 2
    token_h = int(font_px * 1.35)
    space_px = int(font_px * 0.55)
    max_x = x0 + w_px - pad_px

    for t, ww in zip(toks, wn):
        s = t
        if not s.strip():
            continue

        tw = _approx_text_w_px(s, font_px)
        box_w = tw + int(font_px * 0.9)

        if x + box_w > max_x:
            x = x0 + pad_px
            y += token_h + line_gap_px

        if y > y0 + h_px - pad_px:
            dwg.add(
                dwg.text(
                    "…",
                    insert=(x0 + w_px - pad_px - font_px, y0 + h_px - pad_px),
                    font_size=font_px + 6,
                    fill="#6B7280",
                    font_family="system-ui, -apple-system, Segoe UI, sans-serif",
                )
            )
            break

        fill_col, fill_opacity = colour(float(ww))

        dwg.add(
            dwg.rect(
                insert=(x, y - token_h + 4),
                size=(box_w, token_h),
                rx=4,
                ry=4,
                fill=fill_col,
                fill_opacity=fill_opacity,
                stroke="rgb(0,0,0)",
                stroke_opacity=0.08,
                stroke_width=1,
            )
        )
        dwg.add(
            dwg.text(
                s,
                insert=(x + int(font_px * 0.45), y),
                font_size=font_px,
                fill="#111827",
                font_family="system-ui, -apple-system, Segoe UI, sans-serif",
            )
        )
        x += box_w + space_px


def _draw_cell_title(dwg, x: int, y: int, text: str, font_px: int) -> None:
    dwg.add(
        dwg.text(
            text,
            insert=(x, y),
            font_size=font_px,
            fill="#111827",
            font_weight="600",
            font_family="system-ui, -apple-system, Segoe UI, sans-serif",
        )
    )


def render_big_svg(
    out_svg: Path,
    *,
    cve_id: str,
    metric: str,
    gt_idx: int,
    gt_label: str,
    description: str,
    model_infos: List[Dict],
    methods: List[str],
    attribs: Dict[Tuple[str, str], Tuple[List[str], np.ndarray, bool]],
    width_px: int,
    font_px: int,
) -> None:
    if svgwrite is None:
        raise RuntimeError("svgwrite not installed. Install inside your env: pip install svgwrite")

    # Layout constants
    margin = 24
    col_header_h = 72
    row_header_w = 150
    row_gap = 16
    col_gap = 16

    # Header typography
    title_font = font_px + 14
    desc_font = font_px + 4
    gap1 = 14   # title -> desc label
    gap3 = 18   # paragraph -> grid
    desc_line_h = int(desc_font * 1.35)

    # Cell width depends on model count
    cell_w = int(
        (width_px - (margin * 2) - row_header_w - (len(model_infos) - 1) * col_gap)
        / max(1, len(model_infos))
    )

    # Measure header height (NO DRAWING YET)
    desc_max_w = width_px - (2 * margin)
    title_y = margin + title_font
    desc_label_y = title_y + gap1 + desc_font
    desc_text_y  = desc_label_y + desc_line_h + 6   # <-- pushes paragraph below label
    desc_h = _measure_wrapped_paragraph_height(description, desc_max_w, desc_font)
    header_h = (desc_text_y - margin) + desc_h + gap3

    # Choose one cell height for everything, based on the largest token set
    pad_px = 10
    line_gap_px = 8
    max_cell_h = 0
    for mi in model_infos:
        for m in methods:
            key = (mi["label"], m)
            toks = attribs.get(key, ([], np.asarray([], dtype=float), True))[0]
            h = _measure_tokens_height(toks, font_px, cell_w, pad_px=pad_px, line_gap_px=line_gap_px)
            if h > max_cell_h:
                max_cell_h = h
    cell_h = max_cell_h + 10

    # Total SVG height
    height_px = (
        margin * 2
        + header_h
        + col_header_h
        + len(methods) * cell_h
        + (len(methods) - 1) * row_gap
    )

    # Now create drawing
    dwg = svgwrite.Drawing(str(out_svg), size=(width_px, height_px))
    dwg.add(dwg.rect(
    insert=(0, 0),
    size=(width_px, height_px),
    fill="white"
    ))

    # ---- Draw header ONCE ----
    x = margin
    dwg.add(
        dwg.text(
            f"{cve_id}  |  Metric {metric.upper()}  |  GT: {gt_label} ({gt_idx})",
            insert=(x, title_y),
            font_size=title_font,
            font_weight="800",
            fill="#111827",
            font_family="system-ui, -apple-system, Segoe UI, sans-serif",
        )
    )

    dwg.add(
        dwg.text(
            "Description:",
            insert=(x, desc_label_y),
            font_size=desc_font,
            font_weight="700",
            fill="#111827",
            font_family="system-ui, -apple-system, Segoe UI, sans-serif",
        )
    )

    _draw_wrapped_paragraph(
        dwg,
        x=x,
        y=desc_text_y,
        text=description,
        max_width_px=desc_max_w,
        font_px=desc_font,
    )

    # ---- Grid start ----
    top = margin + header_h
    left = margin + row_header_w

    # Column headers (models)
    for j, mi in enumerate(model_infos):
        cx = left + j * (cell_w + col_gap)
        cy = top

        dwg.add(
            dwg.rect(
                insert=(cx, cy),
                size=(cell_w, col_header_h),
                fill="white",
                stroke="#E5E7EB",
                stroke_width=1,
                rx=8,
                ry=8,
            )
        )

        label = mi["label"]
        pred_label = mi["pred_label"]
        pred_idx = mi["pred_idx"]
        prob = mi["prob"]

        _draw_cell_title(dwg, cx + 10, cy + 26, label, font_px)
        dwg.add(
            dwg.text(
                f"Pred: {pred_label} ({pred_idx})  p={prob:.3f}",
                insert=(cx + 10, cy + 52),
                font_size=font_px - 2,
                fill="#374151",
                font_family="system-ui, -apple-system, Segoe UI, sans-serif",
            )
        )

    # Rows and cells
    grid_top = top + col_header_h + 16
    for i, method in enumerate(methods):
        ry = grid_top + i * (cell_h + row_gap)

        # row header
        dwg.add(
            dwg.rect(
                insert=(margin, ry),
                size=(row_header_w - 16, cell_h),
                fill="white",
                stroke="#E5E7EB",
                stroke_width=1,
                rx=8,
                ry=8,
            )
        )
        dwg.add(
            dwg.text(
                method,
                insert=(margin + 16, ry + 38),
                font_size=font_px + 4,
                font_weight="800",
                fill="#111827",
                font_family="system-ui, -apple-system, Segoe UI, sans-serif",
            )
        )

        for j, mi in enumerate(model_infos):
            cx = left + j * (cell_w + col_gap)
            key = (mi["label"], method)

            if key not in attribs:
                dwg.add(
                    dwg.rect(
                        insert=(cx, ry),
                        size=(cell_w, cell_h),
                        fill="white",
                        stroke="#E5E7EB",
                        stroke_width=1,
                        rx=8,
                        ry=8,
                    )
                )
                dwg.add(
                    dwg.text(
                        "(not available)",
                        insert=(cx + 10, ry + 28),
                        font_size=font_px,
                        fill="#6B7280",
                        font_family="system-ui, -apple-system, Segoe UI, sans-serif",
                    )
                )
                continue

            toks, wts, signed = attribs[key]
            _draw_wrapped_tokens(
                dwg,
                cx,
                ry,
                cell_w,
                cell_h,
                toks,
                wts,
                signed=signed,
                font_px=font_px,
                pad_px=pad_px,
                line_gap_px=line_gap_px,
            )

    dwg.save()


# ---------------- main ----------------
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
    args = ap.parse_args()

    cve_id = str(args.cve_id).strip()
    out_svg = Path(args.out_svg).resolve()

    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    if not run_dirs:
        raise SystemExit("No --run_dirs provided")

    # ---- load models ----
    models = []
    metric_name = None
    for rd in run_dirs:
        cfg, tok, model, mtype = load_model_and_tokenizer(rd)
        if metric_name is None:
            metric_name = infer_metric_from_run_dir(rd)
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

    base_cfg = models[0]["cfg"]
    metric_display = metric_name if metric_name else "N/A"
    metric_classes = METRIC_TO_CLASSES.get(metric_name, None)

    # ---- load dataset with IDs/texts/labels ----
    csv_path = REPO_ROOT / "cve_data" / "extended_analysis" / base_cfg["data_file"]
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find model CSV at: {csv_path}")

    ids, texts, labels = read_cvss_csv_with_ids(
        str(csv_path),
        base_cfg["label_position"],
        base_cfg["classes"],
    )

    use_norm = bool(base_cfg.get("use_normalised_tokens", False))
    texts_norm = [normalise_text(t, enabled=use_norm) for t in texts]

    id_to_idx: Dict[str, int] = {str(cid): i for i, cid in enumerate(ids)}
    if cve_id not in id_to_idx:
        alt = None
        cve_id_norm = cve_id.strip().upper()
        for k in id_to_idx.keys():
            if str(k).strip().upper() == cve_id_norm:
                alt = k
                break
        if alt is None:
            raise SystemExit(f"CVE not found in dataset CSV: {cve_id}")
        cve_id = alt

    idx = id_to_idx[cve_id]
    text_raw = texts[idx]
    text_for_model = texts_norm[idx]
    gt_idx = int(labels[idx])
    try:
        gt_label = base_cfg["classes"][gt_idx]
    except Exception:
        gt_label = str(gt_idx)

    # ---- compute predictions and attributions ----
    model_infos: List[Dict] = []
    attribs: Dict[Tuple[str, str], Tuple[List[str], np.ndarray, bool]] = {}

    methods: List[str] = ["IG", "LRP"]
    if args.with_lime:
        methods.append("LIME")
    if args.with_shap:
        methods.append("SHAP")

    for m in models:
        pred_idx, prob, _ = predict_class_and_prob(m["model"], m["tok"], text_for_model, max_length=m["max_len"])

        if metric_classes is not None and 0 <= pred_idx < len(metric_classes):
            pred_label = metric_classes[pred_idx]
        else:
            try:
                pred_label = base_cfg["classes"][pred_idx]
            except Exception:
                pred_label = str(pred_idx)

        model_infos.append(dict(
            label=m["label"],
            pred_idx=pred_idx,
            pred_label=pred_label,
            prob=prob,
        ))

        ig_toks, ig_w = ig_attrib(m["model"], m["tok"], text_for_model, target=pred_idx, max_length=m["max_len"])
        attribs[(m["label"], "IG")] = (ig_toks, ig_w, True)

        lrp_res = lrp_attrib_pred_only(m["model"], m["tok"], text_for_model, pred=pred_idx, max_length=m["max_len"])
        if lrp_res is not None:
            lrp_toks, lrp_w = lrp_res
            attribs[(m["label"], "LRP")] = (lrp_toks, lrp_w, False)
        else:
            attribs[(m["label"], "LRP")] = ([], np.asarray([], dtype=float), False)

        if args.with_lime:
            try:
                lime_toks, lime_w = lime_attrib(
                    m["model"], m["tok"], text_for_model, target=pred_idx,
                    max_length=m["max_len"], batch_size=8, num_samples=int(args.lime_samples)
                )
                attribs[(m["label"], "LIME")] = (lime_toks, lime_w, True)
            except Exception as e:
                attribs[(m["label"], "LIME")] = ([f"LIME error: {e}"], np.asarray([0.0], dtype=float), True)

        if args.with_shap:
            try:
                shap_toks, shap_w = shap_attrib(
                    m["model"], m["tok"], text_for_model, target=pred_idx,
                    max_length=m["max_len"], batch_size=8, nsamples=int(args.shap_evals)
                )
                attribs[(m["label"], "SHAP")] = (shap_toks, shap_w, True)
            except Exception as e:
                attribs[(m["label"], "SHAP")] = ([f"SHAP error: {e}"], np.asarray([0.0], dtype=float), True)

    render_big_svg(
        out_svg,
        cve_id=cve_id,
        metric=metric_display,
        gt_idx=gt_idx,
        gt_label=gt_label,
        description=text_raw,
        model_infos=model_infos,
        methods=methods,
        attribs=attribs,
        width_px=int(args.svg_width),
        font_px=int(args.svg_font),
    )

    print(f"Wrote SVG: {out_svg}")


if __name__ == "__main__":
    main()
