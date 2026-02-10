#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, sys, json, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import pandas as pd
from safetensors.torch import load_file as st_load

# ---- Optional deps (LIME/SHAP) ----
try:
    import shap
except Exception:
    shap = None
try:
    from lime.lime_text import LimeTextExplainer
except Exception:
    LimeTextExplainer = None

# ---- repo-root for local imports ----
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# your utilities
from utils.cvss_data_engine_joana.CVSSDataset_modified import read_cvss_csv_with_ids
from utils.text_normalise import normalise_text

# transformers pieces (strict loader)
from transformers import (
    AutoTokenizer, AutoConfig,
    BertConfig, DistilBertConfig,
    XLNetForSequenceClassification, XLNetConfig,
)

# ---------------- metric mapping ----------------
METRIC_TO_CLASSES = {
    "av": ["NETWORK", "ADJACENT_NETWORK", "LOCAL", "PHYSICAL"],
    "ac": ["LOW", "HIGH"],
    "pr": ["NONE", "LOW", "HIGH"],
    "ui": ["NONE", "REQUIRED"],
    "s":  ["UNCHANGED", "CHANGED"],
    "c":  ["NONE", "LOW", "HIGH"],
    "i":  ["NONE", "LOW", "HIGH"],
    "a":  ["NONE", "LOW", "HIGH"],
}

# ---------------- justification columns ----------------
RATING_COLS = [
    "JustificationValue",
    "Explicitness",
    "Inference",
    "Completeness",
    "Ambiguity",
    "Contradiction",
]

TEXT_COLS = [
    "ExternalResources",
    "JustificationComment",
]

# ---------------- device ----------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# ---------------- token utilities ----------------
_WORDPIECE_RE = re.compile(r"^##")
_SENTPIECE_PREFIX = "▁"

def _clean_token(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t or ""

def _project_subwords_to_words(tokens: List[str], weights: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Map BERT/XLNet subwords to whole-word tokens with summed weights.
    (Useful if you want word-level introspection; for HTML we’ll keep subwords.)
    """
    words, wsum = [], []
    cur, acc = "", 0.0
    for tok, w in zip(tokens, weights):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith(_SENTPIECE_PREFIX):
            if cur:
                wc = _clean_token(cur)
                if wc:
                    words.append(wc); wsum.append(acc)
            cur = tok.lstrip(_SENTPIECE_PREFIX)
            acc = float(w)
        elif _WORDPIECE_RE.match(tok):
            cur += tok[2:]
            acc += float(w)
        else:
            if cur:
                wc = _clean_token(cur)
                if wc:
                    words.append(wc); wsum.append(acc)
            cur = tok
            acc = float(w)
    if cur:
        wc = _clean_token(cur)
        if wc:
            words.append(wc); wsum.append(acc)
    return words, np.asarray(wsum, dtype=float)

def _float_list(arr) -> List[float]:
    arr = np.asarray(arr, dtype=float)
    if arr.size:
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return [float(x) for x in arr.tolist()]

# ---------------- HTML helpers ----------------
CSS = """
<style>
body {
  margin: 24px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f9fafb;
}
.h1 {
  font-size: 22px;
  font-weight: 800;
  margin-bottom: 4px;
}
.meta {
  font-size: 12px;
  color: #6b7280;
  margin-bottom: 16px;
}
.cve-card {
  background: #ffffff;
  border-radius: 10px;
  border: 1px solid #e5e7eb;
  margin-bottom: 18px;
  padding: 14px 16px;
}
.cve-header {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 8px;
}
.cve-title {
  font-size: 18px;
  font-weight: 700;
}
.cve-sub {
  font-size: 12px;
  color: #6b7280;
}
.section-title {
  font-size: 14px;
  font-weight: 700;
  margin-top: 8px;
  margin-bottom: 4px;
}
.desc-box {
  background: #f9fafb;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 13px;
}
.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 10px;
  margin-top: 8px;
}
.model-card {
  border-radius: 8px;
  border: 1px solid #e5e7eb;
  padding: 8px;
  background: #fdfdfd;
}
.model-header {
  font-size: 13px;
  font-weight: 600;
}
.badge {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 999px;
  font-size: 11px;
  margin-left: 6px;
}
.tp { background:#e6ffed; color:#036b26; }
.fp { background:#ffeaea; color:#8a0b0b; }
.fn { background:#fff3cd; color:#8a6d3b; }
.tn { background:#e6f0ff; color:#1a4fcc; }
.mis { background:#fff3cd; color:#8a6d3b; }
.small {
  font-size: 11px;
  color: #6b7280;
}
.attr-block {
  margin-top: 6px;
  font-size: 12px;
  line-height: 1.7;
}
.attr-label {
  font-weight: 600;
  margin-bottom: 2px;
}
.comment-block {
  margin-top: 6px;
  padding: 6px 8px;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
  background: #f9fafb;
}
.comment-meta {
  font-size: 11px;
  color: #6b7280;
  margin-bottom: 2px;
}
.comment-text {
  font-size: 12px;
}
.token-span {
  padding: 0 2px;
  margin: 1px 1px 1px 0;
  border-radius: 3px;
  display: inline-block;
}
.topk-list {
  font-size: 11px;
  color: #374151;
  margin-top: 2px;
}
.topk-token {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
</style>
"""

def normalise_signed(w: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(w)) if w.size else 0
    return w / m if m > 0 else w

def normalise_nonneg(w: np.ndarray) -> np.ndarray:
    m = float(np.max(w)) if w.size else 0.0
    return (w / m) if m > 0 else w

def colour_for_weight_signed(w: float) -> str:
    # Red for positive, blue for negative
    w = max(min(w, 1.0), -1.0)
    a = int(30 + 200 * abs(w))
    alpha = a / 255.0
    if w >= 0:
        return f"rgba(255,0,0,{alpha:.3f})"
    else:
        return f"rgba(0,0,255,{alpha:.3f})"

def colour_for_weight_orange(w: float) -> str:
    # Orange scale for non-negative (LRP)
    w = max(min(w, 1.0), 0.0)
    a = int(30 + 200 * w)
    alpha = a / 255.0
    return f"rgba(255,165,0,{alpha:.3f})"

def render_inline(tokens: List[str], weights: np.ndarray, signed: bool = True) -> str:
    if not tokens or weights.size == 0:
        return "<span class='small'>(no tokens)</span>"
    weights = np.asarray(weights, dtype=float)
    if signed:
        w_norm = normalise_signed(weights)
        colour_fn = colour_for_weight_signed
    else:
        weights = np.clip(weights, 0.0, None)
        w_norm = normalise_nonneg(weights)
        colour_fn = colour_for_weight_orange

    spans = []
    for t, w in zip(tokens, w_norm):
        # strip special prefixes for readability
        tok = t.replace("▁", " ")
        safe = (
            tok.replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;")
        )
        spans.append(
            f"<span class='token-span' style='background:{colour_fn(float(w))}'>"
            f"{safe}</span>"
        )
    return "".join(spans)

def html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

def render_topk_list(tokens: List[str],
                     weights: np.ndarray,
                     k: int = 10,
                     signed: bool = True) -> str:
    """
    Render a small 'Top tokens' line: token (weight), sorted by |weight| (signed)
    or by value (non-negative).
    """
    if not tokens or weights is None:
        return ""

    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return ""

    if signed:
        scores = np.abs(w)
    else:
        scores = np.clip(w, 0.0, None)

    if scores.size == 0:
        return ""

    order = np.argsort(-scores)
    top = []
    for idx in order:
        if len(top) >= k:
            break
        if scores[idx] <= 0:
            # ignore zero-score tokens
            continue
        raw_tok = tokens[idx].replace("▁", " ")
        tok = raw_tok.strip()
        if not tok:
            continue
        val = float(w[idx])
        top.append(
            f"<span class='topk-token'>{html_escape(tok)}</span> "
            f"({val:.3f})"
        )

    if not top:
        return ""

    return "<div class='topk-list'><b>Top tokens:</b> " + ", ".join(top) + "</div>"

# ---------------- strict model loading ----------------
def load_cfg(run_dir: Path) -> Dict:
    p = run_dir / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"config.json not found under {run_dir}")
    return json.loads(p.read_text(encoding="utf-8"))

def _find_model_weights_strict(run_dir: Path):
    p = run_dir / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"Expected model.safetensors under {run_dir}")
    return st_load(str(p))

def _ensure_pad_token(tok, model_type: str):
    if tok.pad_token_id is None:
        if getattr(tok, "eos_token", None):
            tok.pad_token = tok.eos_token
        elif getattr(tok, "sep_token", None):
            tok.pad_token = tok.sep_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    if model_type == "xlnet":
        tok.padding_side = "left"

def _enforce_vocab_alignment(tok, model):
    emb = model.get_input_embeddings()
    if emb is None:
        return
    tok_len = int(len(tok))
    emb_len = int(emb.num_embeddings)
    if emb_len != tok_len:
        model.resize_token_embeddings(tok_len)

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

def _lrp_bert_load(run_dir: Path, tok):
    from utils.distilbert_lrp.bert_explainability.distilbert.BertForSequenceClassification import \
        BertForSequenceClassification as LRPBertCls
    hf_cfg = BertConfig.from_pretrained(run_dir, local_files_only=True)
    hf_cfg.vocab_size = len(tok)
    model = LRPBertCls(hf_cfg)
    state = _find_model_weights_strict(run_dir)
    model.load_state_dict(state, strict=False)
    model.resize_token_embeddings(len(tok))
    return model

def _lrp_distilbert_load(run_dir: Path, tok):
    from utils.distilbert_lrp.bert_explainability.distilbert.DistilBertForSequenceClassification import \
        DistilBertForSequenceClassification as LRPDistilCls
    hf_cfg = DistilBertConfig.from_pretrained(run_dir, local_files_only=True)
    hf_cfg.vocab_size = len(tok)
    model = LRPDistilCls(hf_cfg)
    state = _find_model_weights_strict(run_dir)
    model.load_state_dict(state, strict=False)
    model.resize_token_embeddings(len(tok))
    return model

def _xlnet_load(run_dir: Path, tok):
    state = _find_model_weights_strict(run_dir)
    emb_w = state["transformer.word_embedding.weight"]
    vocab_size_ckpt, d_model = emb_w.shape
    q_shape = state["transformer.layer.0.rel_attn.q"].shape
    n_head = int(q_shape[1]); d_head = int(q_shape[2])
    d_inner = int(state["transformer.layer.0.ff.layer_1.weight"].shape[0])
    n_layer = 0
    while f"transformer.layer.{n_layer}.ff.layer_1.weight" in state:
        n_layer += 1
    if "logits_proj.weight" in state:
        num_labels = int(state["logits_proj.weight"].shape[0])
    else:
        num_labels = int(load_cfg(run_dir).get("num_labels", 2))
    hf_cfg = XLNetConfig(
        model_type="xlnet",
        d_model=d_model, n_head=n_head, d_head=d_head,
        d_inner=d_inner, n_layer=n_layer,
        ff_activation="gelu", untie_r=True,
        vocab_size=vocab_size_ckpt, num_labels=num_labels,
    )
    model = XLNetForSequenceClassification(hf_cfg)
    model.load_state_dict(state, strict=False)
    if len(tok) != vocab_size_ckpt and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tok))
    return model

def load_model_and_tokenizer(run_dir: Path):
    cfg = load_cfg(run_dir)
    hfconf = AutoConfig.from_pretrained(run_dir, local_files_only=True)
    model_type = getattr(hfconf, "model_type", "").lower()
    use_fast = False if model_type == "xlnet" else True
    tok = AutoTokenizer.from_pretrained(run_dir, local_files_only=True, use_fast=use_fast)

    _ensure_pad_token(tok, model_type)
    if model_type == "xlnet":
        if (getattr(tok, "model_max_length", None) is None
            or tok.model_max_length <= 0
            or tok.model_max_length > 10000):
            tok.model_max_length = int(cfg.get("max_length", 512))

    if model_type == "xlnet":
        model = _xlnet_load(run_dir, tok)
    elif model_type == "bert":
        model = _lrp_bert_load(run_dir, tok)
    elif model_type == "distilbert":
        model = _lrp_distilbert_load(run_dir, tok)
    else:
        raise RuntimeError(f"Unsupported model_type for strict loader: {model_type}")
    _enforce_vocab_alignment(tok, model)
    model.to(DEVICE).eval()
    return cfg, tok, model, model_type

# ---------------- predictions ----------------
@torch.inference_mode()
def probs_from_texts(model, tok, texts: List[str], max_length=512, batch_size=8) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        ids  = enc["input_ids"].long()
        mask = enc["attention_mask"]
        ids  = _sanitize_ids_for_model(ids, tok, model)
        ids  = ids.to(DEVICE, non_blocking=True)
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
from captum.attr import IntegratedGradients

def ig_attrib(model, tok, text: str, target: int, max_length: int = 512) -> Tuple[List[str], np.ndarray]:
    enc  = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    ids  = enc["input_ids"].long()
    mask = enc["attention_mask"]

    ids  = _sanitize_ids_for_model(ids, tok, model)
    ids  = ids.to(DEVICE, non_blocking=True)
    mask = mask.to(DEVICE, non_blocking=True)

    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        embeds = emb_layer(ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        pad_vec = emb_layer.weight[pad_id: pad_id + 1]
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
    ibs     = int(os.getenv("IG_INTERNAL_BS", "8"))
    attr = ig.attribute(
        embeds,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=ibs,
        return_convergence_delta=False,
    )
    per_tok = attr.sum(-1).squeeze(0).detach().cpu().numpy().astype(np.float32)

    ids_np  = ids[0].detach().cpu().numpy()
    mask_np = mask[0].detach().cpu().numpy().astype(np.float32)
    per_tok = per_tok * mask_np

    # --- strip special tokens consistently ---
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

    toks_all = tok.convert_ids_to_tokens(ids_np.tolist())
    toks = [t for t, k in zip(toks_all, keep) if k]
    per_tok = per_tok[keep]

    return toks, per_tok


def lrp_attrib_pred_only(model, tok, text: str, pred: int, max_length=512) -> Optional[Tuple[List[str], np.ndarray]]:
    if not hasattr(model, "relprop"):
        return None

    enc  = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    ids  = _sanitize_ids_for_model(enc["input_ids"].long(), tok, model)
    mask = enc["attention_mask"]

    ids  = ids.to(DEVICE, non_blocking=True)
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
    ids_np  = ids[0].detach().cpu().numpy()
    mask_np = mask[0].detach().cpu().numpy().astype(np.float32)

    R *= mask_np

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

    # zero out specials
    R[special_mask] = 0.0

    # and then drop them completely
    keep = (~special_mask) & (mask_np > 0)

    toks_all = tok.convert_ids_to_tokens(ids_np.tolist())
    toks = [t for t, k in zip(toks_all, keep) if k]
    R = R[keep]
    R = np.clip(R, 0.0, None)

    return toks, R


def lime_attrib(model, tok, text: str, target: int,
                max_length: int = 512, batch_size: int = 8, num_samples: int = 1000
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

def shap_attrib(model, tok, text: str, target: int,
                max_length: int = 512, batch_size: int = 8, nsamples: int = 500
               ) -> Tuple[List[str], np.ndarray]:
    if shap is None:
        raise RuntimeError("SHAP not installed (pip install shap)")
    def f(X) -> np.ndarray:
        if isinstance(X, np.ndarray): X = X.ravel().tolist()
        elif isinstance(X, str): X = [X]
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

# ---------------- justification loading ----------------
def load_justifications(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).convert_dtypes()
    df.columns = df.columns.str.strip()
    df["CollectedDate"] = pd.to_datetime(df["CollectedDate"], dayfirst=True)

    int_cols = ["QIndex"] + list(RATING_COLS)
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    return df

def get_commented_cves(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Return mapping CVEID -> all rows (do not filter to only commented responses).
    """
    sub = df.copy()
    out: Dict[str, pd.DataFrame] = {}
    for cve, group in sub.groupby("CVEID"):
        out[str(cve)] = group.sort_values(["QuestionSetID", "QIndex"])
    return out

# ---------------- helpers ----------------
def infer_metric_from_run_dir(run_dir: Path) -> Optional[str]:
    last = run_dir.name.lower()
    if last in METRIC_TO_CLASSES:
        return last
    parent = run_dir.parent.name.lower()
    if parent in METRIC_TO_CLASSES:
        return parent
    return None

def conf_badge_binary(gt:int, pred:int) -> Tuple[str,str]:
    if gt==1 and pred==1: return "TP","tp"
    if gt==0 and pred==0: return "TN","tn"
    if gt==1 and pred==0: return "FN","fn"
    if gt==0 and pred==1: return "FP","fp"
    return "MIS","mis"

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Reviewer alignment HTML view: CVE comments vs model attributions.")
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="One or more run_dirs (…/saved_models/<dataset>/<seed>/<model>/<metric>)")
    ap.add_argument("--just_csv", required=True,
                    help="Justification CSV file (cve_survey_justification_responses*.csv)")
    ap.add_argument("--out_html", type=str, default="reviewer_alignment.html",
                    help="Output HTML path.")
    ap.add_argument("--max_cases", type=int, default=50,
                    help="Maximum number of commented CVEs to process (after dedup).")
    ap.add_argument("--with-lime", action="store_true",
                    help="Also run LIME (slow).")
    ap.add_argument("--with-shap", action="store_true",
                    help="Also run SHAP (slow).")
    args = ap.parse_args()

    def safe_str(val) -> str:
        """Convert possibly-NA value to a cleaned string ('' if NA/None)."""
        return "" if pd.isna(val) else str(val).strip()

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
            rd=rd, cfg=cfg, tok=tok, model=model, mtype=mtype,
            label=label, max_len=max_len
        ))

    base_cfg = models[0]["cfg"]
    metric_display = metric_name.upper() if metric_name else "N/A"

    # ---- rebuild full dataset with IDs/texts ----
    csv_path = REPO_ROOT / "cve_data" / "extended_analysis" / base_cfg["data_file"]
    ids, texts, labels = read_cvss_csv_with_ids(str(csv_path), base_cfg["label_position"], base_cfg["classes"])
    use_norm = bool(base_cfg.get("use_normalised_tokens", False))
    texts_norm = [normalise_text(t, enabled=use_norm) for t in texts]

    id_to_idx: Dict[str, int] = {}
    for i, cid in enumerate(ids):
        id_to_idx[str(cid)] = i

    # class list if metric known
    metric_classes = METRIC_TO_CLASSES.get(metric_name, None)

    # ---- load justifications + CVEs ----
    jdf = load_justifications(args.just_csv)

    # Choose top 10 most reviewed CVEs (by number of rows/ratings in the justification file)
    counts = (
        jdf["CVEID"]
        .astype("string")
        .dropna()
        .value_counts()
    )
    top10_ids = [str(x) for x in counts.head(10).index.tolist()]

    cve_to_rows = get_commented_cves(jdf)
    to_process = top10_ids

    if not to_process:
        print("No CVEs found in justification file.")
        return

    print(f"Rendering top 10 most reviewed CVEs to HTML: {len(to_process)} CVEs.")

    # ---- build HTML ----
    out_path = Path(args.out_html)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        fh.write("<title>Reviewer Alignment</title>")
        fh.write(CSS)
        fh.write("</head><body>\n")
        fh.write(f"<div class='h1'>Reviewer alignment – metric {metric_display}</div>\n")
        model_list = ", ".join(m["label"] for m in models)
        fh.write(f"<div class='meta'>Models: {model_list}<br>")
        fh.write(f"Justifications file: {os.path.basename(args.just_csv)}<br>")
        fh.write(f"Rendered CVEs: {len(to_process)} (top 10 most reviewed by count in justification file)</div>\n")

        for cve_id in to_process:
            if cve_id not in id_to_idx:
                # still show judgements even if no model text exists
                fh.write("<div class='cve-card'>")
                fh.write(f"<div class='cve-header'><div class='cve-title'>{cve_id}</div>")
                fh.write("<div class='cve-sub'>&nbsp;No matching CVE in model CSV.</div></div>")
                rows = cve_to_rows.get(cve_id, pd.DataFrame())
                fh.write("<div class='section-title'>Reviewer judgements / comments</div>")
                for _, r in rows.iterrows():
                    jv = r.get("JustificationValue", pd.NA)
                    expl = r.get("Explicitness", pd.NA)
                    inf  = r.get("Inference", pd.NA)
                    comp = r.get("Completeness", pd.NA)
                    amb  = r.get("Ambiguity", pd.NA)
                    contra = r.get("Contradiction", pd.NA)
                    ext = safe_str(r.get("ExternalResources"))
                    com = safe_str(r.get("JustificationComment"))

                    fh.write("<div class='comment-block'>")
                    fh.write("<div class='comment-meta'>")
                    fh.write(
                        f"JustificationValue (/7)={jv} · Likert ratings (/5): "
                        f"Explicitness={expl} · Inference={inf} · Completeness={comp} · "
                        f"Ambiguity={amb} · Contradiction={contra}"
                    )
                    fh.write("</div>")
                    if ext:
                        fh.write(
                            "<div class='comment-text'><b>ExternalResources:</b> "
                            + (ext.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
                            + "</div>"
                        )
                    if com:
                        fh.write(
                            "<div class='comment-text'><b>JustificationComment:</b> "
                            + (com.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
                            + "</div>"
                        )
                    fh.write("</div>")
                fh.write("</div>")
                continue

            idx = id_to_idx[cve_id]
            text_raw = texts[idx]
            text = texts_norm[idx]
            gt = int(labels[idx])
            try:
                gt_label = base_cfg["classes"][gt]
            except Exception:
                gt_label = None

            # Card header
            fh.write("<div class='cve-card'>\n")
            fh.write("<div class='cve-header'>")
            fh.write(f"<div class='cve-title'>{cve_id}</div>")
            if gt_label is not None:
                fh.write(f"<div class='cve-sub'>· Metric {metric_display} · GT={gt_label} ({gt})</div>")
            else:
                fh.write(f"<div class='cve-sub'>· Metric {metric_display} · GT index={gt}</div>")
            fh.write("</div>\n")

            # Description
            fh.write("<div class='section-title'>Description</div>")
            safe_desc = (
                text_raw.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
            )
            fh.write(f"<div class='desc-box'>{safe_desc}</div>\n")

            # Models + attributions
            fh.write("<div class='section-title'>Model predictions and attributions</div>\n")
            fh.write("<div class='models-grid'>\n")

            for m in models:
                pred_idx, p, _ = predict_class_and_prob(
                    m["model"], m["tok"], text, max_length=m["max_len"]
                )
                if metric_classes is not None and 0 <= pred_idx < len(metric_classes):
                    pred_label = metric_classes[pred_idx]
                else:
                    try:
                        pred_label = base_cfg["classes"][pred_idx]
                    except Exception:
                        pred_label = str(pred_idx)

                # badge
                num_classes = len(base_cfg.get("classes", []))
                if num_classes == 2:
                    tag, css = conf_badge_binary(gt, pred_idx)
                else:
                    tag, css = ("TP", "tp") if pred_idx == gt else ("MIS", "mis")

                fh.write("<div class='model-card'>")
                fh.write("<div class='model-header'>")
                fh.write(f"{m['label']} · Pred={pred_label} ({pred_idx}) · p={p:.3f}")
                fh.write(f"<span class='badge {css}'>{tag}</span>")
                fh.write("</div>")
                fh.write("<div class='small'>Rows: IG · LRP(+pred) · LIME (words) · SHAP (words)</div>")

                # IG
                ig_toks, ig_w = ig_attrib(m["model"], m["tok"], text, target=pred_idx, max_length=m["max_len"])
                ig_html = render_inline(ig_toks, ig_w, signed=True)
                fh.write("<div class='attr-block'>")
                fh.write("<div class='attr-label'>IG</div>")
                fh.write(ig_html)
                top_ig = render_topk_list(ig_toks, ig_w, k=10, signed=True)
                if top_ig:
                    fh.write(top_ig)
                fh.write("</div>")

                # LRP
                lrp_res = lrp_attrib_pred_only(m["model"], m["tok"], text, pred=pred_idx, max_length=m["max_len"])
                fh.write("<div class='attr-block'>")
                fh.write("<div class='attr-label'>LRP (+pred, positive only)</div>")
                if lrp_res is not None:
                    lrp_toks, lrp_w = lrp_res
                    lrp_html = render_inline(lrp_toks, lrp_w, signed=False)
                    fh.write(lrp_html)
                    top_lrp = render_topk_list(lrp_toks, lrp_w, k=10, signed=False)
                    if top_lrp:
                        fh.write(top_lrp)
                else:
                    fh.write("<span class='small'>n/a for this model</span>")
                fh.write("</div>")

                # LIME
                fh.write("<div class='attr-block'>")
                fh.write("<div class='attr-label'>LIME</div>")
                if args.with_lime:
                    try:
                        lime_tokens, lime_w = lime_attrib(
                            m["model"], m["tok"], text, target=pred_idx,
                            max_length=m["max_len"], batch_size=8, num_samples=1000
                        )
                        lime_html = render_inline(lime_tokens, lime_w, signed=True)
                        fh.write(lime_html)
                        top_lime = render_topk_list(lime_tokens, lime_w, k=10, signed=True)
                        if top_lime:
                            fh.write(top_lime)
                    except Exception as e:
                        fh.write(f"<span class='small'>LIME error: {e}</span>")
                else:
                    fh.write("<span class='small'>(not run; pass --with-lime)</span>")
                fh.write("</div>")

                # SHAP
                fh.write("<div class='attr-block'>")
                fh.write("<div class='attr-label'>SHAP</div>")
                if args.with_shap:
                    try:
                        shap_tokens, shap_w = shap_attrib(
                            m["model"], m["tok"], text, target=pred_idx,
                            max_length=m["max_len"], batch_size=8, nsamples=500
                        )
                        shap_html = render_inline(shap_tokens, shap_w, signed=True)
                        fh.write(shap_html)
                        top_shap = render_topk_list(shap_tokens, shap_w, k=10, signed=True)
                        if top_shap:
                            fh.write(top_shap)
                    except Exception as e:
                        fh.write(f"<span class='small'>SHAP error: {e}</span>")
                else:
                    fh.write("<span class='small'>(not run; pass --with-shap)</span>")
                fh.write("</div>")

                fh.write("</div>")  # /model-card

            fh.write("</div>")  # /models-grid

            # Reviewer judgements / comments
            rows = cve_to_rows.get(cve_id, pd.DataFrame())
            fh.write("<div class='section-title'>Reviewer judgements / comments</div>")
            for _, r in rows.iterrows():
                jv = r.get("JustificationValue", pd.NA)
                expl = r.get("Explicitness", pd.NA)
                inf  = r.get("Inference", pd.NA)
                comp = r.get("Completeness", pd.NA)
                amb  = r.get("Ambiguity", pd.NA)
                contra = r.get("Contradiction", pd.NA)
                ext = safe_str(r.get("ExternalResources"))
                com = safe_str(r.get("JustificationComment"))

                fh.write("<div class='comment-block'>")
                fh.write("<div class='comment-meta'>")
                fh.write(
                    f"JustificationValue (/7)={jv} · Likert ratings (/5): "
                    f"Explicitness={expl} · Inference={inf} · Completeness={comp} · "
                    f"Ambiguity={amb} · Contradiction={contra}"
                )
                fh.write("</div>")
                if ext:
                    fh.write(
                        "<div class='comment-text'><b>ExternalResources:</b> "
                        + (ext.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
                        + "</div>"
                    )
                if com:
                    fh.write(
                        "<div class='comment-text'><b>JustificationComment:</b> "
                        + (com.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
                        + "</div>"
                    )
                fh.write("</div>")

            fh.write("</div>\n")  # /cve-card

        fh.write("</body></html>\n")

    print(f"HTML written to {out_path}")


if __name__ == "__main__":
    main()
