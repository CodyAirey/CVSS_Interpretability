#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, sys, gc, re, random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
from safetensors.torch import load_file as st_load
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# Captum attributions
from captum.attr import IntegratedGradients  # IG on inputs_embeds

# Optional deps (LIME/SHAP) — we REQUIRE them; fail fast if missing
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

# ---------------- sanity / guards ----------------
def _require_optional_deps():
    missing = []
    if LimeTextExplainer is None:
        missing.append("lime (pip install lime)")
    if shap is None:
        missing.append("shap (pip install shap)")
    if missing:
        raise RuntimeError("Missing required optional dependencies: " + ", ".join(missing))

def _enforce_vocab_alignment(tok, model):
    """Ensure model's embedding size matches tokenizer length."""
    emb = model.get_input_embeddings()
    if emb is None:
        return
    tok_len = int(len(tok))
    emb_len = int(emb.num_embeddings)
    if emb_len != tok_len:
        model.resize_token_embeddings(tok_len)

def _sanitize_ids_for_model(input_ids: torch.Tensor, tok, model) -> torch.Tensor:
    """
    Map any token id >= model's embedding size to UNK to prevent CUDA indexSelect asserts.
    Call this *before* moving tensors to device.
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

def sanitize_for_filename(s: str) -> str:
    import re
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))
    s = re.sub(r'_+', '_', s).strip('_')
    return s or "model"

def _float_list(arr) -> List[float]:
    arr = np.asarray(arr, dtype=float)
    if arr.size:
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return [float(x) for x in arr.tolist()]

def _choose_from_pool(pool_idx: np.ndarray, k: int, seed: int) -> List[int]:
    if pool_idx.size == 0:
        return []
    rng = np.random.RandomState(seed)
    if pool_idx.size <= k:
        return pool_idx.tolist()
    sel = rng.choice(pool_idx, size=k, replace=False)
    return sel.tolist()

# ---------- Common token space + top-k ----------
_WORDPIECE_RE = re.compile(r"^##")
_SENTPIECE_PREFIX = "▁"

def _clean_token(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t or ""

def _project_subwords_to_words(tokens: List[str], weights: np.ndarray) -> Tuple[List[str], np.ndarray]:
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

def _topk_tokens(tokens: List[str], weights: np.ndarray, k: int, score_mode: str) -> List[str]:
    if score_mode == "pos":
        scores = np.clip(weights, 0.0, None)
    else:
        scores = np.abs(weights)
    order = np.argsort(-scores)
    out, seen = [], set()
    for i in order:
        t = _clean_token(tokens[i])
        if not t or t in seen:
            continue
        out.append(t); seen.add(t)
        if len(out) == k:
            break
    return out

def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not (A or B):
        return 1.0
    return len(A & B) / float(len(A | B))

# ---------------- config / device ----------------
def load_cfg(run_dir: Path) -> Dict:
    p = run_dir / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"config.json not found under {run_dir}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# ---------------- strict weights ----------------
def _find_model_weights_strict(run_dir: Path):
    p = run_dir / "model.safetensors"
    if not p.exists():
        raise FileNotFoundError(f"Expected model.safetensors under {run_dir}")
    return st_load(str(p))

# ---------------- strict tokenizer+model loading ----------------
from transformers import (
    AutoTokenizer, AutoConfig,
    BertConfig, DistilBertConfig,
    XLNetForSequenceClassification, XLNetConfig,
)

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

# ---------------- data rebuild (same split) ----------------
def rebuild_test_with_ids(cfg: Dict):
    csv_path = REPO_ROOT / "cve_data" / "extended_analysis" / cfg["data_file"]
    ids, texts, labels = read_cvss_csv_with_ids(str(csv_path), cfg["label_position"], cfg["classes"])
    _, x_test, _, y_test, _, id_test = train_test_split(
        texts, labels, ids,
        test_size=cfg["splits"]["requested"]["test_size"],
        random_state=cfg["seed"],
        stratify=labels
    )
    return x_test, list(map(int, y_test)), id_test

# ---------------- lengths ----------------
def effective_max_len(tok, model_cfg, cfg_max_len: int) -> int:
    tok_limit = getattr(tok, "model_max_length", None)
    if tok_limit is None or tok_limit <= 0 or tok_limit > 10000:
        tok_limit = cfg_max_len
    model_limit = getattr(model_cfg, "max_position_embeddings", None)
    if model_limit is None or model_limit <= 0:
        model_limit = cfg_max_len
    out = min(int(cfg_max_len), int(tok_limit), int(model_limit))
    if out <= 0 or out > 4096:
        out = int(cfg_max_len if cfg_max_len > 0 else 512)
    return out

# ---------------- predictions ----------------
@torch.inference_mode()
def probs_from_texts(model, tok, texts: List[str], max_length=512, batch_size=1) -> np.ndarray:
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
    return np.vstack(out) if out else np.zeros((0, getattr(getattr(model, "config", None), "num_labels", 1)))

def predict_classes(model, tok, texts: List[str], max_length=512, batch_size=32) -> Tuple[np.ndarray, np.ndarray]:
    probs = probs_from_texts(model, tok, texts, max_length=max_length, batch_size=batch_size)
    preds = probs.argmax(axis=1) if probs.size else np.zeros((0,), dtype=int)
    return preds, probs

# ---------------- sampling ----------------
def sample_indices_stratified(labels: List[int], k: int, seed: int) -> List[int]:
    k = min(k, len(labels))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=k, random_state=seed)
    (_, sample_idx), = sss.split(np.zeros(len(labels)), labels)
    return list(sample_idx)

def sample_balanced_by_reference(labels: List[int], ref_preds: List[int], k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    buckets: Dict[Tuple[int,int], List[int]] = {}
    for i, (g, p) in enumerate(zip(labels, ref_preds)):
        buckets.setdefault((int(g), int(p)), []).append(i)
    nonempty = [c for c, idxs in buckets.items() if idxs]
    if not nonempty:
        return sample_indices_stratified(labels, k, seed)
    k = min(k, len(labels))
    base = max(1, k // len(nonempty)); rem = k - base*len(nonempty)
    out: List[int] = []
    for c in nonempty:
        pool = buckets[c][:]; rng.shuffle(pool)
        out += pool[:min(base, len(pool))]
    if len(out) < k:
        need = k - len(out)
        flat = [i for c in nonempty for i in buckets[c] if i not in out]
        rng.shuffle(flat); out += flat[:need]
    return out[:k]

# ---------------- rendering helpers ----------------
def normalise_signed(w: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(w)) if w.size else 0
    return w / m if m > 0 else w

def normalise_nonneg(w: np.ndarray) -> np.ndarray:
    m = float(np.max(w)) if w.size else 0.0
    return (w / m) if m > 0 else w

def colour_for_weight_signed(w: float) -> str:
    w = max(min(w, 1.0), -1.0)
    a = int(30 + 200 * abs(w))
    return (f"rgba(255,0,0,{a/255.0:.3f})" if w >= 0 else f"rgba(0,0,255,{a/255.0:.3f})")

def colour_for_weight_orange(w: float) -> str:
    w = max(min(w, 1.0), 0.0)
    a = int(30 + 200 * w)
    return f"rgba(255,165,0,{a/255.0:.3f})"

def render_inline(tokens: List[str], weights: np.ndarray) -> str:
    weights = normalise_signed(weights)
    spans = []
    for t, w in zip(tokens, weights):
        safe = t.replace("<", "&lt;").replace(">", "&gt;")
        spans.append(
            f"<span style='background:{colour_for_weight_signed(float(w))}; "
            "padding:1px 2px; margin:1px; border-radius:3px'>"
            f"{safe}</span>"
        )
    return " ".join(spans)

def render_inline_lrp_pos(tokens: List[str], weights_pos: np.ndarray) -> str:
    w = np.clip(weights_pos, 0.0, None)
    w = normalise_nonneg(w)
    spans = []
    for t, val in zip(tokens, w):
        safe = t.replace("<", "&lt;").replace(">", "&gt;")
        spans.append(
            f"<span style='background:{colour_for_weight_orange(float(val))}; "
            "padding:1px 2px; margin:1px; border-radius:3px'>"
            f"{safe}</span>"
        )
    return " ".join(spans)

# ---- IG (inputs_embeds, model-agnostic) ----
def ig_attrib(model, tok, text: str, target: int, max_length: int = 512) -> Tuple[List[str], np.ndarray]:
    enc  = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    ids  = enc["input_ids"].long()           # [1, L]
    mask = enc["attention_mask"]             # [1, L]
    ids = _sanitize_ids_for_model(ids, tok, model)
    ids  = ids.to(DEVICE, non_blocking=True)
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
    ibs     = int(os.getenv("IG_INTERNAL_BS", "8"))
    attr = ig.attribute(
        embeds,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=ibs,
        return_convergence_delta=False
    )
    per_tok = attr.sum(-1).squeeze(0).detach().cpu().numpy()
    per_tok = per_tok * mask[0].detach().cpu().numpy().astype(np.float32)
    toks = tok.convert_ids_to_tokens(ids[0].detach().cpu().tolist())
    return toks, per_tok

# ---- LRP (Chefer-style) ----
def lrp_attrib_pred_only(model, tok, text: str, pred: int, max_length=512):
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
    spec_mask = np.zeros_like(ids_np, dtype=bool)
    for tid in (tok.pad_token_id, tok.cls_token_id, tok.sep_token_id):
        if tid is not None:
            spec_mask |= (ids_np == tid)
    R *= mask_np
    pad_id = getattr(tok, "pad_token_id", None)
    cls_id = getattr(tok, "cls_token_id", None)
    sep_id = getattr(tok, "sep_token_id", None)
    if pad_id is not None: R[ids_np == pad_id] = 0.0
    if cls_id is not None: R[ids_np == cls_id] = 0.0
    if sep_id is not None: R[ids_np == sep_id] = 0.0
    R = np.clip(R, 0.0, None)
    toks = tok.convert_ids_to_tokens(ids[0].detach().cpu().tolist())
    return toks, R

# ---- LIME (word-level) ----
def lime_attrib(model, tok, text: str, target: int,
                max_length: int = 512, batch_size: int = 16, num_samples: int = 1000
               ) -> Tuple[List[str], np.ndarray]:
    _require_optional_deps()
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

# ---- SHAP (word-level) ----
def shap_attrib(model, tok, text: str, target: int, max_length=512, batch_size=16, nsamples=500):
    _require_optional_deps()
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

# ---------------- tiny HTML helpers ----------------
CSS = """
<style>
body{margin:24px;font-family:ui-sans-serif,system-ui}
.grid{display:grid;grid-template-columns:1fr;gap:18px}
.example{border:1px solid #e5e7eb;border-radius:10px;padding:12px}
.h1{font-size:20px;font-weight:800;margin:0 0 6px}
.h2{font-size:16px;font-weight:700;margin:10px 0 6px}
.meta{color:#6b7280;font-size:12px}
.row{display:flex;gap:12px;margin-top:6px}
.col{flex:1;border:1px solid #e5e7eb;border-radius:8px;padding:8px}
.badge{display:inline-block;padding:2px 8px;border-radius:9999px;font-size:12px;margin-left:8px}
.tp{background:#e6ffed;color:#036b26}
.fp{background:#ffeaea;color:#8a0b0b}
.fn{background:#fff3cd;color:#8a6d3b}
.tn{background:#e6f0ff;color:#1a4fcc}
.mis{background:#fff3cd;color:#8a6d3b}
.small{font-size:12px;color:#6b7280}
</style>
"""

def conf_badge_binary(gt:int, pred:int)->str:
    if gt==1 and pred==1: return "TP","tp"
    if gt==0 and pred==0: return "TN","tn"
    if gt==1 and pred==0: return "FN","fn"
    if gt==0 and pred==1: return "FP","fp"
    return "MIS","mis"

def class_name(cfg: Dict, y: int) -> str:
    try: return str(cfg["classes"][int(y)])
    except Exception: return str(y)

# ---------------- bucket helper (your requested semantics) ----------------
def get_bucket_name(preds_for_example: np.ndarray, gt: int, cfg: Dict) -> str:
    """
    Buckets solely by how many models predict the GT:
      - agree_correct (all M correct)
      - agree_incorrect (all M same non-GT)
      - mixed_correctX  (X correct, where 1 <= X <= M-1)
    """
    preds = preds_for_example.astype(int).tolist()
    M = len(preds)
    correct_count = sum(1 for p in preds if int(p) == int(gt))
    unique = set(preds)
    if len(unique) == 1:
        only_label = next(iter(unique))
        if int(only_label) == int(gt):
            return "agree_correct"
        else:
            return "agree_incorrect"
    return f"mixed_correct{correct_count}"

# ---------------- main rendering function (render one bucket) ----------------
def render_bucket(models, texts_norm, labels_np, ids, base_cfg, args, indices, bucket_name, out_dir, seed):
    """
    Process the provided indices (list of ints) and produce:
      - HTML file for this bucket
      - raw_rows and full_raw_rows lists (returned)
    """
    raw_rows = []
    full_raw_rows = []
    metric = models[0]["run_dir"].name
    k = len(indices)
    out_html = out_dir / f"anecdotal_compare_{bucket_name}_k{k}_jaccard.html"
    example_meta = {}
    per_example_exports = {}

    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><title>Cross-Model Anecdotes</title>")
        fh.write(CSS); fh.write("</head><body>\n")
        # ---- inlined JS copied exactly from your original script ----
        fh.write("""
            <script>
            function rgba(r,g,b,a){return "rgba("+r+","+g+","+b+","+a+")";}
            function colourSigned(v){
            v = Math.max(-1, Math.min(1, v));
            var a = (30 + 200*Math.abs(v))/255.0;
            return v >= 0 ? rgba(255,0,0,a) : rgba(0,0,255,a);
            }
            function colourOrange(v){
            v = Math.max(0, Math.min(1, v));
            var a = (30 + 200*v)/255.0;
            return rgba(255,165,0,a);
            }
            function rgbaToFill(rgbaStr){
            var m = rgbaStr.match(/rgba\\((\\d+),(\\d+),(\\d+),([\\d\\.]+)\\)/);
            if(!m) return {rgb:"rgb(0,0,0)", a:1.0};
            return {rgb:"rgb("+m[1]+","+m[2]+","+m[3]+")", a:parseFloat(m[4])};
            }
            function normaliseSigned(arr){
            var maxa = 0; for (var i=0;i<arr.length;i++){maxa=Math.max(maxa, Math.abs(arr[i]));}
            if(maxa===0) return arr.slice();
            return arr.map(function(x){return x/maxa;});
            }
            function normaliseNonneg(arr){
            var m = 0; for (var i=0;i<arr.length;i++){m=Math.max(m, arr[i]);}
            if(m===0) return arr.slice();
            return arr.map(function(x){return x/m;});
            }
            function escapeHtml(s){
            return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;");
            }

            /* ----- FIXED: safer blob download ----- */
            function downloadText(filename, text){
            try {
                const blob = new Blob([text], {type: "application/octet-stream"});
                const a = document.createElement("a");
                a.href = URL.createObjectURL(blob);
                a.download = filename;
                a.style.display = "none";
                document.body.appendChild(a);
                a.click();
                setTimeout(()=>{document.body.removeChild(a);URL.revokeObjectURL(a.href);},1000);
            } catch(err){ alert("Download failed: "+err); }
            }

            /* ---------- token renderer ---------- */
            function tokensToSVG(opts){
            var tokens = opts.tokens.slice();
            var W = opts.weights.slice();
            var signed = !!opts.signed;
            var fontSize = opts.fontSize || 14;
            var fontFamily = opts.fontFamily || "DejaVu Sans Mono, monospace";
            var maxCharsPerLine = opts.maxCharsPerLine || 90;
            var Wn = signed ? normaliseSigned(W) : normaliseNonneg(W.map(v=>Math.max(0,v)));
            var colourFn = signed ? colourSigned : colourOrange;

            var lines = [], cur=[], curLen=0;
            for (var i=0;i<tokens.length;i++){
                var t = (tokens[i]||"").replace(/▁/g," ");
                var tok = t.length ? t : " ";
                var step = tok.length + 1;
                if(curLen + step > maxCharsPerLine && cur.length){ lines.push(cur); cur=[]; curLen=0; }
                cur.push({t: tok, w: Wn[i]}); curLen += step;
            }
            if(cur.length) lines.push(cur);

            var xPad=10, yPad=12, hGap=6, vGap=8;
            var chW = fontSize * 0.6, lineH = fontSize + vGap;
            var width = Math.round(2*xPad + maxCharsPerLine * chW);
            var height = Math.round(2*yPad + lines.length * lineH + (opts.title ? 20 : 0));

            var svg=[];
            svg.push("<svg xmlns='http://www.w3.org/2000/svg' width='"+width+"' height='"+height+"'>");
            svg.push("<rect x='0' y='0' width='"+width+"' height='"+height+"' fill='white'/>");
            var y = yPad;
            if(opts.title){
                svg.push("<text x='"+xPad+"' y='"+y+"' font-family='"+escapeHtml(fontFamily)+"' font-size='"+fontSize+"' font-weight='bold' fill='black'>"+escapeHtml(opts.title)+"</text>");
                y += lineH;
            }
            for (var li=0; li<lines.length; li++){
                var x = xPad;
                var line = lines[li];
                for (var j=0;j<line.length;j++){
                var tok = line[j].t, v = line[j].w;
                var boxW = Math.round(tok.length * chW + hGap);
                var rgba = colourFn(v); var fill = rgbaToFill(rgba);
                svg.push("<rect x='"+x+"' y='"+(y-(fontSize-2))+"' width='"+boxW+"' height='"+(fontSize+4)+"' fill='"+fill.rgb+"' fill-opacity='"+fill.a+"' rx='3' ry='3'/>");
                svg.push("<text x='"+(x+2)+"' y='"+y+"' font-family='"+escapeHtml(fontFamily)+"' font-size='"+fontSize+"' fill='black'>"+escapeHtml(tok)+"</text>");
                x += boxW;
                }
                y += lineH;
            }
            svg.push("</svg>");
            return svg.join("\\n"); // FIX: must be literal \\n
            }

            /* ---------- per-method download ---------- */
            function downloadSVGsFor(exId){
            var tag=document.getElementById('data-'+exId);
            if(!tag){alert('No data for '+exId);return;}
            var payload=JSON.parse(tag.textContent||tag.innerText||"[]");
            if(!payload.length){alert('Nothing to export');return;}
            for(var i=0;i<payload.length;i++){
                var p=payload[i];
                var svg=tokensToSVG({
                tokens:p.tokens,weights:p.weights,signed:p.signed,
                title:p.title,maxCharsPerLine:p.maxCharsPerLine||90,
                fontSize:p.fontSize||14,fontFamily:p.fontFamily||"DejaVu Sans Mono, monospace"
                });
                downloadText(p.filename,svg);
            }
            }

            /* ---------- composite download ---------- */
            function _svgSize(svgStr){
            var m=svgStr.match(/<svg[^>]*width=['"](\\d+)['"][^>]*height=['"](\\d+)['"]/i);
            return m?{w:parseInt(m[1],10),h:parseInt(m[2],10)}:{w:0,h:0};
            }
            function _toDataURI(svgStr){
            var b64 = btoa(unescape(encodeURIComponent(svgStr)));
            return "data:image/svg+xml;base64," + b64;
            }
            function _methodFromFilename(name){
            if(/_IG\\.svg$/i.test(name))return"IG";
            if(/_LRP\\.svg$/i.test(name))return"LRP(+Pred)";
            if(/_LIME\\.svg$/i.test(name))return"LIME";
            if(/_SHAP\\.svg$/i.test(name))return"SHAP";
            return"UNK";
            }
            function _modelFromTitle(title){
            var m=String(title).split("·");
            return m.length>=2?m[1].trim():"model";
            }

            function buildCompositeSVG(exId, payload){
            var panels = payload.map(function(p){
                var svg = tokensToSVG({
                tokens: p.tokens, weights: p.weights, signed: p.signed,
                title: p.title, maxCharsPerLine: p.maxCharsPerLine || 90,
                fontSize: p.fontSize || 14, fontFamily: p.fontFamily || "DejaVu Sans Mono, monospace"
                });
                var sz = _svgSize(svg);
                return {
                svg: svg,
                w: sz.w, h: sz.h,
                method: _methodFromFilename(p.filename || ""),
                model: _modelFromTitle(p.title || "")
                };
            });
                 
            function _extractInnerSVG(svgStr){
            return svgStr.replace(/^<svg[^>]*>/i, "").replace(/<\/svg>\\s*$/i, "");
            }

            var byModel={};
            panels.forEach(p=>(byModel[p.model]=byModel[p.model]||[]).push(p));
            var pad=12,gapX=10,gapY=10,headerH=18,titleH=22;
            var models=Object.keys(byModel);
            var order=["IG","LRP(+Pred)","LIME","SHAP"];
            var panelW=0,panelH=0;
            panels.forEach(p=>{panelW=Math.max(panelW,p.w);panelH=Math.max(panelH,p.h);});
            var blockH=headerH+(panelH*2+gapY);
            var blockW=(panelW*2+gapX);
            var width=pad*2+blockW;
            var height=pad*2+titleH+models.length*blockH+(models.length?(models.length-1)*gapY:0);

            var out=[];
            out.push("<svg xmlns='http://www.w3.org/2000/svg' width='"+width+"' height='"+height+"'>");
            out.push("<rect x='0' y='0' width='"+width+"' height='"+height+"' fill='white'/>");
            var y=pad;
            var meta = (window.EXAMPLE_META && window.EXAMPLE_META[exId]) || {};
            var cve = meta.cve || exId;
            var metric = meta.metric || (window.PAGE_METRIC || "Metric");
            var gt = meta.gt || "GT?";
            out.push(
            "<text x='"+pad+"' y='"+(y+18)+"' font-family='ui-sans-serif,system-ui' font-size='18' font-weight='700' fill='black'>"
            + escapeHtml(cve + " · Metric: " + metric + " · GT: " + gt + " · All Models")
            + "</text>"
            );
            y+=titleH;

            models.forEach(function(model){
                var x=pad;
                out.push("<text x='"+x+"' y='"+(y+headerH-6)+"' font-family='ui-sans-serif,system-ui' font-size='14' font-weight='700' fill='black'>"+escapeHtml(model)+"</text>");
                var top=y+headerH;
                var pmap={};
                byModel[model].forEach(p=>pmap[p.method]=p);

                var m1=pmap[order[0]],m2=pmap[order[1]];
                if(m1)out.push("<g transform='translate("+x+","+top+")'>"+_extractInnerSVG(m1.svg)+"</g>");
                if(m2)out.push("<g transform='translate("+(x+panelW+gapX)+","+top+")'>"+_extractInnerSVG(m2.svg)+"</g>");

                var m3=pmap[order[2]],m4=pmap[order[3]];
                var y2=top+panelH+gapY;
                if(m3)out.push("<g transform='translate("+x+","+y2+")'>"+_extractInnerSVG(m3.svg)+"</g>");
                if(m4)out.push("<g transform='translate("+(x+panelW+gapX)+","+y2+")'>"+_extractInnerSVG(m4.svg)+"</g>");

                y+=blockH+gapY;
            });

            out.push("</svg>");
            return out.join("\\n");
            }

            function downloadCompositeFor(exId){
            var tag=document.getElementById('data-'+exId);
            if(!tag){alert('No data for '+exId);return;}
            var payload=JSON.parse(tag.textContent||tag.innerText||"[]");
            if(!payload.length){alert('Nothing to export');return;}
            var svg=buildCompositeSVG(exId,payload);
            downloadText(exId+"_ALL.svg",svg);
            }
            </script>
        """)
        # ---- end inlined JS ----

        # Header/meta
        metric = models[0]["run_dir"].name
        seed = models[0]["run_dir"].parent.parent.name
        dataset = models[0]["run_dir"].parent.parent.parent.name
        model_list = ", ".join(m["label"] for m in models)
        fh.write(f"<script>window.PAGE_METRIC = {json.dumps(metric)};</script>\n")
        fh.write(f"<div class='h1'>Cross-Model Anecdotes – {dataset} · seed={seed} · metric={metric} · bucket={bucket_name}</div>")
        fh.write(f"<div class='meta'>Models: {model_list}</div><br>")
        fh.write("<div class='grid'>\n")

        for count, idx in enumerate(indices, start=1):
            cve_id = ids[idx]
            text = texts_norm[idx]
            gt = int(labels_np[idx])
            gt_n = class_name(base_cfg, gt)

            example_id = f"ex{count:03d}_cve{cve_id}"
            example_meta[example_id] = {
                "cve": f"CVE-{cve_id}",
                "metric": metric,
                "gt": gt_n,
                "bucket": bucket_name
            }
            per_example_exports[example_id] = []

            row_html = [f"<div class='example'><div class='h2'>#{count} · cve_id {cve_id} · {metric}</div>"]
            row_html.append(f"<div class='meta'>GT={gt_n} ({gt}) · Bucket={bucket_name}</div>")
            row_html.append("<div class='row'>")

            # per-model columns
            for m in models:
                pred = int(m["preds"][idx])
                p_pred = float(m["probs"][idx][pred])
                num_classes = len(m["cfg"].get("classes", []))
                if num_classes == 2:
                    tag, css = conf_badge_binary(gt, pred)
                else:
                    tag, css = ("TP","tp") if pred==gt else ("MIS","mis")
                badge = f"<span class='badge {css}'>{tag}</span>"

                ig_tokens, ig_weights = ig_attrib(m["model"], m["tok"], text, target=pred, max_length=m["max_len"])
                lrp_res = lrp_attrib_pred_only(m["model"], m["tok"], text, pred=pred, max_length=m["max_len"])
                lime_tokens, lime_weights = lime_attrib(
                    m["model"], m["tok"], text, target=pred,
                    max_length=m["max_len"], batch_size=args.batch_size,
                    num_samples=args.lime_samples
                )
                shap_tokens, shap_weights = shap_attrib(
                    m["model"], m["tok"], text, target=pred,
                    max_length=m["max_len"], batch_size=args.batch_size,
                    nsamples=args.shap_nsamples
                )

                ig_block_html = render_inline(ig_tokens, ig_weights)
                if lrp_res is not None:
                    lrp_tokens, lrp_weights = lrp_res
                    lrp_block_html = render_inline_lrp_pos(lrp_tokens, lrp_weights)
                else:
                    lrp_tokens, lrp_weights = [], np.array([], dtype=float)
                    lrp_block_html = "<span class='small'>n/a</span>"
                lime_block_html = render_inline(lime_tokens, lime_weights)
                shap_block_html = render_inline(shap_tokens, shap_weights)

                # project + topk
                ig_wtokens, ig_wweights = _project_subwords_to_words(ig_tokens, ig_weights)
                ig_top = _topk_tokens(ig_wtokens, ig_wweights, k=args.topk, score_mode=args.score_mode)
                if lrp_res is not None:
                    lrp_wtokens, lrp_wweights = _project_subwords_to_words(lrp_tokens, lrp_weights)
                    lrp_top = _topk_tokens(lrp_wtokens, lrp_wweights, k=args.topk, score_mode=args.score_mode)
                else:
                    lrp_top = []
                lime_top = _topk_tokens(lime_tokens, lime_weights, k=args.topk, score_mode=args.score_mode)
                shap_top = _topk_tokens(shap_tokens, shap_weights, k=args.topk, score_mode=args.score_mode)

                pairs = [("IG","LRP",ig_top,lrp_top), ("IG","LIME",ig_top,lime_top),
                        ("IG","SHAP",ig_top,shap_top), ("LRP","LIME",lrp_top,lime_top),
                        ("LRP","SHAP",lrp_top,shap_top), ("LIME","SHAP",lime_top,shap_top)]
                jacc = {f"{a}_{b}": _jaccard(x,y) for (a,b,x,y) in pairs if (x and y)}

                raw_rows.append({
                    "example_id": example_id,
                    "cve_id": cve_id,
                    "metric": metric,
                    "model": m["label"],
                    "gt": int(gt),
                    "pred": int(pred),
                    "p_pred": float(p_pred),
                    "topk": int(args.topk),
                    "score_mode": args.score_mode,
                    "tokens": {"IG": ig_top, "LRP": lrp_top, "LIME": lime_top, "SHAP": shap_top},
                    "jaccard": jacc,
                    "bucket": bucket_name,
                })

                full_raw_rows.append({
                    "example_id": example_id,
                    "cve_id": cve_id,
                    "metric": metric,
                    "model": m["label"],
                    "gt": int(gt),
                    "pred": int(pred),
                    "p_pred": float(p_pred),
                    "text_norm": text,
                    "topk_used_in_html": int(args.topk),
                    "methods": {
                        "IG": {
                            "tokens": ig_tokens,
                            "weights": _float_list(ig_weights),
                            "word_tokens": ig_wtokens,
                            "word_weights": _float_list(ig_wweights),
                            "score_mode_html": args.score_mode,
                        },
                        "LRP": None if not lrp_tokens else {
                            "tokens": lrp_tokens,
                            "weights": _float_list(lrp_weights),
                            "word_tokens": lrp_wtokens,
                            "word_weights": _float_list(lrp_wweights),
                            "score_mode_html": "pos",
                        },
                        "LIME": {
                            "tokens": lime_tokens,
                            "weights": _float_list(lime_weights),
                            "score_mode_html": args.score_mode,
                        },
                        "SHAP": {
                            "tokens": shap_tokens,
                            "weights": _float_list(shap_weights),
                            "score_mode_html": args.score_mode,
                        },
                    },
                    "bucket": bucket_name
                })

                export_entries = []
                export_entries.append({
                    "filename": f"{example_id}_{m['label']}_IG.svg",
                    "title": f"IG · {m['label']} · Pred={class_name(m['cfg'], pred)} ({pred}) · p={p_pred:.2f}",
                    "tokens": ig_tokens,
                    "weights": _float_list(ig_weights),
                    "signed": True
                })
                if lrp_res is not None:
                    export_entries.append({
                        "filename": f"{example_id}_{m['label']}_LRP.svg",
                        "title": f"LRP(+Pred) · {m['label']} · Pred={class_name(m['cfg'], pred)} ({pred}) · p={p_pred:.2f}",
                        "tokens": lrp_tokens,
                        "weights": _float_list(lrp_weights),
                        "signed": False
                    })
                export_entries.append({
                    "filename": f"{example_id}_{m['label']}_LIME.svg",
                    "title": f"LIME · {m['label']} · Pred={class_name(m['cfg'], pred)} ({pred}) · p={p_pred:.2f}",
                    "tokens": lime_tokens,
                    "weights": _float_list(lime_weights),
                    "signed": True
                })
                export_entries.append({
                    "filename": f"{example_id}_{m['label']}_SHAP.svg",
                    "title": f"SHAP · {m['label']} · Pred={class_name(m['cfg'], pred)} ({pred}) · p={p_pred:.2f}",
                    "tokens": shap_tokens,
                    "weights": _float_list(shap_weights),
                    "signed": True
                })

                per_example_exports[example_id].extend(export_entries)

                row_html.append(
                    "<div class='col'>"
                    + f"<div class='h2'>{m['label']} · Pred={class_name(m['cfg'], pred)} ({pred}) · p={p_pred:.2f} {badge}</div>"
                    + "<div class='small'>Row 1: IG / LRP(+Pred) · Row 2: LIME / SHAP</div>"
                    + "<div class='row'>"
                        + "<div class='col'>IG (subwords)<br>" + ig_block_html + "</div>"
                        + "<div class='col'>LRP (+Pred, pos-only)<br>" + lrp_block_html + "</div>"
                    + "</div>"
                    + "<div class='row'>"
                        + "<div class='col'>LIME (words)<br>" + lime_block_html + "</div>"
                        + "<div class='col'>SHAP (words)<br>" + shap_block_html + "</div>"
                    + "</div>"
                    + "</div>"
                )

                gc.collect()

            row_html.append("</div>")  # /row
            payload_json = json.dumps(per_example_exports[example_id], ensure_ascii=False).replace("</", "<\\/")
            row_html.append(
                f"<div class='small' style='margin-top:10px; display:flex; gap:10px'>"
                f"<button onclick=\"downloadSVGsFor('{example_id}')\">Download method SVGs</button>"
                f"<button onclick=\"downloadCompositeFor('{example_id}')\">Download ALL-in-one SVG</button>"
                f"</div>"
                f"<script id='data-{example_id}' type='application/json'>{payload_json}</script>"
            )
            row_html.append("</div>")  # close .example
            fh.write("".join(row_html) + "\n")
            fh.flush()

        fh.write(f"<script>window.EXAMPLE_META = {json.dumps(example_meta)};</script>\n")
        fh.write("</div>\n</body></html>\n")

    print("HTML written to", out_html)
    return raw_rows, full_raw_rows

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Cross-model anecdotal interpretability (strict loader).")
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="Space-separated list of run_dirs (…/saved_models/<dataset>/<seed>/<model>/<metric>)")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--shap_nsamples", type=int, default=500)
    ap.add_argument("--lime_samples", type=int, default=1000)
    ap.add_argument("--sample_mode", choices=["stratified","balanced_by_first"], default="stratified")
    ap.add_argument(
        "--focus",
        choices=["none", "mixed", "all_mis", "any_mis", "disagree", "agree", "agree_correct", "agree_incorrect", "all"],
        default="none",
        help="Which buckets to process. 'mixed' = all mixed buckets; 'all' = all buckets."
    )
    ap.add_argument("--export_jsonl", type=str, default="",
                help="(deprecated) Path to write raw per-example JSONL (kept for compatibility).")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--score_mode", choices=["abs","pos"], default="abs",
                    help="Ranking for top-k: 'abs' uses |weight|; 'pos' uses positive-only.")
    ap.add_argument(
        "--export_jsonl_full",
        action="store_true",
        help="Write per-example, per-model FULL attribution payload (all tokens + weights) as JSONL."
)
    args = ap.parse_args()

    _require_optional_deps()

    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    if not run_dirs:
        raise SystemExit("No run_dirs provided")

    # Load all models with strict loader
    models: List[Dict] = []
    for rd in run_dirs:
        cfg, tok, model, mtype = load_model_and_tokenizer(rd)
        cfg_max_len = int(cfg.get("max_length", args.max_length))
        max_len_eff = effective_max_len(tok, getattr(model, "config", None), cfg_max_len)
        raw_label = rd.parent.name
        label = sanitize_for_filename(raw_label)
        models.append(dict(run_dir=rd, cfg=cfg, tok=tok, model=model, mtype=mtype,
                           label=label, max_len=max_len_eff))

    base_cfg = models[0]["cfg"]
    texts, labels, ids = rebuild_test_with_ids(base_cfg)
    labels_np = np.array(labels, dtype=int)
    use_norm = bool(base_cfg.get("use_normalised_tokens", False))
    texts_norm = [normalise_text(t, enabled=use_norm) for t in texts]

    # Predictions per model
    for m in models:
        preds, probs = predict_classes(
            m["model"], m["tok"], texts_norm,
            max_length=m["max_len"], batch_size=args.batch_size
        )
        m["preds"] = preds
        m["probs"] = probs

    N = len(texts)
    M = len(models)
    pred_mat = np.stack([m["preds"] for m in models], axis=0)      # [M, N]

    # compute buckets for every example
    bucket_of = []
    for i in range(N):
        bucket_of.append(get_bucket_name(pred_mat[:, i], int(labels_np[i]), base_cfg))

    # group example indices by bucket
    buckets = defaultdict(list)
    for idx, b in enumerate(bucket_of):
        buckets[b].append(idx)

    # decide which buckets to process based on args.focus
    def buckets_from_focus(focus: str):
        if focus == "none":
            return []  # nothing
        if focus == "all":
            return sorted(list(buckets.keys()))
        if focus == "mixed":
            return sorted([b for b in buckets.keys() if b.startswith("mixed_correct")])
        if focus == "agree_correct":
            return ["agree_correct"] if "agree_correct" in buckets else []
        if focus == "agree_incorrect":
            return ["agree_incorrect"] if "agree_incorrect" in buckets else []
        # legacy mappings
        if focus == "agree":
            return [b for b in buckets.keys() if b.startswith("agree")]
        if focus == "disagree":
            return [b for b in buckets.keys() if b.startswith("mixed_correct")]
        if focus == "any_mis":
            return [b for b in buckets.keys() if b != "agree_correct"]
        if focus == "all_mis":
            return [b for b in buckets.keys() if b != "agree_correct"]
        return []

    selected_buckets = buckets_from_focus(args.focus)
    if not selected_buckets:
        print("No buckets selected by --focus=", args.focus)
        return

    # sampling per bucket (seeded)
    seed = int(base_cfg.get("seed", 0))
    rng = np.random.RandomState(seed)
    Kcap = int(args.K)
    sampled_by_bucket: Dict[str, List[int]] = {}
    for b in selected_buckets:
        idxs = buckets.get(b, [])
        if not idxs:
            sampled_by_bucket[b] = []
            continue
        if len(idxs) <= Kcap:
            sampled = idxs[:]
        else:
            sampled = list(rng.choice(idxs, size=Kcap, replace=False))
        sampled_by_bucket[b] = sampled

    out_dir = models[0]["run_dir"] / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # process each selected bucket independently: run attribution, write per-bucket HTML and JSONL
    overall_raw = []
    overall_full = {}
    for b in sorted(sampled_by_bucket.keys()):
        idxs = sampled_by_bucket[b]
        if not idxs:
            print(f"Skipping empty bucket {b}")
            continue
        print(f"Processing bucket {b} -> {len(idxs)} examples (K={Kcap})")
        raw_rows, full_raw_rows = render_bucket(models, texts_norm, labels_np, ids, base_cfg, args, idxs, b, out_dir, seed)
        overall_raw.extend(raw_rows)
        overall_full[b] = full_raw_rows

        # write per-bucket raw JSONL
        fname = f"{b}_{len(raw_rows)}.jsonl"
        out_path = out_dir / fname
        with open(out_path, "w", encoding="utf-8") as fjs:
            for r in raw_rows:
                fjs.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("Wrote", out_path)

        # write per-bucket full JSONL if requested
        if args.export_jsonl_full:
            # Required format: full_attribs_${FOCUS}_${K}.jsonl
            fname_full = f"full_attribs_{b}_{int(args.K)}.jsonl"
            out_path_full = out_dir / fname_full
            with open(out_path_full, "w", encoding="utf-8") as fjs:
                for r in full_raw_rows:
                    fjs.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("Wrote", out_path_full)


    # write bucket summary
    bucket_summary = {b: {"count": len(sampled_by_bucket[b])} for b in sampled_by_bucket}
    summary_path = out_dir / "buckets_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh_sum:
        json.dump(bucket_summary, fh_sum, ensure_ascii=False, indent=2)
    print("Bucket summary written to", summary_path)
    for b, info in sorted(bucket_summary.items(), key=lambda x: -x[1]["count"]):
        print(f"{b:25s} {info['count']:4d}")

if __name__ == "__main__":
    main()
