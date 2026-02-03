from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients

from utils.cvss_data_engine_joana.CVSSDataset_modified import read_cvss_csv_with_ids
from utils.text_normalise import normalise_text
from utils.cvss_mappings import METRIC_TO_CLASSES


# ---------------- device ----------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()


# ---------------- token safety ----------------
def sanitize_ids_for_model(input_ids: torch.Tensor, tok, model) -> torch.Tensor:
    """
    Map any token id >= model's embedding size to UNK to prevent index errors.
    This is defensive: ideally tokeniser and embedding sizes match.
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
def probs_from_texts(model, tok, texts: List[str], max_length: int = 512, batch_size: int = 8) -> np.ndarray:
    out: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

        ids = sanitize_ids_for_model(enc["input_ids"].long(), tok, model)
        mask = enc["attention_mask"]

        ids = ids.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True)

        logits = model(input_ids=ids, attention_mask=mask).logits
        probs = torch.softmax(logits, -1).detach().cpu().numpy()
        out.append(probs)

        del enc, ids, mask, logits

    if not out:
        num_labels = int(getattr(getattr(model, "config", None), "num_labels", 1))
        return np.zeros((0, num_labels))
    return np.vstack(out)


def predict_class_and_prob(model, tok, text: str, max_length: int = 512) -> Tuple[int, float, np.ndarray]:
    probs = probs_from_texts(model, tok, [text], max_length=max_length, batch_size=1)
    if probs.size == 0:
        return 0, 0.0, probs
    pred = int(probs.argmax(axis=1)[0])
    p = float(probs[0, pred])
    return pred, p, probs


# ---------------- attribution helpers ----------------
def strip_specials(tok, ids_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
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


def ig_attrib(
    model,
    tok,
    text: str,
    target: int,
    *,
    max_length: int = 512,
    ig_steps: int = 128,
    ig_internal_bs: int = 8,
    ig_use_amp: bool = True,
) -> Tuple[List[str], np.ndarray]:
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

    ids = sanitize_ids_for_model(enc["input_ids"].long(), tok, model)
    mask = enc["attention_mask"]

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

    def forward_from_embeds(e):
        am = mask_for_model
        # am is [1, T]; Captum may call with e as [B, T, D]
        if am.dim() == 1:
            am = am.unsqueeze(0)
        if am.size(0) != e.size(0):
            am = am.expand(e.size(0), -1).contiguous()

        use_amp = ig_use_amp and (DEVICE.type == "cuda")
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(inputs_embeds=e, attention_mask=am).logits
        else:
            logits = model(inputs_embeds=e, attention_mask=am).logits
        return logits[:, target]


    ig = IntegratedGradients(forward_from_embeds)
    attr = ig.attribute(
        embeds,
        baselines=baseline,
        n_steps=int(ig_steps),
        internal_batch_size=int(ig_internal_bs),
        return_convergence_delta=False,
    )

    per_tok = attr.sum(-1).squeeze(0).detach().cpu().numpy().astype(np.float32)

    ids_np = ids[0].detach().cpu().numpy()
    mask_np = mask[0].detach().cpu().numpy().astype(np.float32)
    per_tok = per_tok * mask_np

    keep = strip_specials(tok, ids_np, mask_np)
    toks_all = tok.convert_ids_to_tokens(ids_np.tolist())

    toks = [t for t, k in zip(toks_all, keep) if k]
    per_tok = per_tok[keep]
    return toks, per_tok


def lrp_attrib_pred_only(
    model, tok, text: str, pred: int, *, max_length: int = 512
) -> Optional[Tuple[List[str], np.ndarray]]:
    """
    LRP attribution for the predicted class only.
    Requires an LRP-capable model with a .relprop method.
    """
    if not hasattr(model, "relprop"):
        return None

    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    ids = sanitize_ids_for_model(enc["input_ids"].long(), tok, model)
    mask = enc["attention_mask"]

    ids = ids.to(DEVICE, non_blocking=True)
    mask = mask.to(DEVICE, non_blocking=True)

    model.zero_grad(set_to_none=True)
    logits = model(input_ids=ids, attention_mask=mask).logits

    seed = torch.zeros_like(logits)
    seed[0, pred] = 1.0

    import inspect

    sig = inspect.signature(model.relprop)
    kwargs = {"alpha": 1.0}

    if "input_ids" in sig.parameters:
        kwargs["input_ids"] = ids
    if "attention_mask" in sig.parameters:
        kwargs["attention_mask"] = mask

    R_all = model.relprop(seed, **kwargs)

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
    keep = strip_specials(tok, ids_np, mask_np)

    toks_all = tok.convert_ids_to_tokens(ids_np.tolist())
    toks = [t for t, k in zip(toks_all, keep) if k]
    R = np.clip(R[keep], 0.0, None)

    return toks, R


def lime_attrib(
    model,
    tok,
    text: str,
    target: int,
    *,
    max_length: int = 512,
    batch_size: int = 8,
    num_samples: int = 1000,
) -> Tuple[List[str], np.ndarray]:
    try:
        from lime.lime_text import LimeTextExplainer  # type: ignore
    except Exception as e:
        raise RuntimeError("LIME not installed (pip install lime)") from e

    try:
        C = int(getattr(getattr(model, "config", None), "num_labels", 2))
    except Exception:
        C = 2

    explainer = LimeTextExplainer(class_names=[str(i) for i in range(C)], bow=False)

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
    *,
    max_length: int = 512,
    batch_size: int = 8,
    nsamples: int = 500,
) -> Tuple[List[str], np.ndarray]:
    try:
        import shap  # type: ignore
    except Exception as e:
        raise RuntimeError("SHAP not installed (pip install shap)") from e

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


# ---------------- high-level orchestration ----------------
def load_cve_from_run_cfg(
    *,
    repo_root: Path,
    base_cfg: Dict,
    cve_id: str,
) -> Tuple[str, str, str, int, str]:
    """
    Returns:
      (resolved_cve_id, raw_text, text_for_model, gt_idx, gt_label)
    """
    csv_path = repo_root / "cve_data" / "extended_analysis" / base_cfg["data_file"]
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
    resolved = cve_id
    if resolved not in id_to_idx:
        cve_id_norm = cve_id.strip().upper()
        alt = None
        for k in id_to_idx.keys():
            if str(k).strip().upper() == cve_id_norm:
                alt = k
                break
        if alt is None:
            raise SystemExit(f"CVE not found in dataset CSV: {cve_id}")
        resolved = alt

    idx = id_to_idx[resolved]
    text_raw = texts[idx]
    text_for_model = texts_norm[idx]
    gt_idx = int(labels[idx])

    try:
        gt_label = base_cfg["classes"][gt_idx]
    except Exception:
        gt_label = str(gt_idx)

    return resolved, text_raw, text_for_model, gt_idx, gt_label


def compute_case(
    *,
    models: List[Dict],
    cve_id: str,
    repo_root: Path,
    with_lime: bool,
    with_shap: bool,
    lime_samples: int,
    shap_evals: int,
    ig_steps: int,
    ig_internal_bs: int,
    ig_use_amp: bool,
) -> Dict:
    """
    Compute predictions + attributions for one CVE across multiple models.

    Input `models` is your existing list of dicts from model_loader:
      {label, tok, model, max_len, cfg, ...}

    Returns a dict suitable for rendering.
    """
    if not models:
        raise ValueError("models list is empty")

    base_cfg = models[0]["cfg"]
    metric_name = None
    for m in models:
        rd = m.get("rd")
        if metric_name is None and isinstance(rd, Path):
            metric_name = infer_metric_from_run_dir(rd)

    metric_display = metric_name if metric_name else "N/A"
    metric_classes = METRIC_TO_CLASSES.get(metric_name, None)

    resolved_id, text_raw, text_for_model, gt_idx, gt_label = load_cve_from_run_cfg(
        repo_root=repo_root,
        base_cfg=base_cfg,
        cve_id=cve_id,
    )

    methods: List[str] = ["IG", "LRP"]
    if with_lime:
        methods.append("LIME")
    if with_shap:
        methods.append("SHAP")

    model_infos: List[Dict] = []
    attribs: Dict[Tuple[str, str], Tuple[List[str], np.ndarray, bool]] = {}

    for m in models:
        pred_idx, prob, _ = predict_class_and_prob(
            m["model"], m["tok"], text_for_model, max_length=m["max_len"]
        )

        if metric_classes is not None and 0 <= pred_idx < len(metric_classes):
            pred_label = metric_classes[pred_idx]
        else:
            try:
                pred_label = base_cfg["classes"][pred_idx]
            except Exception:
                pred_label = str(pred_idx)

        model_infos.append(
            dict(
                label=m["label"],
                pred_idx=pred_idx,
                pred_label=pred_label,
                prob=prob,
            )
        )

        ig_toks, ig_w = ig_attrib(
            m["model"],
            m["tok"],
            text_for_model,
            target=pred_idx,
            max_length=m["max_len"],
            ig_steps=ig_steps,
            ig_internal_bs=ig_internal_bs,
            ig_use_amp=ig_use_amp,
        )
        attribs[(m["label"], "IG")] = (ig_toks, ig_w, True)

        lrp_res = lrp_attrib_pred_only(
            m["model"], m["tok"], text_for_model, pred=pred_idx, max_length=m["max_len"]
        )
        if lrp_res is not None:
            lrp_toks, lrp_w = lrp_res
            attribs[(m["label"], "LRP")] = (lrp_toks, lrp_w, False)
        else:
            attribs[(m["label"], "LRP")] = ([], np.asarray([], dtype=float), False)

        if with_lime:
            try:
                lime_toks, lime_w = lime_attrib(
                    m["model"],
                    m["tok"],
                    text_for_model,
                    target=pred_idx,
                    max_length=m["max_len"],
                    batch_size=8,
                    num_samples=int(lime_samples),
                )
                attribs[(m["label"], "LIME")] = (lime_toks, lime_w, True)
            except Exception as e:
                attribs[(m["label"], "LIME")] = ([f"LIME error: {e}"], np.asarray([0.0], dtype=float), True)

        if with_shap:
            try:
                shap_toks, shap_w = shap_attrib(
                    m["model"],
                    m["tok"],
                    text_for_model,
                    target=pred_idx,
                    max_length=m["max_len"],
                    batch_size=8,
                    nsamples=int(shap_evals),
                )
                attribs[(m["label"], "SHAP")] = (shap_toks, shap_w, True)
            except Exception as e:
                attribs[(m["label"], "SHAP")] = ([f"SHAP error: {e}"], np.asarray([0.0], dtype=float), True)

    return dict(
        cve_id=resolved_id,
        metric=metric_display,
        gt_idx=gt_idx,
        gt_label=gt_label,
        description=text_raw,
        model_infos=model_infos,
        methods=methods,
        attribs=attribs,
    )
