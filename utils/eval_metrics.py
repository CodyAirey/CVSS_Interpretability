# utils/eval_metrics.py
from __future__ import annotations
import numpy as np
from typing import Any, Tuple
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
)

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ---------- helpers ----------
def _to_numpy_2d(x: Any) -> np.ndarray:
    """
    Accept (logits,) tuples from HF, torch.Tensor, lists, or ndarrays
    and return a 2D float array [N, C].
    """
    # HF often gives (logits,) or (loss, logits, ...)
    if isinstance(x, (tuple, list)):
        # prefer the first ndarray-like item
        for item in x:
            if _looks_like_array(item):
                x = item
                break

    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)
    if x.ndim == 1:
        # single example edge case -> make it [1, C]
        x = x[None, :]
    return x


def _looks_like_array(x: Any) -> bool:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return True
    try:
        _ = np.asarray(x)
        return True
    except Exception:
        return False


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

def _labels_preds_from_eval_pred(eval_pred) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hugging Face passes (predictions, labels) but 'predictions' may be either
    logits or a tuple (logits, ...). We normalise it here.
    """
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    y_true = labels
    y_pred = np.argmax(preds, axis=-1)
    return y_true, y_pred

# ---------- public API ----------
def preds_from_logits(logits: Any) -> np.ndarray:
    arr = _to_numpy_2d(logits)
    return arr.argmax(axis=1)


def compute_all_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    """
    Your post-hoc evaluation (called on the test set in train.py).
    """
    y_pred = np.argmax(logits, axis=-1)
    return {
        "accuracy":            accuracy_score(y_true, y_pred),
        "balanced_accuracy":   balanced_accuracy_score(y_true, y_pred),
        "f1_weighted":         f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":            f1_score(y_true, y_pred, average="macro",   zero_division=0),
        "f1_micro":            f1_score(y_true, y_pred, average="micro",   zero_division=0),
        "precision_weighted":  precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":     recall_score(y_true, y_pred, average="weighted",  zero_division=0),
    }

def hf_compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Returns a metrics dict whose keys line up with Trainer's expectations,
    e.g. 'eval_f1_macro' when you set metric_for_best_model='eval_f1_macro'.
    """
    y_true, y_pred = _labels_preds_from_eval_pred(eval_pred)
    return {
        "accuracy":            accuracy_score(y_true, y_pred),
        "balanced_accuracy":   balanced_accuracy_score(y_true, y_pred),
        "f1_weighted":         f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro":            f1_score(y_true, y_pred, average="macro",   zero_division=0),
        "f1_micro":            f1_score(y_true, y_pred, average="micro",   zero_division=0),
        "precision_weighted":  precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":     recall_score(y_true, y_pred, average="weighted",  zero_division=0),
    }


def trainer_predict(trainer, dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, logits, probs, y_pred)."""
    out = trainer.predict(dataset)
    logits = _to_numpy_2d(out.predictions)
    probs = _softmax(logits, axis=1)
    y_pred = logits.argmax(axis=1)
    return out.label_ids, logits, probs, y_pred
