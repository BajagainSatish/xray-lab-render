"""Binary evaluation metrics, threshold selection, and CAM localization checks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def _arrays(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y_true.ndim != 1 or scores.ndim != 1 or y_true.shape != scores.shape:
        raise ValueError("y_true and scores must be 1D arrays of equal length.")
    if y_true.size == 0:
        raise ValueError("Empty evaluation arrays.")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Labels must be binary 0/1.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain non-finite values.")
    return y_true, scores


def binary_metrics(y_true, scores, threshold: float) -> dict:
    y_true, scores = _arrays(y_true, scores)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be in [0, 1].")
    predicted = scores >= threshold
    positive = y_true == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    ppv = tp / (tp + fp) if tp + fp else None
    npv = tn / (tn + fn) if tn + fn else None
    result = {
        "threshold": float(threshold),
        "n": int(y_true.size),
        "positives": int(positive.sum()),
        "negatives": int((~positive).sum()),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": (tp + tn) / y_true.size,
        "auroc": None,
        "auprc": None,
        "brier": float(brier_score_loss(y_true, scores)),
    }
    if positive.any() and (~positive).any():
        result["auroc"] = float(roc_auc_score(y_true, scores))
        result["auprc"] = float(average_precision_score(y_true, scores))
    return result


def youden_threshold(y_true, scores) -> tuple[float, float]:
    """Pick the threshold maximizing sensitivity + specificity - 1."""
    y_true, scores = _arrays(y_true, scores)
    if len(np.unique(y_true)) != 2:
        raise ValueError("Threshold selection needs both classes.")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j = tpr - fpr
    index = int(np.argmax(j))
    threshold = float(thresholds[index])
    if not np.isfinite(threshold):
        threshold = 1.0
    return threshold, float(j[index])


def bootstrap_ci(y_true, scores, threshold: float, *, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """Percentile bootstrap for AUROC, AUPRC, sensitivity, and specificity."""
    y_true, scores = _arrays(y_true, scores)
    rng = np.random.default_rng(seed)
    collected = {key: [] for key in ("auroc", "auprc", "sensitivity", "specificity")}
    for _ in range(n_boot):
        indices = rng.integers(0, y_true.size, y_true.size)
        sample_y = y_true[indices]
        if len(np.unique(sample_y)) != 2:
            continue
        metrics = binary_metrics(sample_y, scores[indices], threshold)
        for key in collected:
            value = metrics[key]
            if value is not None:
                collected[key].append(value)
    result = {}
    for key, values in collected.items():
        if values:
            result[key] = [float(np.percentile(values, 2.5)),
                           float(np.percentile(values, 97.5))]
        else:
            result[key] = None
    return {"n_boot": n_boot, "seed": seed, "ci95": result}


def cam_pointing(mask, cam) -> dict | None:
    """Does the CAM peak fall inside the lesion mask, and how much mass does it put there?"""
    mask = np.asarray(mask).astype(bool)
    cam = np.asarray(cam, dtype=np.float64)
    if mask.shape != cam.shape:
        raise ValueError("Mask and CAM shapes differ.")
    if not np.isfinite(cam).all() or cam.min() < 0 or cam.max() > 1:
        raise ValueError("CAM must be finite and normalized to [0, 1].")
    area = float(mask.mean())
    if area == 0:
        return None
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    hit = bool(mask[peak])
    total = float(cam.sum())
    mass_inside = float(cam[mask].sum() / total) if total > 0 else 0.0
    return {
        "hit": hit,
        "peak_xy": [int(peak[1]), int(peak[0])],
        "mask_fraction": area,
        "cam_mass_inside": mass_inside,
        "lift": mass_inside / area if area > 0 else None,
    }
