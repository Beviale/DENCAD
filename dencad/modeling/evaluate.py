"""Compute the metrics for anomaly detection - model and dataset agnostic"""

from __future__ import annotations

from typing import Optional

import numpy as np


PARTS = ("train", "validation", "test")


def segments(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end of every run of consecutive positives, as half-open bounds."""
    edge = np.diff(np.r_[0, (y != 0).astype(int), 0])
    return np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)


def metrics(y: np.ndarray, score: np.ndarray,
            threshold: Optional[float] = None) -> dict:
    """Threshold-free scores, plus the ones that need a cut-off."""
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)

    usable = y.any() and not y.all()
    out = {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, score)) if usable else None,
        "average_precision": float(average_precision_score(y, score)) if y.any()
        else None,
    }
    if threshold is None:
        grid = np.quantile(score, np.linspace(0.50, 0.9999, 400))
        f1s = [f1_score(y, score >= t, zero_division=0) for t in grid]
        threshold = float(grid[int(np.argmax(f1s))])
    
    pred = (score >= threshold).astype("uint8")
    starts, ends = segments(y)
    caught = sum(1 for a, b in zip(starts, ends) if pred[a:b].any())
    out.update({
        "threshold": float(threshold),
        "flagged": int(pred.sum()),
        "flagged_share": float(pred.mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "segments": int(len(starts)),
        "segments_caught": int(caught),
        "segment_recall": float(caught / len(starts)) if len(starts) else None,
    })
    return out



__all__ = ["metrics", "segments", "PARTS"]
