"""Evaluation metrics. All numbers come from real predictions on real held-out data."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, p_fake: np.ndarray, n_bins: int = 10) -> float:
    """ECE of the predicted-class confidence (standard top-label ECE)."""
    y_true = np.asarray(y_true)
    pred = (p_fake >= 0.5).astype(int)
    conf = np.where(pred == 1, p_fake, 1.0 - p_fake)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None
) -> dict:
    """Metrics for the positive (fake) class plus macro-F1 and confusion matrix.

    `scores` may be probabilities or decision values; only the ranking matters for ROC-AUC.
    """
    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fake": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_fake": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["real", "fake"],
        "n_samples": int(len(y_true)),
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        report["roc_auc"] = float(roc_auc_score(y_true, scores))
    return report


def calibration_report(y_true: np.ndarray, p_fake: np.ndarray) -> dict:
    return {
        "ece": expected_calibration_error(y_true, p_fake),
        "brier": float(brier_score_loss(y_true, p_fake)),
    }
