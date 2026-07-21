"""Metric computation shared by every experiment (baseline, stream, active
learning, hybrid). Kept as a single source of truth so numbers are
comparable across arms — every table in results/tables/ gets the full
metric set below simply because they all funnel through compute_metrics().
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute the standard metric set for a binary (0=benign, 1=attack) IDS task.

    Positive class is attack=1. Benign-side metrics (specificity / per-class
    recall_0) are reported explicitly alongside the attack-side ones because
    in IDS, a false positive (benign flagged as attack) has real operational
    cost — a headline F1/accuracy that looks good purely from class
    imbalance can hide a model that flags most benign traffic as malicious.

    ROC-AUC is only included when y_proba is provided and both classes are
    present in y_true (otherwise sklearn would raise).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # a.k.a. detection rate
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # true negative rate

    try:
        mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    except ValueError:
        mcc = 0.0

    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0,
    )

    roc_auc = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            roc_auc = None

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "detection_rate": recall,
        "f1_score": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "specificity": specificity,
        "true_negative_rate": specificity,
        "mcc": mcc,
        "per_class_precision_0": float(per_class_precision[0]),
        "per_class_recall_0": float(per_class_recall[0]),
        "per_class_f1_0": float(per_class_f1[0]),
        "per_class_precision_1": float(per_class_precision[1]),
        "per_class_recall_1": float(per_class_recall[1]),
        "per_class_f1_1": float(per_class_f1[1]),
        "roc_auc": roc_auc,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "n_samples": int(len(y_true)),
    }


def metrics_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of compute_metrics()-shaped dicts into a tidy DataFrame."""
    return pd.DataFrame(rows)


def best_model_by_f1(df: pd.DataFrame, name_col: str = "model") -> str | None:
    """Return the name of the row with the highest f1_score, or None if empty."""
    if df.empty or "f1_score" not in df.columns:
        return None
    return str(df.loc[df["f1_score"].idxmax(), name_col])


def compute_composite_score(
    macro_f1: float, false_positive_rate: float, label_query_percentage: float | None,
) -> float:
    """Single practical-IDS-comparison score trading off detection quality,
    false-alarm cost, and labeling cost:

        composite_score = macro_f1 - 0.5 * false_positive_rate - 0.002 * label_query_percentage

    Picking a "best" method by F1 alone ignores that a method achieving
    slightly lower F1 at a much lower false-positive rate and label budget
    can be the more deployable choice — this score makes that trade-off
    explicit and comparable across arms. `label_query_percentage=None`
    (not applicable / not a budgeted method) is treated as 0 cost, not
    penalized.
    """
    lq = 0.0 if label_query_percentage is None or (isinstance(label_query_percentage, float) and pd.isna(label_query_percentage)) else float(label_query_percentage)
    return float(macro_f1) - 0.5 * float(false_positive_rate) - 0.002 * lq
