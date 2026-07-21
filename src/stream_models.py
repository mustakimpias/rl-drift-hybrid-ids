"""River-based online/adaptive classifiers for the streaming IDS experiments.

Wraps river estimators behind a tiny common interface (predict_one /
predict_proba_one / learn_one) so hybrid_ids.py and active_learning.py don't
need to know which specific online model backs the "adaptive ML" layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from river import forest, linear_model, preprocessing, tree

from .evaluation import compute_metrics
from .utils import optional_import

logger = logging.getLogger("rl_drift_ids")


def get_stream_models(seed: int = 42, use_adaptive_random_forest: bool = True) -> dict[str, Any]:
    """Build the dict of {model_name: river estimator pipeline} to benchmark online."""
    models: dict[str, Any] = {
        "hoeffding_tree": preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier(),
        "online_logistic_regression": preprocessing.StandardScaler() | linear_model.LogisticRegression(),
    }
    if use_adaptive_random_forest:
        try:
            models["adaptive_random_forest"] = (
                preprocessing.StandardScaler() | forest.ARFClassifier(seed=seed)
            )
        except Exception as exc:  # pragma: no cover - defensive, river version differences
            logger.warning("ARFClassifier unavailable (%s); skipping adaptive_random_forest.", exc)
    return models


def _row_to_dict(x: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    return {name: float(val) for name, val in zip(feature_names, x)}


@dataclass
class StreamRunResult:
    predictions: list[int]
    probabilities: list[float]
    rolling_f1: list[float]
    metrics: dict[str, Any]


def run_stream_prequential(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    window_size: int = 500,
) -> StreamRunResult:
    """Prequential (test-then-train) evaluation: predict on each sample, then
    learn from it, in arrival order. Returns per-sample predictions plus a
    rolling F1 curve computed over `window_size`-sample windows.
    """
    from sklearn.metrics import f1_score

    preds: list[int] = []
    probas: list[float] = []
    window_true: list[int] = []
    window_pred: list[int] = []
    rolling_f1: list[float] = []

    for i in range(len(X)):
        x_dict = _row_to_dict(X[i], feature_names)
        proba_dict = model.predict_proba_one(x_dict)
        proba_attack = float(proba_dict.get(1, 0.0)) if proba_dict else 0.0
        pred = 1 if proba_attack >= 0.5 else 0
        if not proba_dict:
            pred = model.predict_one(x_dict) or 0

        preds.append(pred)
        probas.append(proba_attack)
        model.learn_one(x_dict, int(y[i]))

        window_true.append(int(y[i]))
        window_pred.append(pred)
        if len(window_true) > window_size:
            window_true.pop(0)
            window_pred.pop(0)
        if (i + 1) % window_size == 0 or i == len(X) - 1:
            rolling_f1.append(f1_score(window_true, window_pred, zero_division=0))

    metrics = compute_metrics(y, preds, probas)
    return StreamRunResult(predictions=preds, probabilities=probas, rolling_f1=rolling_f1, metrics=metrics)


def train_and_evaluate_stream_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
    window_size: int = 500,
    use_adaptive_random_forest: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, StreamRunResult], dict[str, Any]]:
    """Run every stream model prequentially and return (metric_rows, run_results, fitted_models)."""
    models = get_stream_models(seed=seed, use_adaptive_random_forest=use_adaptive_random_forest)
    rows: list[dict[str, Any]] = []
    run_results: dict[str, StreamRunResult] = {}

    for name, model in models.items():
        logger.info("Running stream model: %s", name)
        result = run_stream_prequential(model, X, y, feature_names, window_size=window_size)
        row = dict(result.metrics)
        row["model"] = name
        rows.append(row)
        run_results[name] = result
        logger.info("%s: prequential f1=%.4f acc=%.4f", name, row["f1_score"], row["accuracy"])

    return rows, run_results, models
