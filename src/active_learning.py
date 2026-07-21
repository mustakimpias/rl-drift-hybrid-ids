"""Active-learning label-query strategies for the streaming IDS setting.

Three strategies are compared, all operating under a shared label budget
(as a fraction of the stream length), via the shared `run_budgeted_stream`
engine in stream_utils.py (also reused by the drift-triggered hybrid IDS
arm, so the query/warmup/drift logic is identical in both places):

- random: batch-wise random k-of-batch selection.
- uncertainty: batch-wise top-k most-uncertain selection (margin sampling).
- drift_triggered_uncertainty: queries an entire batch when a drift event
  fired within `drift_window` samples before it; otherwise falls back to
  top-k uncertainty sampling at the normal rate.

A short unconditional warm-up phase (see stream_utils.compute_warmup_size)
labels an initial seed set before any strategy-driven decisions are made.
Without it, a freshly-initialized Hoeffding tree can become falsely
overconfident after just 1-2 samples and then stop querying entirely —
collapsing to a degenerate always-benign predictor. This was a real bug in
the previous per-sample greedy-threshold implementation (F1=0 for both
uncertainty-based strategies at every budget); the batch-wise top-k
selection also guarantees the realized label_query_percentage tracks the
configured budget, which a per-sample stochastic/threshold rule did not.

Design note on the drift signal: the drift detector is fed the model's
prediction-error indicator on every sample (using the ground-truth label
available in this offline simulation), which mirrors standard practice in
the drift-detection literature for benchmarking detector behavior. Query
*decisions* for a batch are made from the drift status as of the start of
that batch — i.e. strictly before that batch's own labels are used — so no
decision is informed by a label it hasn't been "told" yet. This is separate
from — and does not spend — the label-query budget, which only tracks how
often a label is used to update the model.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from river import preprocessing, tree

from .stream_utils import BudgetedStreamResult, run_budgeted_stream

logger = logging.getLogger("rl_drift_ids")


def _fresh_model() -> Any:
    return preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()


def run_active_learning_strategy(
    strategy: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    label_budget_fraction: float = 0.15,
    uncertainty_margin_threshold: float = 0.15,
    drift_window: int = 50,
    detector_type: str = "adwin",
    adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50,
    page_hinkley_min_instances: int = 30,
    batch_size: int = 1000,
    warmup_fraction: float = 0.01,
    warmup_min: int = 100,
    window_size: int = 500,
    seed: int = 42,
) -> BudgetedStreamResult:
    """Run one active-learning strategy over the stream and return its results."""
    result, _ = run_budgeted_stream(
        model_factory=_fresh_model,
        X=X, y=y, feature_names=feature_names,
        strategy=strategy,
        label_budget_fraction=label_budget_fraction,
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
        drift_window=drift_window,
        batch_size=batch_size,
        warmup_fraction=warmup_fraction, warmup_min=warmup_min,
        window_size=window_size,
        seed=seed,
    )
    return result


def run_all_active_learning_strategies(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    strategies: list[str],
    label_budgets: list[float] | float = 0.15,
    uncertainty_margin_threshold: float = 0.15,
    detector_type: str = "adwin",
    adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50,
    page_hinkley_min_instances: int = 30,
    batch_size: int = 1000,
    warmup_fraction: float = 0.01,
    warmup_min: int = 100,
    window_size: int = 500,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, BudgetedStreamResult]]:
    """Run every (strategy, label_budget) combination and return (metric_rows,
    results_by_"strategy@budget"). `label_budgets` is normally a list (e.g.
    config['active_learning_budgets']) so the cost/label-budget trade-off can
    be swept, but a single float is accepted too.
    """
    budgets = label_budgets if isinstance(label_budgets, list) else [label_budgets]
    rows: list[dict[str, Any]] = []
    results: dict[str, BudgetedStreamResult] = {}
    for strategy in strategies:
        for budget in budgets:
            logger.info("Running active learning strategy: %s (budget=%s)", strategy, budget)
            result = run_active_learning_strategy(
                strategy, X, y, feature_names,
                label_budget_fraction=budget,
                uncertainty_margin_threshold=uncertainty_margin_threshold,
                detector_type=detector_type, adwin_delta=adwin_delta,
                page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
                batch_size=batch_size,
                warmup_fraction=warmup_fraction, warmup_min=warmup_min,
                window_size=window_size,
                seed=seed,
            )
            row = dict(result.metrics)
            row["strategy"] = strategy
            row["label_budget_fraction"] = budget
            row["label_query_percentage"] = result.label_query_percentage
            row["drift_count"] = result.drift_count
            row["queried_count"] = len(result.queried_labels)
            row["queried_attack_count"] = result.queried_attack_count
            row["queried_benign_count"] = result.queried_benign_count
            row["feedback_mode"] = "budgeted_active_learning"
            rows.append(row)
            results[f"{strategy}@{budget}"] = result
            logger.info(
                "%s (budget=%s): f1=%.4f label_query_pct=%.2f%% drift_count=%d queried=%d (attack=%d, benign=%d)",
                strategy, budget, row["f1_score"], result.label_query_percentage, result.drift_count,
                len(result.queried_labels), result.queried_attack_count, result.queried_benign_count,
            )
    return rows, results
