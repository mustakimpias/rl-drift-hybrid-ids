"""Shared primitives for streaming/online experiments (active learning,
drift-triggered hybrid IDS). Centralized here so the query/warmup/drift
logic is identical between run_active_learning.py and hybrid_ids.py rather
than duplicated and drifting apart.

Causality contract (see rl_controller.py / hybrid_ids.py for the callers
that must honor it): everything in this module that a *decision* (query
selection, action selection) depends on — uncertainty, drift status, recent
performance — must be computable from information available strictly
*before* the current sample's true label is looked at. The true label may
only be used afterward, to compute reward/error and to actually learn from
a sample once a query has been decided. `run_budgeted_stream` below enforces
this: within a batch, query selection is made from a frozen pre-batch model
and a `drift_window` flag computed from drift events *prior to* the batch.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .drift_detection import DriftDetector
from .evaluation import compute_metrics

logger = logging.getLogger("rl_drift_ids")


def row_to_dict(x: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    return {name: float(val) for name, val in zip(feature_names, x)}


def uncertainty_from_proba(proba_dict: dict[Any, float] | None) -> tuple[float, float]:
    """Returns (proba_attack, uncertainty) from a river predict_proba_one() dict.

    Fallback when the model has no prediction yet (predict_proba_one returns
    an empty dict before any training has happened, or a model without
    probability support is used): proba_attack=0.5, uncertainty=1.0 (treat
    as maximally uncertain — safest default, since the model genuinely knows
    nothing yet).
    """
    if not proba_dict:
        return 0.5, 1.0
    proba_attack = float(proba_dict.get(1, 0.0))
    uncertainty = 1.0 - abs(2.0 * proba_attack - 1.0)
    return proba_attack, uncertainty


def compute_warmup_size(
    n: int, warmup_fraction: float = 0.01, warmup_min: int = 100, budget_cap: int | None = None,
) -> int:
    """Size of the unconditional warm-up label seed set: enough samples to
    give the online model *some* basis for uncertainty estimates before any
    query-selection logic trusts its confidence. Without this, a model with
    zero training sees every sample as maximally uncertain OR (worse, for
    tree models) can become falsely overconfident after 1-2 lucky samples
    and then stop querying entirely — a real cold-start failure mode this
    fixes.
    """
    size = max(warmup_min, int(round(warmup_fraction * n)))
    size = min(size, n)
    if budget_cap is not None:
        size = min(size, budget_cap)
    return max(0, size)


def select_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest-scoring elements (descending), stable-ish via argsort."""
    if k <= 0 or len(scores) == 0:
        return np.array([], dtype=int)
    k = min(k, len(scores))
    return np.argsort(scores)[::-1][:k]


def stratified_warmup_indices(
    y: np.ndarray, eligible_idx: np.ndarray, k: int, rng: np.random.Generator, balanced: bool = True,
) -> np.ndarray:
    """Pick k indices for the warm-up seed set as a class-*balanced* random
    sample drawn from across the *whole* eligible stream — not the first k
    chronological arrivals, and not just proportional to (often severely
    imbalanced) class prevalence.

    This matters a lot on real, naturally block-ordered, imbalanced data:
    CICIoT2023's combined stream is grouped by attack-category folder
    (alphabetical), so a tiny label budget whose entire allowance goes to
    warm-up would, with a chronological-prefix warm-up, end up training
    almost exclusively on whichever category happens to sit first (observed:
    494/500 warm-up samples were benign purely because Benign_Final sorts
    right after the tiny Backdoor_Malware folder, versus benign being ~2% of
    the true stream) — collapsing the model to "always predict benign" and
    tanking F1 to ~0. Even with a class-*proportional* random warm-up, a
    severely imbalanced stream (e.g. 98% attack) would still hand the model
    almost no benign examples to learn from, risking the opposite collapse
    ("always predict attack" — high F1 purely from prevalence, FPR≈1.0 on
    the minority class). `balanced=True` (the default) instead aims for an
    *equal* number of warm-up examples per class regardless of overall
    prevalence — closer to how a real deployment's initial labeled seed set
    is normally curated (deliberately sampled to cover both classes) than
    either "whatever arrived first" or "whatever's most common."
    """
    if k <= 0 or len(eligible_idx) == 0:
        return np.array([], dtype=int)
    k = min(k, len(eligible_idx))
    y_eligible = y[eligible_idx]
    classes, counts = np.unique(y_eligible, return_counts=True)
    n_classes = len(classes)
    selected: list[int] = []
    if balanced:
        equal_share = k // n_classes
        remainder = k - equal_share * n_classes
        for i, (cls, count) in enumerate(zip(classes, counts)):
            share = equal_share + (1 if i < remainder else 0)
            share = min(share, int(count))
            cls_idx = eligible_idx[y_eligible == cls]
            chosen = rng.choice(cls_idx, size=share, replace=False)
            selected.extend(chosen.tolist())
        # Any shortfall (a class had fewer available rows than its equal
        # share) is redistributed from classes with more availability so the
        # seed set still totals ~k where the data allows it.
        shortfall = k - len(selected)
        if shortfall > 0:
            already = set(selected)
            remaining_pool = np.array([i for i in eligible_idx if i not in already])
            if len(remaining_pool) > 0:
                extra = rng.choice(remaining_pool, size=min(shortfall, len(remaining_pool)), replace=False)
                selected.extend(extra.tolist())
    else:
        for cls, count in zip(classes, counts):
            share = max(1, min(int(count), int(round(k * count / len(eligible_idx)))))
            cls_idx = eligible_idx[y_eligible == cls]
            chosen = rng.choice(cls_idx, size=min(share, len(cls_idx)), replace=False)
            selected.extend(chosen.tolist())
    selected_arr = np.array(selected, dtype=int)
    if len(selected_arr) > k:
        selected_arr = rng.choice(selected_arr, size=k, replace=False)
    return np.sort(selected_arr)


def tune_decision_threshold(
    warmup_probs: np.ndarray, warmup_labels: np.ndarray, default: float = 0.5,
) -> float:
    """Pick a decision threshold via grid search over the (already-trained-
    on) warm-up set, maximizing balanced accuracy. A fixed 0.5 cutoff can
    produce a degenerate "always predict the majority class" classifier —
    especially early on, when the model is young and its probability
    estimates are poorly calibrated — which shows up as recall=1.0 but
    specificity≈0 (or vice versa) despite a deceptively fine-looking F1 on
    an imbalanced stream. Falls back to `default` if there's no warm-up data
    or only one class is represented in it (balanced accuracy is undefined).
    """
    if len(warmup_probs) == 0 or len(np.unique(warmup_labels)) < 2:
        return default
    from sklearn.metrics import balanced_accuracy_score

    best_threshold, best_score = default, -1.0
    for t in np.arange(0.30, 0.71, 0.05):
        preds = (warmup_probs >= t).astype(int)
        score = balanced_accuracy_score(warmup_labels, preds)
        if score > best_score:
            best_score, best_threshold = score, float(round(t, 2))
    return best_threshold


@dataclass
class BudgetedStreamResult:
    predictions: list[int]
    query_flags: list[bool]
    queried_labels: list[int]
    rolling_f1: list[float]
    metrics: dict[str, Any]
    label_query_percentage: float
    drift_count: int
    drift_points: list[dict[str, Any]]
    queried_attack_count: int = field(init=False, default=0)
    queried_benign_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.queried_attack_count = sum(1 for v in self.queried_labels if v == 1)
        self.queried_benign_count = sum(1 for v in self.queried_labels if v == 0)


def run_budgeted_stream(
    model_factory: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    strategy: str,
    label_budget_fraction: float,
    detector_type: str = "adwin",
    adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50,
    page_hinkley_min_instances: int = 30,
    drift_window: int = 50,
    batch_size: int = 1000,
    warmup_fraction: float = 0.01,
    warmup_min: int = 100,
    window_size: int = 500,
    rule_matched_mask: np.ndarray | None = None,
    rule_preds: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[BudgetedStreamResult, Any]:
    """Run one label-budgeted online-learning strategy over a stream, with a
    warm-up phase and batch-wise top-k query selection so the budget is
    actually spent close to its target (fixes the cold-start collapse where
    a per-sample greedy-threshold query rule starves itself after 1-2
    samples). Optionally routes samples through a pre-fitted rule layer
    (`rule_matched_mask`/`rule_preds`): matched samples are resolved by the
    rule layer and never consume label budget; the ML layer still scores
    them (for continuous drift monitoring) but its prediction is discarded.

    The warm-up seed set is a class-stratified *random* sample drawn from
    across the whole stream (see stratified_warmup_indices) and is used for
    training only — it is excluded from the returned predictions/metrics,
    matching standard active-learning practice of treating an initial seed
    set as separate from the stream being evaluated (and avoiding the
    alternative of scoring those exact rows again post-training, which would
    just be evaluating the model on what it already memorized). Evaluation
    therefore covers the remaining `n - warmup_n` samples, in their original
    chronological order.

    strategy: "random" | "uncertainty" | "drift_triggered_uncertainty"
    """
    if strategy not in {"random", "uncertainty", "drift_triggered_uncertainty"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    n = len(X)
    budget_total = max(1, int(round(label_budget_fraction * n)))
    eligible_mask = np.ones(n, dtype=bool) if rule_matched_mask is None else ~rule_matched_mask
    eligible_idx_all = np.where(eligible_mask)[0]
    warmup_target = compute_warmup_size(n, warmup_fraction, warmup_min, budget_cap=budget_total)

    model = model_factory()
    detector = DriftDetector(
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    rng = np.random.default_rng(seed)

    warmup_idx = stratified_warmup_indices(y, eligible_idx_all, warmup_target, rng)
    warmup_idx_set = set(warmup_idx.tolist())
    budget_remaining = budget_total

    preds: list[int] = []
    query_flags: list[bool] = []
    queried_labels: list[int] = []
    window_true: list[int] = []
    window_pred: list[int] = []
    rolling_f1: list[float] = []
    steps_since_drift = 10**9
    n_evaluated = 0

    from sklearn.metrics import f1_score

    decision_threshold = 0.5  # tuned below, once warm-up training has happened

    def _score_sample(i: int) -> tuple[dict[str, float], int, int, float]:
        """Returns (x_dict, final_pred, ml_pred, uncertainty) using the
        *current* (not-yet-updated-for-i) model — no label used.
        """
        x_dict = row_to_dict(X[i], feature_names)
        proba = model.predict_proba_one(x_dict)
        p_attack, uncertainty = uncertainty_from_proba(proba)
        ml_pred = 1 if p_attack >= decision_threshold else 0
        if rule_matched_mask is not None and rule_matched_mask[i]:
            final_pred = int(rule_preds[i])
        else:
            final_pred = ml_pred
        return x_dict, final_pred, ml_pred, uncertainty

    def _post_sample(i: int, x_dict: dict, final_pred: int, ml_pred: int, queried: bool) -> None:
        nonlocal steps_since_drift, n_evaluated
        true_label = int(y[i])  # true label only used from here on (post decision)
        error_indicator = float(ml_pred != true_label)
        fired = detector.update(error_indicator)
        steps_since_drift = 0 if (fired["adwin"] or fired["page_hinkley"]) else steps_since_drift + 1

        preds.append(final_pred)
        if queried:
            model.learn_one(x_dict, true_label)
            query_flags.append(True)
            queried_labels.append(true_label)
        else:
            query_flags.append(False)

        window_true.append(true_label)
        window_pred.append(final_pred)
        if len(window_true) > window_size:
            window_true.pop(0)
            window_pred.pop(0)
        n_evaluated += 1
        if n_evaluated % window_size == 0 or i == n - 1:
            rolling_f1.append(f1_score(window_true, window_pred, zero_division=0))

    # --- Warm-up: train-only on the stratified seed set (order doesn't
    # affect drift/rolling-F1 bookkeeping since these rows never reach
    # _post_sample / the evaluation stream at all). ---
    for i in sorted(warmup_idx_set):
        x_dict = row_to_dict(X[i], feature_names)
        model.learn_one(x_dict, int(y[i]))
        query_flags.append(True)
        queried_labels.append(int(y[i]))
        budget_remaining -= 1

    # --- Threshold tuning: score the now-trained model on its own warm-up
    # set (post-training) and pick the 0.5-cutoff replacement that
    # maximizes balanced accuracy — guards against the model collapsing to
    # "always predict the majority class" under class imbalance. ---
    if warmup_idx_set:
        warmup_probs = np.array([
            uncertainty_from_proba(model.predict_proba_one(row_to_dict(X[i], feature_names)))[0]
            for i in sorted(warmup_idx_set)
        ])
        warmup_labels = np.array([int(y[i]) for i in sorted(warmup_idx_set)])
        decision_threshold = tune_decision_threshold(warmup_probs, warmup_labels)
        logger.info("Tuned decision threshold from warm-up set: %.2f", decision_threshold)

    # --- Batch-wise phase over the remaining (non-warm-up) samples, in
    # original chronological order, scoring a frozen-model batch and
    # picking top-k before applying. ---
    remaining_idx = [i for i in range(n) if i not in warmup_idx_set]
    pos = 0
    while pos < len(remaining_idx):
        idxs = remaining_idx[pos: pos + batch_size]
        batch_info = [_score_sample(j) for j in idxs]
        scores = np.array([info[3] for info in batch_info])

        # Query-eligible = not already resolved by the rule layer.
        eligible = np.array([
            rule_matched_mask is None or not rule_matched_mask[j] for j in idxs
        ])
        in_drift_window = steps_since_drift <= drift_window  # status entering this batch, pre-label

        target_k = int(round(label_budget_fraction * len(idxs)))
        if strategy == "random":
            eligible_idx = np.where(eligible)[0]
            k = min(budget_remaining, target_k, len(eligible_idx))
            selected = rng.choice(eligible_idx, size=k, replace=False) if k > 0 else np.array([], dtype=int)
        elif strategy == "uncertainty":
            masked_scores = np.where(eligible, scores, -np.inf)
            k = min(budget_remaining, target_k)
            selected = select_topk(masked_scores, k)
        else:  # drift_triggered_uncertainty
            masked_scores = np.where(eligible, scores, -np.inf)
            k = min(budget_remaining, len(idxs) if in_drift_window else target_k)
            selected = select_topk(masked_scores, k)

        selected_set = set(int(s) for s in selected)

        for offset, j in enumerate(idxs):
            x_dict, final_pred, ml_pred, _ = batch_info[offset]
            queried = offset in selected_set and budget_remaining > 0
            if queried:
                budget_remaining -= 1
            _post_sample(j, x_dict, final_pred, ml_pred, queried=queried)

        pos += batch_size

    metrics = compute_metrics(y[remaining_idx], preds)
    result = BudgetedStreamResult(
        predictions=preds,
        query_flags=query_flags,
        queried_labels=queried_labels,
        rolling_f1=rolling_f1,
        metrics=metrics,
        label_query_percentage=100.0 * sum(query_flags) / n,
        drift_count=detector.total_drift_count(),
        drift_points=detector.drift_points(),
    )
    return result, model
