"""The IDS arms compared in the thesis experiment (see run_hybrid_comparison
for the full list — rule-only, static ML, adaptive ML, drift-aware adaptive
ML, rule+ML hybrid, drift-triggered AL hybrid, and RL-guided hybrid).

The "rule layer" is explicitly a simulated stand-in for a signature engine
(e.g. Snort/Suricata): a shallow decision tree whose verdict is only
"trusted" (treated as a signature match) when its leaf-node class purity
clears a confidence threshold. Below that threshold, traffic is "unmatched"
and falls through to the ML layer. Where the dataset carries a multiclass
attack-category column (UNSW's attack_cat, CICIDS's named Label), the tree
is additionally restricted at *training* time to benign rows plus rows from
a small set of well-represented ("known") attack categories — rows from
rare/unlisted categories never enter its training set, simulating that a
hand-written signature engine was never built to catch them. This uses
TRAINING labels/categories only; predict() is always feature-only and never
touches a label — see RuleBasedLayer.predict() and tests/test_sanity.py.

Evaluation-period design: all arms that fit something offline (rule layer,
static ML) are trained on a small chronological *prefix* of the combined
stream (`rule_train_fraction`, e.g. the first 20%); every arm is then
evaluated over the *rest* of the stream. This keeps the injected/real
concept-drift point inside the monitored evaluation window (previously, a
70/30 train/test split put the drift point entirely inside the "training"
portion, so hybrid arms trivially inherited an already-adapted model and
never observed a live drift event — this made drift_count always read 0 in
the hybrid experiment even though the standalone stream experiment, which
monitors the whole series, detected it). It also removes a real leakage bug
where "static ML" previously reused a baseline model fit on a *shuffled*
train/test split, which overlapped with the chronological stream it was
then "tested" on.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from river import preprocessing, tree as river_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from .drift_detection import DriftDetector
from .evaluation import compute_composite_score, compute_metrics
from .rl_controller import (
    ACTION_ADJUST_THRESHOLD,
    ACTION_NO_QUERY,
    ACTION_QUERY_AND_UPDATE,
    ACTION_UPDATE_IF_DRIFT,
    RLController,
)
from .stream_utils import (
    compute_warmup_size, row_to_dict, run_budgeted_stream, select_topk, stratified_warmup_indices,
    tune_decision_threshold, uncertainty_from_proba,
)
from .utils import set_seed, timer

logger = logging.getLogger("rl_drift_ids")


# --------------------------------------------------------------------------
# Simulated rule / signature layer
# --------------------------------------------------------------------------

class RuleBasedLayer:
    """Simulated signature-matching layer: a shallow decision tree whose
    predictions are only trusted where a leaf is highly pure (confident),
    standing in for a small set of hand-written detection rules. Traffic
    landing in an impure leaf is reported as "unmatched" and left to the ML
    layer, mirroring how real signature engines only cover known patterns.

    predict() takes only X — never y — by construction (see
    tests/test_sanity.py::test_rule_layer_predict_never_takes_labels), so it
    cannot use true test labels to decide a match.
    """

    def __init__(
        self, confidence_threshold: float = 0.97, max_depth: int = 4, random_state: int = 42,
        known_categories_top_k: int = 5,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.known_categories_top_k = known_categories_top_k
        self.known_categories: list[str] | None = None
        self.mode = "unfit"
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, category_train: np.ndarray | None = None) -> "RuleBasedLayer":
        """Fit on training features/labels. If `category_train` (a per-row
        multiclass attack-category array aligned with X_train/y_train) is
        given, the tree is trained only on benign rows plus rows from the
        `known_categories_top_k` most frequent attack categories seen in
        training — rows from rarer/novel categories are excluded from its
        own training set, so it never learns to recognize them. Both
        `y_train` and `category_train` are training-time-only; this never
        touches test data.
        """
        if category_train is not None:
            category_train = np.asarray(category_train)
            attack_mask = y_train == 1
            values, counts = np.unique(category_train[attack_mask], return_counts=True)
            order = np.argsort(counts)[::-1]
            self.known_categories = [str(values[i]) for i in order[: self.known_categories_top_k]]
            self.mode = "category_aware"
            keep_mask = (~attack_mask) | np.isin(category_train, self.known_categories)
            fit_X, fit_y = X_train[keep_mask], y_train[keep_mask]
            logger.info(
                "Rule layer (category-aware): known categories=%s, training on %d/%d rows",
                self.known_categories, int(keep_mask.sum()), len(X_train),
            )
        else:
            self.mode = "feature_threshold_binary"
            self.known_categories = None
            fit_X, fit_y = X_train, y_train
        self.tree.fit(fit_X, fit_y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (matched, predictions) from features only. Where matched
        is False, predictions default to 0 (benign) — a rule-only system
        cannot flag what it doesn't recognize, which is exactly the
        limitation this thesis targets.
        """
        if not self._fitted:
            raise RuntimeError("RuleBasedLayer.fit() must be called before predict().")
        proba = self.tree.predict_proba(X)
        confidence = proba.max(axis=1)
        matched = confidence >= self.confidence_threshold
        raw_pred = self.tree.predict(X)
        predictions = np.where(matched, raw_pred, 0)
        return matched, predictions

    def coverage(self, X: np.ndarray) -> float:
        matched, _ = self.predict(X)
        return float(np.mean(matched))

    def describe(self) -> str:
        if self.mode == "category_aware":
            return f"category-aware simulated rule layer (known categories: {self.known_categories})"
        return "feature-threshold simulated rule layer (binary-only dataset — no attack category info available)"


def rule_layer_precision_recall(
    matched: np.ndarray, rule_preds: np.ndarray, y: np.ndarray,
) -> tuple[float | None, float]:
    """precision = trustworthiness of a rule hit (of samples it claimed to
    recognize, how many were right). recall = of ALL true attacks in y, how
    many did the rule layer alone (matched-and-correct) catch — this is
    penalized by low coverage exactly as a real signature engine would be.
    """
    if matched.any():
        precision = float(precision_score(y[matched], rule_preds[matched], zero_division=0))
    else:
        precision = None
    recall = float(recall_score(y, np.where(matched, rule_preds, 0), zero_division=0))
    return precision, recall


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def _fresh_river_model() -> Any:
    return preprocessing.StandardScaler() | river_tree.HoeffdingTreeClassifier()


def _stratified_category_prefix_indices(
    category: np.ndarray, target_total: int, rng: np.random.Generator,
) -> np.ndarray:
    """Select `target_total` rows for rule/static-ML training, stratified by
    attack_cat (proportional to each category's share of the stream) rather
    than a chronological prefix cut — see run_hybrid_comparison for why a
    prefix cut badly misrepresents category diversity on CICIoT2023's
    category-block-ordered stream once benign is a large block.
    """
    n = len(category)
    categories, counts = np.unique(category, return_counts=True)
    selected: list[int] = []
    for cat, count in zip(categories, counts):
        share = max(1, int(round(target_total * count / n)))
        idx = np.where(category == cat)[0]
        chosen = rng.choice(idx, size=min(share, len(idx)), replace=False)
        selected.extend(chosen.tolist())
    selected_arr = np.array(selected, dtype=int)
    if len(selected_arr) > target_total:
        selected_arr = rng.choice(selected_arr, size=target_total, replace=False)
    return np.sort(selected_arr)


def build_rule_train_stream_split(
    X_train_chrono: np.ndarray, y_train_chrono: np.ndarray,
    X_test_chrono: np.ndarray, y_test_chrono: np.ndarray,
    rule_train_fraction: float,
    seed: int,
    category_train_chrono: np.ndarray | None = None,
    category_test_chrono: np.ndarray | None = None,
) -> dict[str, Any]:
    """Combine train+test chronological arrays into one stream, then split
    into a small rule/static-ML training prefix (stratified by attack
    category — or by binary label if no category column exists — NOT a
    chronological cut; see run_hybrid_comparison's module-level docstring
    for why) plus the remaining evaluation stream.

    Extracted so run_hybrid_comparison, run_rl_reward_tuning, and the
    strict-causal experiment script (scripts/run_strict_causal_experiment.py)
    all reproduce byte-identical splits given the same seed/config, instead
    of three copies of this logic silently drifting apart.
    """
    combined_X = np.concatenate([X_train_chrono, X_test_chrono], axis=0)
    combined_y = np.concatenate([y_train_chrono, y_test_chrono], axis=0)
    combined_category = None
    if category_train_chrono is not None and category_test_chrono is not None:
        combined_category = np.concatenate([category_train_chrono, category_test_chrono], axis=0)

    n_total = len(combined_X)
    target_train = max(1, min(n_total - 1, int(round(n_total * rule_train_fraction))))
    rng = np.random.default_rng(seed)
    if combined_category is not None:
        train_idx = _stratified_category_prefix_indices(combined_category, target_train, rng)
    else:
        train_idx = stratified_warmup_indices(combined_y, np.arange(n_total), target_train, rng, balanced=False)
    train_idx_set = set(train_idx.tolist())
    stream_idx = np.array([i for i in range(n_total) if i not in train_idx_set])

    X_rule_train, y_rule_train = combined_X[train_idx], combined_y[train_idx]
    X_stream, y_stream = combined_X[stream_idx], combined_y[stream_idx]
    category_rule_train = combined_category[train_idx] if combined_category is not None else None

    return {
        "X_rule_train": X_rule_train, "y_rule_train": y_rule_train,
        "category_rule_train": category_rule_train,
        "X_stream": X_stream, "y_stream": y_stream,
        "train_idx": train_idx, "stream_idx": stream_idx,
        "n_total": n_total,
    }


@dataclass
class ArmResult:
    name: str
    predictions: list[int]
    metrics: dict[str, Any]
    train_time_sec: float
    inference_time_sec: float
    label_query_percentage: float
    drift_count: int
    rule_layer_coverage: float
    ml_layer_coverage: float
    rule_precision: float | None = None
    rule_recall: float | None = None
    rule_layer_mode: str | None = None
    # How labels reach the ML layer for this arm — distinguishes an
    # unconstrained "learn on everything" design from a genuinely
    # budget-constrained active-learning/RL query policy, so
    # label_query_percentage isn't misread as a chosen budget when it's
    # really just "100% of whatever reaches the ML layer, no cost control."
    #   "none"                        — no online learning at all
    #   "full_feedback"               — learns on every sample (100%)
    #   "rule_filtered_full_feedback" — learns on every rule-unmatched sample, no budget
    #   "budgeted_active_learning"    — a real label-budget constraint applies
    #   "budgeted_rl_controlled"      — budget constraint + RL controller decides when to spend it
    feedback_mode: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Arm: Rule-only
# --------------------------------------------------------------------------

def run_rule_only(rule_layer: RuleBasedLayer, X_stream: np.ndarray, y_stream: np.ndarray) -> ArmResult:
    with timer() as infer_t:
        matched, preds = rule_layer.predict(X_stream)
    coverage = float(np.mean(matched))
    precision, recall = rule_layer_precision_recall(matched, preds, y_stream)
    metrics = compute_metrics(y_stream, preds)
    return ArmResult(
        name="rule_only",
        predictions=list(preds),
        metrics=metrics,
        train_time_sec=0.0,
        inference_time_sec=infer_t["seconds"],
        label_query_percentage=0.0,
        drift_count=0,
        rule_layer_coverage=coverage * 100,
        ml_layer_coverage=0.0,
        feedback_mode="none",
        rule_precision=precision,
        rule_recall=recall,
        rule_layer_mode=rule_layer.describe(),
    )


# --------------------------------------------------------------------------
# Arm: Static ML-only
# --------------------------------------------------------------------------

def run_static_ml_only(
    X_train: np.ndarray, y_train: np.ndarray, X_stream: np.ndarray, y_stream: np.ndarray, seed: int = 42,
) -> ArmResult:
    """Fits its own RandomForest on the same leakage-safe chronological
    training prefix used by the rule layer — deliberately NOT the baseline
    model from run_baselines.py, which is fit on a *shuffled* split that
    overlaps with whatever chronological "stream" period is used for
    evaluation here (a real leakage bug in the original implementation).
    """
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    with timer() as train_t:
        model.fit(X_train, y_train)
    with timer() as infer_t:
        preds = model.predict(X_stream)
    metrics = compute_metrics(y_stream, preds)
    return ArmResult(
        name="static_ml_only",
        predictions=list(preds),
        metrics=metrics,
        train_time_sec=train_t["seconds"],
        inference_time_sec=infer_t["seconds"],
        label_query_percentage=0.0,
        drift_count=0,
        rule_layer_coverage=0.0,
        ml_layer_coverage=100.0,
        feedback_mode="none",
    )


# --------------------------------------------------------------------------
# Arm: Adaptive ML-only (pure online, no offline pretraining)
# --------------------------------------------------------------------------

def run_adaptive_ml_only(
    X_stream: np.ndarray, y_stream: np.ndarray, feature_names: list[str],
    detector_type: str = "adwin", adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50, page_hinkley_min_instances: int = 30,
) -> ArmResult:
    model = _fresh_river_model()
    detector = DriftDetector(
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    preds: list[int] = []
    with timer() as infer_t:
        for i in range(len(X_stream)):
            x_dict = row_to_dict(X_stream[i], feature_names)
            proba = model.predict_proba_one(x_dict)
            p_attack, _ = uncertainty_from_proba(proba)
            pred = 1 if p_attack >= 0.5 else 0
            preds.append(pred)
            true_label = int(y_stream[i])
            detector.update(float(pred != true_label))
            model.learn_one(x_dict, true_label)

    metrics = compute_metrics(y_stream, preds)
    return ArmResult(
        name="adaptive_ml_only",
        predictions=preds,
        metrics=metrics,
        train_time_sec=0.0,
        inference_time_sec=infer_t["seconds"],
        label_query_percentage=100.0,  # every sample is learned from
        drift_count=detector.total_drift_count(),
        rule_layer_coverage=0.0,
        ml_layer_coverage=100.0,
        feedback_mode="full_feedback",
    )


# --------------------------------------------------------------------------
# Arm: Drift-aware adaptive ML (resets model on detected drift)
# --------------------------------------------------------------------------

def run_drift_aware_adaptive_ml(
    X_stream: np.ndarray, y_stream: np.ndarray, feature_names: list[str],
    detector_type: str = "adwin", adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50, page_hinkley_min_instances: int = 30,
) -> ArmResult:
    """Same as adaptive_ml_only, but actively *acts* on drift: reinitializes
    the model from scratch whenever the detector fires, simulating a simple
    detect-and-retrain policy. Distinguishes this ablation row from plain
    "Adaptive ML only" (which computes drift_count for reporting purposes
    but never changes behavior because of it).
    """
    model = _fresh_river_model()
    detector = DriftDetector(
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    preds: list[int] = []
    reset_count = 0
    with timer() as infer_t:
        for i in range(len(X_stream)):
            x_dict = row_to_dict(X_stream[i], feature_names)
            proba = model.predict_proba_one(x_dict)
            p_attack, _ = uncertainty_from_proba(proba)
            pred = 1 if p_attack >= 0.5 else 0
            preds.append(pred)
            true_label = int(y_stream[i])
            fired = detector.update(float(pred != true_label))
            model.learn_one(x_dict, true_label)
            if fired["adwin"] or fired["page_hinkley"]:
                model = _fresh_river_model()
                reset_count += 1

    metrics = compute_metrics(y_stream, preds)
    return ArmResult(
        name="drift_aware_adaptive_ml",
        predictions=preds,
        metrics=metrics,
        train_time_sec=0.0,
        inference_time_sec=infer_t["seconds"],
        label_query_percentage=100.0,
        drift_count=detector.total_drift_count(),
        rule_layer_coverage=0.0,
        ml_layer_coverage=100.0,
        feedback_mode="full_feedback",
        extra={"model_reset_count": reset_count},
    )


# --------------------------------------------------------------------------
# Arm: Rule + ML hybrid (rule routes; ML learns on every unmatched sample,
# no budget, no drift-specific behavior — the simplest combination)
# --------------------------------------------------------------------------

def run_rule_ml_hybrid(
    rule_layer: RuleBasedLayer, X_stream: np.ndarray, y_stream: np.ndarray, feature_names: list[str],
) -> ArmResult:
    model = _fresh_river_model()
    matched_mask, rule_preds = rule_layer.predict(X_stream)
    preds: list[int] = []
    queried = 0
    with timer() as infer_t:
        for i in range(len(X_stream)):
            if matched_mask[i]:
                preds.append(int(rule_preds[i]))
                continue
            x_dict = row_to_dict(X_stream[i], feature_names)
            proba = model.predict_proba_one(x_dict)
            p_attack, _ = uncertainty_from_proba(proba)
            pred = 1 if p_attack >= 0.5 else 0
            preds.append(pred)
            model.learn_one(x_dict, int(y_stream[i]))
            queried += 1

    metrics = compute_metrics(y_stream, preds)
    coverage = float(np.mean(matched_mask))
    precision, recall = rule_layer_precision_recall(matched_mask, rule_preds, y_stream)
    return ArmResult(
        name="rule_ml_hybrid",
        predictions=preds,
        metrics=metrics,
        train_time_sec=0.0,
        inference_time_sec=infer_t["seconds"],
        # This arm makes no active query *decision* at all — it unconditionally
        # trains on every rule-unmatched sample, so there is no label-budget
        # concept to report here. label_query_percentage is 0 by definition
        # (nothing is *queried* on demand); how much traffic reaches the ML
        # layer at all is a separate, already-reported quantity:
        # ml_layer_coverage. Conflating the two previously made this arm look
        # like it was spending an 86%+ "query budget" when it isn't budgeted
        # at all — feedback_mode="rule_filtered_full_feedback" is the
        # honest description of what's actually happening.
        label_query_percentage=0.0,
        drift_count=0,
        rule_layer_coverage=coverage * 100,
        ml_layer_coverage=(1 - coverage) * 100,
        feedback_mode="rule_filtered_full_feedback",
        rule_precision=precision,
        rule_recall=recall,
        rule_layer_mode=rule_layer.describe(),
        extra={"auto_trained_count": queried},
    )


# --------------------------------------------------------------------------
# Arm: Drift-triggered active-learning hybrid (rule + budgeted AL via the
# shared stream_utils engine)
# --------------------------------------------------------------------------

def run_drift_triggered_al_hybrid(
    rule_layer: RuleBasedLayer,
    X_stream: np.ndarray, y_stream: np.ndarray, feature_names: list[str],
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
    seed: int = 42,
) -> ArmResult:
    matched_mask, rule_preds = rule_layer.predict(X_stream)
    with timer() as infer_t:
        result, _ = run_budgeted_stream(
            model_factory=_fresh_river_model,
            X=X_stream, y=y_stream, feature_names=feature_names,
            strategy="drift_triggered_uncertainty",
            label_budget_fraction=label_budget_fraction,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            drift_window=drift_window, batch_size=batch_size,
            warmup_fraction=warmup_fraction, warmup_min=warmup_min,
            rule_matched_mask=matched_mask, rule_preds=rule_preds,
            seed=seed,
        )

    coverage = float(np.mean(matched_mask))
    precision, recall = rule_layer_precision_recall(matched_mask, rule_preds, y_stream)
    return ArmResult(
        name="drift_triggered_al_hybrid",
        predictions=result.predictions,
        metrics=result.metrics,
        train_time_sec=0.0,
        inference_time_sec=infer_t["seconds"],
        label_query_percentage=result.label_query_percentage,
        drift_count=result.drift_count,
        rule_layer_coverage=coverage * 100,
        ml_layer_coverage=(1 - coverage) * 100,
        feedback_mode="budgeted_active_learning",
        rule_precision=precision,
        rule_recall=recall,
        rule_layer_mode=rule_layer.describe(),
        extra={
            "queried_count": len(result.queried_labels),
            "queried_attack_count": result.queried_attack_count,
            "queried_benign_count": result.queried_benign_count,
        },
    )


# --------------------------------------------------------------------------
# Arm: RL-guided hybrid
# --------------------------------------------------------------------------

def run_rl_guided_hybrid(
    rule_layer: RuleBasedLayer,
    X_stream: np.ndarray, y_stream: np.ndarray, feature_names: list[str],
    rl_config: dict[str, Any],
    label_budget_fraction: float = 0.15,
    detector_type: str = "adwin",
    adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50,
    page_hinkley_min_instances: int = 30,
    drift_recency_window: int = 50,
    metrics_window: int = 200,
    warmup_fraction: float = 0.01,
    warmup_min: int = 100,
    seed: int = 42,
) -> tuple[ArmResult, RLController]:
    """Rule layer filters known/confident traffic; everything else is handed
    to the ML layer, whose query/update/threshold behavior is chosen each
    step by the RL controller based on (uncertainty, drift, recent
    performance, remaining label budget).

    Causality: for each sample, uncertainty/state/action are all computed
    from information available *before* that sample's true label is looked
    at (uncertainty comes from predict_proba_one on features only; the
    drift flag reflects detector status as of the *previous* sample, not
    this one; recent_f1/recent_fpr come from the historical window). The
    true label is read only afterward, to compute the reward, decide
    whether to actually call learn_one, and update the drift detector for
    the *next* step. See tests/test_sanity.py for a structural check that
    RLController.select_action/discretize_state take no label argument.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    ml_model = _fresh_river_model()
    controller = RLController(rl_config=rl_config, seed=seed)
    detector = DriftDetector(
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    n = len(X_stream)
    budget_total = max(1, int(round(label_budget_fraction * n)))
    matched_mask, rule_preds = rule_layer.predict(X_stream)

    # Warm-up seed set: class-stratified random sample across the whole
    # stream (not a chronological prefix) — see
    # stream_utils.stratified_warmup_indices for why a prefix warm-up badly
    # misrepresents class balance on real, category-block-ordered data.
    # Train-only; excluded from the evaluation predictions/metrics below.
    eligible_idx_all = np.where(~matched_mask)[0]
    warmup_target = compute_warmup_size(n, warmup_fraction, warmup_min, budget_cap=budget_total)
    warmup_idx = stratified_warmup_indices(y_stream, eligible_idx_all, warmup_target, rng)
    warmup_idx_set = set(warmup_idx.tolist())
    for i in sorted(warmup_idx_set):
        x_dict = row_to_dict(X_stream[i], feature_names)
        ml_model.learn_one(x_dict, int(y_stream[i]))
    budget_remaining = budget_total - len(warmup_idx_set)

    # Tune the starting decision threshold from the just-trained warm-up set
    # (see stream_utils.tune_decision_threshold) instead of a hardcoded 0.5
    # — guards against an early "always predict majority class" collapse.
    # ACTION_ADJUST_THRESHOLD can still move it further during the run.
    decision_threshold = 0.5
    if warmup_idx_set:
        warmup_probs = np.array([
            uncertainty_from_proba(ml_model.predict_proba_one(row_to_dict(X_stream[i], feature_names)))[0]
            for i in sorted(warmup_idx_set)
        ])
        warmup_labels = np.array([int(y_stream[i]) for i in sorted(warmup_idx_set)])
        decision_threshold = tune_decision_threshold(warmup_probs, warmup_labels)
        logger.info("RL-guided hybrid: tuned starting decision threshold from warm-up set: %.2f", decision_threshold)

    steps_since_drift = 10**9
    recent_true: list[int] = []
    recent_pred: list[int] = []

    from sklearn.metrics import f1_score

    preds: list[int] = []
    queried_flags: list[bool] = [True] * len(warmup_idx_set)
    remaining_idx = [i for i in range(n) if i not in warmup_idx_set]
    action_trace: list[dict[str, Any]] = []

    with timer() as infer_t:
        for i in remaining_idx:
            x_dict = row_to_dict(X_stream[i], feature_names)
            proba = ml_model.predict_proba_one(x_dict)
            p_attack, uncertainty = uncertainty_from_proba(proba)
            ml_pred_monitor = 1 if p_attack >= 0.5 else 0

            # --- Everything up to and including action selection uses only
            # information available before this sample's true label. ---
            drift_flag_for_state = steps_since_drift <= drift_recency_window
            recent_f1 = f1_score(recent_true, recent_pred, zero_division=0) if recent_true else 1.0
            recent_fp = sum(1 for t, p in zip(recent_true, recent_pred) if t == 0 and p == 1)
            recent_neg = sum(1 for t in recent_true if t == 0)
            recent_fpr = recent_fp / recent_neg if recent_neg > 0 else 0.0
            budget_frac = budget_remaining / budget_total

            queried = False
            updated = False

            if matched_mask[i]:
                final_pred = int(rule_preds[i])
            else:
                state = controller.discretize_state(uncertainty, drift_flag_for_state, recent_f1, recent_fpr, budget_frac)
                action = controller.select_action(state, budget_remaining)  # no label used

                if action == ACTION_QUERY_AND_UPDATE and budget_remaining > 0:
                    final_pred = 1 if p_attack >= decision_threshold else 0
                    queried = True
                elif action == ACTION_UPDATE_IF_DRIFT:
                    final_pred = 1 if p_attack >= decision_threshold else 0
                    if drift_flag_for_state and budget_remaining > 0:
                        queried = True
                elif action == ACTION_ADJUST_THRESHOLD:
                    decision_threshold = min(0.9, decision_threshold + 0.05)
                    final_pred = 1 if p_attack >= decision_threshold else 0
                else:  # ACTION_NO_QUERY
                    final_pred = 1 if p_attack >= decision_threshold else 0

                if action != ACTION_ADJUST_THRESHOLD and decision_threshold > 0.5:
                    decision_threshold = max(0.5, decision_threshold - 0.01)

            # --- True label is read only from here on. ---
            true_label = int(y_stream[i])

            if not matched_mask[i] and queried:
                ml_model.learn_one(x_dict, true_label)
                budget_remaining -= 1
                updated = True

            if not matched_mask[i]:
                reward = controller.compute_reward(true_label, final_pred, queried, updated)
                next_state = controller.discretize_state(
                    uncertainty, drift_flag_for_state, recent_f1, recent_fpr, budget_remaining / budget_total,
                )
                controller.update(state, action, reward, next_state)
                controller.decay_epsilon()
                action_trace.append({
                    "sample_index": i,
                    "state_uncertainty_bin": state[0],
                    "state_drift_flag": state[1],
                    "state_recent_f1_bin": state[2],
                    "state_recent_fpr_bin": state[3],
                    "state_budget_remaining_bin": state[4],
                    "action": action,
                    "queried": queried,
                    "updated": updated,
                    "predicted_label": final_pred,
                    "true_label": true_label,
                    "reward": reward,
                    "budget_remaining": budget_remaining,
                })

            error_indicator = float(ml_pred_monitor != true_label)
            fired = detector.update(error_indicator)
            steps_since_drift = 0 if (fired["adwin"] or fired["page_hinkley"]) else steps_since_drift + 1

            preds.append(final_pred)
            queried_flags.append(queried)
            recent_true.append(true_label)
            recent_pred.append(final_pred)
            if len(recent_true) > metrics_window:
                recent_true.pop(0)
                recent_pred.pop(0)

    metrics = compute_metrics(y_stream[remaining_idx], preds)
    coverage = float(np.mean(matched_mask))
    precision, recall = rule_layer_precision_recall(matched_mask, rule_preds, y_stream)
    result = ArmResult(
        name="rl_guided_hybrid",
        predictions=preds,
        metrics=metrics,
        train_time_sec=0.0,
        inference_time_sec=infer_t["seconds"],
        label_query_percentage=100.0 * sum(queried_flags) / n,
        drift_count=detector.total_drift_count(),
        rule_layer_coverage=coverage * 100,
        ml_layer_coverage=(1 - coverage) * 100,
        feedback_mode="budgeted_rl_controlled",
        rule_precision=precision,
        rule_recall=recall,
        rule_layer_mode=rule_layer.describe(),
        extra={"rl_summary": controller.summary(), "action_trace": action_trace},
    )
    return result, controller


# --------------------------------------------------------------------------
# Arm: Drift-triggered active-learning hybrid, strict-causal variant
# --------------------------------------------------------------------------

def run_drift_triggered_al_hybrid_strict_causal(
    rule_layer: RuleBasedLayer,
    X_stream: np.ndarray, y_stream: np.ndarray, feature_names: list[str],
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
    seed: int = 42,
) -> ArmResult:
    """Strict-causal counterpart of run_drift_triggered_al_hybrid: identical
    routing/query policy, but drift-detector updates for samples whose label
    was NOT queried use only the unsupervised (prediction-uncertainty)
    signal instead of the true-label error indicator — see
    stream_utils.run_budgeted_stream(strict_causal=True).
    """
    matched_mask, rule_preds = rule_layer.predict(X_stream)
    with timer() as infer_t:
        result, _ = run_budgeted_stream(
            model_factory=_fresh_river_model,
            X=X_stream, y=y_stream, feature_names=feature_names,
            strategy="drift_triggered_uncertainty",
            label_budget_fraction=label_budget_fraction,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            drift_window=drift_window, batch_size=batch_size,
            warmup_fraction=warmup_fraction, warmup_min=warmup_min,
            rule_matched_mask=matched_mask, rule_preds=rule_preds,
            seed=seed, strict_causal=True,
        )

    coverage = float(np.mean(matched_mask))
    precision, recall = rule_layer_precision_recall(matched_mask, rule_preds, y_stream)
    return ArmResult(
        name="drift_triggered_al_hybrid_strict_causal",
        predictions=result.predictions,
        metrics=result.metrics,
        train_time_sec=0.0,
        inference_time_sec=infer_t["seconds"],
        label_query_percentage=result.label_query_percentage,
        drift_count=result.drift_count,
        rule_layer_coverage=coverage * 100,
        ml_layer_coverage=(1 - coverage) * 100,
        feedback_mode="budgeted_active_learning_strict_causal",
        rule_precision=precision,
        rule_recall=recall,
        rule_layer_mode=rule_layer.describe(),
        extra={
            "queried_count": len(result.queried_labels),
            "queried_attack_count": result.queried_attack_count,
            "queried_benign_count": result.queried_benign_count,
            "unsupervised_drift_count": result.unsupervised_drift_count,
        },
    )


# --------------------------------------------------------------------------
# Arm: RL-guided hybrid, strict-causal variant
#
# See run_rl_guided_hybrid's docstring for the shared routing/action design.
# This function differs from it ONLY in what happens to the true label after
# an action has been chosen — see the docstring below for the exact gating.
# The archived/original run_rl_guided_hybrid above is left completely
# unmodified so prior results remain reproducible; this is a parallel,
# separately-invoked implementation, not a patched version of it.
# --------------------------------------------------------------------------

def run_rl_guided_hybrid_strict_causal(
    rule_layer: RuleBasedLayer,
    X_stream: np.ndarray, y_stream: np.ndarray, feature_names: list[str],
    rl_config: dict[str, Any],
    label_budget_fraction: float = 0.15,
    detector_type: str = "adwin",
    adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50,
    page_hinkley_min_instances: int = 30,
    drift_recency_window: int = 50,
    metrics_window: int = 200,
    warmup_fraction: float = 0.01,
    warmup_min: int = 100,
    seed: int = 42,
    disable_threshold_adjustment: bool = False,
) -> tuple[ArmResult, RLController, pd.DataFrame]:
    """Strict-causal counterpart of run_rl_guided_hybrid.

    `disable_threshold_adjustment` is a diagnostic-only ablation switch (see
    STRICT_CAUSAL_RL_FAILURE_DIAGNOSTICS.md): when True, ACTION_ADJUST_THRESHOLD
    can still be *selected* by the controller (so the action distribution and
    Q-learning dynamics are unaffected), but its real-world effect — actually
    moving `decision_threshold` — is suppressed. This isolates "does the
    threshold-raising behavior itself hurt F1" from "does the controller
    choose to raise the threshold too often."

    Relative to run_rl_guided_hybrid, every consumer of the true label other
    than (a) the classifier update — already correctly gated on `queried` in
    the original — and (b) the evaluator's own final metrics (computed once,
    after the whole run, exactly like the original) is now also gated on
    `queried`:

      - Reward: controller.compute_reward(true_label, ...) is called ONLY
        when queried. Otherwise controller.compute_proxy_reward() (a fixed
        0.0 that never touches true_label) is used instead, and a normal
        Q-table transition is still recorded with that proxy reward so
        Q-learning bookkeeping (visitation counts, epsilon decay) stays
        intact across every step, not just queried ones.
      - Drift detection: a supervised DriftDetector (fed the true-label
        error indicator) is updated ONLY when queried or during warm-up.
        A second, independent unsupervised DriftDetector (fed prediction
        uncertainty, which never requires a label) is updated on every
        single step. The "drift recently" flag consulted by state
        discretization and by ACTION_UPDATE_IF_DRIFT is the OR of both
        detectors' recency windows, so drift-awareness doesn't silently
        degrade to "only aware when we happened to query."
      - recent_f1 / recent_fpr (part of the RL state): computed only from
        the subsequence of (true_label, prediction) pairs where true_label
        was actually queried — never from an unqueried sample's hidden
        label. This makes these two state features a biased, budget-sized
        window rather than the true rolling performance (a known, reported
        limitation — see STRICT_CAUSAL_AUDIT_REPORT.md), but they are never
        computed from information the algorithm would not really have.

    Returns (ArmResult, RLController, trace_df): trace_df is a per-instance
    audit log — one row per sample in X_stream (both warm-up and evaluated)
    — with exactly the columns needed to check every rule in the strict
    causal protocol (stream_index, routed_to_rule_layer/ml_layer,
    y_true_available_to_algorithm, queried, warmup_label, action,
    classifier_updated, q_table_updated, supervised_reward_used,
    unsupervised_or_zero_reward_used, supervised_drift_updated,
    unsupervised_drift_updated, recent_f1_fpr_updated, prediction,
    final_metric_label_available_to_evaluator).
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)
    ml_model = _fresh_river_model()
    controller = RLController(rl_config=rl_config, seed=seed)
    supervised_detector = DriftDetector(
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    unsupervised_detector = DriftDetector(
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    n = len(X_stream)
    budget_total = max(1, int(round(label_budget_fraction * n)))
    matched_mask, rule_preds = rule_layer.predict(X_stream)

    eligible_idx_all = np.where(~matched_mask)[0]
    warmup_target = compute_warmup_size(n, warmup_fraction, warmup_min, budget_cap=budget_total)
    warmup_idx = stratified_warmup_indices(y_stream, eligible_idx_all, warmup_target, rng)
    warmup_idx_set = set(warmup_idx.tolist())

    trace_rows: list[dict[str, Any]] = []

    # Warm-up: a genuinely labeled seed set (train-only, excluded from
    # evaluated predictions/metrics below, exactly as in run_rl_guided_hybrid)
    # — these labels ARE legitimately available to the algorithm by
    # construction, so no gating applies here.
    for i in sorted(warmup_idx_set):
        x_dict = row_to_dict(X_stream[i], feature_names)
        ml_model.learn_one(x_dict, int(y_stream[i]))
        trace_rows.append({
            "stream_index": int(i),
            "routed_to_rule_layer": bool(matched_mask[i]),
            "routed_to_ml_layer": not bool(matched_mask[i]),
            "y_true_available_to_algorithm": True,
            "queried": False,
            "warmup_label": True,
            "action": None,
            "classifier_updated": True,
            "q_table_updated": False,
            "supervised_reward_used": False,
            "unsupervised_or_zero_reward_used": False,
            "supervised_drift_updated": False,
            "unsupervised_drift_updated": False,
            "recent_f1_fpr_updated": False,
            "prediction": None,
            "final_metric_label_available_to_evaluator": True,
            "uncertainty": None,
            "recent_f1": None,
            "recent_fpr": None,
            "decision_threshold": None,
            "drift_flag_for_state": None,
        })
    budget_remaining = budget_total - len(warmup_idx_set)

    decision_threshold = 0.5
    if warmup_idx_set:
        warmup_probs = np.array([
            uncertainty_from_proba(ml_model.predict_proba_one(row_to_dict(X_stream[i], feature_names)))[0]
            for i in sorted(warmup_idx_set)
        ])
        warmup_labels = np.array([int(y_stream[i]) for i in sorted(warmup_idx_set)])
        decision_threshold = tune_decision_threshold(warmup_probs, warmup_labels)
        logger.info(
            "RL-guided hybrid (strict-causal): tuned starting decision threshold from warm-up set: %.2f",
            decision_threshold,
        )

    steps_since_drift_supervised = 10**9
    steps_since_drift_unsupervised = 10**9
    recent_true_queried: list[int] = []
    recent_pred_queried: list[int] = []

    from sklearn.metrics import f1_score

    def _recent_f1_fpr() -> tuple[float, float]:
        if not recent_true_queried:
            return 1.0, 0.0
        f1 = f1_score(recent_true_queried, recent_pred_queried, zero_division=0)
        fp = sum(1 for t, p in zip(recent_true_queried, recent_pred_queried) if t == 0 and p == 1)
        neg = sum(1 for t in recent_true_queried if t == 0)
        fpr = fp / neg if neg > 0 else 0.0
        return f1, fpr

    preds: list[int] = []
    queried_flags: list[bool] = [True] * len(warmup_idx_set)
    remaining_idx = [i for i in range(n) if i not in warmup_idx_set]
    action_trace: list[dict[str, Any]] = []

    with timer() as infer_t:
        for i in remaining_idx:
            x_dict = row_to_dict(X_stream[i], feature_names)
            proba = ml_model.predict_proba_one(x_dict)
            p_attack, uncertainty = uncertainty_from_proba(proba)
            ml_pred_monitor = 1 if p_attack >= 0.5 else 0

            # --- Everything up to and including action selection uses only
            # information available before this sample's true label is
            # looked at: uncertainty from features only, drift flags from
            # detector state as of the *previous* sample, recent_f1/fpr from
            # the queried-only history so far. ---
            drift_flag_for_state = (
                steps_since_drift_supervised <= drift_recency_window
                or steps_since_drift_unsupervised <= drift_recency_window
            )
            recent_f1, recent_fpr = _recent_f1_fpr()
            budget_frac = budget_remaining / budget_total

            queried = False
            updated = False
            action = None
            state = None
            routed_to_rule = bool(matched_mask[i])

            if routed_to_rule:
                final_pred = int(rule_preds[i])
            else:
                state = controller.discretize_state(uncertainty, drift_flag_for_state, recent_f1, recent_fpr, budget_frac)
                action = controller.select_action(state, budget_remaining)  # no label used

                if action == ACTION_QUERY_AND_UPDATE and budget_remaining > 0:
                    final_pred = 1 if p_attack >= decision_threshold else 0
                    queried = True
                elif action == ACTION_UPDATE_IF_DRIFT:
                    final_pred = 1 if p_attack >= decision_threshold else 0
                    if drift_flag_for_state and budget_remaining > 0:
                        queried = True
                elif action == ACTION_ADJUST_THRESHOLD:
                    if not disable_threshold_adjustment:
                        decision_threshold = min(0.9, decision_threshold + 0.05)
                    final_pred = 1 if p_attack >= decision_threshold else 0
                else:  # ACTION_NO_QUERY
                    final_pred = 1 if p_attack >= decision_threshold else 0

                if action != ACTION_ADJUST_THRESHOLD and decision_threshold > 0.5:
                    decision_threshold = max(0.5, decision_threshold - 0.01)

            # --- True label is read only from here on, and is only actually
            # USED (for anything besides the evaluator's own final metrics,
            # computed once at the very end from the full y_stream) when
            # `queried` is True. ---
            true_label = int(y_stream[i])
            y_true_available = queried

            classifier_updated = False
            if (not routed_to_rule) and queried:
                ml_model.learn_one(x_dict, true_label)
                budget_remaining -= 1
                updated = True
                classifier_updated = True

            recent_f1_fpr_updated = False
            if (not routed_to_rule) and queried:
                recent_true_queried.append(true_label)
                recent_pred_queried.append(final_pred)
                if len(recent_true_queried) > metrics_window:
                    recent_true_queried.pop(0)
                    recent_pred_queried.pop(0)
                recent_f1_fpr_updated = True

            # --- Drift detection: supervised detector only sees the true
            # label when it was actually revealed; every other step instead
            # feeds the always-available uncertainty signal to the
            # unsupervised detector. Rule 8: supervised drift update is
            # allowed only for queried/warm-up labels. ---
            supervised_drift_updated = False
            if queried:
                error_indicator = float(ml_pred_monitor != true_label)
                fired_sup = supervised_detector.update(error_indicator)
                steps_since_drift_supervised = (
                    0 if (fired_sup["adwin"] or fired_sup["page_hinkley"]) else steps_since_drift_supervised + 1
                )
                supervised_drift_updated = True
            else:
                steps_since_drift_supervised += 1  # decays recency only; no label used

            fired_unsup = unsupervised_detector.update(uncertainty)
            steps_since_drift_unsupervised = (
                0 if (fired_unsup["adwin"] or fired_unsup["page_hinkley"]) else steps_since_drift_unsupervised + 1
            )
            unsupervised_drift_updated = True

            supervised_reward_used = False
            unsupervised_or_zero_reward_used = False
            q_table_updated = False
            if not routed_to_rule:
                next_drift_flag = (
                    steps_since_drift_supervised <= drift_recency_window
                    or steps_since_drift_unsupervised <= drift_recency_window
                )
                next_recent_f1, next_recent_fpr = _recent_f1_fpr()
                next_state = controller.discretize_state(
                    uncertainty, next_drift_flag, next_recent_f1, next_recent_fpr, budget_remaining / budget_total,
                )
                if queried:
                    reward = controller.compute_reward(true_label, final_pred, queried, updated)
                    supervised_reward_used = True
                else:
                    reward = controller.compute_proxy_reward()
                    unsupervised_or_zero_reward_used = True
                controller.update(state, action, reward, next_state)
                controller.decay_epsilon()
                q_table_updated = True
                action_trace.append({
                    "sample_index": i,
                    "state_uncertainty_bin": state[0],
                    "state_drift_flag": state[1],
                    "state_recent_f1_bin": state[2],
                    "state_recent_fpr_bin": state[3],
                    "state_budget_remaining_bin": state[4],
                    "action": action,
                    "queried": queried,
                    "updated": updated,
                    "predicted_label": final_pred,
                    "true_label": true_label if queried else None,
                    "reward": reward,
                    "supervised_reward_used": supervised_reward_used,
                    "budget_remaining": budget_remaining,
                })

            preds.append(final_pred)
            queried_flags.append(queried)

            trace_rows.append({
                "stream_index": int(i),
                "routed_to_rule_layer": routed_to_rule,
                "routed_to_ml_layer": not routed_to_rule,
                "y_true_available_to_algorithm": y_true_available,
                "queried": queried,
                "warmup_label": False,
                "action": action,
                "classifier_updated": classifier_updated,
                "q_table_updated": q_table_updated,
                "supervised_reward_used": supervised_reward_used,
                "unsupervised_or_zero_reward_used": unsupervised_or_zero_reward_used,
                "supervised_drift_updated": supervised_drift_updated,
                "unsupervised_drift_updated": unsupervised_drift_updated,
                "recent_f1_fpr_updated": recent_f1_fpr_updated,
                "prediction": final_pred,
                "final_metric_label_available_to_evaluator": True,
                "uncertainty": float(uncertainty),
                "recent_f1": float(recent_f1),
                "recent_fpr": float(recent_fpr),
                "decision_threshold": float(decision_threshold),
                "drift_flag_for_state": bool(drift_flag_for_state),
            })

    # Evaluator-only step: uses the full true-label array to score
    # predictions AFTER the run, exactly like run_rl_guided_hybrid and every
    # other arm — this is the one place y_stream is used unconditionally,
    # and it never feeds back into the algorithm above.
    metrics = compute_metrics(y_stream[remaining_idx], preds)
    coverage = float(np.mean(matched_mask))
    precision, recall = rule_layer_precision_recall(matched_mask, rule_preds, y_stream)
    result = ArmResult(
        name="rl_guided_hybrid_strict_causal",
        predictions=preds,
        metrics=metrics,
        train_time_sec=0.0,
        inference_time_sec=infer_t["seconds"],
        label_query_percentage=100.0 * sum(queried_flags) / n,
        drift_count=supervised_detector.total_drift_count(),
        rule_layer_coverage=coverage * 100,
        ml_layer_coverage=(1 - coverage) * 100,
        feedback_mode="budgeted_rl_controlled_strict_causal",
        rule_precision=precision,
        rule_recall=recall,
        rule_layer_mode=rule_layer.describe(),
        extra={
            "rl_summary": controller.summary(),
            "action_trace": action_trace,
            "unsupervised_drift_count": unsupervised_detector.total_drift_count(),
            "warmup_count": len(warmup_idx_set),
            "supervised_drift_points": supervised_detector.drift_points(),
            "unsupervised_drift_points": unsupervised_detector.drift_points(),
        },
    )
    trace_df = pd.DataFrame(trace_rows)
    return result, controller, trace_df


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_hybrid_comparison(
    X_train_chrono: np.ndarray, y_train_chrono: np.ndarray,
    X_test_chrono: np.ndarray, y_test_chrono: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
    seed: int = 42,
    category_train_chrono: np.ndarray | None = None,
    category_test_chrono: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, ArmResult], dict[float, RLController]]:
    """Run every arm and return (metric_rows, results_by_arm_key, rl_controllers_by_budget).

    Combines X_train_chrono + X_test_chrono back into one chronological
    stream, then re-splits it at `rule_train_fraction` (small, e.g. 20%):
    that prefix trains the rule layer and the static-ML arm; every arm is
    *evaluated* on the remainder. See module docstring for why.
    """
    hybrid_cfg = config.get("hybrid_ids", {}).get("rule_layer", {})
    al_cfg = config.get("active_learning", {})
    drift_cfg = config.get("drift_detection", {})
    rl_cfg = config.get("rl", {})

    confidence_threshold = hybrid_cfg.get("confidence_threshold", 0.97)
    known_categories_top_k = hybrid_cfg.get("known_categories_top_k", 5)
    rule_train_fraction = config.get("hybrid_ids", {}).get("rule_train_fraction", 0.2)
    uncertainty_margin_threshold = al_cfg.get("uncertainty_margin_threshold", 0.15)
    warmup_fraction = al_cfg.get("warmup_fraction", 0.01)
    warmup_min = al_cfg.get("warmup_min", 100)
    batch_size = config.get("batch_size", 1000)
    detector_type = config.get("drift_detector", "adwin")
    adwin_delta = drift_cfg.get("adwin_delta", 0.002)
    page_hinkley_threshold = drift_cfg.get("page_hinkley_threshold", 50)
    page_hinkley_min_instances = drift_cfg.get("page_hinkley_min_instances", 30)

    budgets = config.get("active_learning_budgets", [0.15])
    if not isinstance(budgets, list):
        budgets = [budgets]

    # Stratified by attack_cat (or binary label, if no category column
    # exists), not a chronological prefix cut: CICIoT2023's combined stream
    # is category-block-ordered, and with balanced_binary sampling the
    # benign block alone can be tens of thousands of rows sorting right
    # after a single tiny attack-category block — a plain prefix cut can
    # then hand the rule/static-ML training set almost no attack diversity
    # at all (observed: rule_only and static_ml_only both collapsed to
    # F1 < 0.16 when the prefix happened to contain exactly one attack
    # category). Stratifying guarantees every category contributes some
    # training rows regardless of block order. See build_rule_train_stream_split.
    split = build_rule_train_stream_split(
        X_train_chrono, y_train_chrono, X_test_chrono, y_test_chrono,
        rule_train_fraction, seed, category_train_chrono, category_test_chrono,
    )
    X_rule_train, y_rule_train = split["X_rule_train"], split["y_rule_train"]
    X_stream, y_stream = split["X_stream"], split["y_stream"]
    category_rule_train = split["category_rule_train"]
    logger.info(
        "Hybrid comparison: %d rows reserved for rule/static training, %d rows in the evaluation stream",
        len(X_rule_train), len(X_stream),
    )

    rule_layer = RuleBasedLayer(
        confidence_threshold=confidence_threshold, random_state=seed, known_categories_top_k=known_categories_top_k,
    )
    rule_layer.fit(X_rule_train, y_rule_train, category_train=category_rule_train)
    logger.info("Rule layer: %s", rule_layer.describe())

    results: dict[str, ArmResult] = {}
    rows: list[dict[str, Any]] = []
    rl_controllers_by_budget: dict[float, RLController] = {}

    def _add_row(key: str, arm_name: str, res: ArmResult, budget: float | None) -> None:
        results[key] = res
        row = dict(res.metrics)
        row["arm"] = arm_name
        row["label_budget_fraction"] = budget
        row["train_time_sec"] = res.train_time_sec
        row["inference_time_sec"] = res.inference_time_sec
        row["label_query_percentage"] = res.label_query_percentage
        row["drift_count"] = res.drift_count
        row["rule_layer_coverage_pct"] = res.rule_layer_coverage
        row["ml_layer_coverage_pct"] = res.ml_layer_coverage
        row["rule_precision"] = res.rule_precision
        row["rule_recall"] = res.rule_recall
        row["rule_layer_mode"] = res.rule_layer_mode
        row["feedback_mode"] = res.feedback_mode
        rows.append(row)
        logger.info(
            "%s%s: f1=%.4f acc=%.4f label_query_pct=%.2f%% drift_count=%d",
            arm_name, f" (budget={budget})" if budget is not None else "",
            row["f1_score"], row["accuracy"], res.label_query_percentage, res.drift_count,
        )

    logger.info("Running hybrid arm: rule_only")
    _add_row("rule_only", "rule_only", run_rule_only(rule_layer, X_stream, y_stream), None)

    logger.info("Running hybrid arm: static_ml_only")
    _add_row(
        "static_ml_only", "static_ml_only",
        run_static_ml_only(X_rule_train, y_rule_train, X_stream, y_stream, seed=seed),
        None,
    )

    logger.info("Running hybrid arm: adaptive_ml_only")
    _add_row(
        "adaptive_ml_only", "adaptive_ml_only",
        run_adaptive_ml_only(
            X_stream, y_stream, feature_names,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
        ),
        None,
    )

    logger.info("Running hybrid arm: drift_aware_adaptive_ml")
    _add_row(
        "drift_aware_adaptive_ml", "drift_aware_adaptive_ml",
        run_drift_aware_adaptive_ml(
            X_stream, y_stream, feature_names,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
        ),
        None,
    )

    logger.info("Running hybrid arm: rule_ml_hybrid")
    _add_row("rule_ml_hybrid", "rule_ml_hybrid", run_rule_ml_hybrid(rule_layer, X_stream, y_stream, feature_names), None)

    for budget in budgets:
        logger.info("Running hybrid arm: drift_triggered_al_hybrid (budget=%s)", budget)
        dt_result = run_drift_triggered_al_hybrid(
            rule_layer, X_stream, y_stream, feature_names,
            label_budget_fraction=budget,
            uncertainty_margin_threshold=uncertainty_margin_threshold,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            batch_size=batch_size, warmup_fraction=warmup_fraction, warmup_min=warmup_min,
            seed=seed,
        )
        _add_row(f"drift_triggered_al_hybrid@{budget}", "drift_triggered_al_hybrid", dt_result, budget)

        logger.info("Running hybrid arm: rl_guided_hybrid (budget=%s)", budget)
        rl_result, rl_controller = run_rl_guided_hybrid(
            rule_layer, X_stream, y_stream, feature_names,
            rl_config=rl_cfg,
            label_budget_fraction=budget,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            warmup_fraction=warmup_fraction, warmup_min=warmup_min,
            seed=seed,
        )
        _add_row(f"rl_guided_hybrid@{budget}", "rl_guided_hybrid", rl_result, budget)
        rl_controllers_by_budget[budget] = rl_controller

    return rows, results, rl_controllers_by_budget


# --------------------------------------------------------------------------
# RL reward sensitivity tuning (small grid search)
# --------------------------------------------------------------------------

def run_rl_reward_tuning(
    X_train_chrono: np.ndarray, y_train_chrono: np.ndarray,
    X_test_chrono: np.ndarray, y_test_chrono: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
    seed: int = 42,
    category_train_chrono: np.ndarray | None = None,
    category_test_chrono: np.ndarray | None = None,
    fp_penalties: tuple[float, ...] = (-2, -4, -6),
    fn_penalties: tuple[float, ...] = (-6, -8, -10),
    label_query_costs: tuple[float, ...] = (-0.5, -1.0, -2.0),
    sample_size: int = 12000,
    budget_fraction: float = 0.1,
) -> pd.DataFrame:
    """Small reward-sensitivity grid search for the RL controller: holds
    correct_attack_reward=5 and correct_benign_reward=1 fixed and sweeps
    false_positive_penalty x false_negative_penalty x label_query_cost
    (3x3x3 = 27 combinations by default), evaluating RL-guided hybrid once
    per combination on the *same* rule-layer/evaluation-stream split used by
    run_hybrid_comparison, so results are directly comparable to the main
    reported numbers.

    The evaluation stream is stratified-subsampled to `sample_size` rows
    (not a chronological prefix, for the same category-block-ordering
    reason as elsewhere in this module) purely to keep a 27-combination
    sweep tractable — this is a tuning pass, not the final reported result.
    """
    hybrid_cfg = config.get("hybrid_ids", {}).get("rule_layer", {})
    drift_cfg = config.get("drift_detection", {})
    al_cfg = config.get("active_learning", {})
    confidence_threshold = hybrid_cfg.get("confidence_threshold", 0.97)
    known_categories_top_k = hybrid_cfg.get("known_categories_top_k", 5)
    rule_train_fraction = config.get("hybrid_ids", {}).get("rule_train_fraction", 0.2)
    warmup_fraction = al_cfg.get("warmup_fraction", 0.01)
    warmup_min = al_cfg.get("warmup_min", 100)
    detector_type = config.get("drift_detector", "adwin")
    adwin_delta = drift_cfg.get("adwin_delta", 0.002)
    page_hinkley_threshold = drift_cfg.get("page_hinkley_threshold", 50)
    page_hinkley_min_instances = drift_cfg.get("page_hinkley_min_instances", 30)

    split = build_rule_train_stream_split(
        X_train_chrono, y_train_chrono, X_test_chrono, y_test_chrono,
        rule_train_fraction, seed, category_train_chrono, category_test_chrono,
    )
    X_rule_train, y_rule_train = split["X_rule_train"], split["y_rule_train"]
    X_stream, y_stream = split["X_stream"], split["y_stream"]
    category_rule_train = split["category_rule_train"]
    rng = np.random.default_rng(seed)

    rule_layer = RuleBasedLayer(
        confidence_threshold=confidence_threshold, random_state=seed, known_categories_top_k=known_categories_top_k,
    )
    rule_layer.fit(X_rule_train, y_rule_train, category_train=category_rule_train)

    if len(X_stream) > sample_size:
        sub_idx = np.sort(stratified_warmup_indices(y_stream, np.arange(len(X_stream)), sample_size, rng, balanced=False))
        X_tune, y_tune = X_stream[sub_idx], y_stream[sub_idx]
    else:
        X_tune, y_tune = X_stream, y_stream

    logger.info(
        "RL reward tuning: %d combinations over a %d-row stratified subsample of the evaluation stream",
        len(fp_penalties) * len(fn_penalties) * len(label_query_costs), len(X_tune),
    )

    rl_base_cfg = dict(config.get("rl", {}))
    rows: list[dict[str, Any]] = []
    for fp_penalty in fp_penalties:
        for fn_penalty in fn_penalties:
            for label_query_cost in label_query_costs:
                rl_cfg = dict(rl_base_cfg)
                rl_cfg.update({
                    "false_positive_penalty": fp_penalty,
                    "false_negative_penalty": fn_penalty,
                    "label_query_cost": label_query_cost,
                    "correct_attack_reward": 5,
                    "correct_benign_reward": 1,
                })
                result, controller = run_rl_guided_hybrid(
                    rule_layer, X_tune, y_tune, feature_names,
                    rl_config=rl_cfg,
                    label_budget_fraction=budget_fraction,
                    detector_type=detector_type, adwin_delta=adwin_delta,
                    page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
                    warmup_fraction=warmup_fraction, warmup_min=warmup_min,
                    seed=seed,
                )
                m = result.metrics
                total_reward = controller.summary()["total_reward"]
                composite = compute_composite_score(m["macro_f1"], m["false_positive_rate"], result.label_query_percentage)
                rows.append({
                    "fp_penalty": fp_penalty,
                    "fn_penalty": fn_penalty,
                    "label_query_cost": label_query_cost,
                    "f1_score": m["f1_score"],
                    "macro_f1": m["macro_f1"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "recall": m["recall"],
                    "specificity": m["specificity"],
                    "false_positive_rate": m["false_positive_rate"],
                    "mcc": m["mcc"],
                    "label_query_percentage": result.label_query_percentage,
                    "total_reward": total_reward,
                    "composite_score": composite,
                })
                logger.info(
                    "RL reward tuning [fp=%s fn=%s lq=%s]: f1=%.4f fpr=%.4f composite=%.4f",
                    fp_penalty, fn_penalty, label_query_cost, m["f1_score"], m["false_positive_rate"], composite,
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Unseen-attack-category generalization experiment (category_holdout split)
# --------------------------------------------------------------------------

def run_unseen_category_experiment(
    X: np.ndarray, y: np.ndarray, category: np.ndarray, feature_names: list[str],
    rl_config: dict[str, Any],
    detector_type: str = "adwin", adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50, page_hinkley_min_instances: int = 30,
    held_out_fraction: float = 0.3,
    benign_test_fraction: float = 0.3,
    budget_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Train on benign + a subset of attack categories ("known"), test on a
    disjoint set of benign rows + the *held-out* attack categories
    ("unseen") — a category_holdout split. `category` is used only to build
    this split; it is never passed to any model as a feature. This measures
    generalization to genuinely novel attack types, which a random/
    chronological split cannot: those splits still let the model see every
    category during training, so a high score there only shows it can
    recognize attacks it has already met, not ones it hasn't.

    Compares four methods on the *identical* known/unseen split: Static ML
    only (offline baseline), Drift-aware adaptive ML (warm-started on known
    categories, then adapts — and resets on detected drift — during the
    unseen-category test phase, faithful to its definition elsewhere in the
    pipeline; a reset discards the known-category learning, which is a
    real, reportable trade-off this experiment can surface, not a bug),
    Drift-triggered active learning (warm-started, then a budgeted
    top-k-uncertainty query policy over the unseen test phase), and
    RL-guided hybrid (warm-started, then the tabular RL controller decides
    query/update/threshold actions over the unseen test phase — no rule
    layer here, since a rule layer restricted to `known` categories by
    definition wouldn't recognize the held-out ones and would just be
    ml_layer_coverage=100% again).
    """
    from sklearn.metrics import f1_score as _f1_score

    rng = np.random.default_rng(seed)
    attack_categories = sorted(set(category[y == 1].tolist()))
    n_holdout = max(1, int(round(len(attack_categories) * held_out_fraction)))
    held_out = set(rng.choice(attack_categories, size=n_holdout, replace=False).tolist())
    known = [c for c in attack_categories if c not in held_out]

    benign_idx = np.where(y == 0)[0]
    rng.shuffle(benign_idx)
    split = int(len(benign_idx) * (1 - benign_test_fraction))
    benign_train_idx, benign_test_idx = set(benign_idx[:split].tolist()), set(benign_idx[split:].tolist())

    known_set = set(known)
    train_idx = np.array(sorted(
        i for i in range(len(y))
        if (y[i] == 1 and category[i] in known_set) or (y[i] == 0 and i in benign_train_idx)
    ))
    test_idx = np.array(sorted(
        i for i in range(len(y))
        if (y[i] == 1 and category[i] in held_out) or (y[i] == 0 and i in benign_test_idx)
    ))

    logger.info(
        "Unseen-attack-category experiment: %d known categories -> %d train rows; "
        "%d held-out categories (%s) -> %d test rows",
        len(known), len(train_idx), len(held_out), sorted(held_out), len(test_idx),
    )

    rows: list[dict[str, Any]] = []

    def _record(method: str, method_role: str, feedback_mode: str, preds: list[int], extra: dict[str, Any] | None = None) -> None:
        m = compute_metrics(y[test_idx], preds)
        row = {
            "method": method, "method_role": method_role, "feedback_mode": feedback_mode, **m,
            "n_known_categories": len(known), "n_held_out_categories": len(held_out),
            "known_categories": "; ".join(known), "held_out_categories": "; ".join(sorted(held_out)),
            "n_train": len(train_idx), "n_test": len(test_idx),
        }
        if extra:
            row.update(extra)
        row["composite_score"] = compute_composite_score(
            row["macro_f1"], row["false_positive_rate"], row.get("label_query_percentage"),
        )
        rows.append(row)
        logger.info(
            "Unseen-attack-category [%s]: f1=%.4f recall=%.4f specificity=%.4f",
            method, m["f1_score"], m["recall"], m["specificity"],
        )

    # --- 1. Static ML only (offline baseline, no adaptation at all) ---
    static_model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    static_model.fit(X[train_idx], y[train_idx])
    _record("Static ML only", "baseline", "none", list(static_model.predict(X[test_idx])))

    # --- 2. Drift-aware adaptive ML (warm-start on known, reset-on-drift over unseen) ---
    model2 = _fresh_river_model()
    for i in train_idx:
        model2.learn_one(row_to_dict(X[i], feature_names), int(y[i]))
    detector2 = DriftDetector(
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    preds2: list[int] = []
    reset_count2 = 0
    for i in test_idx:
        x_dict = row_to_dict(X[i], feature_names)
        p_attack, _ = uncertainty_from_proba(model2.predict_proba_one(x_dict))
        pred = 1 if p_attack >= 0.5 else 0
        preds2.append(pred)
        true_label = int(y[i])
        fired = detector2.update(float(pred != true_label))
        model2.learn_one(x_dict, true_label)
        if fired["adwin"] or fired["page_hinkley"]:
            model2 = _fresh_river_model()  # faithful to drift_aware_adaptive_ml's definition elsewhere
            reset_count2 += 1
    _record(
        "Drift-aware adaptive ML", "full_feedback_upper_bound", "full_feedback", preds2,
        {"drift_count": detector2.total_drift_count(), "label_query_percentage": 100.0,
         "model_reset_count": reset_count2},
    )

    # --- 3. Drift-triggered active learning (warm-start, then budgeted top-k-uncertainty over unseen) ---
    model3 = _fresh_river_model()
    for i in train_idx:
        model3.learn_one(row_to_dict(X[i], feature_names), int(y[i]))
    detector3 = DriftDetector(
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    budget_total3 = max(1, int(round(budget_fraction * len(test_idx))))
    budget_remaining3 = budget_total3
    preds3: list[int] = []
    steps_since_drift3 = 10**9
    batch_size_local = 500
    test_list = list(test_idx)
    pos = 0
    while pos < len(test_list):
        batch = test_list[pos: pos + batch_size_local]
        scored = []
        for i in batch:
            x_dict = row_to_dict(X[i], feature_names)
            p_attack, unc = uncertainty_from_proba(model3.predict_proba_one(x_dict))
            pred = 1 if p_attack >= 0.5 else 0
            scored.append((i, x_dict, pred, unc))
        in_drift_window = steps_since_drift3 <= 50
        target_k = int(round(budget_fraction * len(batch)))
        k = min(budget_remaining3, len(batch) if in_drift_window else target_k)
        scores_arr = np.array([s[3] for s in scored])
        selected_set = set(int(s) for s in select_topk(scores_arr, k))
        for offset, (i, x_dict, pred, _unc) in enumerate(scored):
            preds3.append(pred)
            true_label = int(y[i])
            fired = detector3.update(float(pred != true_label))
            steps_since_drift3 = 0 if (fired["adwin"] or fired["page_hinkley"]) else steps_since_drift3 + 1
            if offset in selected_set and budget_remaining3 > 0:
                model3.learn_one(x_dict, true_label)
                budget_remaining3 -= 1
        pos += batch_size_local
    _record(
        "Drift-triggered active learning", "active_learning_baseline", "budgeted_active_learning", preds3,
        {"drift_count": detector3.total_drift_count(),
         "label_query_percentage": 100.0 * (budget_total3 - budget_remaining3) / len(test_idx)},
    )

    # --- 4. RL-guided hybrid (warm-start, then RL-controlled actions over unseen; no rule layer) ---
    model4 = _fresh_river_model()
    for i in train_idx:
        model4.learn_one(row_to_dict(X[i], feature_names), int(y[i]))
    controller4 = RLController(rl_config=rl_config, seed=seed)
    detector4 = DriftDetector(
        detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    budget_total4 = max(1, int(round(budget_fraction * len(test_idx))))
    budget_remaining4 = budget_total4
    decision_threshold4 = 0.5
    steps_since_drift4 = 10**9
    recent_true4: list[int] = []
    recent_pred4: list[int] = []
    preds4: list[int] = []
    queried4 = 0
    for i in test_idx:
        x_dict = row_to_dict(X[i], feature_names)
        p_attack, uncertainty = uncertainty_from_proba(model4.predict_proba_one(x_dict))
        ml_pred_monitor = 1 if p_attack >= 0.5 else 0

        drift_flag = steps_since_drift4 <= 50
        recent_f1 = _f1_score(recent_true4, recent_pred4, zero_division=0) if recent_true4 else 1.0
        recent_fp = sum(1 for t, p in zip(recent_true4, recent_pred4) if t == 0 and p == 1)
        recent_neg = sum(1 for t in recent_true4 if t == 0)
        recent_fpr = recent_fp / recent_neg if recent_neg > 0 else 0.0
        budget_frac = budget_remaining4 / budget_total4

        state = controller4.discretize_state(uncertainty, drift_flag, recent_f1, recent_fpr, budget_frac)
        action = controller4.select_action(state, budget_remaining4)  # no label used

        queried, updated = False, False
        if action == ACTION_QUERY_AND_UPDATE and budget_remaining4 > 0:
            final_pred = 1 if p_attack >= decision_threshold4 else 0
            queried = True
        elif action == ACTION_UPDATE_IF_DRIFT:
            final_pred = 1 if p_attack >= decision_threshold4 else 0
            if drift_flag and budget_remaining4 > 0:
                queried = True
        elif action == ACTION_ADJUST_THRESHOLD:
            decision_threshold4 = min(0.9, decision_threshold4 + 0.05)
            final_pred = 1 if p_attack >= decision_threshold4 else 0
        else:
            final_pred = 1 if p_attack >= decision_threshold4 else 0
        if action != ACTION_ADJUST_THRESHOLD and decision_threshold4 > 0.5:
            decision_threshold4 = max(0.5, decision_threshold4 - 0.01)

        true_label = int(y[i])  # true label read only from here on
        if queried:
            model4.learn_one(x_dict, true_label)
            budget_remaining4 -= 1
            updated = True
            queried4 += 1

        reward = controller4.compute_reward(true_label, final_pred, queried, updated)
        next_state = controller4.discretize_state(uncertainty, drift_flag, recent_f1, recent_fpr, budget_remaining4 / budget_total4)
        controller4.update(state, action, reward, next_state)
        controller4.decay_epsilon()

        fired = detector4.update(float(ml_pred_monitor != true_label))
        steps_since_drift4 = 0 if (fired["adwin"] or fired["page_hinkley"]) else steps_since_drift4 + 1

        preds4.append(final_pred)
        recent_true4.append(true_label)
        recent_pred4.append(final_pred)
        if len(recent_true4) > 200:
            recent_true4.pop(0)
            recent_pred4.pop(0)

    _record(
        "RL-guided hybrid", "proposed", "budgeted_rl_controlled", preds4,
        {"drift_count": detector4.total_drift_count(),
         "label_query_percentage": 100.0 * queried4 / len(test_idx),
         "rl_steps": len(test_idx), "total_reward": controller4.summary()["total_reward"]},
    )

    return pd.DataFrame(rows)
