"""Invariant tests for the strict-causal limited-feedback protocol
(run_rl_guided_hybrid_strict_causal / run_budgeted_stream(strict_causal=True)).

These check the specific leakage the thesis reviewer flagged in the
archived (non-strict) implementation: for any stream sample whose label was
NOT queried and which is not part of the warm-up seed set, the true label
must never reach the classifier, the reward/Q-update, the supervised drift
detector, or the recent-F1/FPR state features. Fast, self-contained (no
prior pipeline run needed) — uses a small synthetic dataset, same pattern as
tests/test_sanity.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data_loader import generate_synthetic_dataset
from src.hybrid_ids import (
    RuleBasedLayer,
    run_drift_triggered_al_hybrid_strict_causal,
    run_rl_guided_hybrid_strict_causal,
)
from src.preprocessing import preprocess_pipeline
from src.stream_utils import run_budgeted_stream


def _build_stream(seed: int = 7, n_rows: int = 4000):
    df = generate_synthetic_dataset(n_rows=n_rows, n_features=8, seed=seed)
    result = preprocess_pipeline(df, id_like_patterns=[], test_size=0.3, scale=True, seed=seed)
    X = np.concatenate([result.X_train_chrono, result.X_test_chrono])
    y = np.concatenate([result.y_train_chrono, result.y_test_chrono])
    return X, y, result.feature_names


def _fit_rule_layer(X: np.ndarray, y: np.ndarray, seed: int = 7) -> RuleBasedLayer:
    cut = max(1, int(0.2 * len(X)))
    layer = RuleBasedLayer(confidence_threshold=0.97, random_state=seed)
    layer.fit(X[:cut], y[:cut])
    return layer


@pytest.fixture(scope="module")
def strict_causal_run():
    X, y, feature_names = _build_stream()
    rule_layer = _fit_rule_layer(X, y)
    cut = max(1, int(0.2 * len(X)))
    X_stream, y_stream = X[cut:], y[cut:]
    rl_config = {"epsilon": 0.2, "alpha": 0.15, "gamma": 0.9}
    result, controller, trace_df = run_rl_guided_hybrid_strict_causal(
        rule_layer, X_stream, y_stream, feature_names,
        rl_config=rl_config, label_budget_fraction=0.1,
        warmup_fraction=0.02, warmup_min=20, seed=7,
    )
    return result, controller, trace_df


def test_trace_has_expected_columns(strict_causal_run):
    _, _, trace_df = strict_causal_run
    expected = {
        "stream_index", "routed_to_rule_layer", "routed_to_ml_layer",
        "y_true_available_to_algorithm", "queried", "warmup_label", "action",
        "classifier_updated", "q_table_updated", "supervised_reward_used",
        "unsupervised_or_zero_reward_used", "supervised_drift_updated",
        "unsupervised_drift_updated", "recent_f1_fpr_updated", "prediction",
        "final_metric_label_available_to_evaluator",
    }
    assert expected.issubset(set(trace_df.columns))
    assert len(trace_df) > 0


def test_unqueried_nonwarmup_rows_never_see_the_label(strict_causal_run):
    """Core strict-causal invariant: queried=False and warmup_label=False
    implies the true label was not available to the algorithm and none of
    the label-dependent updates ran.
    """
    _, _, trace_df = strict_causal_run
    hidden = trace_df[(~trace_df["queried"]) & (~trace_df["warmup_label"])]
    assert len(hidden) > 0, "test is vacuous if the run never left a label unqueried"

    assert not hidden["y_true_available_to_algorithm"].any()
    assert not hidden["classifier_updated"].any()
    assert not hidden["supervised_reward_used"].any()
    assert not hidden["supervised_drift_updated"].any()
    assert not hidden["recent_f1_fpr_updated"].any()


def test_q_table_updates_on_unqueried_rows_use_only_proxy_reward(strict_causal_run):
    """Q-table updates DO happen on unqueried, ML-routed rows (Rule 6), but
    only ever with the documented zero/neutral proxy reward — never with a
    reward derived from the hidden true label.
    """
    _, _, trace_df = strict_causal_run
    hidden_ml_routed = trace_df[
        (~trace_df["queried"]) & (~trace_df["warmup_label"]) & (trace_df["routed_to_ml_layer"])
    ]
    updated = hidden_ml_routed[hidden_ml_routed["q_table_updated"]]
    assert len(updated) > 0, "expected at least one no-query Q-update in this run"
    assert not updated["supervised_reward_used"].any()
    assert updated["unsupervised_or_zero_reward_used"].all()


def test_queried_rows_do_use_the_label(strict_causal_run):
    """Sanity check on the other side: queried rows (or warm-up) SHOULD show
    label-dependent updates — otherwise the gating above would be vacuously
    true because nothing is ever gated on.
    """
    _, _, trace_df = strict_causal_run
    revealed = trace_df[trace_df["queried"] | trace_df["warmup_label"]]
    assert len(revealed) > 0
    assert revealed["y_true_available_to_algorithm"].any() or revealed["warmup_label"].any()
    queried_ml = trace_df[trace_df["queried"] & trace_df["routed_to_ml_layer"]]
    if len(queried_ml) > 0:
        assert queried_ml["classifier_updated"].all()
        assert queried_ml["supervised_drift_updated"].all()
        assert queried_ml["recent_f1_fpr_updated"].all()


def test_evaluator_always_has_the_final_label(strict_causal_run):
    """The evaluator-side flag is unconditionally True: final metrics are
    always computed post-hoc from the complete label array, regardless of
    what the algorithm itself was allowed to see during the run.
    """
    _, _, trace_df = strict_causal_run
    assert trace_df["final_metric_label_available_to_evaluator"].all()


def test_label_query_percentage_matches_trace_counts(strict_causal_run):
    result, _, trace_df = strict_causal_run
    n = len(trace_df)
    revealed = int((trace_df["queried"] | trace_df["warmup_label"]).sum())
    expected_pct = 100.0 * revealed / n
    assert result.label_query_percentage == pytest.approx(expected_pct, abs=1e-6)


def test_budgeted_stream_strict_causal_runs_and_reports_unsupervised_drift():
    """run_budgeted_stream(strict_causal=True) — the drift_triggered_al_hybrid
    sibling — should run without error and separately report a supervised
    (query-gated) and an unsupervised (always-on) drift count.
    """
    X, y, feature_names = _build_stream(seed=11)
    from src.hybrid_ids import _fresh_river_model  # same model factory used by the arms

    result, _ = run_budgeted_stream(
        model_factory=_fresh_river_model,
        X=X, y=y, feature_names=feature_names,
        strategy="drift_triggered_uncertainty",
        label_budget_fraction=0.1,
        warmup_fraction=0.02, warmup_min=20,
        seed=11, strict_causal=True,
    )
    assert result.label_query_percentage > 0
    assert result.unsupervised_drift_count >= 0


def test_drift_triggered_al_hybrid_strict_causal_arm_runs():
    X, y, feature_names = _build_stream(seed=3)
    rule_layer = _fit_rule_layer(X, y, seed=3)
    cut = max(1, int(0.2 * len(X)))
    X_stream, y_stream = X[cut:], y[cut:]
    arm_result = run_drift_triggered_al_hybrid_strict_causal(
        rule_layer, X_stream, y_stream, feature_names,
        label_budget_fraction=0.1, warmup_fraction=0.02, warmup_min=20, seed=3,
    )
    assert arm_result.feedback_mode == "budgeted_active_learning_strict_causal"
    assert arm_result.label_query_percentage > 0
    assert "unsupervised_drift_count" in arm_result.extra
