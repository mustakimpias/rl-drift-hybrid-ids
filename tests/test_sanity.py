"""Sanity checks requested in the codebase audit:

1. No label column appears in X features.
2. Synthetic flag appears in all results.
3. Active learning consumes nonzero budget.
4. RL action does not use true label before reward.
5. Rule layer does not use true test labels directly.

These are fast, self-contained unit tests (no dependency on a prior
pipeline run) so they can run in CI or locally via `pytest tests/`.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.active_learning import run_active_learning_strategy
from src.data_loader import generate_synthetic_dataset, load_dataset
from src.hybrid_ids import RuleBasedLayer
from src.preprocessing import preprocess_pipeline
from src.rl_controller import RLController
from src.utils import add_synthetic_flag


# --- 1. No label column appears in X features. ---

def test_label_column_not_in_features():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "feature_a": rng.normal(size=n),
        "feature_b": rng.normal(size=n),
        "label": rng.choice(["benign", "attack"], size=n),
    })
    result = preprocess_pipeline(df, id_like_patterns=[], test_size=0.3, scale=True, seed=0)
    assert result.label_column == "label"
    assert "label" not in result.feature_names
    assert result.n_features == len(result.feature_names) == result.X_train.shape[1]


# --- 2. Synthetic flag appears in all results. ---

def test_add_synthetic_flag_stamps_column():
    df = pd.DataFrame({"f1_score": [0.9, 0.8]})
    flagged_true = add_synthetic_flag(df, True)
    flagged_false = add_synthetic_flag(df, False)
    assert "synthetic_data" in flagged_true.columns
    assert bool(flagged_true["synthetic_data"].iloc[0]) is True
    assert bool(flagged_false["synthetic_data"].iloc[0]) is False
    # original frame must be untouched (add_synthetic_flag copies)
    assert "synthetic_data" not in df.columns


def test_load_dataset_synthetic_flag_and_refusal(tmp_path):
    empty_raw = tmp_path / "raw"
    empty_raw.mkdir()

    df, info = load_dataset(
        "auto", empty_raw, allow_synthetic_fallback=True, synthetic_rows=500, synthetic_features=5, seed=0,
    )
    assert info.is_synthetic is True
    assert len(df) > 0

    with pytest.raises(FileNotFoundError):
        load_dataset("auto", empty_raw, allow_synthetic_fallback=False)


# --- 3. Active learning consumes nonzero budget. ---

@pytest.mark.parametrize("strategy", ["random", "uncertainty", "drift_triggered_uncertainty"])
def test_active_learning_consumes_nonzero_budget(strategy):
    df = generate_synthetic_dataset(n_rows=3000, n_features=8, seed=1)
    result = preprocess_pipeline(df, id_like_patterns=[], test_size=0.3, scale=True, seed=1)
    X = np.concatenate([result.X_train_chrono, result.X_test_chrono])
    y = np.concatenate([result.y_train_chrono, result.y_test_chrono])

    al_result = run_active_learning_strategy(
        strategy, X, y, result.feature_names,
        label_budget_fraction=0.2, batch_size=200,
        warmup_fraction=0.01, warmup_min=20, window_size=200, seed=1,
    )
    assert al_result.label_query_percentage > 0, f"{strategy} queried zero labels — budget starvation bug"
    assert len(al_result.queried_labels) > 0


# --- 4. RL action does not use true label before reward. ---

def test_rl_controller_action_selection_takes_no_label():
    forbidden = {"y", "label", "true_label", "y_true"}
    select_action_params = set(inspect.signature(RLController.select_action).parameters)
    discretize_state_params = set(inspect.signature(RLController.discretize_state).parameters)
    assert forbidden.isdisjoint(select_action_params), (
        f"select_action() must not accept a label argument, got {select_action_params}"
    )
    assert forbidden.isdisjoint(discretize_state_params), (
        f"discretize_state() must not accept a label argument, got {discretize_state_params}"
    )
    # compute_reward is the only place a label may be used, and only after
    # an action has already been chosen.
    reward_params = set(inspect.signature(RLController.compute_reward).parameters)
    assert "true_label" in reward_params


def test_rl_controller_reward_only_after_action():
    controller = RLController(rl_config={"epsilon": 0.5, "alpha": 0.1, "gamma": 0.9}, seed=0)
    state = controller.discretize_state(uncertainty=0.8, drift_detected=False, recent_f1=0.9, recent_fpr=0.01, budget_remaining_frac=1.0)
    action = controller.select_action(state, budget_remaining=10)
    assert action in (0, 1, 2, 3)
    # Reward can only be computed once a prediction/action already exist.
    reward = controller.compute_reward(true_label=1, predicted_label=1, queried_label=True, updated_model=True)
    assert isinstance(reward, float)
    assert controller.query_count == 1
    assert controller.update_count == 1
    assert controller.true_positive_count == 1


# --- 5. Rule layer does not use true test labels directly. ---

def test_rule_layer_predict_signature_takes_only_features():
    params = list(inspect.signature(RuleBasedLayer.predict).parameters.keys())
    assert params == ["self", "X"], f"RuleBasedLayer.predict must take only (self, X), got {params}"


def test_rule_layer_predict_is_deterministic_given_only_features():
    rng = np.random.default_rng(2)
    X_train = rng.normal(size=(300, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = rng.normal(size=(100, 4))

    layer = RuleBasedLayer(confidence_threshold=0.9, random_state=2)
    layer.fit(X_train, y_train)
    matched1, preds1 = layer.predict(X_test)
    matched2, preds2 = layer.predict(X_test)  # no y_test ever passed in
    assert np.array_equal(matched1, matched2)
    assert np.array_equal(preds1, preds2)
