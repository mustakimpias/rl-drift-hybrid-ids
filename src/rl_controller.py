"""Lightweight RL-lite controller: tabular Q-learning over a small
discretized state space, used to decide IDS *behavior* (query a label,
update the model, adjust the decision threshold) — not to classify traffic.

This is intentionally not a deep RL agent. State discretization keeps the
table small and the whole thing reproducible without a replay buffer or
neural network, per the thesis scope constraints.

State = (uncertainty_bin, drift_flag, recent_f1_bin, recent_fpr_bin, budget_bin)
Actions:
    0 = do not query label
    1 = query label and update model
    2 = update only if drift detected
    3 = adjust decision threshold conservatively (raise it, fewer false positives)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("rl_drift_ids")

ACTION_NO_QUERY = 0
ACTION_QUERY_AND_UPDATE = 1
ACTION_UPDATE_IF_DRIFT = 2
ACTION_ADJUST_THRESHOLD = 3
N_ACTIONS = 4

State = tuple[int, int, int, int, int]


@dataclass
class RewardWeights:
    correct_attack_reward: float = 5.0
    correct_benign_reward: float = 1.0
    false_positive_penalty: float = -2.0
    false_negative_penalty: float = -6.0
    label_query_cost: float = -0.5
    model_update_cost: float = -0.1


def _digitize(value: float, n_bins: int, lo: float = 0.0, hi: float = 1.0) -> int:
    """Map value in [lo, hi] to an integer bin in [0, n_bins - 1]."""
    if n_bins <= 1:
        return 0
    value = min(max(value, lo), hi)
    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    return int(np.digitize([value], edges)[0])


class RLController:
    """Tabular Q-learning controller over a discretized IDS decision state."""

    def __init__(self, rl_config: dict[str, Any] | None = None, seed: int = 42) -> None:
        """Build the controller from the flat `rl:` config block, e.g.:

            rl:
              epsilon: 0.1
              alpha: 0.1
              gamma: 0.9
              false_positive_penalty: -2
              false_negative_penalty: -6
              correct_attack_reward: 5
              correct_benign_reward: 1
              label_query_cost: -0.5
              model_update_cost: -0.1

        `epsilon_min` / `epsilon_decay` / `state_bins` are optional extras;
        if omitted, epsilon stays constant (no decay) and state bins default
        to 3 per dimension.
        """
        cfg = rl_config or {}
        self.alpha = cfg.get("alpha", 0.15)
        self.gamma = cfg.get("gamma", 0.9)
        self.epsilon = cfg.get("epsilon", 0.1)
        self.epsilon_min = cfg.get("epsilon_min", self.epsilon)
        self.epsilon_decay = cfg.get("epsilon_decay", 1.0)  # no decay unless explicitly configured
        self.reward_weights = RewardWeights(
            correct_attack_reward=cfg.get("correct_attack_reward", 5.0),
            correct_benign_reward=cfg.get("correct_benign_reward", 1.0),
            false_positive_penalty=cfg.get("false_positive_penalty", -2.0),
            false_negative_penalty=cfg.get("false_negative_penalty", -6.0),
            label_query_cost=cfg.get("label_query_cost", -0.5),
            model_update_cost=cfg.get("model_update_cost", -0.1),
        )
        bins = cfg.get("state_bins", {}) or {}
        self.n_bins = {
            "uncertainty": bins.get("uncertainty", 3),
            "recent_f1": bins.get("recent_f1", 3),
            "recent_fpr": bins.get("recent_fpr", 3),
            "budget_remaining": bins.get("budget_remaining", 3),
        }
        self.rng = np.random.default_rng(seed)
        self.q_table: dict[State, np.ndarray] = defaultdict(lambda: np.zeros(N_ACTIONS))

        self.reward_history: list[float] = []
        self.action_counts: dict[int, int] = defaultdict(int)
        self.query_count = 0
        self.update_count = 0
        self.true_positive_count = 0
        self.true_negative_count = 0
        self.false_positive_count = 0
        self.false_negative_count = 0

    def discretize_state(
        self,
        uncertainty: float,
        drift_detected: bool,
        recent_f1: float,
        recent_fpr: float,
        budget_remaining_frac: float,
    ) -> State:
        return (
            _digitize(uncertainty, self.n_bins["uncertainty"]),
            int(bool(drift_detected)),
            _digitize(recent_f1, self.n_bins["recent_f1"]),
            _digitize(recent_fpr, self.n_bins["recent_fpr"]),
            _digitize(budget_remaining_frac, self.n_bins["budget_remaining"]),
        )

    def select_action(self, state: State, budget_remaining: int) -> int:
        """Epsilon-greedy action selection. Query-consuming actions (1) are
        masked out once the label budget is exhausted.
        """
        valid_actions = [ACTION_NO_QUERY, ACTION_UPDATE_IF_DRIFT, ACTION_ADJUST_THRESHOLD]
        if budget_remaining > 0:
            valid_actions.append(ACTION_QUERY_AND_UPDATE)

        if self.rng.random() < self.epsilon:
            action = int(self.rng.choice(valid_actions))
        else:
            q_values = self.q_table[state]
            masked = np.full(N_ACTIONS, -np.inf)
            masked[valid_actions] = q_values[valid_actions]
            action = int(np.argmax(masked))

        self.action_counts[action] += 1
        return action

    def compute_reward(
        self,
        true_label: int,
        predicted_label: int,
        queried_label: bool,
        updated_model: bool,
    ) -> float:
        """Reward combines detection correctness with labeling/update costs.

        Called strictly *after* an action has already been chosen (see
        select_action / discretize_state, neither of which take a label
        argument) — true_label is only ever consulted here, for reward
        bookkeeping, never for the decision itself.
        """
        w = self.reward_weights
        if predicted_label == 1 and true_label == 1:
            r = w.correct_attack_reward
            self.true_positive_count += 1
        elif predicted_label == 0 and true_label == 0:
            r = w.correct_benign_reward
            self.true_negative_count += 1
        elif predicted_label == 1 and true_label == 0:
            r = w.false_positive_penalty
            self.false_positive_count += 1
        else:  # predicted 0, true 1
            r = w.false_negative_penalty
            self.false_negative_count += 1

        if queried_label:
            r += w.label_query_cost
            self.query_count += 1
        if updated_model:
            r += w.model_update_cost
            self.update_count += 1
        return r

    def update(self, state: State, action: int, reward: float, next_state: State) -> None:
        """Standard tabular Q-learning update rule."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        self.reward_history.append(reward)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def summary(self) -> dict[str, Any]:
        return {
            "total_reward": float(np.sum(self.reward_history)) if self.reward_history else 0.0,
            "mean_reward": float(np.mean(self.reward_history)) if self.reward_history else 0.0,
            "n_steps": len(self.reward_history),
            "final_epsilon": self.epsilon,
            "n_states_visited": len(self.q_table),
            "query_count": self.query_count,
            "update_count": self.update_count,
            "true_positive_count": self.true_positive_count,
            "true_negative_count": self.true_negative_count,
            # Every false positive/negative incurs exactly one reward
            # penalty in this design, so *_count and *_penalty_count are
            # numerically identical — both are exposed since downstream
            # consumers ask for either name.
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "false_positive_penalty_count": self.false_positive_count,
            "false_negative_penalty_count": self.false_negative_count,
            "action_counts": {
                "no_query": self.action_counts.get(ACTION_NO_QUERY, 0),
                "query_and_update": self.action_counts.get(ACTION_QUERY_AND_UPDATE, 0),
                "update_if_drift": self.action_counts.get(ACTION_UPDATE_IF_DRIFT, 0),
                "adjust_threshold": self.action_counts.get(ACTION_ADJUST_THRESHOLD, 0),
            },
        }
