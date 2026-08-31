#!/usr/bin/env python
"""Diagnostics for why the strict-causal RL-guided hybrid arm collapses
(F1 ~0.71-0.72 flat across 1-25% label budgets, vs. the drift-triggered
active-learning baseline which scales normally with budget — see
STRICT_CAUSAL_AUDIT_REPORT.md for the headline finding this investigates).

Tests six hypotheses (reward sparsity, state-discretization collapse,
action/query mismatch, drift-signal mismatch, threshold-adjustment harm, and
"RL adds no benefit over a deterministic policy") using the SAME processed
data, split, and seed as the strict-causal main experiment. Writes only to
results_ciciot2023_strict_causal/diagnostics/ — no archived results, and no
file under results_ciciot2023_strict_causal/tables/ (the Table-5.1-equivalent
outputs from run_strict_causal_experiment.py), are touched or overwritten.

This script does NOT tune anything: the "simple rule" diagnostic baseline
(Hypothesis 6c) reuses the uncertainty-margin threshold already in
configs/default.yaml (active_learning.uncertainty_margin_threshold), not a
threshold fitted on this stream.

Usage:
    python scripts/run_strict_causal_diagnostics.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.drift_detection import DriftDetector
from src.evaluation import compute_metrics
from src.hybrid_ids import (
    RuleBasedLayer,
    _fresh_river_model,
    build_rule_train_stream_split,
    run_drift_triggered_al_hybrid_strict_causal,
    run_rl_guided_hybrid_strict_causal,
)
from src.rl_controller import ACTION_ADJUST_THRESHOLD, ACTION_NO_QUERY, ACTION_QUERY_AND_UPDATE, ACTION_UPDATE_IF_DRIFT
from src.stream_utils import (
    compute_warmup_size, row_to_dict, run_budgeted_stream, stratified_warmup_indices,
    tune_decision_threshold, uncertainty_from_proba,
)
from src.utils import load_config, load_processed_data, resolve_config, set_seed

logger = logging.getLogger("rl_drift_ids")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

ACTION_NAMES = {
    ACTION_NO_QUERY: "no_query",
    ACTION_QUERY_AND_UPDATE: "query_and_update",
    ACTION_UPDATE_IF_DRIFT: "update_if_drift",
    ACTION_ADJUST_THRESHOLD: "adjust_threshold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the strict-causal RL-guided hybrid F1 collapse.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0],
        help="Label budgets to sweep (default matches the strict-causal main experiment).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# Hypothesis 6c: a deterministic, non-RL diagnostic baseline. Self-contained
# here (not added to src/) since it is diagnostic-only, per instructions not
# to grow the tuned library surface. Query iff uncertainty exceeds the
# EXISTING (not stream-tuned) config threshold, or a drift flag is active,
# under the same strict-causal drift bookkeeping as the RL arm.
# --------------------------------------------------------------------------

def run_simple_uncertainty_or_drift_rule(
    rule_layer: RuleBasedLayer,
    X_stream: np.ndarray, y_stream: np.ndarray, feature_names: list[str],
    label_budget_fraction: float,
    uncertainty_threshold: float,
    drift_recency_window: int = 50,
    detector_type: str = "adwin",
    adwin_delta: float = 0.002,
    page_hinkley_threshold: float = 50,
    page_hinkley_min_instances: int = 30,
    warmup_fraction: float = 0.01,
    warmup_min: int = 100,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    ml_model = _fresh_river_model()
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
    for i in sorted(warmup_idx_set):
        x_dict = row_to_dict(X_stream[i], feature_names)
        ml_model.learn_one(x_dict, int(y_stream[i]))
    budget_remaining = budget_total - len(warmup_idx_set)

    decision_threshold = 0.5
    if warmup_idx_set:
        warmup_probs = np.array([
            uncertainty_from_proba(ml_model.predict_proba_one(row_to_dict(X_stream[i], feature_names)))[0]
            for i in sorted(warmup_idx_set)
        ])
        warmup_labels = np.array([int(y_stream[i]) for i in sorted(warmup_idx_set)])
        decision_threshold = tune_decision_threshold(warmup_probs, warmup_labels)

    steps_since_drift_sup = 10**9
    steps_since_drift_unsup = 10**9
    preds: list[int] = []
    queried_flags: list[bool] = [True] * len(warmup_idx_set)
    remaining_idx = [i for i in range(n) if i not in warmup_idx_set]

    for i in remaining_idx:
        x_dict = row_to_dict(X_stream[i], feature_names)
        proba = ml_model.predict_proba_one(x_dict)
        p_attack, uncertainty = uncertainty_from_proba(proba)
        ml_pred_monitor = 1 if p_attack >= 0.5 else 0
        drift_flag = steps_since_drift_sup <= drift_recency_window or steps_since_drift_unsup <= drift_recency_window
        routed_to_rule = bool(matched_mask[i])

        if routed_to_rule:
            final_pred = int(rule_preds[i])
            queried = False
        else:
            final_pred = 1 if p_attack >= decision_threshold else 0
            queried = (uncertainty >= uncertainty_threshold or drift_flag) and budget_remaining > 0

        true_label = int(y_stream[i])
        if (not routed_to_rule) and queried:
            ml_model.learn_one(x_dict, true_label)
            budget_remaining -= 1

        if queried:
            error_indicator = float(ml_pred_monitor != true_label)
            fired_sup = supervised_detector.update(error_indicator)
            steps_since_drift_sup = 0 if (fired_sup["adwin"] or fired_sup["page_hinkley"]) else steps_since_drift_sup + 1
        else:
            steps_since_drift_sup += 1
        fired_unsup = unsupervised_detector.update(uncertainty)
        steps_since_drift_unsup = 0 if (fired_unsup["adwin"] or fired_unsup["page_hinkley"]) else steps_since_drift_unsup + 1

        preds.append(final_pred)
        queried_flags.append(queried)

    metrics = compute_metrics(y_stream[remaining_idx], preds)
    return {
        "metrics": metrics,
        "label_query_percentage": 100.0 * sum(queried_flags) / n,
        "supervised_drift_count": supervised_detector.total_drift_count(),
        "unsupervised_drift_count": unsupervised_detector.total_drift_count(),
    }


def _drift_events_to_stream_index(drift_points: list[dict], position_list: list[int]) -> list[int]:
    """DriftDetector event `index` is a sequential call counter, not a raw
    stream_index — map it back using the ordered list of stream_index values
    at which that detector was actually called (see module docstring)."""
    out = []
    for e in drift_points:
        k = e["index"]
        if 0 <= k < len(position_list):
            out.append(position_list[k])
    return out


def _before_after_window(trace_df: pd.DataFrame, event_stream_indices: list[int], window: int, y_by_stream_index: dict) -> dict:
    """Query rate and prediction-error rate in a `window`-row band (by trace
    row position, i.e. chronological stream order excluding warm-up) before
    vs. after each detected drift event, averaged across all events."""
    df = trace_df[~trace_df["warmup_label"]].reset_index(drop=True)
    pos_by_stream_index = {v: i for i, v in enumerate(df["stream_index"].tolist())}
    before_query, after_query, before_err, after_err, n_events = [], [], [], [], 0
    for si in event_stream_indices:
        if si not in pos_by_stream_index:
            continue
        pos = pos_by_stream_index[si]
        b = df.iloc[max(0, pos - window):pos]
        a = df.iloc[pos:min(len(df), pos + window)]
        if len(b) == 0 or len(a) == 0:
            continue
        n_events += 1
        before_query.append(b["queried"].mean())
        after_query.append(a["queried"].mean())
        b_err = [int(row.prediction != y_by_stream_index[row.stream_index]) for row in b.itertuples() if row.prediction is not None]
        a_err = [int(row.prediction != y_by_stream_index[row.stream_index]) for row in a.itertuples() if row.prediction is not None]
        if b_err:
            before_err.append(np.mean(b_err))
        if a_err:
            after_err.append(np.mean(a_err))
    return {
        "n_events": n_events,
        "mean_query_rate_before": float(np.mean(before_query)) if before_query else None,
        "mean_query_rate_after": float(np.mean(after_query)) if after_query else None,
        "mean_error_rate_before": float(np.mean(before_err)) if before_err else None,
        "mean_error_rate_after": float(np.mean(after_err)) if after_err else None,
    }


def run(config_path_or_dict, budgets: list[float]):
    config = resolve_config(config_path_or_dict)
    seed = config.get("random_state", 42)
    set_seed(seed)

    diag_dir = PROJECT_ROOT / "results_ciciot2023_strict_causal" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    arrays, metadata = load_processed_data(config)
    feature_names = metadata["feature_names"]

    hybrid_cfg = config.get("hybrid_ids", {}).get("rule_layer", {})
    confidence_threshold = hybrid_cfg.get("confidence_threshold", 0.97)
    known_categories_top_k = hybrid_cfg.get("known_categories_top_k", 5)
    rule_train_fraction = config.get("hybrid_ids", {}).get("rule_train_fraction", 0.2)
    al_cfg = config.get("active_learning", {})
    warmup_fraction = al_cfg.get("warmup_fraction", 0.01)
    warmup_min = al_cfg.get("warmup_min", 100)
    uncertainty_margin_threshold = al_cfg.get("uncertainty_margin_threshold", 0.15)
    drift_cfg = config.get("drift_detection", {})
    detector_type = config.get("drift_detector", "adwin")
    adwin_delta = drift_cfg.get("adwin_delta", 0.002)
    page_hinkley_threshold = drift_cfg.get("page_hinkley_threshold", 50)
    page_hinkley_min_instances = drift_cfg.get("page_hinkley_min_instances", 30)
    batch_size = config.get("batch_size", 1000)

    split = build_rule_train_stream_split(
        arrays["X_train_chrono"], arrays["y_train_chrono"],
        arrays["X_test_chrono"], arrays["y_test_chrono"],
        rule_train_fraction, seed,
        arrays.get("category_train_chrono"), arrays.get("category_test_chrono"),
    )
    X_rule_train, y_rule_train = split["X_rule_train"], split["y_rule_train"]
    X_stream, y_stream = split["X_stream"], split["y_stream"]
    category_rule_train = split["category_rule_train"]
    n_stream = len(X_stream)
    y_by_stream_index = {i: int(y_stream[i]) for i in range(n_stream)}
    logger.info("Diagnostics: %d rule/static rows, %d evaluation-stream rows (same split as the main strict-causal run)",
                len(X_rule_train), n_stream)

    rule_layer = RuleBasedLayer(
        confidence_threshold=confidence_threshold, random_state=seed, known_categories_top_k=known_categories_top_k,
    )
    rule_layer.fit(X_rule_train, y_rule_train, category_train=category_rule_train)
    matched_mask, rule_preds = rule_layer.predict(X_stream)

    budget_comparison_rows: list[dict] = []
    action_by_budget_rows: list[dict] = []
    reward_density_rows: list[dict] = []
    state_visitation_rows: list[dict] = []
    query_accounting_rows: list[dict] = []
    drift_analysis: dict[str, dict] = {}
    q_value_report: dict[str, list[dict]] = {}
    threshold_trace_summary: dict[str, dict] = {}
    recent_f1_availability: dict[str, dict] = {}

    def _add_comparison(method: str, budget: float, m: dict, label_query_percentage: float,
                         drift_count, extra_drift=None) -> None:
        budget_comparison_rows.append({
            "method": method, "budget": budget,
            "f1": m["f1_score"], "macro_f1": m["macro_f1"], "recall": m["recall"],
            "precision": m["precision"], "fpr": m["false_positive_rate"], "mcc": m["mcc"],
            "total_labels_revealed": int(round(label_query_percentage / 100.0 * n_stream)),
            "label_query_percentage": label_query_percentage,
            "drift_count": drift_count,
        })

    for budget in budgets:
        logger.info("=== budget=%s ===", budget)

        # --- RL-guided hybrid, strict-causal (normal) ---
        rl_res, rl_controller, trace_df = run_rl_guided_hybrid_strict_causal(
            rule_layer, X_stream, y_stream, feature_names,
            rl_config=config.get("rl", {}), label_budget_fraction=budget,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            warmup_fraction=warmup_fraction, warmup_min=warmup_min, seed=seed,
        )
        _add_comparison("RL-guided hybrid (strict-causal)", budget, rl_res.metrics, rl_res.label_query_percentage, rl_res.drift_count)
        logger.info("  RL-guided (normal): f1=%.4f query_pct=%.2f%%", rl_res.metrics["f1_score"], rl_res.label_query_percentage)

        s = rl_controller.summary()
        action_trace = rl_res.extra.get("action_trace", [])
        rl_steps = len(action_trace)
        supervised_rewards = sum(1 for r in action_trace if r["supervised_reward_used"])
        proxy_rewards = rl_steps - supervised_rewards
        for act_id, act_name in ACTION_NAMES.items():
            cnt = s["action_counts"][act_name]
            action_by_budget_rows.append({
                "budget": budget, "action": act_name, "count": cnt,
                "percentage": 100.0 * cnt / rl_steps if rl_steps else 0.0,
            })
        reward_density_rows.append({
            "budget": budget, "rl_steps": rl_steps, "supervised_rewards": supervised_rewards,
            "proxy_or_zero_rewards": proxy_rewards, "q_updates": rl_steps,
            "reward_density": supervised_rewards / rl_steps if rl_steps else 0.0,
        })

        # State visitation (from the states actually visited during action selection)
        state_counts = Counter(
            (r["state_uncertainty_bin"], r["state_drift_flag"], r["state_recent_f1_bin"], r["state_recent_fpr_bin"], r["state_budget_remaining_bin"])
            for r in action_trace
        )
        for state, cnt in state_counts.items():
            state_visitation_rows.append({
                "budget": budget, "state": str(state), "count": cnt,
                "percentage": 100.0 * cnt / rl_steps if rl_steps else 0.0,
            })

        # Query accounting
        n_warmup = rl_res.extra.get("warmup_count", 0)
        rule_handled = int(np.round(rl_res.rule_layer_coverage / 100.0 * n_stream))
        ml_routed = n_stream - rule_handled
        update_if_drift_query_count = sum(1 for r in action_trace if r["action"] == ACTION_UPDATE_IF_DRIFT and r["queried"])
        total_revealed = n_warmup + s["query_count"]
        query_accounting_rows.append({
            "budget": budget, "warmup_labels": n_warmup, "controller_queries": s["query_count"],
            "query_and_update_queries": s["action_counts"]["query_and_update"],
            "update_if_drift_queries": update_if_drift_query_count,
            "total_labels_revealed": total_revealed,
            "full_stream_percentage": 100.0 * total_revealed / n_stream,
            "ml_routed_percentage": 100.0 * ml_routed / n_stream,
        })

        # Drift analysis (Hypothesis 4): map detector event indices back to
        # stream positions and measure query-rate / error-rate before vs after.
        sup_positions = [r["stream_index"] for r in trace_df[trace_df["queried"] & ~trace_df["warmup_label"]].to_dict("records")]
        unsup_positions = trace_df[~trace_df["warmup_label"]]["stream_index"].tolist()
        sup_events = _drift_events_to_stream_index(rl_res.extra["supervised_drift_points"], sup_positions)
        unsup_events = _drift_events_to_stream_index(rl_res.extra["unsupervised_drift_points"], unsup_positions)
        drift_analysis[str(budget)] = {
            "supervised_drift_count": rl_res.drift_count,
            "unsupervised_drift_count": rl_res.extra.get("unsupervised_drift_count", 0),
            "supervised_event_stream_indices": sup_events,
            "unsupervised_event_stream_indices": unsup_events,
            "supervised_before_after": _before_after_window(trace_df, sup_events, 500, y_by_stream_index),
            "unsupervised_before_after": _before_after_window(trace_df, unsup_events, 500, y_by_stream_index),
        }

        # Threshold trace summary
        thr = trace_df[~trace_df["warmup_label"]]["decision_threshold"].astype(float)
        threshold_trace_summary[str(budget)] = {
            "min": float(thr.min()), "max": float(thr.max()), "mean": float(thr.mean()),
            "final": float(thr.iloc[-1]), "pct_steps_above_0.5": float((thr > 0.5).mean() * 100),
        }

        # recent_f1/fpr availability ("stale"/"unknown")
        ml_routed_rows = trace_df[(~trace_df["warmup_label"]) & (trace_df["routed_to_ml_layer"])]
        recent_f1_availability[str(budget)] = {
            "pct_steps_never_queried_yet_default": float((ml_routed_rows["recent_f1"] == 1.0).mean() * 100),
            "pct_steps_stale_this_step": float((~ml_routed_rows["recent_f1_fpr_updated"]).mean() * 100),
        }

        # Q-values for top-10 most-visited states
        top_states = state_counts.most_common(10)
        q_rows = []
        for state, cnt in top_states:
            q_values = rl_controller.q_table.get(state)
            q_rows.append({
                "state": str(state), "visit_count": cnt,
                "q_no_query": float(q_values[ACTION_NO_QUERY]) if q_values is not None else None,
                "q_query_and_update": float(q_values[ACTION_QUERY_AND_UPDATE]) if q_values is not None else None,
                "q_update_if_drift": float(q_values[ACTION_UPDATE_IF_DRIFT]) if q_values is not None else None,
                "q_adjust_threshold": float(q_values[ACTION_ADJUST_THRESHOLD]) if q_values is not None else None,
            })
        q_value_report[str(budget)] = q_rows

        # --- RL-guided hybrid, strict-causal, threshold-adjustment DISABLED (Hypothesis 5) ---
        rl_res_noThresh, _, _ = run_rl_guided_hybrid_strict_causal(
            rule_layer, X_stream, y_stream, feature_names,
            rl_config=config.get("rl", {}), label_budget_fraction=budget,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            warmup_fraction=warmup_fraction, warmup_min=warmup_min, seed=seed,
            disable_threshold_adjustment=True,
        )
        _add_comparison(
            "RL-guided hybrid (strict-causal, threshold-disabled)", budget,
            rl_res_noThresh.metrics, rl_res_noThresh.label_query_percentage, rl_res_noThresh.drift_count,
        )
        logger.info("  RL-guided (threshold-disabled): f1=%.4f query_pct=%.2f%%",
                    rl_res_noThresh.metrics["f1_score"], rl_res_noThresh.label_query_percentage)

        # --- Hypothesis 6a: drift-triggered active-learning baseline (already have this arm) ---
        dt_res = run_drift_triggered_al_hybrid_strict_causal(
            rule_layer, X_stream, y_stream, feature_names,
            label_budget_fraction=budget, detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            batch_size=batch_size, warmup_fraction=warmup_fraction, warmup_min=warmup_min, seed=seed,
        )
        _add_comparison("Drift-triggered AL hybrid (strict-causal)", budget, dt_res.metrics, dt_res.label_query_percentage, dt_res.drift_count)
        logger.info("  Drift-triggered AL: f1=%.4f query_pct=%.2f%%", dt_res.metrics["f1_score"], dt_res.label_query_percentage)

        # --- Hypothesis 6b: uncertainty-only active learning, strict-causal ---
        unc_result, _ = run_budgeted_stream(
            model_factory=_fresh_river_model, X=X_stream, y=y_stream, feature_names=feature_names,
            strategy="uncertainty", label_budget_fraction=budget,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            batch_size=batch_size, warmup_fraction=warmup_fraction, warmup_min=warmup_min,
            rule_matched_mask=matched_mask, rule_preds=rule_preds, seed=seed, strict_causal=True,
        )
        _add_comparison("Uncertainty-only AL (strict-causal)", budget, unc_result.metrics, unc_result.label_query_percentage, unc_result.drift_count)
        logger.info("  Uncertainty-only AL: f1=%.4f query_pct=%.2f%%", unc_result.metrics["f1_score"], unc_result.label_query_percentage)

        # --- Hypothesis 6c: simple deterministic uncertainty-or-drift rule ---
        simple_result = run_simple_uncertainty_or_drift_rule(
            rule_layer, X_stream, y_stream, feature_names,
            label_budget_fraction=budget, uncertainty_threshold=uncertainty_margin_threshold,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            warmup_fraction=warmup_fraction, warmup_min=warmup_min, seed=seed,
        )
        _add_comparison(
            "Simple rule: uncertainty-or-drift (diagnostic)", budget, simple_result["metrics"],
            simple_result["label_query_percentage"], simple_result["supervised_drift_count"],
        )
        logger.info("  Simple rule: f1=%.4f query_pct=%.2f%%", simple_result["metrics"]["f1_score"], simple_result["label_query_percentage"])

    # --- Write CSVs (task D) ---
    pd.DataFrame(budget_comparison_rows).to_csv(diag_dir / "strict_causal_budget_comparison.csv", index=False)
    pd.DataFrame(action_by_budget_rows).to_csv(diag_dir / "strict_causal_rl_action_by_budget.csv", index=False)
    pd.DataFrame(reward_density_rows).to_csv(diag_dir / "strict_causal_rl_reward_density.csv", index=False)
    pd.DataFrame(state_visitation_rows).to_csv(diag_dir / "strict_causal_state_visitation.csv", index=False)
    pd.DataFrame(query_accounting_rows).to_csv(diag_dir / "strict_causal_query_accounting_by_budget.csv", index=False)
    logger.info("Saved 5 diagnostic CSVs under %s", diag_dir)

    # --- Write a JSON of everything not naturally tabular, for the report to cite verbatim ---
    summary = {
        "budgets": budgets,
        "n_stream": n_stream,
        "uncertainty_margin_threshold_used": uncertainty_margin_threshold,
        "drift_analysis": drift_analysis,
        "threshold_trace_summary": threshold_trace_summary,
        "recent_f1_availability": recent_f1_availability,
        "q_value_report_top10_states": q_value_report,
    }
    with open(diag_dir / "diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved %s", diag_dir / "diagnostics_summary.json")

    return summary


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg, args.budgets)
