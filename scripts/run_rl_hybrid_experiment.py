#!/usr/bin/env python
"""Run the full hybrid IDS arm comparison: rule-only, static ML-only,
adaptive ML-only, drift-aware adaptive ML, rule+ML hybrid, drift-triggered
active-learning hybrid, and the RL-guided hybrid controller (see
hybrid_ids.run_hybrid_comparison for the full arm list and evaluation-period
design).

Usage:
    python scripts/run_rl_hybrid_experiment.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation import metrics_to_dataframe
from src.hybrid_ids import run_hybrid_comparison, run_rl_reward_tuning, run_unseen_category_experiment
from src.utils import (
    add_synthetic_flag, apply_cli_overrides, get_output_dirs, load_config, load_processed_data,
    resolve_config, set_seed, setup_logging,
)
from src.visualization import plot_rl_hybrid_comparison

logger = logging.getLogger("rl_drift_ids")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RL-guided hybrid IDS comparison.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=None)
    return parser.parse_args()


def run(config_path_or_dict):
    config = resolve_config(config_path_or_dict)
    seed = config.get("random_state", 42)
    set_seed(seed)
    dirs = get_output_dirs(config)
    setup_logging(dirs["logs"], name="rl_drift_ids")

    arrays, metadata = load_processed_data(config)
    feature_names = metadata["feature_names"]
    is_synthetic = metadata.get("is_synthetic", False)

    rows, results, rl_controllers_by_budget = run_hybrid_comparison(
        arrays["X_train_chrono"], arrays["y_train_chrono"],
        arrays["X_test_chrono"], arrays["y_test_chrono"],
        feature_names, config, seed=seed,
        category_train_chrono=arrays.get("category_train_chrono"),
        category_test_chrono=arrays.get("category_test_chrono"),
    )
    df = metrics_to_dataframe(rows)
    df = add_synthetic_flag(df, is_synthetic)
    table_path = dirs["tables"] / "rl_hybrid_metrics.csv"
    df.to_csv(table_path, index=False)
    logger.info("Saved %s", table_path)

    plot_rl_hybrid_comparison(df, dirs["figures"] / "rl_hybrid_comparison.png")

    # --- RL controller summary: one row per swept budget ---
    summary_rows = []
    for budget, controller in rl_controllers_by_budget.items():
        s = controller.summary()
        summary_rows.append({
            "label_budget_fraction": budget,
            "total_reward": s["total_reward"],
            "mean_reward": s["mean_reward"],
            "n_steps": s["n_steps"],
            "final_epsilon": s["final_epsilon"],
            "n_states_visited": s["n_states_visited"],
            "query_count": s["query_count"],
            "update_count": s["update_count"],
            "true_positive_count": s["true_positive_count"],
            "true_negative_count": s["true_negative_count"],
            "false_positive_count": s["false_positive_count"],
            "false_negative_count": s["false_negative_count"],
            "false_positive_penalty_count": s["false_positive_penalty_count"],
            "false_negative_penalty_count": s["false_negative_penalty_count"],
            "action_no_query": s["action_counts"]["no_query"],
            "action_query_and_update": s["action_counts"]["query_and_update"],
            "action_update_if_drift": s["action_counts"]["update_if_drift"],
            "action_adjust_threshold": s["action_counts"]["adjust_threshold"],
        })
    rl_summary_df = pd.DataFrame(summary_rows)
    rl_summary_df = add_synthetic_flag(rl_summary_df, is_synthetic)
    rl_summary_path = dirs["tables"] / "rl_controller_summary.csv"
    rl_summary_df.to_csv(rl_summary_path, index=False)
    logger.info("Saved %s", rl_summary_path)

    # --- rl_action_trace.csv: per-decision-step trace (state/action/reward)
    # for every swept budget, straight from the RL controller's own record
    # of what it did — not reconstructed after the fact. ---
    trace_rows = []
    for key, res in results.items():
        if not key.startswith("rl_guided_hybrid@"):
            continue
        budget = float(key.split("@", 1)[1])
        trace = res.extra.get("action_trace", [])
        for step_idx, row in enumerate(trace):
            trace_rows.append({"label_budget_fraction": budget, "step": step_idx, **row})
    if trace_rows:
        trace_df = pd.DataFrame(trace_rows)
        trace_df = add_synthetic_flag(trace_df, is_synthetic)
        trace_path = dirs["tables"] / "rl_action_trace.csv"
        trace_df.to_csv(trace_path, index=False)
        logger.info("Saved %s (%d rows across %d budgets)", trace_path, len(trace_df), len(rl_controllers_by_budget))
    else:
        logger.warning("rl_action_trace.csv not written: RL controller took zero decision steps at every budget.")

    # --- Merge this experiment's drift events into the shared drift_points.csv
    # (already written by run_stream_experiment.py), so drift is recorded
    # consistently across both experiments instead of one script clobbering
    # the other's output. ---
    hybrid_drift_rows = []
    for key, res in results.items():
        if res.drift_count > 0:
            # ArmResult doesn't carry per-event indices for these arms (only
            # a count); record the count as a single summary row per arm so
            # it's still visible without inventing per-index detail.
            hybrid_drift_rows.append({
                "index": None, "detector": config.get("drift_detector", "adwin"),
                "source_model": res.name, "source_experiment": f"hybrid:{key}",
            })
    drift_table_path = dirs["tables"] / "drift_points.csv"
    if hybrid_drift_rows:
        new_df = pd.DataFrame(hybrid_drift_rows)
        new_df = add_synthetic_flag(new_df, is_synthetic)
        if drift_table_path.exists():
            existing = pd.read_csv(drift_table_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(drift_table_path, index=False)
        logger.info("Merged %d hybrid-experiment drift row(s) into %s", len(hybrid_drift_rows), drift_table_path)

    # --- Optional: RL reward-sensitivity grid search (config-flag-gated,
    # off by default -- a 27-combination sweep is only meant to be run when
    # actively re-tuning the RL controller, not on every pipeline run). ---
    tuning_cfg = config.get("rl_reward_tuning", {})
    if tuning_cfg.get("enabled", False):
        logger.info("=== RL reward-sensitivity tuning (config-flag enabled) ===")
        tuning_df = run_rl_reward_tuning(
            arrays["X_train_chrono"], arrays["y_train_chrono"],
            arrays["X_test_chrono"], arrays["y_test_chrono"],
            feature_names, config, seed=seed,
            category_train_chrono=arrays.get("category_train_chrono"),
            category_test_chrono=arrays.get("category_test_chrono"),
            fp_penalties=tuple(tuning_cfg.get("false_positive_penalty", [-2, -4, -6])),
            fn_penalties=tuple(tuning_cfg.get("false_negative_penalty", [-6, -8, -10])),
            label_query_costs=tuple(tuning_cfg.get("label_query_cost", [-0.5, -1.0, -2.0])),
            sample_size=tuning_cfg.get("sample_size", 12000),
            budget_fraction=tuning_cfg.get("budget_fraction", 0.1),
        )
        tuning_df = add_synthetic_flag(tuning_df, is_synthetic)
        tuning_path = dirs["tables"] / "rl_reward_tuning.csv"
        tuning_df.to_csv(tuning_path, index=False)
        logger.info("Saved %s (%d combinations)", tuning_path, len(tuning_df))

    # --- Unseen-attack-category generalization experiment (category_holdout
    # split): only meaningful when the dataset actually carries a multiclass
    # attack-category column (e.g. CICIoT2023's folder-derived attack_cat). ---
    category_train_chrono = arrays.get("category_train_chrono")
    category_test_chrono = arrays.get("category_test_chrono")
    if category_train_chrono is not None and category_test_chrono is not None:
        combined_X = np.concatenate([arrays["X_train_chrono"], arrays["X_test_chrono"]], axis=0)
        combined_y = np.concatenate([arrays["y_train_chrono"], arrays["y_test_chrono"]], axis=0)
        combined_category = np.concatenate([category_train_chrono, category_test_chrono], axis=0)
        unseen_cfg = config.get("unseen_attack_experiment", {})
        drift_cfg = config.get("drift_detection", {})
        unseen_df = run_unseen_category_experiment(
            combined_X, combined_y, combined_category, feature_names,
            rl_config=config.get("rl", {}),
            detector_type=config.get("drift_detector", "adwin"),
            adwin_delta=drift_cfg.get("adwin_delta", 0.002),
            page_hinkley_threshold=drift_cfg.get("page_hinkley_threshold", 50),
            page_hinkley_min_instances=drift_cfg.get("page_hinkley_min_instances", 30),
            held_out_fraction=unseen_cfg.get("held_out_fraction", 0.3),
            benign_test_fraction=unseen_cfg.get("benign_test_fraction", 0.3),
            budget_fraction=unseen_cfg.get("budget_fraction", 0.1),
            seed=seed,
        )
        unseen_df = add_synthetic_flag(unseen_df, is_synthetic)
        unseen_path = dirs["tables"] / "unseen_attack_metrics.csv"
        unseen_df.to_csv(unseen_path, index=False)
        logger.info("Saved %s", unseen_path)

        try:
            from src.visualization import plot_unseen_attack_comparison
            plot_unseen_attack_comparison(unseen_df, dirs["figures"] / "unseen_attack_comparison")
        except Exception as exc:  # pragma: no cover - figure generation must not fail the run
            logger.warning("Could not generate unseen_attack_comparison figure: %s", exc)
    else:
        logger.warning(
            "unseen_attack_metrics.csv not written: dataset has no multiclass attack-category "
            "column (category_holdout requires one)."
        )

    return df, results, rl_controllers_by_budget


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args.dataset, args.raw_dir, args.output_dir, args.max_samples, args.batch_size, args.random_state)
    run(cfg)
