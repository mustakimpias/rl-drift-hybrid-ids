#!/usr/bin/env python
"""Compare active-learning label-query strategies (random, uncertainty,
drift-triggered uncertainty) under a shared label budget, over the full
chronological stream.

Usage:
    python scripts/run_active_learning.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.active_learning import run_all_active_learning_strategies
from src.evaluation import metrics_to_dataframe
from src.utils import (
    add_synthetic_flag, apply_cli_overrides, get_output_dirs, load_config, load_processed_data,
    resolve_config, set_seed, setup_logging,
)
from src.visualization import plot_label_budget_vs_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare active learning label-query strategies.")
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
    logger = setup_logging(dirs["logs"], name="rl_drift_ids")

    arrays, metadata = load_processed_data(config)
    X_stream = np.concatenate([arrays["X_train_chrono"], arrays["X_test_chrono"]], axis=0)
    y_stream = np.concatenate([arrays["y_train_chrono"], arrays["y_test_chrono"]], axis=0)
    feature_names = metadata["feature_names"]

    al_cfg = config.get("active_learning", {})
    drift_cfg = config.get("drift_detection", {})

    batch_size = config.get("batch_size", 1000)
    rows, results = run_all_active_learning_strategies(
        X_stream, y_stream, feature_names,
        strategies=al_cfg.get("strategies", ["random", "uncertainty", "drift_triggered_uncertainty"]),
        label_budgets=config.get("active_learning_budgets", [0.15]),
        uncertainty_margin_threshold=al_cfg.get("uncertainty_margin_threshold", 0.15),
        detector_type=config.get("drift_detector", "adwin"),
        adwin_delta=drift_cfg.get("adwin_delta", 0.002),
        page_hinkley_threshold=drift_cfg.get("page_hinkley_threshold", 50),
        page_hinkley_min_instances=drift_cfg.get("page_hinkley_min_instances", 30),
        batch_size=batch_size,
        warmup_fraction=al_cfg.get("warmup_fraction", 0.01),
        warmup_min=al_cfg.get("warmup_min", 100),
        window_size=batch_size,
        seed=seed,
    )
    df = metrics_to_dataframe(rows)
    df = add_synthetic_flag(df, metadata.get("is_synthetic", False))
    table_path = dirs["tables"] / "active_learning_metrics.csv"
    df.to_csv(table_path, index=False)
    logger.info("Saved %s", table_path)
    for strategy in al_cfg.get("strategies", []):
        for budget in config.get("active_learning_budgets", [0.15]):
            pct = df.loc[(df["strategy"] == strategy) & (df["label_budget_fraction"] == budget), "label_query_percentage"]
            if not pct.empty and float(pct.iloc[0]) <= 0.0:
                logger.warning(
                    "Active learning strategy '%s' at budget=%s consumed ZERO label budget — check batch_size/warmup config.",
                    strategy, budget,
                )

    plot_label_budget_vs_f1(df, dirs["figures"] / "label_budget_vs_f1.png")

    return df, results


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args.dataset, args.raw_dir, args.output_dir, args.max_samples, args.batch_size, args.random_state)
    run(cfg)
