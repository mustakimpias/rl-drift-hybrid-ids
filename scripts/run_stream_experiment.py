#!/usr/bin/env python
"""Run River online/adaptive classifiers prequentially over the full
chronological stream (train_chrono + test_chrono, in original order), and
run drift detection (ADWIN / PageHinkley) over the resulting error signal.

Usage:
    python scripts/run_stream_experiment.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.drift_detection import DriftDetector
from src.evaluation import metrics_to_dataframe
from src.stream_models import train_and_evaluate_stream_models
from src.utils import (
    add_synthetic_flag, apply_cli_overrides, get_output_dirs, load_config, load_processed_data,
    resolve_config, set_seed, setup_logging,
)
from src.visualization import plot_drift_points, plot_stream_f1_over_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive/online stream ML experiments + drift detection.")
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
    logger.info("Chronological stream length: %d samples", len(X_stream))

    stream_cfg = config.get("stream", {})
    batch_size = config.get("batch_size", 1000)
    rows, run_results, fitted_models = train_and_evaluate_stream_models(
        X_stream, y_stream, feature_names,
        seed=seed,
        window_size=batch_size,
        use_adaptive_random_forest=stream_cfg.get("use_adaptive_random_forest", True),
    )
    df = metrics_to_dataframe(rows)
    df = add_synthetic_flag(df, metadata.get("is_synthetic", False))
    table_path = dirs["tables"] / "stream_metrics.csv"
    df.to_csv(table_path, index=False)
    logger.info("Saved %s", table_path)

    plot_stream_f1_over_time(run_results, batch_size, dirs["figures"] / "stream_f1_over_time.png")

    # --- Drift detection over the primary model's prediction-error signal ---
    primary_model_name = "hoeffding_tree" if "hoeffding_tree" in run_results else next(iter(run_results))
    primary_result = run_results[primary_model_name]
    error_signal = [int(p != t) for p, t in zip(primary_result.predictions, y_stream)]

    drift_cfg = config.get("drift_detection", {})
    detector = DriftDetector(
        detector_type=config.get("drift_detector", "adwin"),
        adwin_delta=drift_cfg.get("adwin_delta", 0.002),
        page_hinkley_threshold=drift_cfg.get("page_hinkley_threshold", 50),
        page_hinkley_min_instances=drift_cfg.get("page_hinkley_min_instances", 30),
    )
    for e in error_signal:
        detector.update(float(e))

    drift_points = detector.drift_points()
    drift_df = pd.DataFrame(drift_points) if drift_points else pd.DataFrame(columns=["index", "detector"])
    drift_df["source_model"] = primary_model_name
    drift_df["source_experiment"] = "stream_experiment"
    drift_df = add_synthetic_flag(drift_df, metadata.get("is_synthetic", False))
    drift_table_path = dirs["tables"] / "drift_points.csv"
    drift_df.to_csv(drift_table_path, index=False)
    logger.info("Saved %s (%d drift events detected via %s)", drift_table_path, len(drift_points), primary_model_name)

    plot_drift_points(drift_points, len(X_stream), dirs["figures"] / "drift_points_plot.png", error_signal=error_signal)

    return df, drift_df, run_results


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args.dataset, args.raw_dir, args.output_dir, args.max_samples, args.batch_size, args.random_state)
    run(cfg)
