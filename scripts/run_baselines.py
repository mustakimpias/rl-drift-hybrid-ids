#!/usr/bin/env python
"""Train and evaluate static baseline classifiers (RF, Extra Trees, Gradient
Boosting, Logistic Regression, XGBoost/LightGBM if installed) on the
preprocessed shuffled train/test split.

Usage:
    python scripts/run_baselines.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from src.baseline_models import train_and_evaluate_baselines
from src.evaluation import best_model_by_f1, metrics_to_dataframe
from src.utils import (
    add_synthetic_flag, apply_cli_overrides, get_output_dirs, load_config, load_processed_data,
    resolve_config, resolve_path, set_seed, setup_logging,
)
from src.visualization import plot_bar_comparison, plot_confusion_matrix_fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run static baseline ML experiments.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=None)
    return parser.parse_args()


def run(config_path_or_dict):
    """Accepts either a config dict (already resolved, e.g. by run_all.py so
    CLI overrides apply consistently across every stage) or a path to a
    YAML file (for standalone use).
    """
    config = resolve_config(config_path_or_dict)
    seed = config.get("random_state", 42)
    set_seed(seed)
    dirs = get_output_dirs(config)
    logger = setup_logging(dirs["logs"], name="rl_drift_ids")

    arrays, metadata = load_processed_data(config)
    logger.info("Loaded processed data: %s", {k: v.shape for k, v in arrays.items() if hasattr(v, "shape")})

    bm_cfg = config.get("baseline_models", {})
    rows, fitted_models = train_and_evaluate_baselines(
        arrays["X_train"], arrays["y_train"], arrays["X_test"], arrays["y_test"],
        random_state=seed,
        n_estimators=bm_cfg.get("n_estimators", 200),
        use_xgboost=bm_cfg.get("use_xgboost", True),
        use_lightgbm=bm_cfg.get("use_lightgbm", True),
    )
    df = metrics_to_dataframe(rows)
    df = add_synthetic_flag(df, metadata.get("is_synthetic", False))
    table_path = dirs["tables"] / "baseline_metrics.csv"
    df.to_csv(table_path, index=False)
    logger.info("Saved %s", table_path)

    plot_bar_comparison(df, "model", "f1_score", "Baseline Model F1 Comparison", "F1 Score",
                         dirs["figures"] / "baseline_f1_comparison.png")
    plot_bar_comparison(df, "model", "false_positive_rate", "Baseline Model False Positive Rate", "FPR",
                         dirs["figures"] / "baseline_fpr_comparison.png")

    best_name = best_model_by_f1(df, "model")
    if best_name is not None:
        best_model = fitted_models[best_name]
        y_pred = best_model.predict(arrays["X_test"])
        plot_confusion_matrix_fig(arrays["y_test"], y_pred, best_name, dirs["figures"] / "confusion_matrix_best_model.png")
        logger.info("Best baseline model by F1: %s", best_name)

    model_dir = resolve_path(config.get("processed_dir", "data/processed"))
    joblib.dump(fitted_models, model_dir / "baseline_models.joblib")

    return df, fitted_models, best_name


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args.dataset, args.raw_dir, args.output_dir, args.max_samples, args.batch_size, args.random_state)
    run(cfg)
