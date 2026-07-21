#!/usr/bin/env python
"""Load raw dataset CSVs, clean/encode/scale them, and cache the resulting
train/test splits (both shuffled and chronological) under data/processed/.

Usage:
    python scripts/run_preprocessing.py --config configs/default.yaml
    python scripts/run_preprocessing.py --dataset ciciot2023 --raw_dir data/raw/ciciot2023/CIC_IOT_Dataset2023_CSV/CSV --output_dir results_ciciot2023_test --max_samples 50000

If allow_synthetic_fallback is false (the default) and no usable CSVs are
found under raw_dir, this exits with a clear error instead of silently
falling back to synthetic data — synthetic runs are for code smoke-testing
only and must be opted into explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.data_loader import load_dataset
from src.preprocessing import preprocess_pipeline
from src.utils import apply_cli_overrides, get_output_dirs, load_config, resolve_path, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw IDS dataset CSVs.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="unsw | cicids | ciciot2023 | auto")
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None, help="Overrides output_dir")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=None)
    return parser.parse_args()


def run(config: dict) -> dict:
    """config must already be fully resolved (loaded from YAML with any CLI
    overrides applied) — see scripts/run_all.py or the __main__ block below.
    """
    seed = config.get("random_state", 42)
    set_seed(seed)
    dirs = get_output_dirs(config)
    logger = setup_logging(dirs["logs"], name="rl_drift_ids")

    raw_dir = config.get("raw_dir", "data/raw")
    allow_synthetic_fallback = config.get("allow_synthetic_fallback", False)
    sampling_cfg = config.get("sampling", {})
    if sampling_cfg:
        logger.info("Sampling config: %s", sampling_cfg)
    try:
        df, dataset_info = load_dataset(
            dataset_name=config.get("dataset", "auto"),
            raw_dir=raw_dir,
            max_rows=config.get("max_samples"),
            allow_synthetic_fallback=allow_synthetic_fallback,
            synthetic_rows=config.get("synthetic_rows", 20000),
            synthetic_features=config.get("synthetic_features", 20),
            seed=seed,
            sampling_config=sampling_cfg,
        )
    except FileNotFoundError as exc:
        logger.error("=" * 78)
        logger.error("DATASET NOT FOUND — refusing to proceed (allow_synthetic_fallback=false).")
        logger.error(str(exc))
        logger.error(
            "Place real UNSW-NB15, CICIDS2017, or CICIoT2023 CSVs under '%s' (see data/README.md), "
            "or set allow_synthetic_fallback: true in your config for CODE TESTING ONLY "
            "(not valid thesis results).", raw_dir,
        )
        logger.error("=" * 78)
        raise SystemExit(1) from None

    logger.info("Loaded dataset '%s' (synthetic_data=%s), shape=%s", dataset_info.name, dataset_info.is_synthetic, df.shape)
    if dataset_info.loaded_files_count is not None:
        logger.info("Loaded files count: %d", dataset_info.loaded_files_count)
    if dataset_info.source_files:
        logger.info("First few loaded files: %s", dataset_info.source_files[:5])
    if dataset_info.label_source:
        logger.info("Detected label logic: %s", dataset_info.label_source)
    if dataset_info.is_synthetic:
        logger.warning(
            "!!! synthetic_data=true for this run. Every output table/report is flagged "
            "accordingly. These results are NOT valid thesis results. !!!"
        )

    prep_cfg = config.get("preprocessing", {})
    result = preprocess_pipeline(
        df,
        id_like_patterns=prep_cfg.get("id_like_patterns", []),
        max_onehot_cardinality=prep_cfg.get("max_onehot_cardinality", 20),
        test_size=config.get("test_size", 0.3),
        scale=prep_cfg.get("scale", True),
        seed=seed,
    )
    logger.info(
        "Preprocessed: n_samples=%d n_features=%d class_distribution=%s",
        result.n_samples, result.n_features, result.class_distribution,
    )
    if result.dropped_columns:
        logger.info("Dropped columns (%d): %s", len(result.dropped_columns), result.dropped_columns)
    if result.category_column:
        logger.info("Multiclass attack-category column preserved for rule-layer use: '%s'", result.category_column)

    # --- results/tables/leakage_check_report.csv ---
    leakage_path = dirs["tables"] / "leakage_check_report.csv"
    result.leakage_report.to_csv(leakage_path, index=False)
    logger.info("Saved %s (all checks passed: %s)", leakage_path, bool(result.leakage_report["passed"].all()))

    processed_dir = resolve_path(config.get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_arrays = dict(
        X_train=result.X_train, X_test=result.X_test, y_train=result.y_train, y_test=result.y_test,
        X_train_chrono=result.X_train_chrono, X_test_chrono=result.X_test_chrono,
        y_train_chrono=result.y_train_chrono, y_test_chrono=result.y_test_chrono,
    )
    if result.category_train_chrono is not None:
        split_arrays["category_train_chrono"] = result.category_train_chrono
        split_arrays["category_test_chrono"] = result.category_test_chrono
    np.savez_compressed(processed_dir / "split.npz", **split_arrays)

    if result.scaler is not None:
        joblib.dump(result.scaler, processed_dir / "scaler.joblib")
    if result.scaler_chrono is not None:
        joblib.dump(result.scaler_chrono, processed_dir / "scaler_chrono.joblib")

    metadata = {
        "dataset_name": dataset_info.name,
        "is_synthetic": dataset_info.is_synthetic,
        "source_files": dataset_info.source_files,
        "label_column": result.label_column,
        "category_column": result.category_column,
        "feature_names": result.feature_names,
        "n_samples": result.n_samples,
        "n_features": result.n_features,
        "class_distribution": result.class_distribution,
        "class_distribution_before": result.class_distribution_before,
        "dropped_columns": result.dropped_columns,
        "seed": seed,
    }
    with open(processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # --- results/tables/dataset_metadata.csv — one row, required schema ---
    attack_category_distribution = dataset_info.attack_category_distribution
    if attack_category_distribution is None and result.category_column is not None:
        cats = np.concatenate([result.category_train, result.category_test])
        values, counts = np.unique(cats, return_counts=True)
        attack_category_distribution = {str(v): int(c) for v, c in zip(values, counts)}

    dataset_metadata_row = {
        "dataset_name": dataset_info.name,
        "source_folder": dataset_info.source_folder or str(raw_dir),
        "loaded_files_count": dataset_info.loaded_files_count
        if dataset_info.loaded_files_count is not None else len(dataset_info.source_files),
        "n_samples": result.n_samples,
        "n_features": result.n_features,
        "detected_label_column": result.label_column,
        "label_source": dataset_info.label_source or "csv_label_column",
        "binary_class_distribution": json.dumps(result.class_distribution),
        "attack_category_distribution": json.dumps(attack_category_distribution or {}),
        "synthetic_data": dataset_info.is_synthetic,
        # Additional useful provenance beyond the required schema:
        "category_column": result.category_column or "",
        "class_distribution_before": json.dumps(result.class_distribution_before),
        "dropped_columns": "; ".join(result.dropped_columns) if result.dropped_columns else "",
        "random_state": seed,
        "test_size": config.get("test_size", 0.3),
        "sampling_mode": (dataset_info.sampling_info or {}).get("mode", ""),
        "sampling_details": json.dumps(dataset_info.sampling_info) if dataset_info.sampling_info else "",
    }
    dataset_metadata_path = dirs["tables"] / "dataset_metadata.csv"
    pd.DataFrame([dataset_metadata_row]).to_csv(dataset_metadata_path, index=False)
    logger.info("Saved %s", dataset_metadata_path)

    logger.info("Saved processed data to %s", processed_dir)
    return metadata


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args.dataset, args.raw_dir, args.output_dir, args.max_samples, args.batch_size, args.random_state)
    run(cfg)
