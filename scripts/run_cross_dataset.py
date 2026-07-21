#!/usr/bin/env python
"""Cross-dataset generalization check: train the pipeline's core IDS arms on
one NetFlow-v2-format dataset and evaluate them on a *different* dataset,
using only the feature columns the two datasets have in common.

This is deliberately independent of the CICIoT2023 pipeline (run_all.py /
run_rl_hybrid_experiment.py) — it writes to its own --output_dir and never
touches results_ciciot2023_balanced_test/. It targets the NetFlow v2 CSV
exports (e.g. NF-ToN-IoT-v2.csv, NF-CSE-CIC-IDS2018-v2.csv from the UQ NIDS
datasets project), which share a common ~42-column NetFlow schema, so a
model's features transfer directly between them without per-dataset
feature engineering.

Five arms are run, reusing the exact same arm implementations as the main
hybrid IDS pipeline (src/hybrid_ids.py) — nothing about the RL controller,
drift detector, or rule layer is reimplemented here:
    1. Static ML only            (fit on train only, evaluated on test)
    2. Adaptive ML only          (pure online learning over the test stream)
    3. Drift-aware adaptive ML   (online + reset-on-drift over the test stream)
    4. Drift-triggered active learning (rule layer trained on train + budgeted
       uncertainty sampling over the test stream)
    5. RL-guided hybrid IDS      (rule layer trained on train + tabular RL
       controller deciding query/update/threshold actions over the test stream)

The rule layer (arms 4 and 5) is fit on the train dataset only, in
feature-only mode (no attack-category awareness) — the two datasets do not
share a common attack-category taxonomy, so a category-aware rule layer
fit on one would be meaningless applied to the other.

Usage:
    python scripts/run_cross_dataset.py \
        --train_dataset nf_ton_iot_v2 --train_raw_dir data/raw/nf_ton_iot_v2 \
        --test_dataset nf_cse_cic_ids2018_v2 --test_raw_dir data/raw/nf_cse_cic_ids2018_v2 \
        --output_dir results_cross_dataset_smoke \
        --max_train_samples 5000 --max_test_samples 5000 --random_state 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.evaluation import compute_composite_score
from src.hybrid_ids import (
    RuleBasedLayer,
    run_adaptive_ml_only,
    run_drift_aware_adaptive_ml,
    run_drift_triggered_al_hybrid,
    run_rl_guided_hybrid,
    run_static_ml_only,
)
from src.preprocessing import handle_missing_inf, remove_duplicate_value_columns, remove_id_like_columns
from src.utils import get_output_dirs, load_config, set_seed, setup_logging
from src.visualization import plot_bar_comparison

logger = logging.getLogger("rl_drift_ids")

# --------------------------------------------------------------------------
# Label / category / leakage-column detection (task spec, section 4/5/6/7)
# --------------------------------------------------------------------------

# Priority-ordered label-column candidates (task requirement 4). Checked as
# exact column-name matches, in this order, so a dataset carrying both
# "Label" and "Attack" always picks the binary "Label" column.
LABEL_COLUMN_CANDIDATES = ["Label", "label", "Attack", "attack", "class", "Class", "binary_label", "binary_class"]

# Extra candidates considered for the *attack-category* column once the
# binary label column has already been chosen (task requirement 6: preserve
# category if available, never use it as a feature).
CATEGORY_COLUMN_CANDIDATES = ["Attack", "attack", "class", "Class", "category", "Category", "attack_category"]

BENIGN_TOKENS = {"benign", "normal", "0"}

# Literal leakage-column list from the task spec (requirement 7).
LEAKAGE_LITERAL_PATTERNS = [
    "Label", "label", "Attack", "attack", "class", "Class", "category", "Category", "attack_category",
    "flow_id", "src_ip", "dst_ip", "source_ip", "destination_ip", "timestamp", "time", "date",
]
# NetFlow v2's actual column names for the same source/destination IP and
# port concept the literal list above covers generically (e.g. "src_ip") —
# NF-ToN-IoT-v2.csv / NF-CSE-CIC-IDS2018-v2.csv use IPV4_SRC_ADDR /
# L4_SRC_PORT naming, not "src_ip". Left as raw strings, these columns are
# also not usable as numeric features at all (dotted IP strings), so
# dropping them is both a leakage-hygiene step and a correctness necessity.
NETFLOW_ID_ALIAS_PATTERNS = [
    "ipv4_src_addr", "ipv4_dst_addr", "ipv6_src_addr", "ipv6_dst_addr", "l4_src_port", "l4_dst_port",
]

CHUNK_SIZE = 200_000
# Safety cap on how many chunks a single file scan will read while trying to
# fill its per-class sample caps, so a pathologically rare minority class
# cannot make the loader scan an entire multi-GB file. Logged clearly if hit.
MAX_CHUNKS_SCANNED = 60


def _detect_label_and_category_columns(columns: list[str]) -> tuple[str, str | None]:
    """Pick the label column (first exact match in LABEL_COLUMN_CANDIDATES
    order) and, separately, an attack-category column if a second, distinct
    candidate is also present.
    """
    label_col = None
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError(
            f"Could not detect a label column among {columns}. "
            f"Expected one of (case-sensitive): {LABEL_COLUMN_CANDIDATES}"
        )
    category_col = None
    for candidate in CATEGORY_COLUMN_CANDIDATES:
        if candidate in columns and candidate != label_col:
            category_col = candidate
            break
    return label_col, category_col


def _map_binary_label(series: pd.Series) -> pd.Series:
    """Benign/benign/Normal/normal/0 -> 0; Attack/attack/malicious/1/any
    other (non-benign) category value -> 1 (task requirement 5).
    """
    if pd.api.types.is_numeric_dtype(series):
        vals = set(pd.unique(series.dropna()))
        if vals <= {0, 1, 0.0, 1.0}:
            return series.astype(int)
    tokens = series.astype(str).str.strip().str.lower()
    return tokens.apply(lambda v: 0 if v in BENIGN_TOKENS else 1).astype(int)


def _find_data_csv_files(raw_dir: Path) -> list[Path]:
    """All CSVs in raw_dir except metadata files such as
    NetFlow_v2_Features.csv (task requirement 3).
    """
    all_csvs = sorted(raw_dir.glob("*.csv"))
    data_csvs = [p for p in all_csvs if "feature" not in p.name.lower()]
    skipped = [p for p in all_csvs if p not in data_csvs]
    if skipped:
        logger.info("Ignoring metadata file(s) in %s: %s", raw_dir, [p.name for p in skipped])
    return data_csvs


def _load_netflow_dataset(
    raw_dir: Path, dataset_name: str, role: str, max_samples: int, seed: int,
) -> dict[str, Any]:
    """Chunked, class-stratified load of one or more NetFlow-v2-format CSVs
    under raw_dir, capped at max_samples rows total. Real data only — no
    synthetic fallback (task requirement 17).

    Reads in chunks and caps each class at max_samples//2 rows collected,
    rather than a plain `nrows=max_samples` head read: a raw head read risks
    landing entirely inside one attack-scenario block (these NetFlow exports
    are not randomly shuffled — see the sample rows logged below), the same
    category-block-ordering pitfall this project hit earlier with CICIoT2023.
    """
    data_files = _find_data_csv_files(raw_dir)
    if not data_files:
        raise RuntimeError(
            f"No usable CSV files found under {raw_dir} (metadata-only files are ignored). "
            "Refusing to fall back to synthetic data — provide a real NetFlow v2 CSV export."
        )

    rng = np.random.default_rng(seed)
    cap_per_class = max(1, max_samples // 2)
    collected: dict[int, list[pd.DataFrame]] = {0: [], 1: []}
    counts: dict[int, int] = {0: 0, 1: 0}
    label_col: str | None = None
    category_col: str | None = None
    n_raw_columns = 0
    chunks_scanned = 0
    files_used: list[str] = []

    for csv_path in data_files:
        if counts[0] >= cap_per_class and counts[1] >= cap_per_class:
            break
        files_used.append(csv_path.name)
        with pd.read_csv(csv_path, chunksize=CHUNK_SIZE) as reader:
            for chunk in reader:
                chunks_scanned += 1
                if label_col is None:
                    label_col, category_col = _detect_label_and_category_columns(list(chunk.columns))
                    n_raw_columns = len(chunk.columns)
                    logger.info(
                        "%s (%s): detected label column '%s', category column %s, %d raw columns",
                        dataset_name, role, label_col, repr(category_col), n_raw_columns,
                    )
                y_chunk = _map_binary_label(chunk[label_col])
                for cls in (0, 1):
                    need = cap_per_class - counts[cls]
                    if need <= 0:
                        continue
                    idx = chunk.index[y_chunk == cls]
                    if len(idx) > need:
                        idx = rng.choice(idx.to_numpy(), size=need, replace=False)
                    if len(idx) > 0:
                        collected[cls].append(chunk.loc[idx])
                        counts[cls] += len(idx)
                if (counts[0] >= cap_per_class and counts[1] >= cap_per_class) or chunks_scanned >= MAX_CHUNKS_SCANNED:
                    break
        if chunks_scanned >= MAX_CHUNKS_SCANNED and not (counts[0] >= cap_per_class and counts[1] >= cap_per_class):
            logger.warning(
                "%s (%s): hit the %d-chunk scan safety cap before filling class caps "
                "(collected benign=%d, attack=%d of %d requested each). Using what was found.",
                dataset_name, role, MAX_CHUNKS_SCANNED, counts[0], counts[1], cap_per_class,
            )

    if label_col is None:
        raise RuntimeError(f"No rows could be read from {raw_dir} for dataset '{dataset_name}'.")

    frames = collected[0] + collected[1]
    if not frames:
        raise RuntimeError(f"No labeled rows found for dataset '{dataset_name}' under {raw_dir}.")
    df = pd.concat(frames, ignore_index=True)
    # Shuffle: rows were appended class-0-block-then-class-1-block above,
    # which would otherwise hand every downstream "stream" arm a single
    # sharp mid-stream class-distribution shift — shuffling avoids that.
    perm = rng.permutation(len(df))
    df = df.iloc[perm].reset_index(drop=True)
    if len(df) > max_samples:
        keep = rng.choice(len(df), size=max_samples, replace=False)
        df = df.iloc[np.sort(keep)].reset_index(drop=True)

    y = _map_binary_label(df[label_col]).to_numpy()
    category = df[category_col].astype(str).to_numpy() if category_col is not None else None

    feature_df = df.drop(columns=[c for c in [label_col, category_col] if c is not None])
    all_patterns = LEAKAGE_LITERAL_PATTERNS + NETFLOW_ID_ALIAS_PATTERNS
    feature_df, dropped_leakage = remove_id_like_columns(feature_df, all_patterns)
    feature_df = handle_missing_inf(feature_df)
    non_numeric = [c for c in feature_df.columns if not pd.api.types.is_numeric_dtype(feature_df[c])]
    if non_numeric:
        logger.info("%s (%s): dropping %d remaining non-numeric column(s): %s", dataset_name, role, len(non_numeric), non_numeric)
        feature_df = feature_df.drop(columns=non_numeric)
    feature_df, dropped_dup_value = remove_duplicate_value_columns(feature_df)
    feature_df = feature_df.astype(np.float64)

    class_distribution = {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()}
    logger.info(
        "%s (%s): loaded %d rows from %s (%s), class_distribution=%s, %d usable feature columns",
        dataset_name, role, len(df), raw_dir, files_used, class_distribution, feature_df.shape[1],
    )

    return {
        "dataset_name": dataset_name,
        "role": role,
        "raw_dir": str(raw_dir),
        "source_files": files_used,
        "feature_df": feature_df,
        "y": y,
        "category": category,
        "label_column": label_col,
        "category_column": category_col,
        "n_samples": len(df),
        "n_raw_columns": n_raw_columns,
        "class_distribution": class_distribution,
        "dropped_columns": sorted(set(dropped_leakage) | set(dropped_dup_value) | set(non_numeric)),
        "max_samples_requested": max_samples,
    }


# --------------------------------------------------------------------------
# Feature-matrix sanitization — the final, authoritative numeric-safety pass
# before any model sees the data. NetFlow v2 exports carry several
# near-constant columns (DNS/FTP/ICMP-specific fields that are 0 for all but
# a handful of rows) — a tiny train-set standard deviation on one of these
# can turn an ordinary-looking raw value into an astronomically large
# (but still float64-finite) z-score, which then overflows to inf the
# moment sklearn's RandomForest internally casts X down to float32 for
# prediction. Percentile clipping (learned from train only) bounds the raw
# values before scaling ever sees them; the final float32-range clip is a
# second, independent safety net that catches any residual blow-up
# regardless of its cause.
# --------------------------------------------------------------------------

EXTREME_VALUE_THRESHOLD = 1e6
FINAL_FLOAT32_CLIP = 1e6


def sanitize_feature_matrix(
    X_train: np.ndarray,
    X_test: np.ndarray,
    output_dir: str | Path,
    feature_names: list[str],
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Make X_train/X_test finite and numerically safe for every downstream
    model (tree-based and river online models alike), fitting every
    statistic (medians, clip bounds, scaler) on X_train only.

    Returns (X_train_safe, X_test_safe) as float32 arrays, guaranteed
    finite and bounded to [-FINAL_FLOAT32_CLIP, FINAL_FLOAT32_CLIP]. Also
    writes a per-feature before/after report to
    output_dir/tables/cross_dataset_feature_sanity_report.csv.
    """
    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)

    # --- 1. Coerce every column to numeric; anything unparseable becomes NaN. ---
    for col in feature_names:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    # --- 2. Stats BEFORE any cleaning (NaN count already reflects coercion
    # failures; inf count is measured separately since NaN != inf). ---
    nan_before_train = train_df.isna().sum()
    nan_before_test = test_df.isna().sum()
    inf_before_train = train_df.apply(lambda s: np.isinf(s.to_numpy(dtype=np.float64)).sum())
    inf_before_test = test_df.apply(lambda s: np.isinf(s.to_numpy(dtype=np.float64)).sum())
    train_abs_max_before = train_df.abs().max()
    extreme_cols = sorted(train_abs_max_before[train_abs_max_before > EXTREME_VALUE_THRESHOLD].index.tolist())

    total_nan_before = int(nan_before_train.sum() + nan_before_test.sum())
    total_inf_before = int(inf_before_train.sum() + inf_before_test.sum())
    logger.info(
        "Feature sanitization: %d NaN value(s) and %d inf value(s) found before cleaning "
        "(train+test combined, post-numeric-coercion).",
        total_nan_before, total_inf_before,
    )
    if extreme_cols:
        logger.info(
            "Feature sanitization: %d column(s) with |value| > %.0e in train (pre-cleaning): %s",
            len(extreme_cols), EXTREME_VALUE_THRESHOLD, extreme_cols,
        )
    else:
        logger.info("Feature sanitization: no columns exceeded the |value| > %.0e extreme-value threshold.", EXTREME_VALUE_THRESHOLD)

    # --- 3. inf -> NaN, then median-impute using TRAIN medians only. ---
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)

    train_medians = train_df.median(numeric_only=True)
    train_medians = train_medians.fillna(0.0)  # column entirely NaN in train -> 0
    train_df = train_df.fillna(train_medians)
    test_df = test_df.fillna(train_medians)

    # --- 4. Percentile clipping, bounds learned from train only. ---
    lower_bounds = train_df.quantile(lower_percentile / 100.0)
    upper_bounds = train_df.quantile(upper_percentile / 100.0)
    train_df = train_df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    test_df = test_df.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    # Defensive: clipping itself cannot introduce inf/NaN, but guard anyway
    # in case a bound was itself NaN (e.g. a column that was constant 0 in
    # train, giving lower_bound == upper_bound == 0 — fine) or otherwise
    # degenerate.
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- 5. Scale: fit on train only, transform both. ---
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_df.to_numpy(dtype=np.float64))
    X_test_scaled = scaler.transform(test_df.to_numpy(dtype=np.float64))

    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=FINAL_FLOAT32_CLIP, neginf=-FINAL_FLOAT32_CLIP)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=FINAL_FLOAT32_CLIP, neginf=-FINAL_FLOAT32_CLIP)
    X_train_scaled = np.clip(X_train_scaled, -FINAL_FLOAT32_CLIP, FINAL_FLOAT32_CLIP).astype(np.float32)
    X_test_scaled = np.clip(X_test_scaled, -FINAL_FLOAT32_CLIP, FINAL_FLOAT32_CLIP).astype(np.float32)

    # --- 6. Validate. ---
    assert np.isfinite(X_train_scaled).all(), "X_train contains non-finite values after sanitization"
    assert np.isfinite(X_test_scaled).all(), "X_test contains non-finite values after sanitization"
    assert float(np.max(np.abs(X_train_scaled))) <= FINAL_FLOAT32_CLIP, "X_train exceeds the safe float32 clip range"
    assert float(np.max(np.abs(X_test_scaled))) <= FINAL_FLOAT32_CLIP, "X_test exceeds the safe float32 clip range"

    # --- 7. Report + final logging. ---
    report_rows = []
    for col in feature_names:
        report_rows.append({
            "feature_name": col,
            "nan_count_train_before": int(nan_before_train[col]),
            "nan_count_test_before": int(nan_before_test[col]),
            "inf_count_train_before": int(inf_before_train[col]),
            "inf_count_test_before": int(inf_before_test[col]),
            "train_abs_max_before": float(train_abs_max_before[col]),
            "is_extreme_train": col in extreme_cols,
            "train_median_used": float(train_medians[col]),
            "clip_lower": float(lower_bounds[col]),
            "clip_upper": float(upper_bounds[col]),
        })
    report_df = pd.DataFrame(report_rows)
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_path = tables_dir / "cross_dataset_feature_sanity_report.csv"
    report_df.to_csv(report_path, index=False)
    logger.info("Saved %s", report_path)

    logger.info(
        "Feature sanitization complete: X_train shape=%s, X_test shape=%s — all values finite, "
        "max |value|=%.4f (train) / %.4f (test), within [-%.0e, %.0e].",
        X_train_scaled.shape, X_test_scaled.shape,
        float(np.max(np.abs(X_train_scaled))), float(np.max(np.abs(X_test_scaled))),
        FINAL_FLOAT32_CLIP, FINAL_FLOAT32_CLIP,
    )
    return X_train_scaled, X_test_scaled


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-dataset generalization check for the hybrid IDS pipeline.")
    parser.add_argument("--train_dataset", type=str, required=True, help="Label for the training dataset (e.g. nf_ton_iot_v2)")
    parser.add_argument("--train_raw_dir", type=str, required=True, help="Directory containing the training dataset's CSV(s)")
    parser.add_argument("--test_dataset", type=str, required=True, help="Label for the test dataset (e.g. nf_cse_cic_ids2018_v2)")
    parser.add_argument("--test_raw_dir", type=str, required=True, help="Directory containing the test dataset's CSV(s)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_test_samples", type=int, default=5000)
    parser.add_argument("--random_state", type=int, default=42)
    # Optional extras (not part of the required CLI contract, but needed to
    # parameterize the reused arms; default to this thesis's already-tuned
    # main-pipeline settings, loaded from --config when available).
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--label_budget_fraction", type=float, default=None)
    parser.add_argument("--rule_confidence_threshold", type=float, default=None)
    return parser.parse_args()


def _load_arm_settings(config_path: str, overrides: argparse.Namespace) -> dict[str, Any]:
    """Pull RL / drift-detector / rule-layer settings from the main
    project's config for consistency with the rest of the thesis, falling
    back to the same hardcoded defaults configs/default.yaml already ships
    with if the file is missing (never fabricated — these are the actual
    composite_score-tuned values already established elsewhere in this
    project, just given a code-level fallback so this script does not hard-
    depend on the config file existing).
    """
    defaults = {
        "rl": {
            "epsilon": 0.1, "alpha": 0.1, "gamma": 0.9,
            "false_positive_penalty": -2, "false_negative_penalty": -6,
            "correct_attack_reward": 5, "correct_benign_reward": 1,
            "label_query_cost": -0.5, "model_update_cost": -0.1,
        },
        "drift_detector": "adwin",
        "drift_detection": {"adwin_delta": 0.002, "page_hinkley_threshold": 50, "page_hinkley_min_instances": 30},
        "active_learning": {"uncertainty_margin_threshold": 0.15, "warmup_fraction": 0.01, "warmup_min": 100},
        "hybrid_ids": {"rule_layer": {"confidence_threshold": 0.97}},
        "unseen_attack_experiment": {"budget_fraction": 0.1},
    }
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.warning("Config '%s' not found; using this script's built-in defaults.", config_path)
        config = {}

    settings = {
        "rl_config": {**defaults["rl"], **config.get("rl", {})},
        "detector_type": config.get("drift_detector", defaults["drift_detector"]),
        "drift_detection": {**defaults["drift_detection"], **config.get("drift_detection", {})},
        "warmup_fraction": config.get("active_learning", {}).get("warmup_fraction", defaults["active_learning"]["warmup_fraction"]),
        "warmup_min": config.get("active_learning", {}).get("warmup_min", defaults["active_learning"]["warmup_min"]),
        "uncertainty_margin_threshold": config.get("active_learning", {}).get(
            "uncertainty_margin_threshold", defaults["active_learning"]["uncertainty_margin_threshold"],
        ),
        "label_budget_fraction": config.get("unseen_attack_experiment", {}).get("budget_fraction", defaults["unseen_attack_experiment"]["budget_fraction"]),
        "rule_confidence_threshold": config.get("hybrid_ids", {}).get("rule_layer", {}).get("confidence_threshold", defaults["hybrid_ids"]["rule_layer"]["confidence_threshold"]),
    }
    if overrides.label_budget_fraction is not None:
        settings["label_budget_fraction"] = overrides.label_budget_fraction
    if overrides.rule_confidence_threshold is not None:
        settings["rule_confidence_threshold"] = overrides.rule_confidence_threshold
    return settings


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dirs = get_output_dirs({"output_dir": args.output_dir})
    setup_logging(dirs["logs"], name="rl_drift_ids")
    set_seed(args.random_state)

    logger.info("=== Cross-dataset generalization check ===")
    logger.info("Train dataset: %s (raw_dir=%s, max_train_samples=%d)", args.train_dataset, args.train_raw_dir, args.max_train_samples)
    logger.info("Test dataset:  %s (raw_dir=%s, max_test_samples=%d)", args.test_dataset, args.test_raw_dir, args.max_test_samples)
    logger.info("random_state=%d, output_dir=%s", args.random_state, args.output_dir)

    train_raw = _load_netflow_dataset(
        Path(args.train_raw_dir), args.train_dataset, "train", args.max_train_samples, args.random_state,
    )
    test_raw = _load_netflow_dataset(
        Path(args.test_raw_dir), args.test_dataset, "test", args.max_test_samples, args.random_state,
    )

    common_features = sorted(set(train_raw["feature_df"].columns) & set(test_raw["feature_df"].columns))
    if not common_features:
        raise RuntimeError(
            f"No common feature columns between '{args.train_dataset}' and '{args.test_dataset}' "
            "after leakage-column removal — cannot run a cross-dataset comparison."
        )
    logger.info(
        "Common feature columns: %d (train had %d, test had %d before intersection)",
        len(common_features), train_raw["feature_df"].shape[1], test_raw["feature_df"].shape[1],
    )

    X_train_raw = train_raw["feature_df"][common_features].to_numpy(dtype=np.float64)
    X_test_raw = test_raw["feature_df"][common_features].to_numpy(dtype=np.float64)
    y_train, y_test = train_raw["y"], test_raw["y"]

    # Final numeric-safety pass: coerce/clean/clip/scale, fitting every
    # statistic (medians, clip bounds, scaler) on X_train only (task
    # requirement 9) — X_test only ever gets transformed with train-learned
    # parameters, never fit.
    X_train, X_test = sanitize_feature_matrix(X_train_raw, X_test_raw, dirs["results"], common_features)

    settings = _load_arm_settings(args.config, args)
    seed = args.random_state
    rule_layer = RuleBasedLayer(
        confidence_threshold=settings["rule_confidence_threshold"], random_state=seed,
    )
    # feature-only mode: no category_train — train/test attack-category
    # taxonomies differ between datasets, so category-aware rule matching
    # would not transfer (task requirement 6: category is preserved as
    # metadata, never used as a model feature or a cross-dataset key).
    rule_layer.fit(X_train, y_train)
    logger.info("Rule layer (feature-only, fit on train only): %s", rule_layer.describe())

    drift_kwargs = dict(
        detector_type=settings["detector_type"],
        adwin_delta=settings["drift_detection"]["adwin_delta"],
        page_hinkley_threshold=settings["drift_detection"]["page_hinkley_threshold"],
        page_hinkley_min_instances=settings["drift_detection"]["page_hinkley_min_instances"],
    )
    budget = settings["label_budget_fraction"]
    logger.info("RL config=%s, drift=%s, label_budget_fraction=%.4f", settings["rl_config"], drift_kwargs, budget)

    logger.info("Running arm: Static ML only")
    static_res = run_static_ml_only(X_train, y_train, X_test, y_test, seed=seed)

    logger.info("Running arm: Adaptive ML only")
    adaptive_res = run_adaptive_ml_only(X_test, y_test, common_features, **drift_kwargs)

    logger.info("Running arm: Drift-aware adaptive ML")
    drift_aware_res = run_drift_aware_adaptive_ml(X_test, y_test, common_features, **drift_kwargs)

    logger.info("Running arm: Drift-triggered active learning")
    drift_al_res = run_drift_triggered_al_hybrid(
        rule_layer, X_test, y_test, common_features,
        label_budget_fraction=budget,
        uncertainty_margin_threshold=settings["uncertainty_margin_threshold"],
        warmup_fraction=settings["warmup_fraction"], warmup_min=settings["warmup_min"],
        seed=seed, **drift_kwargs,
    )

    logger.info("Running arm: RL-guided hybrid IDS")
    rl_res, rl_controller = run_rl_guided_hybrid(
        rule_layer, X_test, y_test, common_features,
        rl_config=settings["rl_config"],
        label_budget_fraction=budget,
        warmup_fraction=settings["warmup_fraction"], warmup_min=settings["warmup_min"],
        seed=seed, **drift_kwargs,
    )

    arms = [
        ("Static ML only", "baseline", static_res, None),
        ("Adaptive ML only", "full_feedback_upper_bound", adaptive_res, None),
        ("Drift-aware adaptive ML", "full_feedback_upper_bound", drift_aware_res, None),
        ("Drift-triggered active learning", "active_learning_baseline", drift_al_res, None),
        ("RL-guided hybrid IDS", "proposed", rl_res, rl_controller),
    ]

    metric_rows: list[dict[str, Any]] = []
    for method_name, method_role, res, controller in arms:
        row = dict(res.metrics)
        row["method"] = method_name
        row["method_role"] = method_role
        row["train_dataset"] = args.train_dataset
        row["test_dataset"] = args.test_dataset
        row["label_query_percentage"] = res.label_query_percentage
        row["label_saving_vs_full_feedback"] = 100.0 - res.label_query_percentage
        row["drift_count"] = res.drift_count
        if controller is not None:
            summary = controller.summary()
            row["rl_steps"] = summary["n_steps"]
            row["total_reward"] = summary["total_reward"]
        else:
            row["rl_steps"] = None
            row["total_reward"] = None
        row["composite_score"] = compute_composite_score(
            row["macro_f1"], row["false_positive_rate"], row["label_query_percentage"],
        )
        metric_rows.append(row)
        logger.info(
            "%s: f1=%.4f macro_f1=%.4f fpr=%.4f label_query_pct=%.4f composite_score=%.4f",
            method_name, row["f1_score"], row["macro_f1"], row["false_positive_rate"],
            row["label_query_percentage"], row["composite_score"],
        )

    lead_cols = [
        "method", "method_role", "train_dataset", "test_dataset",
        "accuracy", "precision", "recall", "detection_rate", "f1_score", "macro_f1", "weighted_f1",
        "balanced_accuracy", "specificity", "false_positive_rate", "false_negative_rate", "mcc",
        "label_query_percentage", "label_saving_vs_full_feedback", "drift_count", "rl_steps",
        "total_reward", "composite_score",
    ]
    metrics_df = pd.DataFrame(metric_rows)
    remaining_cols = [c for c in metrics_df.columns if c not in lead_cols]
    metrics_df = metrics_df[lead_cols + remaining_cols]

    metrics_path = dirs["tables"] / "cross_dataset_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved %s", metrics_path)

    metadata_rows = []
    for raw in (train_raw, test_raw):
        metadata_rows.append({
            "role": raw["role"],
            "dataset_name": raw["dataset_name"],
            "raw_dir": raw["raw_dir"],
            "source_files": "; ".join(raw["source_files"]),
            "n_samples_loaded": raw["n_samples"],
            "max_samples_requested": raw["max_samples_requested"],
            "n_raw_columns": raw["n_raw_columns"],
            "detected_label_column": raw["label_column"],
            "detected_category_column": raw["category_column"],
            "class_distribution": raw["class_distribution"],
            "dropped_leakage_columns": "; ".join(raw["dropped_columns"]),
            "random_state": args.random_state,
        })
    metadata_path = dirs["tables"] / "cross_dataset_dataset_metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    logger.info("Saved %s", metadata_path)

    common_features_path = dirs["tables"] / "common_features.csv"
    pd.DataFrame({"feature_name": common_features}).to_csv(common_features_path, index=False)
    logger.info("Saved %s", common_features_path)

    rl_summary = rl_controller.summary()
    rl_summary_row = {
        "train_dataset": args.train_dataset, "test_dataset": args.test_dataset,
        "label_budget_fraction": budget,
        "total_reward": rl_summary["total_reward"], "mean_reward": rl_summary["mean_reward"],
        "n_steps": rl_summary["n_steps"], "final_epsilon": rl_summary["final_epsilon"],
        "n_states_visited": rl_summary["n_states_visited"],
        "query_count": rl_summary["query_count"], "update_count": rl_summary["update_count"],
        "true_positive_count": rl_summary["true_positive_count"], "true_negative_count": rl_summary["true_negative_count"],
        "false_positive_penalty_count": rl_summary["false_positive_penalty_count"],
        "false_negative_penalty_count": rl_summary["false_negative_penalty_count"],
        "action_no_query": rl_summary["action_counts"]["no_query"],
        "action_query_and_update": rl_summary["action_counts"]["query_and_update"],
        "action_update_if_drift": rl_summary["action_counts"]["update_if_drift"],
        "action_adjust_threshold": rl_summary["action_counts"]["adjust_threshold"],
    }
    rl_summary_path = dirs["tables"] / "cross_dataset_rl_summary.csv"
    pd.DataFrame([rl_summary_row]).to_csv(rl_summary_path, index=False)
    logger.info("Saved %s", rl_summary_path)

    plot_bar_comparison(
        metrics_df, "method", "f1_score",
        f"Cross-Dataset F1: train={args.train_dataset} -> test={args.test_dataset}", "F1 Score",
        dirs["figures"] / "cross_dataset_f1_comparison.png",
    )
    plot_bar_comparison(
        metrics_df, "method", "false_positive_rate",
        f"Cross-Dataset FPR: train={args.train_dataset} -> test={args.test_dataset}", "False Positive Rate",
        dirs["figures"] / "cross_dataset_fpr_comparison.png",
    )
    plot_bar_comparison(
        metrics_df, "method", "composite_score",
        f"Cross-Dataset Composite Score: train={args.train_dataset} -> test={args.test_dataset}", "Composite Score",
        dirs["figures"] / "cross_dataset_composite_score.png",
    )

    _write_summary_md(dirs["results"] / "cross_dataset_summary.md", args, train_raw, test_raw, common_features, metrics_df)

    logger.info(
        "Cross-dataset check complete. Output files: %s, %s, %s, %s, 3 figures under %s, and %s",
        metrics_path, metadata_path, common_features_path, rl_summary_path, dirs["figures"], dirs["results"] / "cross_dataset_summary.md",
    )


def _write_summary_md(
    output_path: Path, args: argparse.Namespace, train_raw: dict, test_raw: dict,
    common_features: list[str], metrics_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Cross-Dataset Generalization Check")
    lines.append("")
    lines.append(f"- **Train dataset:** {args.train_dataset} (`{train_raw['raw_dir']}`, {train_raw['n_samples']} rows, "
                  f"class_distribution={train_raw['class_distribution']})")
    lines.append(f"- **Test dataset:** {args.test_dataset} (`{test_raw['raw_dir']}`, {test_raw['n_samples']} rows, "
                  f"class_distribution={test_raw['class_distribution']})")
    lines.append(f"- **Common feature columns used:** {len(common_features)}")
    lines.append(f"- **random_state:** {args.random_state}")
    if train_raw["n_samples"] < 20000 or test_raw["n_samples"] < 20000:
        lines.append(
            "- **Note:** this run used a small sample size (see counts above) — treat it as a smoke test of "
            "the cross-dataset pipeline, not a final generalization result. Re-run with larger "
            "`--max_train_samples`/`--max_test_samples` for a thesis-reportable cross-dataset number."
        )
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    cols = ["method", "method_role", "f1_score", "macro_f1", "recall", "specificity",
            "false_positive_rate", "mcc", "label_query_percentage", "composite_score"]
    cols = [c for c in cols if c in metrics_df.columns]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in metrics_df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    if not metrics_df.empty:
        best_idx = metrics_df["composite_score"].astype(float).idxmax()
        best = metrics_df.loc[best_idx]
        lines.append(
            f"Best method by composite_score: **{best['method']}** "
            f"(composite_score={best['composite_score']:.4f}, f1={best['f1_score']:.4f}, "
            f"fpr={best['false_positive_rate']:.4f}, label_query_pct={best['label_query_percentage']:.4f}%)."
        )
        lines.append("")

    static_row = metrics_df[metrics_df["method"] == "Static ML only"]
    rl_row = metrics_df[metrics_df["method"] == "RL-guided hybrid IDS"]
    if not static_row.empty and not rl_row.empty:
        s, r = static_row.iloc[0], rl_row.iloc[0]
        lines.append("## Interpretation")
        lines.append("")
        lines.append(
            f"Static ML only, trained on {args.train_dataset} and evaluated cold on {args.test_dataset} with no "
            f"further adaptation, scores F1={s['f1_score']:.4f}, FPR={s['false_positive_rate']:.4f}. "
            f"RL-guided hybrid IDS (same train/test split, but adapting online over the test stream, "
            f"querying labels for {r['label_query_percentage']:.2f}% of it) scores F1={r['f1_score']:.4f}, "
            f"FPR={r['false_positive_rate']:.4f}."
        )
        f1_delta = r["f1_score"] - s["f1_score"]
        lines.append(
            f"RL-guided hybrid IDS {'improves' if f1_delta > 0 else 'does not improve'} F1 over static ML by "
            f"{abs(f1_delta):.4f} under this train-to-test dataset shift. This is a genuine cross-dataset "
            "generalization number (no rows or features from the test dataset were used to fit any model or "
            "scaler) — it should not be assumed to match the in-distribution CICIoT2023 results reported "
            "elsewhere in this thesis, since a real distribution shift between two independently collected "
            "NetFlow datasets is a substantially harder setting."
        )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The rule layer (used by Drift-triggered active learning and RL-guided hybrid IDS) is fit on the "
        "train dataset only, in feature-only mode (no attack-category awareness), since the two datasets do "
        "not share a common attack-category taxonomy."
    )
    lines.append(
        "- All features are restricted to the common column set between train and test; the scaler is fit on "
        "train data only and applied unchanged to test data."
    )
    lines.append(
        "- No synthetic data is used anywhere in this script; if the raw CSVs cannot be found or read, the "
        "script raises an error instead of falling back to synthetic data."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()
