"""Cleaning, label normalization, encoding, and scaling for flow-based IDS
datasets (UNSW-NB15, CICIDS2017, or the synthetic fallback).

Public entry point: preprocess_pipeline().
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("rl_drift_ids")

BENIGN_TOKENS = {"benign", "normal", "0", "0.0", "background"}
LABEL_COLUMN_CANDIDATES = (
    "label", "class", "attack_cat", "attack_category", "outcome", " label",
)
# Substrings that would indicate a label/category/folder-derived column
# accidentally survived into the feature matrix.
LEAKAGE_NAME_PATTERNS = ("label", "class", "category", "attack_cat", "attack_type", "folder", "outcome")


def leakage_check_report(feature_names: list[str], label_col: str, category_col: str | None) -> pd.DataFrame:
    """Sanity-check the final feature set for label/category leakage before
    any model ever sees it. Returns a small pass/fail report (also saved to
    results/tables/leakage_check_report.csv by run_preprocessing.py); raises
    RuntimeError if the label or category column is literally still present
    in the features — that's not something to warn about and continue.
    """
    rows: list[dict[str, Any]] = []

    label_absent = label_col not in feature_names
    rows.append({
        "check": "no_label_column_in_features",
        "passed": label_absent,
        "detail": f"label column: '{label_col}'",
    })

    if category_col is not None:
        category_absent = category_col not in feature_names
        rows.append({
            "check": "no_attack_category_column_in_features",
            "passed": category_absent,
            "detail": f"category column: '{category_col}'",
        })

    suspicious = [f for f in feature_names if any(pat in f.lower() for pat in LEAKAGE_NAME_PATTERNS)]
    rows.append({
        "check": "no_suspicious_label_or_category_like_column_names",
        "passed": len(suspicious) == 0,
        "detail": "; ".join(suspicious) if suspicious else "none found",
    })

    rows.append({
        "check": "feature_count",
        "passed": True,
        "detail": str(len(feature_names)),
    })

    report = pd.DataFrame(rows)
    failed = report[~report["passed"] & report["check"].isin(
        ["no_label_column_in_features", "no_attack_category_column_in_features"]
    )]
    if not failed.empty:
        logger.error("LEAKAGE DETECTED: %s", failed.to_dict("records"))
        raise RuntimeError(f"Label/category leakage detected in features: {failed['detail'].tolist()}")
    if suspicious:
        logger.warning("Leakage check: suspicious (but not confirmed-leaking) column name(s): %s", suspicious)
    else:
        logger.info("Leakage check passed: no label/category column, no suspicious column names, in %d features.", len(feature_names))
    return report


@dataclass
class PreprocessResult:
    # Shuffled, stratified split — for batch/baseline models where row order
    # does not matter.
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler | None
    # Chronological (unshuffled) split — for stream/drift/active-learning/
    # hybrid experiments, where preserving arrival order is required for
    # drift detection and prequential evaluation to be meaningful.
    X_train_chrono: np.ndarray
    X_test_chrono: np.ndarray
    y_train_chrono: np.ndarray
    y_test_chrono: np.ndarray
    scaler_chrono: StandardScaler | None
    feature_names: list[str]
    label_column: str
    n_samples: int
    n_features: int
    class_distribution: dict[str, int]
    class_distribution_before: dict[str, int]
    dropped_columns: list[str]
    # Multiclass attack-category column (e.g. UNSW's attack_cat, CICIDS's
    # named Label), if the dataset carries one distinct from the binary
    # label. None for binary-only datasets (including the synthetic
    # fallback) — used only for rule-layer category-aware reporting, never
    # to make test-time predictions.
    category_column: str | None = None
    category_train: np.ndarray | None = None
    category_test: np.ndarray | None = None
    category_train_chrono: np.ndarray | None = None
    category_test_chrono: np.ndarray | None = None
    leakage_report: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalize column names to lower_snake_case."""
    df = df.copy()
    new_cols = []
    for col in df.columns:
        c = str(col).strip().lower()
        c = re.sub(r"[^\w]+", "_", c)
        c = re.sub(r"_+", "_", c).strip("_")
        new_cols.append(c or "unnamed")
    # De-duplicate any collisions produced by normalization.
    seen: dict[str, int] = {}
    deduped = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            deduped.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            deduped.append(c)
    df.columns = deduped
    return df


def detect_label_column(df: pd.DataFrame) -> str:
    """Find the label/target column by name, preferring exact matches.

    Raises ValueError if no plausible label column exists.
    """
    normalized = {c: c.strip().lower() for c in df.columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        for col, norm in normalized.items():
            if norm == candidate.strip():
                return col
    # Fallback: any column containing "label" or "class" as a substring.
    for col, norm in normalized.items():
        if "label" in norm or norm == "class":
            return col
    raise ValueError(
        f"Could not detect a label column among: {list(df.columns)}. "
        "Expected one of (case-insensitive): "
        f"{LABEL_COLUMN_CANDIDATES}"
    )


def convert_labels_to_binary(series: pd.Series) -> pd.Series:
    """Map benign/normal -> 0, everything else -> 1.

    Handles both string labels ("BENIGN", "DoS Hulk", ...) and numeric
    labels (0/1) transparently.
    """
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) != 0).astype(int)

    def _map(v: Any) -> int:
        token = str(v).strip().lower()
        return 0 if token in BENIGN_TOKENS else 1

    return series.map(_map).astype(int)


def remove_duplicate_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop exact-name duplicate columns (keep first occurrence). This can
    happen when concatenating many per-file CSVs whose headers weren't
    perfectly consistent, or after `clean_column_names` collapses two
    differently-cased source names onto the same normalized one.
    """
    dupe_mask = df.columns.duplicated()
    dropped = list(df.columns[dupe_mask])
    if dropped:
        logger.info("Dropping %d exact-name duplicate column(s): %s", len(dropped), dropped)
    return df.loc[:, ~dupe_mask], dropped


def remove_duplicate_value_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose values are identical to an earlier column (keep
    first occurrence) — a real data-quality issue in some raw flow-feature
    exports where the same measurement ends up recorded under two feature
    names (e.g. a redundant byte-count/size column pair).
    """
    dup_series = df.T.duplicated()
    dropped = list(dup_series[dup_series].index)
    if dropped:
        logger.info("Dropping %d duplicate-value column(s): %s", len(dropped), dropped)
    return df.loc[:, ~dup_series.values], dropped


def remove_id_like_columns(df: pd.DataFrame, patterns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose names match known id/ip/timestamp-like patterns."""
    drop_cols = [
        col for col in df.columns
        if any(pat.lower() in col.lower() for pat in patterns)
    ]
    if drop_cols:
        logger.info("Dropping %d id-like column(s): %s", len(drop_cols), drop_cols)
    return df.drop(columns=drop_cols, errors="ignore"), drop_cols


def handle_missing_inf(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf with NaN, then impute: median for numeric, mode for categorical."""
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                fill_value = 0.0 if pd.isna(fill_value) else fill_value
            else:
                mode = df[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(fill_value)
    return df


def handle_categorical(df: pd.DataFrame, max_onehot_cardinality: int = 20) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode low-cardinality categorical columns; drop high-cardinality ones.

    High-cardinality string columns (e.g. free-text or near-unique service
    names) are dropped rather than encoded, since one-hot would explode
    dimensionality without adding useful signal for a thesis prototype.
    """
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    low_card = [c for c in cat_cols if df[c].nunique(dropna=True) <= max_onehot_cardinality]
    high_card = [c for c in cat_cols if c not in low_card]
    if high_card:
        logger.info("Dropping %d high-cardinality categorical column(s): %s", len(high_card), high_card)
        df = df.drop(columns=high_card)
    if low_card:
        logger.info("One-hot encoding %d categorical column(s): %s", len(low_card), low_card)
        df = pd.get_dummies(df, columns=low_card, dummy_na=False)
    return df, high_card


def preprocess_pipeline(
    df: pd.DataFrame,
    id_like_patterns: list[str] | None = None,
    max_onehot_cardinality: int = 20,
    test_size: float = 0.3,
    scale: bool = True,
    seed: int = 42,
) -> PreprocessResult:
    """Full cleaning -> encoding -> split -> scaling pipeline.

    Order of operations matters: label column must be detected and pulled
    out before id-like-column removal / one-hot encoding touch the rest of
    the frame, and infinities/NaNs must be resolved before scaling.
    """
    id_like_patterns = id_like_patterns or []
    dropped_columns: list[str] = []

    df = clean_column_names(df)
    df, name_dupes_dropped = remove_duplicate_columns(df)
    dropped_columns.extend(name_dupes_dropped)
    label_col = detect_label_column(df)
    logger.info("Detected label column: '%s'", label_col)
    y_raw = df[label_col]
    class_distribution_before = {str(k): int(v) for k, v in y_raw.value_counts(dropna=False).to_dict().items()}
    logger.info("Class distribution BEFORE binary mapping (raw '%s' values): %s", label_col, class_distribution_before)

    X = df.drop(columns=[label_col])

    # Some datasets carry both a binary "label" and multi-class "attack_cat";
    # the second one is kept aside (not dropped outright) as `category_raw`
    # for rule-layer category-aware reporting, then removed from X so it
    # can't leak into features.
    category_col: str | None = None
    category_raw: pd.Series | None = None
    for other in LABEL_COLUMN_CANDIDATES:
        other_clean = re.sub(r"[^\w]+", "_", other.strip().lower()).strip("_")
        if other_clean in X.columns and other_clean != label_col:
            category_col = other_clean
            category_raw = X[other_clean].astype(str)
            X = X.drop(columns=[other_clean])
            dropped_columns.append(other_clean)
            break  # only one extra label-like column is meaningful; rest handled generically below

    y = convert_labels_to_binary(y_raw)

    X, id_like_dropped = remove_id_like_columns(X, id_like_patterns)
    dropped_columns.extend(id_like_dropped)
    X = handle_missing_inf(X)
    X, high_card_dropped = handle_categorical(X, max_onehot_cardinality=max_onehot_cardinality)
    dropped_columns.extend(high_card_dropped)

    # Any remaining non-numeric columns (shouldn't happen, but be defensive).
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        logger.warning("Dropping unexpected non-numeric column(s) after encoding: %s", non_numeric)
        X = X.drop(columns=non_numeric)
        dropped_columns.extend(non_numeric)

    X, value_dupes_dropped = remove_duplicate_value_columns(X)
    dropped_columns.extend(value_dupes_dropped)

    if dropped_columns:
        logger.info("Total dropped columns (%d): %s", len(dropped_columns), dropped_columns)

    X = X.astype(np.float64)
    feature_names = list(X.columns)
    assert label_col not in feature_names, "label column leaked into features"
    leakage_report = leakage_check_report(feature_names, label_col, category_col)
    X_values = X.values
    y_values = y.values
    category_values = category_raw.values if category_raw is not None else None

    # --- Shuffled, stratified split (batch/baseline models) ---
    stratify = y if y.nunique() > 1 else None
    if category_values is not None:
        X_train, X_test, y_train, y_test, category_train, category_test = train_test_split(
            X_values, y_values, category_values, test_size=test_size, random_state=seed, stratify=stratify
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_values, y_values, test_size=test_size, random_state=seed, stratify=stratify
        )
        category_train = category_test = None
    scaler: StandardScaler | None = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # --- Chronological split (stream/drift/AL/hybrid experiments) ---
    # Row order is assumed to reflect original arrival order (no shuffling),
    # which is required for drift detection / prequential evaluation to be
    # meaningful. A separate scaler is fit only on the chronological training
    # prefix to avoid leaking test-period statistics.
    split_idx = int(round(len(X_values) * (1 - test_size)))
    split_idx = max(1, min(len(X_values) - 1, split_idx))
    X_train_chrono, X_test_chrono = X_values[:split_idx], X_values[split_idx:]
    y_train_chrono, y_test_chrono = y_values[:split_idx], y_values[split_idx:]
    if category_values is not None:
        category_train_chrono, category_test_chrono = category_values[:split_idx], category_values[split_idx:]
    else:
        category_train_chrono = category_test_chrono = None
    scaler_chrono: StandardScaler | None = None
    if scale:
        scaler_chrono = StandardScaler()
        X_train_chrono = scaler_chrono.fit_transform(X_train_chrono)
        X_test_chrono = scaler_chrono.transform(X_test_chrono)

    class_distribution = {str(k): int(v) for k, v in y.value_counts().to_dict().items()}
    logger.info("Class distribution AFTER binary mapping: %s", class_distribution)

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        X_train_chrono=X_train_chrono,
        X_test_chrono=X_test_chrono,
        y_train_chrono=y_train_chrono,
        y_test_chrono=y_test_chrono,
        scaler_chrono=scaler_chrono,
        feature_names=feature_names,
        label_column=label_col,
        n_samples=len(X),
        n_features=len(feature_names),
        class_distribution=class_distribution,
        class_distribution_before=class_distribution_before,
        dropped_columns=dropped_columns,
        category_column=category_col,
        category_train=category_train,
        category_test=category_test,
        category_train_chrono=category_train_chrono,
        category_test_chrono=category_test_chrono,
        leakage_report=leakage_report,
    )
