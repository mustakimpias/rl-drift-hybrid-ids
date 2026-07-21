"""Fast, self-contained unit tests for scripts/run_cross_dataset.py's pure
helper functions (label/category detection, binary-label mapping, leakage-
column removal, metadata-file filtering). No dependency on the actual
multi-GB NetFlow CSVs — those are exercised by the smoke-test run instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

from run_cross_dataset import (  # noqa: E402
    _detect_label_and_category_columns,
    _find_data_csv_files,
    _map_binary_label,
    sanitize_feature_matrix,
)
from src.preprocessing import remove_id_like_columns  # noqa: E402


# --- Label / category column detection (task requirement 4/6) ---

def test_detect_label_and_category_columns_netflow_schema():
    columns = [
        "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL",
        "IN_BYTES", "OUT_BYTES", "Label", "Attack",
    ]
    label_col, category_col = _detect_label_and_category_columns(columns)
    assert label_col == "Label"
    assert category_col == "Attack"


def test_detect_label_and_category_columns_priority_order():
    # "label" (lowercase) should win over "class" per the candidate priority order.
    columns = ["feature_a", "class", "label"]
    label_col, category_col = _detect_label_and_category_columns(columns)
    assert label_col == "label"
    assert category_col == "class"


def test_detect_label_column_raises_when_absent():
    with pytest.raises(ValueError):
        _detect_label_and_category_columns(["feature_a", "feature_b"])


def test_detect_category_column_none_when_only_label_present():
    label_col, category_col = _detect_label_and_category_columns(["feature_a", "Label"])
    assert label_col == "Label"
    assert category_col is None


# --- Binary label mapping (task requirement 5) ---

def test_map_binary_label_numeric_passthrough():
    s = pd.Series([0, 1, 1, 0])
    mapped = _map_binary_label(s)
    assert mapped.tolist() == [0, 1, 1, 0]


def test_map_binary_label_string_tokens():
    s = pd.Series(["Benign", "benign", "Normal", "normal", "ransomware", "DDoS", "malicious", "Attack"])
    mapped = _map_binary_label(s)
    assert mapped.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_map_binary_label_category_names_all_map_to_attack():
    # Any non-benign category string (e.g. an attack-family name) -> 1.
    s = pd.Series(["Benign", "SSH-Bruteforce", "Infilteration", "Bot"])
    mapped = _map_binary_label(s)
    assert mapped.tolist() == [0, 1, 1, 1]


# --- Leakage-column removal (task requirement 7), via the shared
# src.preprocessing.remove_id_like_columns helper with this script's pattern list. ---

def test_leakage_columns_removed_from_features():
    from run_cross_dataset import LEAKAGE_LITERAL_PATTERNS, NETFLOW_ID_ALIAS_PATTERNS

    df = pd.DataFrame({
        "IPV4_SRC_ADDR": ["1.2.3.4"], "IPV4_DST_ADDR": ["5.6.7.8"],
        "L4_SRC_PORT": [443], "L4_DST_PORT": [80],
        "IN_BYTES": [100], "OUT_BYTES": [200], "PROTOCOL": [6],
    })
    cleaned, dropped = remove_id_like_columns(df, LEAKAGE_LITERAL_PATTERNS + NETFLOW_ID_ALIAS_PATTERNS)
    assert set(cleaned.columns) == {"IN_BYTES", "OUT_BYTES", "PROTOCOL"}
    assert set(dropped) == {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"}


# --- Metadata-file filtering (task requirement 3) ---

def test_find_data_csv_files_ignores_features_metadata(tmp_path):
    (tmp_path / "NF-ToN-IoT-v2.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "NetFlow_v2_Features.csv").write_text("name,desc\na,b\n", encoding="utf-8")
    files = _find_data_csv_files(tmp_path)
    names = [f.name for f in files]
    assert "NF-ToN-IoT-v2.csv" in names
    assert "NetFlow_v2_Features.csv" not in names


# --- sanitize_feature_matrix: the exact float32-overflow failure mode seen
# at 50k scale (a near-constant column with a tiny train std producing a
# huge z-score, plus literal inf values) must come out finite and bounded. ---

def test_sanitize_feature_matrix_handles_inf_and_extreme_outliers(tmp_path):
    rng = np.random.default_rng(0)
    n_train, n_test = 2000, 500
    feature_names = ["normal_feature", "near_constant_feature", "has_inf_feature"]

    X_train = np.column_stack([
        rng.normal(loc=100, scale=10, size=n_train),
        np.zeros(n_train),  # near-constant in train
        rng.normal(loc=50, scale=5, size=n_train),
    ])
    X_train[0, 1] = 1e-12  # tiny nonzero deviation -> near-zero but nonzero std
    X_train[5, 2] = np.inf
    X_train[6, 2] = -np.inf

    X_test = np.column_stack([
        rng.normal(loc=100, scale=10, size=n_test),
        np.full(n_test, 500.0),  # wildly different from train's near-zero column
        rng.normal(loc=50, scale=5, size=n_test),
    ])
    X_test[0, 2] = np.inf

    X_train_safe, X_test_safe = sanitize_feature_matrix(X_train, X_test, tmp_path, feature_names)

    assert np.isfinite(X_train_safe).all()
    assert np.isfinite(X_test_safe).all()
    assert np.max(np.abs(X_train_safe)) <= 1e6
    assert np.max(np.abs(X_test_safe)) <= 1e6
    assert X_train_safe.dtype == np.float32
    assert X_test_safe.dtype == np.float32
    assert X_train_safe.shape == X_train.shape
    assert X_test_safe.shape == X_test.shape

    report_path = tmp_path / "tables" / "cross_dataset_feature_sanity_report.csv"
    assert report_path.exists()
    report = pd.read_csv(report_path)
    assert set(report["feature_name"]) == set(feature_names)
    assert report.loc[report["feature_name"] == "has_inf_feature", "inf_count_train_before"].iloc[0] == 2
    assert report.loc[report["feature_name"] == "has_inf_feature", "inf_count_test_before"].iloc[0] == 1


def test_sanitize_feature_matrix_fits_only_on_train(tmp_path):
    # A column that is constant in train but has an extreme, unseen value in
    # test must not let that test-only value leak into the train-fit clip
    # bounds or scaler — test is transform-only.
    n_train, n_test = 500, 100
    feature_names = ["f0"]
    X_train = np.zeros((n_train, 1))
    X_test = np.full((n_test, 1), 1e9)

    X_train_safe, X_test_safe = sanitize_feature_matrix(X_train, X_test, tmp_path, feature_names)
    assert np.isfinite(X_train_safe).all()
    assert np.isfinite(X_test_safe).all()
    # Train was constant zero -> its own scaled representation must stay at 0.
    assert np.allclose(X_train_safe, 0.0)
