"""Shared utilities: config loading, logging, seeding, timing, optional imports.

Kept dependency-free beyond pyyaml so every other module can import from here
without risking circular imports.
"""
from __future__ import annotations

import contextlib
import importlib
import logging
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Raises FileNotFoundError with a clear message if the path is wrong,
    rather than letting yaml raise an opaque error later.
    """
    path = Path(config_path)
    if not path.is_absolute():
        # Allow relative paths whether invoked from repo root or elsewhere.
        candidates = [path, PROJECT_ROOT / path]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            f"Expected an absolute path or a path relative to {PROJECT_ROOT}."
        )
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} did not parse to a dict.")
    return config


def resolve_path(path_str: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_seed(seed: int = 42) -> None:
    """Seed python's random and numpy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(log_dir: str | Path | None = None, name: str = "rl_drift_ids") -> logging.Logger:
    """Configure and return a logger that writes to console and (optionally) a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured (e.g. reused across scripts).
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_path = resolve_path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


@contextlib.contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager measuring wall-clock seconds.

    Usage:
        with timer() as t:
            do_work()
        elapsed = t["seconds"]
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def optional_import(module_name: str) -> Any | None:
    """Import a module if available, else return None (used for xgboost/lightgbm)."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def ensure_dirs(*dirs: str | Path) -> None:
    """Create each directory (and parents) if it does not already exist."""
    for d in dirs:
        resolve_path(d).mkdir(parents=True, exist_ok=True)


def apply_cli_overrides(
    config: dict[str, Any],
    dataset: str | None = None,
    raw_dir: str | None = None,
    output_dir: str | None = None,
    max_samples: int | None = None,
    batch_size: int | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Apply the shared CLI overrides to a config dict in place.

    `config["dataset"]` is a flat string ("unsw" | "cicids" | "ciciot2023" |
    "auto"), `config["raw_dir"]` / `config["output_dir"]` are flat paths —
    matching the top-level schema in configs/default.yaml. Every pipeline
    script (run_all.py and each individual run_*.py) applies this the same
    way, so a CLI override given to run_all.py propagates identically to
    every stage instead of only affecting whichever stage happens to read
    argv directly.
    """
    if dataset:
        config["dataset"] = dataset
    if raw_dir:
        config["raw_dir"] = raw_dir
    if output_dir:
        config["output_dir"] = output_dir
    if max_samples is not None:
        config["max_samples"] = max_samples
    if batch_size is not None:
        config["batch_size"] = batch_size
    if random_state is not None:
        config["random_state"] = random_state
    return config


def resolve_config(config_or_path: "dict[str, Any] | str | Path") -> dict[str, Any]:
    """Accept either an already-loaded (and possibly CLI-overridden) config
    dict, or a path to a YAML file, and return a config dict either way.
    Lets every run_*.py's `run()` be called both by run_all.py (passing the
    single already-overridden dict so overrides apply consistently across
    every stage) and standalone from the command line (passing a path).
    """
    if isinstance(config_or_path, dict):
        return config_or_path
    return load_config(config_or_path)


def load_processed_data(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the split.npz + metadata.json produced by run_preprocessing.py.

    Returns (arrays, metadata) where arrays has keys X_train, X_test,
    y_train, y_test, X_train_chrono, X_test_chrono, y_train_chrono,
    y_test_chrono.

    Raises FileNotFoundError with a clear message telling the user to run
    run_preprocessing.py first.
    """
    import json

    import numpy as np

    processed_dir = resolve_path(config.get("processed_dir", "data/processed"))
    split_path = processed_dir / "split.npz"
    meta_path = processed_dir / "metadata.json"
    if not split_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Processed data not found under {processed_dir}. "
            "Run `python scripts/run_preprocessing.py --config <config>` first."
        )
    # allow_pickle=True: category_train_chrono/category_test_chrono (when
    # present) are string object arrays, which numpy can only load with
    # pickling enabled. Safe here since split.npz is always written by our
    # own run_preprocessing.py, never untrusted external input.
    npz = np.load(split_path, allow_pickle=True)
    arrays = {key: npz[key] for key in npz.files}
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return arrays, metadata


def add_synthetic_flag(df: Any, is_synthetic: bool) -> Any:
    """Stamp a `synthetic_data` boolean column onto a results DataFrame
    before it's saved, so every output CSV self-reports whether it came
    from real data or the smoke-test synthetic fallback.
    """
    df = df.copy()
    df["synthetic_data"] = bool(is_synthetic)
    return df


def get_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve and create the standard output directories under config['output_dir']."""
    results_dir = resolve_path(config.get("output_dir", "results"))
    dirs = {
        "results": results_dir,
        "tables": results_dir / "tables",
        "figures": results_dir / "figures",
        "logs": results_dir / "logs",
    }
    ensure_dirs(*dirs.values())
    return dirs
