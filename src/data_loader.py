"""Dataset loading for UNSW-NB15 / CICIDS2017 / CICIoT2023 (or any similar
flow-based CSV intrusion dataset), with a clearly-labeled synthetic fallback
for smoke testing when no raw data is present.

Nothing here does feature engineering — that happens in preprocessing.py.
This module's only job is: find CSVs, load them, concatenate them, and
report basic provenance (dataset name, whether it's synthetic).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("rl_drift_ids")

UNSW_NAME_HINTS = ("unsw", "nb15", "nb-15")
CICIDS_NAME_HINTS = (
    "cicids", "monday", "tuesday", "wednesday", "thursday", "friday",
    "workinghours", "morning", "afternoon",
)
CICIOT_NAME_HINTS = ("ciciot", "cic_iot", "cic-iot", "iot2023", "ciciot2023")
# Column-name fallbacks used when the filename gives no hint — both datasets
# have very distinctive column vocabularies.
UNSW_COLUMN_HINTS = ("attack_cat", "sbytes", "dbytes", "sttl", "dttl", "ct_state_ttl", "ct_srv_src", "swin", "dwin")
CICIDS_COLUMN_HINTS = (
    "flow duration", "fwd packet", "bwd packet", "destination port", "flow bytes/s", "syn flag count",
)
# CICIoT2023's official per-flow feature export has no label column at all;
# these are the possible label-column names IF a differently-exported
# variant of the dataset happens to carry one inline.
CICIOT_LABEL_CANDIDATES = ("label", "class", "attack_type", "category")
BENIGN_FOLDER_TOKENS = {"benign", "normal"}


@dataclass
class DatasetInfo:
    """Provenance metadata carried alongside a loaded DataFrame."""

    name: str
    is_synthetic: bool
    source_files: list[str] = field(default_factory=list)
    n_rows_before_cap: int | None = None
    warnings: list[str] = field(default_factory=list)
    # Populated by folder-structured loaders (currently CICIoT2023) for
    # richer dataset_metadata.csv reporting.
    loaded_files_count: int | None = None
    label_source: str | None = None
    source_folder: str | None = None
    binary_class_distribution: dict[str, int] | None = None
    attack_category_distribution: dict[str, int] | None = None
    # Populated when a non-default sampling mode is used (see
    # _allocate_balanced_binary / _allocate_category_balanced) — recorded so
    # dataset_metadata.csv can show exactly how the sample was constructed.
    sampling_info: dict[str, Any] | None = None


def _find_csv_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*.csv") if p.is_file())


def _detect_dataset_name(csv_files: list[Path], raw_path: Path | None = None) -> str:
    """Detect UNSW-NB15 / CICIDS2017 / CICIoT2023 by path first, then
    filename, then column-name vocabulary (peeking just the header row).
    """
    if raw_path is not None and any(hint in str(raw_path).lower() for hint in CICIOT_NAME_HINTS):
        return "ciciot2023"

    joined = " ".join(p.stem.lower() for p in csv_files)
    joined_folders = " ".join(p.parent.name.lower() for p in csv_files)
    if any(hint in joined for hint in UNSW_NAME_HINTS):
        return "unsw"
    if any(hint in joined for hint in CICIDS_NAME_HINTS):
        return "cicids"
    if any(hint in joined or hint in joined_folders for hint in CICIOT_NAME_HINTS):
        return "ciciot2023"
    # CICIoT2023's official release groups files into per-attack-type
    # folders with very distinctive names (DDoS-*, DoS-*, Mirai-*, Recon-*,
    # Benign_Final) and no inline label column — a strong structural
    # signature even when "ciciot" doesn't appear anywhere in the path.
    ciciot_folder_hints = ("ddos-", "dos-", "mirai-", "recon-", "benign_final")
    if any(hint in joined_folders for hint in ciciot_folder_hints):
        return "ciciot2023"

    for path in csv_files:
        if "features" in path.stem.lower():
            continue
        try:
            header = pd.read_csv(path, nrows=0, encoding_errors="replace").columns
        except Exception:  # pragma: no cover - defensive
            continue
        cols_joined = " ".join(str(c).strip().lower() for c in header)
        if any(hint in cols_joined for hint in UNSW_COLUMN_HINTS):
            logger.info("Detected UNSW-NB15 by column vocabulary in %s", path.name)
            return "unsw"
        if any(hint in cols_joined for hint in CICIDS_COLUMN_HINTS):
            logger.info("Detected CICIDS2017 by column vocabulary in %s", path.name)
            return "cicids"
        break  # only the first readable, non-features CSV is worth peeking at
    return "unknown"


def _load_unsw_features_map(raw_dir: Path) -> list[str] | None:
    """UNSW-NB15's raw UNSW-NB15_1..4.csv part files ship with no header row;
    a companion NUSW-NB15_features.csv provides the column names. If present,
    use it; otherwise the CSVs are assumed to already contain a header.
    """
    for candidate in raw_dir.rglob("*features*.csv"):
        try:
            feat_df = pd.read_csv(candidate, encoding="latin1")
        except Exception:  # pragma: no cover - defensive
            continue
        name_col = next((c for c in feat_df.columns if "name" in c.lower()), None)
        if name_col is not None:
            return [str(x).strip() for x in feat_df[name_col].tolist()]
    return None


def _read_single_csv(path: Path, header_names: list[str] | None) -> pd.DataFrame:
    read_kwargs: dict[str, Any] = {"low_memory": False, "encoding_errors": "replace"}
    try:
        if header_names is not None:
            # Heuristic: part files with no header are all-numeric-ish filenames
            # like UNSW-NB15_1.csv. If the file already has a text header
            # (e.g. testing-set.csv), reading with forced names would corrupt
            # it, so peek at the first row first.
            probe = pd.read_csv(path, nrows=1, header=None, **read_kwargs)
            first_row_is_header = any(
                isinstance(v, str) and not v.replace(".", "", 1).replace("-", "", 1).isdigit()
                for v in probe.iloc[0].tolist()
            )
            if first_row_is_header:
                df = pd.read_csv(path, **read_kwargs)
            else:
                df = pd.read_csv(path, header=None, names=header_names, **read_kwargs)
        else:
            df = pd.read_csv(path, **read_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to read %s (%s); skipping.", path, exc)
        return pd.DataFrame()
    return df


# --------------------------------------------------------------------------
# CICIoT2023
# --------------------------------------------------------------------------

def _is_benign_folder(folder_name: str) -> bool:
    return any(token in folder_name.lower() for token in BENIGN_FOLDER_TOKENS)


def _find_raw_label_column(columns: list[str]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in CICIOT_LABEL_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _fast_row_count(path: Path) -> int:
    """Count data rows (excluding the header) via chunked byte counting —
    orders of magnitude faster than parsing (~7 seconds for CICIoT2023's
    full 8+GB / 46M+ rows across 309 files), used only to allocate the
    proportional per-file read budget below.
    """
    count = 0
    try:
        with open(path, "rb") as f:
            while True:
                buf = f.read(1024 * 1024)
                if not buf:
                    break
                count += buf.count(b"\n")
    except Exception:  # pragma: no cover - defensive
        return 0
    return max(0, count - 1)


def _compute_proportional_row_caps(
    file_row_counts: dict[Path, int], max_rows: int | None, oversample_factor: int = 4, min_rows_per_file: int = 20,
) -> dict[Path, int | None]:
    """Allocate how many rows to read from each file *proportional to that
    file's true share of the full dataset* — not a flat per-file cap, which
    would badly distort the true class balance whenever one class is
    concentrated in a few very large files (CICIoT2023's Benign_Final: 4
    files totaling ~1.1M rows, vs. hundreds of much smaller attack files).
    A small `min_rows_per_file` floor keeps every attack category
    represented at all in the candidate pool even if its true proportional
    share would round down to nothing.

    Returns None (read the whole file) per path if max_rows is not set.
    """
    if max_rows is None:
        return {p: None for p in file_row_counts}
    total_rows = sum(file_row_counts.values())
    if total_rows == 0:
        return {p: None for p in file_row_counts}
    target_pool = min(total_rows, max_rows * oversample_factor)
    caps: dict[Path, int | None] = {}
    for path, count in file_row_counts.items():
        if count == 0:
            caps[path] = 0
            continue
        proportional_share = int(np.ceil(target_pool * count / total_rows))
        caps[path] = min(count, max(min(count, min_rows_per_file), proportional_share))
    return caps


def _allocate_balanced_binary(
    category_counts: dict[str, int],
    max_samples: int,
    benign_attack_ratio: float,
    per_attack_category_cap: int,
    preserve_attack_diversity: bool,
) -> dict[str, int]:
    """Target row count per category for sampling.mode=balanced_binary.

    Benign gets `max_samples * ratio / (1 + ratio)` rows (ratio=1.0 -> 50/50
    benign:attack). The attack side of the budget is spread *across attack
    categories* rather than drawn proportional to true prevalence — CICIoT2023
    is heavily DDoS-dominated, so a plain proportional/random attack sample
    would just be "mostly DDoS vs benign" again, not a meaningful multi-attack
    IDS evaluation. Each attack category is capped at
    `per_attack_category_cap`; if the sum of those caps exceeds the attack
    budget, every category's allocation is scaled down proportionally so the
    total lands on budget while diversity (every category still present) is
    preserved. If `preserve_attack_diversity` is False, falls back to a
    prevalence-proportional split of the attack budget instead.
    """
    benign_categories = [c for c in category_counts if _is_benign_folder(c)]
    attack_categories = [c for c in category_counts if c not in benign_categories]

    benign_target_total = int(round(max_samples * benign_attack_ratio / (1 + benign_attack_ratio)))
    attack_target_total = max_samples - benign_target_total

    benign_available = sum(category_counts[c] for c in benign_categories)
    benign_target_total = min(benign_target_total, benign_available)

    targets: dict[str, int] = {}
    for c in benign_categories:
        share = category_counts[c] / benign_available if benign_available > 0 else 0.0
        targets[c] = int(round(benign_target_total * share))

    if preserve_attack_diversity:
        raw_alloc = {c: min(per_attack_category_cap, category_counts[c]) for c in attack_categories}
        raw_sum = sum(raw_alloc.values())
        if raw_sum > attack_target_total and raw_sum > 0:
            scale = attack_target_total / raw_sum
            targets.update({c: max(1, int(round(v * scale))) for c, v in raw_alloc.items() if v > 0})
        else:
            targets.update(raw_alloc)
    else:
        attack_available = sum(category_counts[c] for c in attack_categories)
        for c in attack_categories:
            share = category_counts[c] / attack_available if attack_available > 0 else 0.0
            targets[c] = int(round(attack_target_total * share))

    return targets


def _allocate_category_balanced(
    category_counts: dict[str, int], per_category_cap: int, max_samples: int | None,
) -> dict[str, int]:
    """Target row count per category for sampling.mode=category_balanced:
    up to `per_category_cap` rows from every category (benign included as
    just another category), then scaled down proportionally if the sum
    would exceed max_samples.
    """
    targets = {c: min(per_category_cap, count) for c, count in category_counts.items()}
    total = sum(targets.values())
    if max_samples is not None and total > max_samples and total > 0:
        scale = max_samples / total
        targets = {c: max(1, int(round(v * scale))) for c, v in targets.items() if v > 0}
    return targets


def _compute_target_based_row_caps(
    category_files: dict[str, list[Path]], file_row_counts: dict[Path, int], targets: dict[str, int],
) -> dict[Path, int | None]:
    """Per-file read caps for a category target allocation: every file in a
    category is allowed to contribute up to that category's target (capped
    by the file's own size), so a category split across many files still
    gets enough candidates from each file for a proper random selection —
    while a category concentrated in one small file just reads what exists.
    """
    caps: dict[Path, int | None] = {}
    for category, files in category_files.items():
        target = targets.get(category, 0)
        for f in files:
            caps[f] = min(file_row_counts.get(f, 0), target) if target > 0 else 0
    return caps


def _finalize_category_targets(df: pd.DataFrame, category_col: str, targets: dict[str, int], seed: int) -> pd.DataFrame:
    """Trim the loaded candidate pool down to exactly `targets[category]`
    rows per category (or all available rows if fewer), preserving relative
    row order within the selection (see _stratified_sample_by_label for why
    order is kept).
    """
    rng = np.random.default_rng(seed)
    selected_idx: list[Any] = []
    for category, target in targets.items():
        if target <= 0:
            continue
        idx = df.index[df[category_col] == category].to_numpy()
        if len(idx) == 0:
            continue
        take = min(target, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        selected_idx.extend(chosen.tolist())
    selected_idx = sorted(selected_idx)
    return df.loc[selected_idx]


def _stratified_sample_by_label(df: pd.DataFrame, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    """Stratified downsample preserving the candidate pool's binary-label
    balance (rather than plain random sampling, which could by chance skew
    an already-imbalanced attack/benign ratio further). Selected rows keep
    their original relative order (sorted by original index) so the
    resulting frame stays usable as a structured "stream" — see the
    CICIoT2023 loader docstring for why this matters for drift detection.
    """
    n = len(df)
    counts = df[label_col].value_counts()
    selected_idx: list[Any] = []
    rng = np.random.default_rng(seed)
    for value, count in counts.items():
        target = max(1, min(int(count), int(round(max_rows * count / n))))
        idx = df.index[df[label_col] == value].to_numpy()
        chosen = rng.choice(idx, size=target, replace=False)
        selected_idx.extend(chosen.tolist())
    if len(selected_idx) > max_rows:
        selected_idx = list(rng.choice(np.array(selected_idx), size=max_rows, replace=False))
    selected_idx = sorted(selected_idx)
    return df.loc[selected_idx]


def _load_ciciot2023(
    raw_dir: Path, max_rows: int | None, seed: int, sampling_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, DatasetInfo]:
    """Load the official CICIoT2023 per-category CSV export: one folder per
    attack type (plus Benign_Final), each holding one or more CSVs of raw
    per-flow numeric features with NO label column. Per the dataset's own
    distribution layout, the label is therefore which folder a file lives
    in — this is not an assumption, it is how the dataset ships. If a
    variant CSV *does* carry an inline label/class/category column, that
    takes precedence over the folder name for that file.

    Row order is preserved as read (files sorted alphabetically, folders
    alphabetically) rather than shuffled: since CICIoT2023 has no single
    continuous timeline spanning attack types, this produces a structured,
    reproducible "stream" where category transitions are real, meaningful
    distribution shifts for the drift detector to catch — shuffling would
    make every downstream chronological/drift/streaming experiment
    meaningless (nothing left to detect in IID-shuffled data).

    sampling_config['mode'] selects how the sample is constructed:
      - "original_distribution" (default): proportional to each file's true
        share of the full dataset — preserves the dataset's real, severe
        class imbalance (~2.35% benign). Good for "what would a naive
        real-traffic sample look like"; bad for training a model that needs
        to see enough benign/minority-attack examples to learn anything.
      - "balanced_binary": benign:attack ratio controlled via
        `benign_attack_ratio`; attack rows spread across all attack
        categories (capped per category) instead of being dominated by
        whichever category happens to be biggest.
      - "category_balanced": up to `per_attack_category_cap` rows from every
        category (benign included as just another category) — for
        multiclass / category-aware analysis.
    """
    csv_files = _find_csv_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir.resolve()}.")

    sampling_config = sampling_config or {}
    mode = sampling_config.get("mode", "original_distribution")
    if mode not in {"original_distribution", "balanced_binary", "category_balanced"}:
        raise ValueError(
            f"Unknown sampling.mode '{mode}'; expected one of "
            "original_distribution, balanced_binary, category_balanced."
        )
    logger.info("CICIoT2023: sampling mode = %s", mode)

    category_files: dict[str, list[Path]] = {}
    for p in csv_files:
        category_files.setdefault(p.parent.name, []).append(p)

    targets: dict[str, int] | None = None
    if mode == "original_distribution" and max_rows is None:
        logger.warning(
            "No max_samples set — reading all %d CICIoT2023 CSV files in full "
            "(dataset is several GB; this may take a while).", len(csv_files),
        )
        row_caps: dict[Path, int | None] = {p: None for p in csv_files}
    else:
        logger.info("CICIoT2023: counting rows across %d files to allocate a read budget...", len(csv_files))
        file_row_counts = {p: _fast_row_count(p) for p in csv_files}
        total_available = sum(file_row_counts.values())
        logger.info("CICIoT2023: %d total rows available across all files.", total_available)
        category_counts = {
            c: sum(file_row_counts[p] for p in files) for c, files in category_files.items()
        }

        if mode == "original_distribution":
            row_caps = _compute_proportional_row_caps(file_row_counts, max_rows)
            pool_est = sum(c for c in row_caps.values() if c)
            logger.info(
                "CICIoT2023: proportional (size-aware) read budget allocated; candidate pool "
                "~%d rows before final stratified downsample to max_samples=%s.", pool_est, max_rows,
            )
        elif mode == "balanced_binary":
            targets = _allocate_balanced_binary(
                category_counts,
                max_rows or 50000,
                sampling_config.get("benign_attack_ratio", 1.0),
                sampling_config.get("per_attack_category_cap", 2000),
                sampling_config.get("preserve_attack_diversity", True),
            )
            row_caps = _compute_target_based_row_caps(category_files, file_row_counts, targets)
            logger.info("CICIoT2023: balanced_binary per-category targets: %s", targets)
        else:  # category_balanced
            targets = _allocate_category_balanced(
                category_counts, sampling_config.get("per_attack_category_cap", 2000), max_rows,
            )
            row_caps = _compute_target_based_row_caps(category_files, file_row_counts, targets)
            logger.info("CICIoT2023: category_balanced per-category targets: %s", targets)

    frames: list[pd.DataFrame] = []
    loaded_files: list[str] = []
    label_sources_used: set[str] = set()
    read_kwargs: dict[str, Any] = {"low_memory": False, "encoding_errors": "replace"}

    for path in csv_files:
        folder_name = path.parent.name
        cap = row_caps.get(path)
        if cap == 0:
            continue
        try:
            df = pd.read_csv(path, nrows=cap, **read_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read %s (%s); skipping.", path, exc)
            continue
        if df.empty:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]  # exact-name duplicate columns within this file

        raw_label_col = _find_raw_label_column(list(df.columns))
        if raw_label_col is not None:
            label_sources_used.add("csv_label_column")
            raw_labels = df[raw_label_col].astype(str)
            df["attack_cat"] = raw_labels
            df["label"] = raw_labels.str.strip().str.lower().map(
                lambda v: 0 if v in {"benign", "normal", "0", "0.0", "background"} else 1
            ).astype(int)
            df = df.drop(columns=[raw_label_col])
        else:
            label_sources_used.add("folder_name")
            df["attack_cat"] = folder_name
            df["label"] = 0 if _is_benign_folder(folder_name) else 1

        frames.append(df)
        loaded_files.append(str(path.relative_to(raw_dir)))

    if not frames:
        raise FileNotFoundError(
            f"Found {len(csv_files)} CICIoT2023 CSV file(s) under {raw_dir} but none could be parsed."
        )

    logger.info("CICIoT2023: loaded %d/%d files.", len(loaded_files), len(csv_files))
    logger.info("CICIoT2023: first few loaded files: %s", loaded_files[:5])

    combined = pd.concat(frames, axis=0, ignore_index=False, sort=False)
    combined = combined.reset_index(drop=True)
    del frames

    candidate_binary_dist = {str(k): int(v) for k, v in combined["label"].value_counts().to_dict().items()}
    candidate_category_dist = {str(k): int(v) for k, v in combined["attack_cat"].value_counts().to_dict().items()}
    logger.info("CICIoT2023 candidate pool (pre-downsample): %d rows, binary=%s", len(combined), candidate_binary_dist)
    logger.info("CICIoT2023 candidate pool attack_category distribution: %s", candidate_category_dist)

    if targets is not None:
        combined = _finalize_category_targets(combined, "attack_cat", targets, seed)
        # Rounding in the target allocation can leave the total a little
        # over max_samples; trim with the same label-stratified logic used
        # by original_distribution rather than an unstratified truncation.
        if max_rows is not None and len(combined) > max_rows:
            combined = _stratified_sample_by_label(combined, "label", max_rows, seed)
    elif max_rows is not None and len(combined) > max_rows:
        combined = _stratified_sample_by_label(combined, "label", max_rows, seed)
    combined = combined.reset_index(drop=True)

    final_binary_dist = {str(k): int(v) for k, v in combined["label"].value_counts().to_dict().items()}
    final_category_dist = {str(k): int(v) for k, v in combined["attack_cat"].value_counts().to_dict().items()}

    if label_sources_used == {"folder_name"}:
        label_source = "folder_name"
    elif label_sources_used == {"csv_label_column"}:
        label_source = "csv_label_column"
    else:
        label_source = "mixed"

    logger.info("CICIoT2023: detected label logic = %s", label_source)
    logger.info("CICIoT2023 final n_samples=%d n_raw_columns=%d", len(combined), combined.shape[1])
    logger.info("CICIoT2023 final binary_class_distribution=%s", final_binary_dist)
    logger.info("CICIoT2023 final attack_category_distribution=%s", final_category_dist)

    sampling_info = {
        "mode": mode,
        "benign_attack_ratio": sampling_config.get("benign_attack_ratio") if mode == "balanced_binary" else None,
        "per_attack_category_cap": sampling_config.get("per_attack_category_cap") if mode != "original_distribution" else None,
        "preserve_attack_diversity": sampling_config.get("preserve_attack_diversity") if mode == "balanced_binary" else None,
        "category_targets": targets,
    }

    info = DatasetInfo(
        name="ciciot2023",
        is_synthetic=False,
        source_files=loaded_files,
        n_rows_before_cap=len(combined),
        loaded_files_count=len(loaded_files),
        label_source=label_source,
        source_folder=str(raw_dir),
        binary_class_distribution=final_binary_dist,
        attack_category_distribution=final_category_dist,
        sampling_info=sampling_info,
    )
    return combined, info


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def load_dataset(
    dataset_name: str,
    raw_dir: str | Path,
    max_rows: int | None = None,
    allow_synthetic_fallback: bool = True,
    synthetic_rows: int = 20000,
    synthetic_features: int = 20,
    seed: int = 42,
    sampling_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, DatasetInfo]:
    """Load a dataset from CSVs under raw_dir, or fall back to synthetic data.

    Args:
        dataset_name: "unsw", "cicids", "ciciot2023", or "auto" to detect.
        raw_dir: directory searched recursively for *.csv files.
        max_rows: if set, subsample down to this many rows (seeded,
            stratified by label for ciciot2023, plain random otherwise).
        allow_synthetic_fallback: if True and no CSVs are found, generate a
            synthetic dataset instead of raising.
        synthetic_rows / synthetic_features: size of the synthetic fallback.
        seed: RNG seed for subsampling / synthetic generation.
        sampling_config: (CICIoT2023 only) the config['sampling'] block —
            {"mode": "original_distribution"|"balanced_binary"|"category_balanced", ...}.
            Ignored for other datasets.

    Returns:
        (dataframe, DatasetInfo)

    Raises:
        FileNotFoundError: if no CSVs are found and allow_synthetic_fallback
            is False.
    """
    raw_path = Path(raw_dir)
    csv_files = _find_csv_files(raw_path)

    if not csv_files:
        if not allow_synthetic_fallback:
            raise FileNotFoundError(
                f"No CSV files found under {raw_path.resolve()} and "
                "allow_synthetic_fallback is False."
            )
        logger.warning(
            "!!! No dataset CSVs found under %s. Falling back to a SYNTHETIC "
            "dataset for smoke-testing only. Results computed on synthetic "
            "data are NOT valid thesis results — download UNSW-NB15, "
            "CICIDS2017, or CICIoT2023 CSVs into that directory before "
            "running real experiments (see data/README.md). !!!",
            raw_path.resolve(),
        )
        df = generate_synthetic_dataset(n_rows=synthetic_rows, n_features=synthetic_features, seed=seed)
        info = DatasetInfo(
            name="synthetic",
            is_synthetic=True,
            source_files=[],
            n_rows_before_cap=len(df),
            warnings=["Synthetic fallback data — not valid for thesis results."],
        )
        return _cap_rows(df, max_rows, seed), info

    resolved_name = dataset_name if dataset_name != "auto" else _detect_dataset_name(csv_files, raw_path)

    if resolved_name == "ciciot2023":
        return _load_ciciot2023(raw_path, max_rows, seed, sampling_config=sampling_config)

    if resolved_name == "unknown":
        logger.warning(
            "Could not auto-detect dataset type from filenames %s; "
            "loading generically (concatenating all CSVs on shared columns).",
            [p.name for p in csv_files],
        )

    header_names = _load_unsw_features_map(raw_path) if resolved_name == "unsw" else None
    feature_files = [p for p in csv_files if "features" not in p.stem.lower()]

    frames = []
    for path in feature_files:
        df = _read_single_csv(path, header_names)
        if not df.empty:
            frames.append(df)
            logger.info("Loaded %s (%d rows, %d cols)", path.name, len(df), df.shape[1])

    if not frames:
        raise FileNotFoundError(
            f"Found {len(csv_files)} CSV file(s) under {raw_path} but none could be parsed."
        )

    combined = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    info = DatasetInfo(
        name=resolved_name,
        is_synthetic=False,
        source_files=[p.name for p in feature_files],
        n_rows_before_cap=len(combined),
        loaded_files_count=len(feature_files),
    )
    return _cap_rows(combined, max_rows, seed), info


def _cap_rows(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
        logger.info("Subsampled dataset down to max_rows=%d", max_rows)
    return df.reset_index(drop=True)


def generate_synthetic_dataset(n_rows: int = 20000, n_features: int = 20, seed: int = 42) -> pd.DataFrame:
    """Generate a small synthetic flow-like dataset for pipeline smoke testing.

    Includes a deliberate mid-stream distribution shift (concept drift) so
    that drift-detection code paths can be exercised even without real data.
    This data must NEVER be used to report thesis results.
    """
    rng = np.random.default_rng(seed)
    n_benign = int(n_rows * 0.8)
    n_attack = n_rows - n_benign

    half = n_rows // 2

    def _block(n: int, label: int, drift_shift: float) -> pd.DataFrame:
        mean = (0.5 if label == 0 else 2.0) + drift_shift
        X = rng.normal(loc=mean, scale=1.0, size=(n, n_features))
        cat = rng.choice(["tcp", "udp", "icmp"], size=n, p=[0.6, 0.3, 0.1])
        cols = {f"feature_{i}": X[:, i] for i in range(n_features)}
        cols["protocol_type"] = cat
        cols["label"] = "attack" if label == 1 else "benign"
        return pd.DataFrame(cols)

    parts = []
    for start_idx, n_block, drift_shift in (
        (0, n_rows // 2, 0.0),
        (n_rows // 2, n_rows - n_rows // 2, 0.75),
    ):
        n_block_benign = int(n_block * 0.8)
        n_block_attack = n_block - n_block_benign
        parts.append(_block(n_block_benign, 0, drift_shift))
        parts.append(_block(n_block_attack, 1, drift_shift))

    df = pd.concat(parts, axis=0, ignore_index=True)
    # Interleave benign/attack within each half instead of leaving them
    # block-sorted, but keep the two halves (pre/post drift) in order.
    first_half = df.iloc[: len(df) // 2].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    second_half = df.iloc[len(df) // 2:].sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    df = pd.concat([first_half, second_half], axis=0, ignore_index=True)
    return df
