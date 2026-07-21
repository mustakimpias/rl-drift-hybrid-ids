"""Matplotlib figure generation for every plot required by the thesis
pipeline. Each function saves directly to a given path and closes its
figure (no interactive display, safe for headless script runs).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger("rl_drift_ids")


def _save(fig: plt.Figure, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", save_path)


def _save_thesis(fig: plt.Figure, save_stem: str | Path) -> None:
    """Save a publication-quality pair (.png + .pdf) at 300 dpi. `save_stem`
    is a path *without* extension — both extensions are appended.
    """
    save_stem = Path(save_stem)
    save_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_stem.with_suffix(".png"), dpi=300)
    fig.savefig(save_stem.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    logger.info("Saved thesis figure: %s.png / .pdf", save_stem)


def plot_bar_comparison(
    df: pd.DataFrame, group_col: str, value_col: str, title: str, ylabel: str, save_path: str | Path,
) -> None:
    """Generic bar chart of value_col grouped by group_col (e.g. model -> F1)."""
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        logger.warning("Skipping plot '%s': empty data or missing columns.", title)
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df[group_col].astype(str), df[value_col].astype(float), color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group_col.replace("_", " ").title())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, save_path)


def plot_stream_f1_over_time(
    run_results: dict[str, Any], window_size: int, save_path: str | Path,
) -> None:
    """Rolling F1 curves for each streaming model (run_results values must
    expose a `.rolling_f1` list attribute, as StreamRunResult does).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    any_plotted = False
    for name, result in run_results.items():
        rolling = getattr(result, "rolling_f1", None)
        if not rolling:
            continue
        x = np.arange(1, len(rolling) + 1) * window_size
        ax.plot(x, rolling, marker="o", markersize=3, label=name)
        any_plotted = True
    if not any_plotted:
        logger.warning("Skipping stream_f1_over_time plot: no rolling F1 data.")
        plt.close(fig)
        return
    ax.set_title("Prequential F1 Over Time (Stream Models)")
    ax.set_xlabel("Samples processed")
    ax.set_ylabel("Rolling F1 (window)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save(fig, save_path)


def plot_drift_points(
    drift_points: list[dict[str, Any]], stream_length: int, save_path: str | Path,
    error_signal: list[float] | None = None,
) -> None:
    """Mark drift detection indices along the stream. If an error signal
    (0/1 per sample) is provided, plot it as light background context.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    if error_signal is not None and len(error_signal) > 0:
        window = max(1, len(error_signal) // 200)
        rolling_err = pd.Series(error_signal).rolling(window=window, min_periods=1).mean()
        ax.plot(rolling_err, color="#999999", linewidth=1, label="rolling error rate")
    for dp in drift_points:
        ax.axvline(x=dp["index"], color="red", linestyle="--", alpha=0.6)
    ax.set_title(f"Detected Drift Points (n={len(drift_points)})")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Error rate")
    ax.set_xlim(0, max(stream_length, 1))
    if error_signal is not None:
        ax.legend()
    _save(fig, save_path)


def plot_label_budget_vs_f1(al_rows: pd.DataFrame, save_path: str | Path) -> None:
    """F1 vs. actual label-query percentage, one line per AL strategy across
    the swept label budgets (config['active_learning_budgets']).
    """
    if al_rows.empty:
        logger.warning("Skipping label_budget_vs_f1 plot: no data.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    strategies = list(al_rows["strategy"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(len(strategies), 1)))
    for strategy, color in zip(strategies, colors):
        sub = al_rows[al_rows["strategy"] == strategy].sort_values("label_query_percentage")
        ax.plot(sub["label_query_percentage"], sub["f1_score"], marker="o", color=color, label=strategy)
    ax.set_xlabel("Actual Label Query Percentage (%)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Label Budget vs. F1 by Active Learning Strategy")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, save_path)


def plot_rl_hybrid_comparison(hybrid_rows: pd.DataFrame, save_path: str | Path) -> None:
    """Grouped bar chart comparing F1 and FPR across all hybrid IDS arms.

    Arms that were swept across label budgets (drift_triggered_al_hybrid,
    rl_guided_hybrid) appear once per budget; the x-tick label disambiguates
    them with the budget value.
    """
    if hybrid_rows.empty:
        logger.warning("Skipping rl_hybrid_comparison plot: no data.")
        return

    if "label_budget_fraction" in hybrid_rows.columns:
        labels = [
            arm if pd.isna(budget) else f"{arm}\n(b={budget:g})"
            for arm, budget in zip(hybrid_rows["arm"], hybrid_rows["label_budget_fraction"])
        ]
    else:
        labels = hybrid_rows["arm"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(hybrid_rows)), 5))
    x = np.arange(len(hybrid_rows))
    width = 0.35
    ax.bar(x - width / 2, hybrid_rows["f1_score"].astype(float), width, label="F1 Score", color="#4C72B0")
    ax.bar(x + width / 2, hybrid_rows["false_positive_rate"].astype(float), width, label="False Positive Rate", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Hybrid IDS Arm Comparison")
    ax.set_ylabel("Score")
    ax.legend()
    _save(fig, save_path)


def plot_confusion_matrix_fig(y_true: Any, y_pred: Any, model_name: str, save_path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"])
    ax.set_yticklabels(["Benign", "Attack"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, save_path)


# --------------------------------------------------------------------------
# Thesis-quality figures (300 dpi, PNG + PDF) — see cost_performance_summary.csv
# and budget_wise_results.csv for the data these draw from.
# --------------------------------------------------------------------------

def plot_static_vs_proposed(
    cost_df: pd.DataFrame, metric_col: str, title: str, ylabel: str, save_stem: str | Path,
    method_a: str = "Static ML only", method_b: str = "RL-guided hybrid",
) -> None:
    """Two-bar comparison of a single metric between the static baseline and
    the proposed RL-guided hybrid — the headline "did the proposed method
    help" figure.
    """
    sub = cost_df[cost_df["method"].isin([method_a, method_b])]
    if sub.empty or metric_col not in sub.columns:
        logger.warning("Skipping %s: missing data for %s vs %s.", title, method_a, method_b)
        return
    sub = sub.set_index("method").reindex([method_a, method_b]).dropna(subset=[metric_col])
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4C72B0", "#55A868"]
    ax.bar(sub.index.astype(str), sub[metric_col].astype(float), color=colors[: len(sub)])
    for i, v in enumerate(sub[metric_col].astype(float)):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _save_thesis(fig, save_stem)


def plot_budget_vs_metric(budget_df: pd.DataFrame, metric_col: str, title: str, ylabel: str, save_stem: str | Path) -> None:
    """Metric vs. actual label-query percentage, one line per method, across
    every swept budget — built from budget_wise_results.csv, so it includes
    the RL-guided hybrid alongside the plain AL strategies (unlike the
    older label_budget_vs_f1.png, which only covered AL strategies).
    """
    if budget_df.empty or metric_col not in budget_df.columns:
        logger.warning("Skipping %s: no data.", title)
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    methods = list(budget_df["method"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(len(methods), 1)))
    for method, color in zip(methods, colors):
        sub = budget_df[budget_df["method"] == method].sort_values("actual_label_query_percentage")
        ax.plot(sub["actual_label_query_percentage"], sub[metric_col], marker="o", color=color, label=method)
    ax.set_xlabel("Actual Label Query Percentage (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save_thesis(fig, save_stem)


def plot_label_saving_vs_performance(cost_df: pd.DataFrame, save_stem: str | Path) -> None:
    """Scatter of label savings (vs. 100%-label full feedback) against F1 —
    the cost/performance trade-off summary for every method in
    cost_performance_summary.csv.
    """
    needed = {"label_saving_vs_full_feedback", "f1_score", "method"}
    if cost_df.empty or not needed.issubset(cost_df.columns):
        logger.warning("Skipping label_saving_vs_performance: missing data.")
        return
    sub = cost_df.dropna(subset=["label_saving_vs_full_feedback", "f1_score"])
    if sub.empty:
        logger.warning("Skipping label_saving_vs_performance: no rows with both fields.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, max(len(sub), 1)))
    for (_, row), color in zip(sub.iterrows(), colors):
        ax.scatter(row["label_saving_vs_full_feedback"], row["f1_score"], s=140, color=color)
        ax.annotate(
            row["method"], (row["label_saving_vs_full_feedback"], row["f1_score"]),
            textcoords="offset points", xytext=(6, 6), fontsize=9,
        )
    ax.set_xlabel("Label Saving vs. Full Feedback (%)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Label-Saving vs. Performance Trade-off")
    ax.grid(alpha=0.3)
    _save_thesis(fig, save_stem)


def plot_rl_action_distribution(action_counts: dict[str, int], save_stem: str | Path) -> None:
    """Bar chart of how often the RL controller took each action — a direct
    look at the learned policy's behavior (e.g. does it lean on
    update_if_drift over blind querying).
    """
    if not action_counts or sum(action_counts.values()) == 0:
        logger.warning("Skipping rl_action_distribution: RL controller took zero actions.")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = list(action_counts.keys())
    values = [action_counts[k] for k in labels]
    ax.bar(labels, values, color="#4C72B0")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_title("RL Controller Action Distribution")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_thesis(fig, save_stem)


def plot_drift_points_timeline(drift_df: pd.DataFrame, save_stem: str | Path) -> None:
    """Timeline scatter of every recorded drift event, grouped/colored by
    which experiment detected it — shows whether drift detection is
    consistent across the stream experiment and the hybrid arms, or
    concentrated in just one.
    """
    if drift_df.empty or "index" not in drift_df.columns:
        logger.warning("Skipping drift_points_timeline: no data.")
        return
    sub = drift_df.dropna(subset=["index"])
    if sub.empty:
        logger.warning("Skipping drift_points_timeline: no rows with a sample index.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    sources = list(sub["source_experiment"].unique()) if "source_experiment" in sub.columns else ["unknown"]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(sources), 1)))
    for source, color in zip(sources, colors):
        s = sub[sub["source_experiment"] == source] if "source_experiment" in sub.columns else sub
        ax.scatter(s["index"], [source] * len(s), color=color, s=40)
    ax.set_xlabel("Sample index")
    ax.set_title(f"Drift Events Timeline (n={len(sub)})")
    ax.grid(alpha=0.3, axis="x")
    _save_thesis(fig, save_stem)


def plot_unseen_attack_comparison(unseen_df: pd.DataFrame, save_stem: str | Path) -> None:
    """Grouped bar chart of F1/recall/specificity across the 4 methods
    evaluated on held-out (never-trained-on) attack categories."""
    needed = {"method", "f1_score", "recall", "specificity"}
    if unseen_df.empty or not needed.issubset(unseen_df.columns):
        logger.warning("Skipping unseen_attack_comparison: missing data.")
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(unseen_df))
    width = 0.25
    ax.bar(x - width, unseen_df["f1_score"].astype(float), width, label="F1", color="#4C72B0")
    ax.bar(x, unseen_df["recall"].astype(float), width, label="Recall (unseen attacks)", color="#DD8452")
    ax.bar(x + width, unseen_df["specificity"].astype(float), width, label="Specificity (benign)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(unseen_df["method"].astype(str), rotation=20, ha="right")
    ax.set_title("Generalization to Held-Out (Unseen) Attack Categories")
    ax.set_ylabel("Score")
    ax.legend()
    _save_thesis(fig, save_stem)
