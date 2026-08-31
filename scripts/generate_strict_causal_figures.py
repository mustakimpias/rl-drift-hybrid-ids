#!/usr/bin/env python
"""Regenerate the Chapter 5 headline figures from strict-causal results only.

Every figure produced here reads exclusively from
results_ciciot2023_strict_causal/ (the main experiment's tables/ and the
diagnostics/ sub-run) -- never from the archived/oracle-feedback pipeline
(results_ciciot2023_balanced_test/). This replaces the six Chapter 5 figures
that previously depicted archived (oracle-feedback) RL-guided numbers:
    static_vs_proposed_f1.png, static_vs_proposed_fpr.png,
    label_saving_vs_performance.png, label_budget_vs_f1.png,
    label_budget_vs_fpr.png, rl_action_distribution.png
See FIGURE_STRICT_CAUSAL_UPDATE_REPORT.md for the full replacement mapping.

Output directory: thesis_latex/thesis_overleaf_assets/figures/strict_causal/
(PNG at 300 dpi + PDF, matching the project's existing figure conventions).

Usage:
    python scripts/generate_strict_causal_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = PROJECT_ROOT / "results_ciciot2023_strict_causal"
MAIN_RESULTS_CSV = RESULTS_DIR / "tables" / "strict_causal_main_results.csv"
BUDGET_COMPARISON_CSV = RESULTS_DIR / "diagnostics" / "strict_causal_budget_comparison.csv"
ACTION_BY_BUDGET_CSV = RESULTS_DIR / "diagnostics" / "strict_causal_rl_action_by_budget.csv"

OUT_DIR = PROJECT_ROOT / "thesis_latex" / "thesis_overleaf_assets" / "figures" / "strict_causal"

# Validated categorical palette (dataviz skill reference palette, fixed slot
# order -- never reassigned per-chart): slot1 blue, slot2 orange, slot3 aqua,
# slot4 yellow, slot7 violet, slot8 red. Chrome/ink from the same reference.
BLUE = "#2a78d6"       # slot 1 -- proposed method (hero series throughout)
ORANGE = "#eb6834"     # slot 2 -- RL-guided controller (negative-result series)
AQUA = "#1baf7a"       # slot 3 -- uncertainty-only baseline
YELLOW = "#eda100"     # slot 4
VIOLET = "#4a3aa7"     # slot 7 -- full-feedback upper bounds / static baseline
RED = "#e34948"        # slot 8

MUTED = "#898781"      # axis/label ink
GRID = "#e1e0d9"       # hairline gridline
AXIS = "#c3c2b7"       # baseline/axis
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": PRIMARY_INK,
    "text.color": PRIMARY_INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.dpi": 150,
})


def _style_axes(ax, y_grid_only: bool = True) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    if y_grid_only:
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)


def _save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / name}.png (+ .pdf)")


def fig_static_vs_proposed_f1(main: pd.DataFrame) -> None:
    static = main[main["method"] == "Static ML only"].iloc[0]
    proposed = main[main["method"] == "Drift-triggered AL hybrid (strict-causal)"].iloc[0]
    labels = ["Static ML only", "Drift-triggered AL hybrid\n(proposed, 25% budget)"]
    values = [static["f1_score"], proposed["f1_score"]]
    colors = [VIOLET, BLUE]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=11, color=PRIMARY_INK, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1-score")
    ax.set_title("Static ML vs. proposed method (strict-causal)", fontsize=12, pad=12)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "static_vs_proposed_f1_strict_causal")


def fig_static_vs_proposed_fpr(main: pd.DataFrame) -> None:
    static = main[main["method"] == "Static ML only"].iloc[0]
    proposed = main[main["method"] == "Drift-triggered AL hybrid (strict-causal)"].iloc[0]
    labels = ["Static ML only", "Drift-triggered AL hybrid\n(proposed, 25% budget)"]
    values = [static["false_positive_rate"], proposed["false_positive_rate"]]
    colors = [VIOLET, BLUE]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(values) * 0.03, f"{v:.5f}",
                 ha="center", va="bottom", fontsize=11, color=PRIMARY_INK, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.35)
    ax.set_ylabel("False positive rate")
    ax.set_title("Static ML vs. proposed method — FPR (strict-causal)", fontsize=12, pad=12)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "static_vs_proposed_fpr_strict_causal")


def fig_label_saving_vs_performance(main: pd.DataFrame) -> None:
    # (method, marker, color, label xytext offset, label ha)
    plot_rows = [
        ("Static ML only", "o", VIOLET, (10, -4), "left"),
        ("Adaptive ML only", "s", AXIS, (-10, 10), "right"),
        ("Drift-aware adaptive ML", "^", AXIS, (-10, 10), "right"),
        ("Rule + ML hybrid", "D", YELLOW, (10, -20), "left"),
        ("Drift-triggered AL hybrid (strict-causal)", "*", BLUE, (8, 12), "left"),
        ("RL-guided hybrid (strict-causal)", "P", ORANGE, (10, -20), "left"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for method, marker, color, offset, ha in plot_rows:
        r = main[main["method"] == method].iloc[0]
        x = 0.0 if pd.isna(r["label_query_percentage"]) else r["label_query_percentage"]
        y = r["f1_score"]
        size = 280 if method == "Drift-triggered AL hybrid (strict-causal)" else 160
        ax.scatter(x, y, s=size, marker=marker, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        display_name = method.replace(" (strict-causal)", "")
        ax.annotate(display_name, (x, y), textcoords="offset points", xytext=offset,
                    fontsize=9.5, color=SECONDARY_INK, ha=ha)
    ax.set_xlabel("Labels queried (% of 40,000-sample stream)")
    ax.set_ylabel("F1-score")
    ax.set_xlim(-8, 118)
    ax.set_ylim(0.855, 1.015)
    ax.set_title("Label cost vs. performance (strict-causal)", fontsize=12, pad=12)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, "label_saving_vs_performance_strict_causal")


def fig_label_budget_vs_metric(budget_df: pd.DataFrame, metric_col: str, ylabel: str, fname: str, title: str) -> None:
    series = [
        ("Drift-triggered AL hybrid (strict-causal)", "Drift-triggered AL hybrid (proposed)", BLUE, "o"),
        ("Uncertainty-only AL (strict-causal)", "Uncertainty-only active learning", AQUA, "s"),
        ("RL-guided hybrid (strict-causal)", "RL-guided hybrid (exploratory extension)", ORANGE, "^"),
    ]
    budgets = sorted(budget_df["budget"].unique())
    x = np.arange(len(budgets))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for csv_name, display_name, color, marker in series:
        sub = budget_df[budget_df["method"] == csv_name].set_index("budget").reindex(budgets)
        ax.plot(x, sub[metric_col].values, marker=marker, markersize=7, linewidth=2.2,
                color=color, label=display_name, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b * 100)}%" for b in budgets])
    ax.set_xlabel("Nominal label budget")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, fname)


def fig_rl_action_distribution(action_df: pd.DataFrame) -> None:
    action_order = ["no_query", "query_and_update", "update_if_drift", "adjust_threshold"]
    action_labels = {
        "no_query": "No query", "query_and_update": "Query and update",
        "update_if_drift": "Update if drift", "adjust_threshold": "Adjust threshold",
    }
    action_colors = {"no_query": MUTED, "query_and_update": BLUE, "update_if_drift": AQUA, "adjust_threshold": YELLOW}

    budgets = sorted(action_df["budget"].unique())
    x = np.arange(len(budgets))
    bottoms = np.zeros(len(budgets))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for action in action_order:
        sub = action_df[action_df["action"] == action].set_index("budget").reindex(budgets)
        values = sub["percentage"].fillna(0).values
        ax.bar(x, values, bottom=bottoms, color=action_colors[action], label=action_labels[action],
               width=0.6, edgecolor="white", linewidth=1.0)
        bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b * 100)}%" for b in budgets])
    ax.set_xlabel("Nominal label budget")
    ax.set_ylabel("Share of RL decision steps (%)")
    ax.set_ylim(0, 105)
    ax.set_title("RL controller action distribution by budget (strict-causal, exploratory extension)", fontsize=11.5, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    _style_axes(ax, y_grid_only=True)
    fig.tight_layout()
    _save(fig, "rl_action_distribution_strict_causal")


def main() -> None:
    main_df = pd.read_csv(MAIN_RESULTS_CSV)
    budget_df = pd.read_csv(BUDGET_COMPARISON_CSV)
    action_df = pd.read_csv(ACTION_BY_BUDGET_CSV)

    fig_static_vs_proposed_f1(main_df)
    fig_static_vs_proposed_fpr(main_df)
    fig_label_saving_vs_performance(main_df)
    fig_label_budget_vs_metric(budget_df, "f1", "F1-score", "label_budget_vs_f1_strict_causal",
                                "Label budget vs. F1-score (strict-causal)")
    fig_label_budget_vs_metric(budget_df, "fpr", "False positive rate", "label_budget_vs_fpr_strict_causal",
                                "Label budget vs. false positive rate (strict-causal)")
    fig_rl_action_distribution(action_df)


if __name__ == "__main__":
    main()
