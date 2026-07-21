#!/usr/bin/env python
"""End-to-end pipeline: preprocessing -> baselines -> stream -> active
learning -> RL-guided hybrid comparison -> final summary table + markdown
report.

Usage:
    python scripts/run_all.py --config configs/default.yaml
    python scripts/run_all.py --dataset unsw --raw_dir data/raw --output_dir results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `import run_preprocessing` etc.

import pandas as pd

import run_active_learning
import run_baselines
import run_preprocessing
import run_rl_hybrid_experiment
import run_stream_experiment
from src.evaluation import compute_composite_score
from src.report_generator import generate_report
from src.utils import add_synthetic_flag, apply_cli_overrides, get_output_dirs, load_config, setup_logging
from src.visualization import (
    plot_budget_vs_metric, plot_drift_points_timeline, plot_label_saving_vs_performance,
    plot_rl_action_distribution, plot_static_vs_proposed,
)

# ablation label -> method_role, per task 1's fixed vocabulary.
METHOD_ROLES = {
    "Static ML only": "baseline",
    "Adaptive ML only": "full_feedback_upper_bound",
    "Drift-aware adaptive ML": "full_feedback_upper_bound",
    "Active learning only": "active_learning_baseline",
    "Drift-triggered active learning": "active_learning_baseline",
    "Rule + ML hybrid": "ablation",
    "Rule + Drift + AL hybrid": "ablation",
    "RL-guided hybrid": "proposed",
}
# The 6 methods required in cost_performance_summary.csv (task 2).
COST_PERFORMANCE_METHODS = [
    "Static ML only", "Adaptive ML only", "Drift-aware adaptive ML",
    "Rule + ML hybrid", "Rule + Drift + AL hybrid", "RL-guided hybrid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full RL-Drift-Hybrid-IDS pipeline end to end.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="unsw | cicids | ciciot2023 | auto")
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Overrides max_samples")
    parser.add_argument("--batch_size", type=int, default=None, help="Overrides batch_size")
    parser.add_argument("--random_state", type=int, default=None, help="Overrides random_state")
    return parser.parse_args()


def build_final_comparison(
    baseline_df: pd.DataFrame, stream_df: pd.DataFrame, al_df: pd.DataFrame, hybrid_df: pd.DataFrame,
    rl_controllers_by_budget: dict[float, "RLController"] | None = None,
) -> pd.DataFrame:
    """The requested ablation rows, each sourced from whichever experiment
    table actually characterizes that configuration — not just "best row
    per table" (a hybrid arm and a plain AL strategy both landing in
    different tables need their own named row here).
    """
    rows: list[dict] = []
    rl_controllers_by_budget = rl_controllers_by_budget or {}

    def _best(df: pd.DataFrame | None, name_col: str, filter_val: str | None = None) -> pd.Series | None:
        if df is None or df.empty or name_col not in df.columns:
            return None
        sub = df if filter_val is None else df[df[name_col] == filter_val]
        if sub.empty:
            return None
        return sub.loc[sub["f1_score"].astype(float).idxmax()]

    def _add(ablation_name: str, row: pd.Series | None, method_col: str | None = None, is_rl: bool = False) -> None:
        if row is None:
            return
        rl_steps, total_reward = None, None
        if is_rl and rl_controllers_by_budget:
            budget = row.get("label_budget_fraction")
            controller = rl_controllers_by_budget.get(budget)
            if controller is not None:
                s = controller.summary()
                rl_steps, total_reward = s["n_steps"], s["total_reward"]
        rows.append({
            "ablation": ablation_name,
            "method_role": METHOD_ROLES.get(ablation_name),
            "feedback_mode": row.get("feedback_mode"),
            "source_method": row.get(method_col) if method_col else ablation_name,
            "label_budget_fraction": row.get("label_budget_fraction"),
            "f1_score": row.get("f1_score"),
            "macro_f1": row.get("macro_f1"),
            "weighted_f1": row.get("weighted_f1"),
            "balanced_accuracy": row.get("balanced_accuracy"),
            "accuracy": row.get("accuracy"),
            "precision": row.get("precision"),
            "recall": row.get("recall"),
            "specificity": row.get("specificity"),
            "false_positive_rate": row.get("false_positive_rate"),
            "false_negative_rate": row.get("false_negative_rate"),
            "mcc": row.get("mcc"),
            "roc_auc": row.get("roc_auc"),
            "label_query_percentage": row.get("label_query_percentage"),
            "drift_count": row.get("drift_count"),
            "rule_layer_coverage": row.get("rule_layer_coverage_pct"),
            "ml_layer_coverage": row.get("ml_layer_coverage_pct"),
            "rule_precision": row.get("rule_precision"),
            "rule_recall": row.get("rule_recall"),
            "rl_steps": rl_steps,
            "total_reward": total_reward,
        })

    _add("Static ML only", _best(hybrid_df, "arm", "static_ml_only"), "arm")
    _add("Adaptive ML only", _best(hybrid_df, "arm", "adaptive_ml_only"), "arm")
    _add("Drift-aware adaptive ML", _best(hybrid_df, "arm", "drift_aware_adaptive_ml"), "arm")
    _add("Active learning only", _best(al_df, "strategy"), "strategy")
    _add("Drift-triggered active learning", _best(al_df, "strategy", "drift_triggered_uncertainty"), "strategy")
    _add("Rule + ML hybrid", _best(hybrid_df, "arm", "rule_ml_hybrid"), "arm")
    _add("Rule + Drift + AL hybrid", _best(hybrid_df, "arm", "drift_triggered_al_hybrid"), "arm")
    _add("RL-guided hybrid", _best(hybrid_df, "arm", "rl_guided_hybrid"), "arm", is_rl=True)
    return pd.DataFrame(rows)


def build_cost_performance_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    """Thesis-ready cost/performance table (task 2): one row per required
    method, with label_saving_vs_full_feedback = 100 - actual label query %
    (full feedback = 100% queried, by definition — this is not a separate
    measurement, just the complement of the already-computed query rate).
    """
    cols = [
        "ablation", "method_role", "f1_score", "macro_f1", "balanced_accuracy", "precision", "recall",
        "specificity", "false_positive_rate", "mcc", "label_query_percentage", "drift_count",
        "rl_steps", "total_reward",
    ]
    sub = final_df[final_df["ablation"].isin(COST_PERFORMANCE_METHODS)][cols].copy()
    sub = sub.rename(columns={"ablation": "method"})
    sub["label_saving_vs_full_feedback"] = sub["label_query_percentage"].apply(
        lambda q: (100.0 - q) if pd.notna(q) else None
    )
    sub["composite_score"] = sub.apply(
        lambda r: compute_composite_score(r["macro_f1"], r["false_positive_rate"], r["label_query_percentage"]),
        axis=1,
    )
    order = [
        "method", "method_role", "f1_score", "macro_f1", "balanced_accuracy", "precision", "recall",
        "specificity", "false_positive_rate", "mcc", "label_query_percentage",
        "label_saving_vs_full_feedback", "composite_score", "drift_count", "rl_steps", "total_reward",
    ]
    # Preserve the requested method order rather than whatever order they
    # landed in final_df.
    sub["_order"] = sub["method"].apply(lambda m: COST_PERFORMANCE_METHODS.index(m))
    sub = sub.sort_values("_order").drop(columns="_order")
    return sub[order].reset_index(drop=True)


def build_budget_wise_results(
    al_df: pd.DataFrame, hybrid_df: pd.DataFrame, rl_controllers_by_budget: dict[float, "RLController"],
) -> pd.DataFrame:
    """Every swept budget (not just the best) for random / uncertainty /
    drift_triggered_uncertainty (task 3) plus rl_guided_hybrid, so the
    budget-vs-performance trade-off is visible for the proposed method too.
    """
    metric_cols = [
        "f1_score", "macro_f1", "balanced_accuracy", "precision", "recall",
        "specificity", "false_positive_rate", "mcc", "drift_count",
    ]
    rows: list[dict] = []

    if al_df is not None and not al_df.empty:
        for strategy in ("random", "uncertainty", "drift_triggered_uncertainty"):
            sub = al_df[al_df["strategy"] == strategy]
            for _, r in sub.iterrows():
                row = {"method": strategy, "budget": r.get("label_budget_fraction"),
                       "actual_label_query_percentage": r.get("label_query_percentage"),
                       "total_reward": None}
                row.update({c: r.get(c) for c in metric_cols})
                rows.append(row)

    if hybrid_df is not None and not hybrid_df.empty:
        sub = hybrid_df[hybrid_df["arm"] == "rl_guided_hybrid"]
        for _, r in sub.iterrows():
            budget = r.get("label_budget_fraction")
            controller = rl_controllers_by_budget.get(budget)
            total_reward = controller.summary()["total_reward"] if controller is not None else None
            row = {"method": "rl_guided_hybrid", "budget": budget,
                   "actual_label_query_percentage": r.get("label_query_percentage"),
                   "total_reward": total_reward}
            row.update({c: r.get(c) for c in metric_cols})
            rows.append(row)

    order = ["method", "budget", "actual_label_query_percentage"] + metric_cols + ["composite_score", "total_reward"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=order)
    df["composite_score"] = df.apply(
        lambda r: compute_composite_score(r["macro_f1"], r["false_positive_rate"], r["actual_label_query_percentage"]),
        axis=1,
    )
    return df[order]


def run(args: argparse.Namespace) -> None:
    # Config is loaded and CLI-overridden exactly once here, then the same
    # resolved dict is passed into every stage below — this is what makes
    # --dataset/--raw_dir/--output_dir/--max_samples/--batch_size/
    # --random_state actually apply consistently across the whole pipeline.
    config = load_config(args.config)
    apply_cli_overrides(
        config, args.dataset, args.raw_dir, args.output_dir,
        args.max_samples, args.batch_size, args.random_state,
    )
    dirs = get_output_dirs(config)
    logger = setup_logging(dirs["logs"], name="rl_drift_ids")

    logger.info("=== Stage 1/5: Preprocessing ===")
    metadata = run_preprocessing.run(config)

    logger.info("=== Stage 2/5: Baseline models ===")
    baseline_df, _, _ = run_baselines.run(config)

    logger.info("=== Stage 3/5: Stream / adaptive models + drift detection ===")
    stream_df, drift_df, _ = run_stream_experiment.run(config)

    logger.info("=== Stage 4/5: Active learning strategies ===")
    al_df, _ = run_active_learning.run(config)

    logger.info("=== Stage 5/5: RL-guided hybrid IDS comparison ===")
    hybrid_df, hybrid_results, rl_controllers_by_budget = run_rl_hybrid_experiment.run(config)

    # run_rl_hybrid_experiment merges its own drift events into
    # drift_points.csv on disk; re-read it so the report reflects the full,
    # merged picture rather than the stage-3-only snapshot still held here.
    drift_points_path = dirs["tables"] / "drift_points.csv"
    if drift_points_path.exists():
        drift_df = pd.read_csv(drift_points_path)

    # run_rl_hybrid_experiment also writes unseen_attack_metrics.csv and
    # (when enabled) rl_reward_tuning.csv directly to disk rather than
    # returning them; re-read here so the report can include both.
    unseen_attack_path = dirs["tables"] / "unseen_attack_metrics.csv"
    unseen_attack_df = pd.read_csv(unseen_attack_path) if unseen_attack_path.exists() else None
    rl_reward_tuning_path = dirs["tables"] / "rl_reward_tuning.csv"
    rl_reward_tuning_df = pd.read_csv(rl_reward_tuning_path) if rl_reward_tuning_path.exists() else None

    logger.info("=== Building final comparison + report ===")
    final_df = build_final_comparison(baseline_df, stream_df, al_df, hybrid_df, rl_controllers_by_budget)
    final_df = add_synthetic_flag(final_df, metadata["is_synthetic"])
    final_path = dirs["tables"] / "final_comparison_summary.csv"
    final_df.to_csv(final_path, index=False)
    logger.info("Saved %s", final_path)

    cost_df = build_cost_performance_summary(final_df)
    cost_df = add_synthetic_flag(cost_df, metadata["is_synthetic"])
    cost_path = dirs["tables"] / "cost_performance_summary.csv"
    cost_df.to_csv(cost_path, index=False)
    logger.info("Saved %s", cost_path)

    budget_df = build_budget_wise_results(al_df, hybrid_df, rl_controllers_by_budget)
    budget_df = add_synthetic_flag(budget_df, metadata["is_synthetic"])
    budget_path = dirs["tables"] / "budget_wise_results.csv"
    budget_df.to_csv(budget_path, index=False)
    logger.info("Saved %s", budget_path)

    # Report the RL controller for whichever budget produced the best
    # rl_guided_hybrid F1, so the reward summary matches the headline result.
    rl_summary = None
    rl_rows = hybrid_df[hybrid_df["arm"] == "rl_guided_hybrid"] if hybrid_df is not None and not hybrid_df.empty else None
    if rl_rows is not None and not rl_rows.empty and rl_controllers_by_budget:
        best_budget = rl_rows.loc[rl_rows["f1_score"].astype(float).idxmax(), "label_budget_fraction"]
        controller = rl_controllers_by_budget.get(best_budget) or next(iter(rl_controllers_by_budget.values()))
        rl_summary = controller.summary()

    # --- Thesis-quality figures (300 dpi, PNG+PDF) ---
    plot_static_vs_proposed(
        cost_df, "f1_score", "Static ML vs. RL-Guided Hybrid — F1 Score", "F1 Score",
        dirs["figures"] / "static_vs_proposed_f1",
    )
    plot_static_vs_proposed(
        cost_df, "false_positive_rate", "Static ML vs. RL-Guided Hybrid — False Positive Rate", "False Positive Rate",
        dirs["figures"] / "static_vs_proposed_fpr",
    )
    plot_budget_vs_metric(
        budget_df, "f1_score", "Label Budget vs. F1 (All Methods)", "F1 Score",
        dirs["figures"] / "label_budget_vs_f1",
    )
    plot_budget_vs_metric(
        budget_df, "false_positive_rate", "Label Budget vs. False Positive Rate (All Methods)", "False Positive Rate",
        dirs["figures"] / "label_budget_vs_fpr",
    )
    plot_label_saving_vs_performance(cost_df, dirs["figures"] / "label_saving_vs_performance")
    if rl_summary:
        plot_rl_action_distribution(rl_summary.get("action_counts", {}), dirs["figures"] / "rl_action_distribution")
    plot_drift_points_timeline(drift_df, dirs["figures"] / "drift_points_timeline")

    generate_report(
        output_path=dirs["results"] / "experiment_summary.md",
        dataset_name=metadata["dataset_name"],
        is_synthetic=metadata["is_synthetic"],
        n_samples=metadata["n_samples"],
        n_features=metadata["n_features"],
        class_distribution=metadata["class_distribution"],
        baseline_df=baseline_df,
        stream_df=stream_df,
        al_df=al_df,
        hybrid_df=hybrid_df,
        rl_summary=rl_summary,
        drift_points_df=drift_df,
        final_comparison_df=final_df,
        cost_performance_df=cost_df,
        sampling_mode=config.get("sampling", {}).get("mode") if config.get("dataset") == "ciciot2023" else None,
        unseen_attack_df=unseen_attack_df,
        rl_reward_tuning_df=rl_reward_tuning_df,
    )
    # --- Thesis-ready table aliases (task 6): plain, unambiguous names for
    # the four tables the thesis write-up references directly. These are
    # copies of tables already built above/by run_rl_hybrid_experiment, not
    # new computations -- kept as aliases so the "working" filenames
    # (cost_performance_summary.csv etc.) stay stable for the rest of the
    # pipeline/tests while the thesis manuscript can cite fixed names.
    cost_df.to_csv(dirs["tables"] / "table_main_balanced_results.csv", index=False)
    budget_df.to_csv(dirs["tables"] / "table_budget_results.csv", index=False)
    if unseen_attack_df is not None:
        unseen_attack_df.to_csv(dirs["tables"] / "table_unseen_attack_results.csv", index=False)
    if rl_reward_tuning_df is not None:
        rl_reward_tuning_df.to_csv(dirs["tables"] / "table_rl_reward_tuning.csv", index=False)
    logger.info("Saved thesis-ready table aliases (table_main_balanced_results.csv, table_budget_results.csv, "
                "table_unseen_attack_results.csv, table_rl_reward_tuning.csv where applicable)")

    logger.info("Pipeline complete. See results/experiment_summary.md and results/tables/*.csv")


if __name__ == "__main__":
    run(parse_args())
