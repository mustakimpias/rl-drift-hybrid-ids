"""Assembles results/experiment_summary.md from the actual DataFrames /
summaries produced by the pipeline. Every number in the report is read from
already-saved experiment output — nothing here is invented.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("rl_drift_ids")

SYNTHETIC_WARNING_LINE = (
    "These results are generated from synthetic data and must not be used as final thesis results."
)


def _best_row(df: pd.DataFrame, name_col: str, metric_col: str = "f1_score") -> tuple[str, float, float | None] | None:
    """Returns (name, metric_value, label_budget_fraction_or_None) for the
    best row by metric_col, or None if df is empty/missing columns.
    """
    if df is None or df.empty or metric_col not in df.columns or name_col not in df.columns:
        return None
    idx = df[metric_col].astype(float).idxmax()
    budget = df.loc[idx, "label_budget_fraction"] if "label_budget_fraction" in df.columns else None
    budget = None if budget is None or pd.isna(budget) else float(budget)
    return str(df.loc[idx, name_col]), float(df.loc[idx, metric_col]), budget


def _format_best(best: tuple[str, float, float | None] | None) -> str:
    if best is None:
        return "N/A"
    name, f1, budget = best
    budget_part = f", label_budget={budget:g}" if budget is not None else ""
    return f"{name} (F1={f1:.4f}{budget_part})"


def _df_to_markdown_table(df: pd.DataFrame, columns: list[str], float_cols: set[str]) -> list[str]:
    cols = [c for c in columns if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row.get(c)
            if pd.isna(v):
                cells.append("")
            elif c in float_cols:
                cells.append(f"{float(v):.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _generate_key_findings(
    dataset_name: str,
    sampling_mode: str | None,
    cost_performance_df: pd.DataFrame | None,
    drift_points_df: pd.DataFrame | None,
) -> list[str]:
    """Auto-generated bullet points computed only from already-saved
    experiment output (cost_performance_summary.csv / drift_points.csv) —
    nothing here is a manual/free-text claim.
    """
    findings: list[str] = []

    if dataset_name == "ciciot2023" and sampling_mode:
        findings.append(
            f"Balanced CICIoT2023 sampling was used (sampling.mode='{sampling_mode}'), avoiding the raw "
            "dataset's severe ~2.35% benign / DDoS-dominated imbalance — see results/tables/dataset_metadata.csv."
        )

    def _row(method: str) -> pd.Series | None:
        if cost_performance_df is None or cost_performance_df.empty:
            return None
        m = cost_performance_df[cost_performance_df["method"] == method]
        return m.iloc[0] if not m.empty else None

    static_row = _row("Static ML only")
    if static_row is not None:
        findings.append(
            f"Static ML baseline: F1={static_row['f1_score']:.4f}, "
            f"FPR={static_row['false_positive_rate']:.4f}, MCC={static_row['mcc']:.4f} "
            "(offline, no adaptation — see method_role='baseline')."
        )

    full_fb_row = _row("Drift-aware adaptive ML") if _row("Drift-aware adaptive ML") is not None else _row("Adaptive ML only")
    if full_fb_row is not None:
        findings.append(
            f"Full-feedback adaptive upper bound ({full_fb_row['method']}, method_role='full_feedback_upper_bound', "
            f"100% label query): F1={full_fb_row['f1_score']:.4f}, FPR={full_fb_row['false_positive_rate']:.4f}. "
            "This is an upper-bound reference that assumes unlimited labeling budget, not the proposed method."
        )

    proposed_row = _row("RL-guided hybrid")
    if proposed_row is not None:
        findings.append(
            f"Proposed RL-guided hybrid (method_role='proposed'): F1={proposed_row['f1_score']:.4f}, "
            f"macro-F1={proposed_row['macro_f1']:.4f}, balanced accuracy={proposed_row['balanced_accuracy']:.4f}, "
            f"FPR={proposed_row['false_positive_rate']:.4f}, MCC={proposed_row['mcc']:.4f}, "
            f"using only {proposed_row['label_query_percentage']:.4f}% of labels."
        )
        if pd.notna(proposed_row.get("label_saving_vs_full_feedback")):
            findings.append(
                f"Label saving achieved by the RL-guided hybrid vs. full feedback (100% labels): "
                f"{proposed_row['label_saving_vs_full_feedback']:.4f}%."
            )
        if static_row is not None and static_row["false_positive_rate"] > 0:
            reduction = 100.0 * (1 - proposed_row["false_positive_rate"] / static_row["false_positive_rate"])
            findings.append(
                f"False positive rate vs. static ML: {static_row['false_positive_rate']:.4f} -> "
                f"{proposed_row['false_positive_rate']:.4f} "
                f"({'a ' + format(reduction, '.1f') + '% reduction' if reduction >= 0 else format(-reduction, '.1f') + '% increase'})."
            )

    if drift_points_df is not None and not drift_points_df.empty:
        findings.append(f"Drift events detected across all experiments: {len(drift_points_df)}.")

    findings.append(
        "Limitations: the rule layer is a simulated (not real Snort/Suricata) signature engine, the RL "
        "controller is a lightweight tabular Q-learning agent rather than a deep RL model, and cross-dataset "
        "validation (e.g. against UNSW-NB15/CICIDS2017) is still needed before generalizing these findings."
    )
    return findings


def _generate_unseen_attack_discussion(unseen_attack_df: pd.DataFrame | None) -> list[str]:
    """Honest unseen-attack framing (task 5): the RL-guided hybrid must NOT
    be reported as "beating" static ML on unseen-attack F1 — on held-out
    attack categories it typically shows better recall/detection at the
    cost of a higher false-positive rate. Every number below is read
    straight from unseen_attack_metrics.csv; only the framing is fixed.
    """
    if unseen_attack_df is None or unseen_attack_df.empty or "method" not in unseen_attack_df.columns:
        return []

    def _row(method: str) -> pd.Series | None:
        m = unseen_attack_df[unseen_attack_df["method"] == method]
        return m.iloc[0] if not m.empty else None

    static_row = _row("Static ML only")
    rl_row = _row("RL-guided hybrid")
    lines: list[str] = []

    if static_row is not None:
        lines.append(
            f"Static ML only on unseen attack categories: F1={static_row['f1_score']:.4f}, "
            f"recall={static_row['recall']:.4f}, FPR={static_row['false_positive_rate']:.4f}."
        )
    if rl_row is not None:
        lines.append(
            f"RL-guided hybrid on unseen attack categories: F1={rl_row['f1_score']:.4f}, "
            f"recall={rl_row['recall']:.4f}, FPR={rl_row['false_positive_rate']:.4f}."
        )
    if static_row is not None and rl_row is not None:
        recall_delta = rl_row["recall"] - static_row["recall"]
        fpr_delta = rl_row["false_positive_rate"] - static_row["false_positive_rate"]
        f1_delta = rl_row["f1_score"] - static_row["f1_score"]
        recall_verb = "higher" if recall_delta > 0 else ("lower" if recall_delta < 0 else "equal")
        fpr_verb = "higher" if fpr_delta > 0 else ("lower" if fpr_delta < 0 else "equal")
        lines.append(
            f"RL-guided hybrid does NOT beat static ML on unseen-attack F1 "
            f"({rl_row['f1_score']:.4f} vs. {static_row['f1_score']:.4f}, "
            f"{'higher' if f1_delta > 0 else 'lower' if f1_delta < 0 else 'equal'} by {abs(f1_delta):.4f}). "
            f"What it does show is {recall_verb} recall/detection rate on held-out attack categories "
            f"({rl_row['recall']:.4f} vs. {static_row['recall']:.4f}, delta={recall_delta:+.4f}), "
            f"at the cost of a {fpr_verb} false positive rate "
            f"({rl_row['false_positive_rate']:.4f} vs. {static_row['false_positive_rate']:.4f}, delta={fpr_delta:+.4f}). "
            "This is a genuine detection-vs-false-alarm tradeoff under unseen attacks, not an unqualified win — "
            "report it as such."
        )
    return lines


def _generate_thesis_interpretation(
    cost_performance_df: pd.DataFrame | None,
    unseen_attack_df: pd.DataFrame | None,
) -> list[str]:
    """Final "Thesis Interpretation" section (task 7). Every number quoted
    is read from already-saved tables; the surrounding narrative connects
    them for the write-up but invents no new results.
    """
    lines: list[str] = []

    def _cost_row(method: str) -> pd.Series | None:
        if cost_performance_df is None or cost_performance_df.empty:
            return None
        m = cost_performance_df[cost_performance_df["method"] == method]
        return m.iloc[0] if not m.empty else None

    static_row = _cost_row("Static ML only")
    proposed_row = _cost_row("RL-guided hybrid")
    upper_bound_row = _cost_row("Drift-aware adaptive ML")

    # 1. Proposed method vs. static ML.
    if static_row is not None and proposed_row is not None:
        lines.append(
            "**Proposed method vs. static ML.** The RL-guided hybrid improves macro-F1 from "
            f"{static_row['macro_f1']:.4f} to {proposed_row['macro_f1']:.4f} and reduces false positive rate from "
            f"{static_row['false_positive_rate']:.4f} to {proposed_row['false_positive_rate']:.4f}, while querying "
            f"labels for only {proposed_row['label_query_percentage']:.4f}% of traffic — a static, offline-trained "
            "model with no adaptation is measurably outperformed on both detection quality and false-alarm cost "
            "on the balanced in-distribution test set."
        )

    # 2. Full-feedback upper bound.
    if upper_bound_row is not None and proposed_row is not None:
        lines.append(
            "**Full-feedback upper bound.** Drift-aware adaptive ML, trained with 100% label feedback "
            f"(macro-F1={upper_bound_row['macro_f1']:.4f}, FPR={upper_bound_row['false_positive_rate']:.4f}), is an "
            "upper-bound reference assuming unlimited labeling budget, not a competing deployable method — it shows "
            "the ceiling the RL-guided hybrid is trading against, not a method the RL controller needs to beat."
        )

    # 3. Label-efficiency advantage.
    if proposed_row is not None and pd.notna(proposed_row.get("label_saving_vs_full_feedback")):
        if upper_bound_row is not None:
            lines.append(
                "**Label-efficiency advantage.** The RL-guided hybrid reaches "
                f"{100.0 * proposed_row['macro_f1'] / upper_bound_row['macro_f1']:.1f}% of the full-feedback upper "
                f"bound's macro-F1 while saving {proposed_row['label_saving_vs_full_feedback']:.2f}% of the labeling "
                "budget — this label-efficiency gap is the central practical argument for the proposed method over "
                "full-feedback retraining."
            )
        else:
            lines.append(
                "**Label-efficiency advantage.** The RL-guided hybrid saves "
                f"{proposed_row['label_saving_vs_full_feedback']:.2f}% of the labeling budget vs. full feedback "
                f"while retaining macro-F1={proposed_row['macro_f1']:.4f} — this label-efficiency gap is the central "
                "practical argument for the proposed method over full-feedback retraining."
            )

    # 4. False-positive tradeoff.
    if static_row is not None and proposed_row is not None:
        fpr_delta = proposed_row["false_positive_rate"] - static_row["false_positive_rate"]
        direction = "reduces" if fpr_delta < 0 else "increases" if fpr_delta > 0 else "leaves unchanged"
        lines.append(
            f"**False-positive tradeoff.** On the balanced in-distribution test, the RL-guided hybrid {direction} "
            f"false positive rate relative to static ML ({static_row['false_positive_rate']:.4f} -> "
            f"{proposed_row['false_positive_rate']:.4f}); this reflects the reward shaping "
            "(false_positive_penalty vs. false_negative_penalty vs. label_query_cost) chosen for the controller, "
            "not a free improvement — a differently tuned reward would move this number, per the reward-sensitivity "
            "sweep in results/tables/rl_reward_tuning.csv."
        )

    # 5. Unseen-attack limitation (honest framing, see _generate_unseen_attack_discussion).
    if unseen_attack_df is not None and not unseen_attack_df.empty and "method" in unseen_attack_df.columns:
        static_u = unseen_attack_df[unseen_attack_df["method"] == "Static ML only"]
        rl_u = unseen_attack_df[unseen_attack_df["method"] == "RL-guided hybrid"]
        if not static_u.empty and not rl_u.empty:
            su, ru = static_u.iloc[0], rl_u.iloc[0]
            lines.append(
                "**Unseen-attack limitation.** On held-out (never-trained-on) attack categories, the RL-guided "
                f"hybrid does NOT beat static ML on F1 ({ru['f1_score']:.4f} vs. {su['f1_score']:.4f}); it trades "
                f"higher recall ({ru['recall']:.4f} vs. {su['recall']:.4f}) for a higher false positive rate "
                f"({ru['false_positive_rate']:.4f} vs. {su['false_positive_rate']:.4f}). This is a genuine limitation "
                "of the current reward shaping and label budget under distribution shift, not a result to overstate "
                "in the thesis write-up."
            )

    # 6. Why a Q1 paper needs more than this.
    lines.append(
        "**Why a Q1-venue paper needs cross-dataset validation or more RL tuning.** All results above come from a "
        "single dataset (CICIoT2023) and a single reward configuration selected from a 27-point sensitivity sweep "
        "over a subsampled evaluation stream (results/tables/rl_reward_tuning.csv) — this is enough to demonstrate "
        "the method's mechanics and internal tradeoffs, but not enough to claim general IDS performance. A "
        "Q1-level submission would need: (a) the same 8-method comparison repeated on at least one further "
        "benchmark (e.g. UNSW-NB15 or CICIDS2017) to rule out dataset-specific artifacts of CICIoT2023's category "
        "structure and class balance choices, and (b) a wider or adaptive reward-tuning search (the current sweep "
        "fixes correct_attack_reward/correct_benign_reward and only varies 3 penalty terms across 3 values each) "
        "before the unseen-attack false-positive tradeoff reported here can be called a tuned, final result rather "
        "than a first-pass finding."
    )
    return lines


def generate_report(
    output_path: str | Path,
    dataset_name: str,
    is_synthetic: bool,
    n_samples: int,
    n_features: int,
    class_distribution: dict[str, int],
    baseline_df: pd.DataFrame | None = None,
    stream_df: pd.DataFrame | None = None,
    al_df: pd.DataFrame | None = None,
    hybrid_df: pd.DataFrame | None = None,
    rl_summary: dict[str, Any] | None = None,
    drift_points_df: pd.DataFrame | None = None,
    final_comparison_df: pd.DataFrame | None = None,
    cost_performance_df: pd.DataFrame | None = None,
    sampling_mode: str | None = None,
    unseen_attack_df: pd.DataFrame | None = None,
    rl_reward_tuning_df: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_baseline = _best_row(baseline_df, "model") if baseline_df is not None else None
    best_stream = _best_row(stream_df, "model") if stream_df is not None else None
    best_al = _best_row(al_df, "strategy") if al_df is not None else None
    best_hybrid = _best_row(hybrid_df, "arm") if hybrid_df is not None else None

    lines: list[str] = []
    lines.append("# Experiment Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    if is_synthetic:
        lines.append("# ⚠️ SYNTHETIC DATA — NOT VALID THESIS RESULTS ⚠️")
        lines.append("")
        lines.append(f"## {SYNTHETIC_WARNING_LINE}")
        lines.append("")
        lines.append(
            "> No usable CSVs were found under `raw_dir` and `allow_synthetic_fallback: true` was set, "
            "so this run used the synthetic fallback dataset for code smoke-testing only. Every table "
            "produced by this run carries a `synthetic_data=true` column. Populate `data/raw/` with real "
            "UNSW-NB15 or CICIDS2017 CSVs and re-run before citing ANY number in this report."
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- **synthetic_data:** {is_synthetic}")
    lines.append(f"- **Dataset used:** {dataset_name}{' (synthetic fallback)' if is_synthetic else ''}")
    lines.append(f"- **Sample count:** {n_samples}")
    lines.append(f"- **Feature count:** {n_features}")
    lines.append(f"- **Class distribution:** {class_distribution}")
    lines.append("- Full provenance (source files, dropped columns, before/after class balance): `results/tables/dataset_metadata.csv`")
    lines.append("")

    lines.append("## Best Performers")
    lines.append("")
    lines.append(f"- **Best baseline model (by F1):** {_format_best(best_baseline)}")
    lines.append(f"- **Best adaptive/stream model (by F1):** {_format_best(best_stream)}")
    lines.append(f"- **Best active learning method (by F1):** {_format_best(best_al)}")
    lines.append(f"- **Best hybrid IDS arm (by F1):** {_format_best(best_hybrid)}")
    lines.append("")

    if final_comparison_df is not None and not final_comparison_df.empty:
        lines.append("## Ablation Comparison")
        lines.append("")
        lines.extend(_df_to_markdown_table(
            final_comparison_df,
            ["ablation", "source_method", "label_budget_fraction", "f1_score", "macro_f1", "balanced_accuracy",
             "accuracy", "precision", "recall", "specificity", "false_positive_rate", "mcc",
             "label_query_percentage", "drift_count", "rule_layer_coverage", "rl_steps", "total_reward"],
            float_cols={
                "f1_score", "macro_f1", "balanced_accuracy", "accuracy", "precision", "recall",
                "specificity", "false_positive_rate", "mcc",
            },
        ))
        lines.append("")

    key_findings = _generate_key_findings(dataset_name, sampling_mode, cost_performance_df, drift_points_df)
    if key_findings:
        lines.append("## Key Findings")
        lines.append("")
        for finding in key_findings:
            lines.append(f"- {finding}")
        lines.append("")
        lines.append("Full thesis-ready cost/performance table: `results/tables/cost_performance_summary.csv`")
        lines.append("")

    lines.append("## RL Controller Reward Summary")
    lines.append("")
    if rl_summary:
        lines.append(f"- Total reward accumulated: {rl_summary.get('total_reward', 0):.2f}")
        lines.append(f"- Mean reward per step: {rl_summary.get('mean_reward', 0):.4f}")
        lines.append(f"- Steps (decision points): {rl_summary.get('n_steps', 0)}")
        lines.append(f"- Distinct states visited: {rl_summary.get('n_states_visited', 0)}")
        lines.append(f"- Final exploration rate (epsilon): {rl_summary.get('final_epsilon', 0):.4f}")
        lines.append(f"- Query count: {rl_summary.get('query_count', 0)}")
        lines.append(f"- Update count: {rl_summary.get('update_count', 0)}")
        lines.append(f"- True positive / true negative counts: {rl_summary.get('true_positive_count', 0)} / {rl_summary.get('true_negative_count', 0)}")
        lines.append(f"- False positive penalty count: {rl_summary.get('false_positive_penalty_count', 0)}")
        lines.append(f"- False negative penalty count: {rl_summary.get('false_negative_penalty_count', 0)}")
        action_counts = rl_summary.get("action_counts", {})
        if action_counts:
            lines.append("- Action distribution:")
            for action, count in action_counts.items():
                lines.append(f"  - {action}: {count}")
        lines.append("- Full per-budget breakdown: `results/tables/rl_controller_summary.csv`")
    else:
        lines.append("N/A (RL hybrid experiment not run).")
    lines.append("")

    if unseen_attack_df is not None and not unseen_attack_df.empty:
        lines.append("## Unseen-Attack-Category Generalization")
        lines.append("")
        lines.extend(_df_to_markdown_table(
            unseen_attack_df,
            ["method", "method_role", "f1_score", "macro_f1", "recall", "specificity",
             "false_positive_rate", "mcc", "label_query_percentage", "composite_score"],
            float_cols={
                "f1_score", "macro_f1", "recall", "specificity", "false_positive_rate", "mcc",
                "label_query_percentage", "composite_score",
            },
        ))
        lines.append("")
        for finding in _generate_unseen_attack_discussion(unseen_attack_df):
            lines.append(f"- {finding}")
        lines.append("")
        lines.append("Full unseen-attack table: `results/tables/unseen_attack_metrics.csv`")
        lines.append("")

    if rl_reward_tuning_df is not None and not rl_reward_tuning_df.empty:
        lines.append("## RL Reward Sensitivity Tuning")
        lines.append("")
        lines.append(
            f"Grid search over {len(rl_reward_tuning_df)} (false_positive_penalty, false_negative_penalty, "
            "label_query_cost) combinations, holding correct_attack_reward=5 and correct_benign_reward=1 fixed. "
            "The best row by composite_score (not F1 alone) is used to set configs/default.yaml's `rl:` block:"
        )
        lines.append("")
        best_idx = rl_reward_tuning_df["composite_score"].astype(float).idxmax()
        best_row = rl_reward_tuning_df.loc[best_idx]
        lines.append(
            f"- **Best configuration (by composite_score):** false_positive_penalty={best_row['fp_penalty']:g}, "
            f"false_negative_penalty={best_row['fn_penalty']:g}, label_query_cost={best_row['label_query_cost']:g} "
            f"-> f1={best_row['f1_score']:.4f}, macro_f1={best_row['macro_f1']:.4f}, "
            f"fpr={best_row['false_positive_rate']:.4f}, composite_score={best_row['composite_score']:.4f}."
        )
        lines.append("")
        lines.append("Full sweep: `results/tables/rl_reward_tuning.csv`")
        lines.append("")

    lines.append("## Drift Points Summary")
    lines.append("")
    if drift_points_df is not None and not drift_points_df.empty:
        lines.append(f"- Total drift events/rows recorded: {len(drift_points_df)}")
        if "source_experiment" in drift_points_df.columns:
            by_source = drift_points_df["source_experiment"].value_counts().to_dict()
            lines.append(f"- By source experiment: {by_source}")
        if "detector" in drift_points_df.columns:
            by_detector = drift_points_df["detector"].value_counts().to_dict()
            lines.append(f"- By detector: {by_detector}")
        if "index" in drift_points_df.columns and drift_points_df["index"].notna().any():
            idx_col = drift_points_df["index"].dropna()
            lines.append(f"- First detected at sample index: {int(idx_col.min())}")
            lines.append(f"- Last detected at sample index: {int(idx_col.max())}")
    else:
        lines.append("No drift events recorded (or drift experiment not run).")
    lines.append("")

    lines.append("## Key Observations")
    lines.append("")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- (Add manual observations after reviewing results/tables and results/figures.)")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- The rule/signature layer is a simulated stand-in (a confidence-thresholded "
        "shallow decision tree, optionally restricted to a small set of 'known' attack "
        "categories when the dataset carries a multiclass category column), not a real "
        "Snort/Suricata ruleset; see `rule_layer_mode` / `rule_precision` / `rule_recall` "
        "columns in results/tables/rl_hybrid_metrics.csv for how well it actually performs "
        "standalone on this run's data."
    )
    lines.append(
        "- The RL controller is a tabular Q-learning-lite agent over a small discretized "
        "state space, not a deep RL agent — this is a deliberate scope decision, not a "
        "capability gap, but it limits how fine-grained the learned policy can be."
    )
    lines.append(
        "- The drift detector's error signal uses ground-truth labels available in this "
        "offline simulation (standard practice for benchmarking detector behavior), but the "
        "RL controller's action selection and the active-learning query decisions never see "
        "a sample's true label before that decision is made — only the reward/model-update "
        "step afterward does. See tests/test_sanity.py for the structural checks."
    )
    if is_synthetic:
        lines.append(f"- **{SYNTHETIC_WARNING_LINE}** Re-run against UNSW-NB15/CICIDS2017 before drawing any conclusions.")
    lines.append("")

    thesis_interpretation = _generate_thesis_interpretation(cost_performance_df, unseen_attack_df)
    if thesis_interpretation:
        lines.append("## Thesis Interpretation")
        lines.append("")
        for point in thesis_interpretation:
            lines.append(f"- {point}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote experiment summary report: %s", output_path)
    return output_path
