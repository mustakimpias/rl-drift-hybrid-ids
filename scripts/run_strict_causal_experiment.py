#!/usr/bin/env python
"""Strict-causal limited-feedback re-run of the main hybrid IDS comparison
(the thesis's Table 5.1 / cost_performance_summary.csv).

Why this script exists: the archived run_rl_hybrid_experiment.py pipeline
(see results_ciciot2023_balanced_test/) computes the RL-guided hybrid arm's
drift-detector updates, reward, Q-table updates, and rolling F1/FPR state
features from the TRUE label on every stream sample, regardless of whether
that sample's label was actually queried under the label budget. That is a
documented simplifying benchmarking assumption in the original code
(hybrid_ids.run_rl_guided_hybrid's docstring), not a bug — but it means the
reported "4.47% labels queried" describes only the classifier's label cost,
while the drift detector and the RL controller's decision-making see 100%
of labels. This script re-runs the same comparison under a genuinely
strict, causal limited-feedback protocol instead: the algorithm (drift
detector, reward, Q-table, recent-F1/FPR state, classifier) may only ever
use a sample's true label on steps where that label was actually revealed
(queried, or part of the initial warm-up seed set). See
src/hybrid_ids.py::run_rl_guided_hybrid_strict_causal and
src/stream_utils.py::run_budgeted_stream(strict_causal=True) for the
implementation, and STRICT_CAUSAL_AUDIT_REPORT.md for the full comparison
against the archived (oracle-feedback) numbers.

This does NOT touch or overwrite any archived results directory. It writes
exclusively to a new `results_ciciot2023_strict_causal/` output directory,
reusing the exact same processed data (data/processed/split.npz — the same
49,992-row CICIoT2023-balanced chronological split already on disk), the
same seed (42), and the same 20%/80% rule-train/evaluation-stream split
logic (hybrid_ids.build_rule_train_stream_split) as the archived run, so the
only thing that differs is the strict-causal gating itself.

Usage:
    python scripts/run_strict_causal_experiment.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation import compute_composite_score, compute_metrics
from src.hybrid_ids import (
    RuleBasedLayer,
    build_rule_train_stream_split,
    run_adaptive_ml_only,
    run_drift_aware_adaptive_ml,
    run_drift_triggered_al_hybrid_strict_causal,
    run_rl_guided_hybrid_strict_causal,
    run_rule_ml_hybrid,
    run_rule_only,
    run_static_ml_only,
)
from src.utils import (
    add_synthetic_flag, apply_cli_overrides, get_output_dirs, load_config, load_processed_data,
    resolve_config, set_seed, setup_logging,
)

logger = logging.getLogger("rl_drift_ids")

# Same required-method vocabulary as scripts/run_all.py's COST_PERFORMANCE_METHODS,
# with the two budgeted arms renamed to make clear they are the strict-causal
# re-implementation, not the archived oracle-feedback one.
METHOD_ROLES = {
    "Rule only": "baseline",
    "Static ML only": "baseline",
    "Adaptive ML only": "full_feedback_upper_bound",
    "Drift-aware adaptive ML": "full_feedback_upper_bound",
    "Rule + ML hybrid": "ablation",
    "Drift-triggered AL hybrid (strict-causal)": "active_learning_baseline",
    "RL-guided hybrid (strict-causal)": "proposed",
}
MAIN_TABLE_METHODS = [
    "Static ML only", "Adaptive ML only", "Drift-aware adaptive ML",
    "Rule + ML hybrid", "Drift-triggered AL hybrid (strict-causal)", "RL-guided hybrid (strict-causal)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict-causal re-run of the main hybrid IDS comparison.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="results_ciciot2023_strict_causal")
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=None,
        help="Label budgets to sweep for the two budgeted arms (default: config's active_learning_budgets).",
    )
    return parser.parse_args()


def _metrics_row(method: str, feedback_mode: str, m: dict, label_budget_fraction, label_query_percentage,
                  drift_count, unsupervised_drift_count, rl_steps=None, total_reward=None) -> dict:
    row = {
        "method": method,
        "method_role": METHOD_ROLES.get(method),
        "feedback_mode": feedback_mode,
        "label_budget_fraction": label_budget_fraction,
        "f1_score": m["f1_score"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "balanced_accuracy": m["balanced_accuracy"],
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "specificity": m["specificity"],
        "false_positive_rate": m["false_positive_rate"],
        "false_negative_rate": m["false_negative_rate"],
        "mcc": m["mcc"],
        "n_samples": m["n_samples"],
        "label_query_percentage": label_query_percentage,
        "label_saving_vs_full_feedback": (100.0 - label_query_percentage) if label_query_percentage is not None else None,
        "drift_count": drift_count,
        "unsupervised_drift_count": unsupervised_drift_count,
        "rl_steps": rl_steps,
        "total_reward": total_reward,
    }
    row["composite_score"] = compute_composite_score(row["macro_f1"], row["false_positive_rate"], label_query_percentage)
    return row


def run(config_path_or_dict, budgets_override=None):
    config = resolve_config(config_path_or_dict)
    seed = config.get("random_state", 42)
    set_seed(seed)
    dirs = get_output_dirs(config)
    setup_logging(dirs["logs"], name="rl_drift_ids")

    arrays, metadata = load_processed_data(config)
    feature_names = metadata["feature_names"]
    is_synthetic = metadata.get("is_synthetic", False)

    hybrid_cfg = config.get("hybrid_ids", {}).get("rule_layer", {})
    confidence_threshold = hybrid_cfg.get("confidence_threshold", 0.97)
    known_categories_top_k = hybrid_cfg.get("known_categories_top_k", 5)
    rule_train_fraction = config.get("hybrid_ids", {}).get("rule_train_fraction", 0.2)
    al_cfg = config.get("active_learning", {})
    warmup_fraction = al_cfg.get("warmup_fraction", 0.01)
    warmup_min = al_cfg.get("warmup_min", 100)
    drift_cfg = config.get("drift_detection", {})
    detector_type = config.get("drift_detector", "adwin")
    adwin_delta = drift_cfg.get("adwin_delta", 0.002)
    page_hinkley_threshold = drift_cfg.get("page_hinkley_threshold", 50)
    page_hinkley_min_instances = drift_cfg.get("page_hinkley_min_instances", 30)
    batch_size = config.get("batch_size", 1000)

    budgets = budgets_override or config.get("active_learning_budgets", [0.1])
    if not isinstance(budgets, list):
        budgets = [budgets]

    # --- Identical split to the archived run_hybrid_comparison: same seed,
    # same rule_train_fraction, same stratified-by-category split logic. ---
    split = build_rule_train_stream_split(
        arrays["X_train_chrono"], arrays["y_train_chrono"],
        arrays["X_test_chrono"], arrays["y_test_chrono"],
        rule_train_fraction, seed,
        arrays.get("category_train_chrono"), arrays.get("category_test_chrono"),
    )
    X_rule_train, y_rule_train = split["X_rule_train"], split["y_rule_train"]
    X_stream, y_stream = split["X_stream"], split["y_stream"]
    category_rule_train = split["category_rule_train"]
    n_stream = len(X_stream)
    logger.info(
        "Strict-causal experiment: %d rows reserved for rule/static training, %d rows in the evaluation stream "
        "(identical split to the archived run given seed=%d)",
        len(X_rule_train), n_stream, seed,
    )

    rule_layer = RuleBasedLayer(
        confidence_threshold=confidence_threshold, random_state=seed, known_categories_top_k=known_categories_top_k,
    )
    rule_layer.fit(X_rule_train, y_rule_train, category_train=category_rule_train)
    logger.info("Rule layer: %s", rule_layer.describe())

    main_rows: list[dict] = []
    accounting_rows: list[dict] = []
    budget_sweep_rows: list[dict] = []
    trace_sample_frames: list[pd.DataFrame] = []

    # --- Arms that are already causal-safe as-is (no per-step label
    # gating concept applies: either fully offline, or already declared
    # full-feedback / rule-filtered-full-feedback with no hidden budget
    # claim) — reused unmodified from src/hybrid_ids.py. ---
    logger.info("Running arm: rule_only")
    rule_res = run_rule_only(rule_layer, X_stream, y_stream)
    main_rows.append(_metrics_row("Rule only", rule_res.feedback_mode, rule_res.metrics, None, 0.0, 0, 0))

    logger.info("Running arm: static_ml_only")
    static_res = run_static_ml_only(X_rule_train, y_rule_train, X_stream, y_stream, seed=seed)
    main_rows.append(_metrics_row("Static ML only", static_res.feedback_mode, static_res.metrics, None, 0.0, 0, 0))

    logger.info("Running arm: adaptive_ml_only")
    adaptive_res = run_adaptive_ml_only(
        X_stream, y_stream, feature_names, detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    main_rows.append(_metrics_row(
        "Adaptive ML only", adaptive_res.feedback_mode, adaptive_res.metrics, None,
        adaptive_res.label_query_percentage, adaptive_res.drift_count, None,
    ))

    logger.info("Running arm: drift_aware_adaptive_ml")
    drift_aware_res = run_drift_aware_adaptive_ml(
        X_stream, y_stream, feature_names, detector_type=detector_type, adwin_delta=adwin_delta,
        page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
    )
    main_rows.append(_metrics_row(
        "Drift-aware adaptive ML", drift_aware_res.feedback_mode, drift_aware_res.metrics, None,
        drift_aware_res.label_query_percentage, drift_aware_res.drift_count, None,
    ))

    logger.info("Running arm: rule_ml_hybrid")
    rule_ml_res = run_rule_ml_hybrid(rule_layer, X_stream, y_stream, feature_names)
    main_rows.append(_metrics_row(
        "Rule + ML hybrid", rule_ml_res.feedback_mode, rule_ml_res.metrics, None,
        rule_ml_res.label_query_percentage, rule_ml_res.drift_count, None,
    ))

    # --- Budgeted arms: strict-causal implementations, swept across every
    # configured label budget. Best-F1 budget becomes the Table 5.1
    # headline row (mirrors scripts/run_all.py::build_final_comparison's
    # "_best" selection so the strict-causal table is structurally
    # comparable to the archived one). ---
    dt_results, rl_results, rl_controllers, rl_traces = {}, {}, {}, {}
    for budget in budgets:
        logger.info("Running arm: drift_triggered_al_hybrid_strict_causal (budget=%s)", budget)
        dt_res = run_drift_triggered_al_hybrid_strict_causal(
            rule_layer, X_stream, y_stream, feature_names,
            label_budget_fraction=budget, detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            batch_size=batch_size, warmup_fraction=warmup_fraction, warmup_min=warmup_min, seed=seed,
        )
        dt_results[budget] = dt_res
        logger.info(
            "  drift_triggered_al_hybrid_strict_causal@%s: f1=%.4f label_query_pct=%.2f%% drift=%d/unsup=%d",
            budget, dt_res.metrics["f1_score"], dt_res.label_query_percentage,
            dt_res.drift_count, dt_res.extra.get("unsupervised_drift_count", 0),
        )

        logger.info("Running arm: rl_guided_hybrid_strict_causal (budget=%s)", budget)
        rl_res, rl_controller, trace_df = run_rl_guided_hybrid_strict_causal(
            rule_layer, X_stream, y_stream, feature_names,
            rl_config=config.get("rl", {}), label_budget_fraction=budget,
            detector_type=detector_type, adwin_delta=adwin_delta,
            page_hinkley_threshold=page_hinkley_threshold, page_hinkley_min_instances=page_hinkley_min_instances,
            warmup_fraction=warmup_fraction, warmup_min=warmup_min, seed=seed,
        )
        rl_results[budget] = rl_res
        rl_controllers[budget] = rl_controller
        rl_traces[budget] = trace_df
        logger.info(
            "  rl_guided_hybrid_strict_causal@%s: f1=%.4f label_query_pct=%.2f%% drift=%d/unsup=%d",
            budget, rl_res.metrics["f1_score"], rl_res.label_query_percentage,
            rl_res.drift_count, rl_res.extra.get("unsupervised_drift_count", 0),
        )

        # --- Label accounting (task E), per budget, for both budgeted arms. ---
        s = rl_controller.summary()
        action_trace = rl_res.extra.get("action_trace", [])
        update_if_drift_query_count = sum(
            1 for r in action_trace if r["action"] == 2 and r["queried"]  # ACTION_UPDATE_IF_DRIFT == 2
        )
        n_warmup = rl_res.extra.get("warmup_count", 0)
        total_revealed = n_warmup + s["query_count"]
        rule_handled = int(np.round(rl_res.rule_layer_coverage / 100.0 * n_stream))
        ml_routed = n_stream - rule_handled
        accounting_rows.append({
            "method": "RL-guided hybrid (strict-causal)",
            "label_budget_fraction": budget,
            "total_stream_samples": n_stream,
            "warmup_labels": n_warmup,
            "controller_query_count": s["query_count"],
            "query_and_update_count": s["action_counts"]["query_and_update"],
            "update_if_drift_query_count": update_if_drift_query_count,
            "total_labels_revealed_to_algorithm": total_revealed,
            "total_labels_revealed_percentage": 100.0 * total_revealed / n_stream,
            "classifier_update_count": n_warmup + s["update_count"],
            "rl_decision_steps": s["n_steps"],
            "ml_routed_steps": ml_routed,
            "rule_layer_handled_steps": rule_handled,
            "supervised_drift_count": rl_res.drift_count,
            "unsupervised_drift_count": rl_res.extra.get("unsupervised_drift_count", 0),
        })
        accounting_rows.append({
            "method": "Drift-triggered AL hybrid (strict-causal)",
            "label_budget_fraction": budget,
            "total_stream_samples": n_stream,
            "warmup_labels": None,  # run_budgeted_stream folds warm-up into query_flags directly
            "controller_query_count": None,
            "query_and_update_count": None,
            "update_if_drift_query_count": None,
            "total_labels_revealed_to_algorithm": int(round(dt_res.label_query_percentage / 100.0 * n_stream)),
            "total_labels_revealed_percentage": dt_res.label_query_percentage,
            "classifier_update_count": dt_res.extra.get("queried_count"),
            "rl_decision_steps": None,
            "ml_routed_steps": ml_routed,
            "rule_layer_handled_steps": rule_handled,
            "supervised_drift_count": dt_res.drift_count,
            "unsupervised_drift_count": dt_res.extra.get("unsupervised_drift_count", 0),
        })

        budget_sweep_rows.append(_metrics_row(
            "RL-guided hybrid (strict-causal)", rl_res.feedback_mode, rl_res.metrics, budget,
            rl_res.label_query_percentage, rl_res.drift_count, rl_res.extra.get("unsupervised_drift_count", 0),
            rl_steps=s["n_steps"], total_reward=s["total_reward"],
        ))
        budget_sweep_rows.append(_metrics_row(
            "Drift-triggered AL hybrid (strict-causal)", dt_res.feedback_mode, dt_res.metrics, budget,
            dt_res.label_query_percentage, dt_res.drift_count, dt_res.extra.get("unsupervised_drift_count", 0),
        ))

    # --- Headline (Table 5.1 equivalent) row per budgeted arm = best F1
    # across the swept budgets, exactly mirroring how the archived
    # cost_performance_summary.csv row was chosen. ---
    best_dt_budget = max(dt_results, key=lambda b: dt_results[b].metrics["f1_score"])
    best_rl_budget = max(rl_results, key=lambda b: rl_results[b].metrics["f1_score"])
    best_dt, best_rl = dt_results[best_dt_budget], rl_results[best_rl_budget]
    best_rl_controller = rl_controllers[best_rl_budget]
    s_best = best_rl_controller.summary()

    main_rows.append(_metrics_row(
        "Drift-triggered AL hybrid (strict-causal)", best_dt.feedback_mode, best_dt.metrics, best_dt_budget,
        best_dt.label_query_percentage, best_dt.drift_count, best_dt.extra.get("unsupervised_drift_count", 0),
    ))
    main_rows.append(_metrics_row(
        "RL-guided hybrid (strict-causal)", best_rl.feedback_mode, best_rl.metrics, best_rl_budget,
        best_rl.label_query_percentage, best_rl.drift_count, best_rl.extra.get("unsupervised_drift_count", 0),
        rl_steps=s_best["n_steps"], total_reward=s_best["total_reward"],
    ))

    # --- Explain the archived run's 4.4675% figure from its own on-disk
    # artifacts (task E), read live rather than hardcoded. ---
    archived_dir = PROJECT_ROOT / "results_ciciot2023_balanced_test" / "tables"
    archived_explainer = None
    try:
        archived_hybrid = pd.read_csv(archived_dir / "rl_hybrid_metrics.csv")
        archived_rl_summary = pd.read_csv(archived_dir / "rl_controller_summary.csv")
        row = archived_hybrid[
            (archived_hybrid["arm"] == "rl_guided_hybrid") & (archived_hybrid["label_budget_fraction"] == 0.1)
        ].iloc[0]
        summ = archived_rl_summary[archived_rl_summary["label_budget_fraction"] == 0.1].iloc[0]
        archived_n = int(row["n_samples"])  # evaluated-stream n (excludes warm-up)
        # n_samples in rl_hybrid_metrics.csv is the *evaluated* stream size
        # (n - warmup); the label_query_percentage denominator is the FULL
        # stream (n), i.e. archived_n + warmup_labels.
        query_count = int(summ["query_count"])
        reported_pct = float(row["label_query_percentage"])
        full_n = round(100.0 * query_count / reported_pct) if reported_pct else None
        # Solve warmup_labels s.t. (warmup + query_count) / (archived_n + warmup) == reported_pct/100
        # -> warmup = (reported_pct/100 * archived_n - query_count) / (1 - reported_pct/100)
        if reported_pct and reported_pct < 100:
            frac = reported_pct / 100.0
            warmup_est = (frac * archived_n - query_count) / (1 - frac)
        else:
            warmup_est = None
        archived_explainer = {
            "evaluated_stream_n_samples": archived_n,
            "controller_query_count": query_count,
            "action_query_and_update": int(summ["action_query_and_update"]),
            "reported_label_query_percentage": reported_pct,
            "implied_full_stream_n": (archived_n + round(warmup_est)) if warmup_est is not None else None,
            "implied_warmup_labels": round(warmup_est) if warmup_est is not None else None,
            "reconstruction": (
                f"{reported_pct}% = 100 * ({round(warmup_est) if warmup_est is not None else '?'} warm-up "
                f"+ {query_count} controller-driven queried labels) / "
                f"{(archived_n + round(warmup_est)) if warmup_est is not None else '?'} total stream samples"
            ),
        }
    except Exception as exc:  # pragma: no cover - archived dir is optional context, not a hard dependency
        logger.warning("Could not read archived results for the 4.4675%% explainer: %s", exc)

    # --- Write outputs ---
    main_df = pd.DataFrame(main_rows)
    main_df = add_synthetic_flag(main_df, is_synthetic)
    main_path = dirs["tables"] / "strict_causal_main_results.csv"
    main_df.to_csv(main_path, index=False)
    logger.info("Saved %s", main_path)

    budget_sweep_df = pd.DataFrame(budget_sweep_rows)
    budget_sweep_df = add_synthetic_flag(budget_sweep_df, is_synthetic)
    budget_sweep_df.to_csv(dirs["tables"] / "strict_causal_budget_sweep.csv", index=False)

    accounting_df = pd.DataFrame(accounting_rows)
    accounting_df = add_synthetic_flag(accounting_df, is_synthetic)
    accounting_path = dirs["tables"] / "strict_causal_label_accounting.csv"
    accounting_df.to_csv(accounting_path, index=False)
    logger.info("Saved %s", accounting_path)

    # --- Per-instance trace sample (task C/G): all "label revealed" rows
    # (queried or warm-up) plus a systematic 1-in-10 sample of the "label
    # hidden" rows, for the headline best-F1 RL budget, capped to keep the
    # file inspectable. ---
    best_trace = rl_traces[best_rl_budget].copy()
    best_trace["label_budget_fraction"] = best_rl_budget
    revealed_mask = best_trace["queried"] | best_trace["warmup_label"]
    hidden_sample = best_trace[~revealed_mask].iloc[::10]
    trace_sample = pd.concat([best_trace[revealed_mask], hidden_sample]).sort_values("stream_index")
    if len(trace_sample) > 8000:
        trace_sample = trace_sample.iloc[:8000]
    trace_sample_path = dirs["tables"] / "strict_causal_protocol_trace_sample.csv"
    trace_sample.to_csv(trace_sample_path, index=False)
    logger.info("Saved %s (%d rows, from the budget=%s RL-guided run)", trace_sample_path, len(trace_sample), best_rl_budget)

    return {
        "main_df": main_df,
        "budget_sweep_df": budget_sweep_df,
        "accounting_df": accounting_df,
        "archived_explainer": archived_explainer,
        "best_dt_budget": best_dt_budget,
        "best_rl_budget": best_rl_budget,
        "dirs": dirs,
    }


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, None, None, args.output_dir, None, None, None)
    out = run(cfg, budgets_override=args.budgets)
    print("\n=== Strict-causal main results (Table 5.1 equivalent) ===")
    print(out["main_df"][[
        "method", "f1_score", "macro_f1", "false_positive_rate", "label_query_percentage",
        "label_budget_fraction", "drift_count", "unsupervised_drift_count",
    ]].to_string(index=False))
    if out["archived_explainer"]:
        print("\n=== Archived run's 4.4675% label-query figure, reconstructed ===")
        for k, v in out["archived_explainer"].items():
            print(f"  {k}: {v}")
