#!/usr/bin/env python
"""Generate the Chapter 5 strict-causal main-results LaTeX table directly from
the verified strict-causal result CSVs, so the thesis table is never
hand-typed.

Sources (both under results_ciciot2023_strict_causal/, produced by
scripts/run_strict_causal_experiment.py and scripts/run_strict_causal_diagnostics.py
respectively -- never the archived/oracle-feedback pipeline):
    tables/strict_causal_main_results.csv
    diagnostics/strict_causal_budget_comparison.csv   (for the uncertainty-only
        active-learning row, which is a diagnostics-only arm not part of the
        main experiment script)

Outputs:
    results_ciciot2023_strict_causal/tables/table_strict_causal_main_results_source.csv
        -- the curated, ordered subset of columns/rows actually used in the
        thesis table, so the LaTeX table's provenance is a single traceable CSV.
    thesis_latex/tables/table_strict_causal_main_results.tex
        -- the \\input-able LaTeX table for Chapter 5, generated (never
        hand-typed) from the CSV above.

Usage:
    python scripts/generate_strict_causal_thesis_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

RESULTS_DIR = PROJECT_ROOT / "results_ciciot2023_strict_causal"
THESIS_DIR = PROJECT_ROOT / "thesis_latex"

MAIN_RESULTS_CSV = RESULTS_DIR / "tables" / "strict_causal_main_results.csv"
BUDGET_COMPARISON_CSV = RESULTS_DIR / "diagnostics" / "strict_causal_budget_comparison.csv"
SOURCE_CSV_OUT = RESULTS_DIR / "tables" / "table_strict_causal_main_results_source.csv"
TEX_OUT = THESIS_DIR / "tables" / "table_strict_causal_main_results.tex"

# Display order and display names for the thesis table. Values are pulled
# live from the CSVs below -- nothing here is a result value.
ROW_ORDER = [
    "Static ML only",
    "Adaptive ML only",
    "Drift-aware adaptive ML",
    "Rule + ML hybrid",
    "Uncertainty-only active learning",
    "Drift-triggered AL hybrid (proposed)",
    "RL-guided hybrid (strict-causal)",
]
UPPER_BOUND_METHODS = {"Adaptive ML only", "Drift-aware adaptive ML"}
NO_QUERY_METHODS = {"Static ML only", "Rule + ML hybrid"}
PROPOSED_METHOD_DISPLAY = "Drift-triggered AL hybrid (proposed)"


def _fmt(x: float, decimals: int = 4) -> str:
    return f"{x:.{decimals}f}"


def build_source_dataframe() -> pd.DataFrame:
    main = pd.read_csv(MAIN_RESULTS_CSV)
    budget = pd.read_csv(BUDGET_COMPARISON_CSV)

    rows: list[dict] = []

    def _from_main(method_csv_name: str, display_name: str, labels_used_note: str | None = None) -> None:
        r = main[main["method"] == method_csv_name].iloc[0]
        n_stream = int(r["n_samples"]) if pd.notna(r["n_samples"]) else 40000
        lq_pct = float(r["label_query_percentage"])
        labels_used = labels_used_note or f"{round(lq_pct / 100.0 * 40000):,} of 40,000 ({lq_pct:.2f}\\%)"
        rows.append({
            "method": display_name,
            "f1_score": float(r["f1_score"]),
            "macro_f1": float(r["macro_f1"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "false_positive_rate": float(r["false_positive_rate"]),
            "mcc": float(r["mcc"]),
            "label_query_percentage": lq_pct,
            "labels_used_display": labels_used,
            "label_budget_fraction": r.get("label_budget_fraction"),
        })

    _from_main("Static ML only", "Static ML only", "0 (offline, no query decision)")
    _from_main("Adaptive ML only", "Adaptive ML only", "40,000 of 40,000 (100.00\\%)")
    _from_main("Drift-aware adaptive ML", "Drift-aware adaptive ML", "40,000 of 40,000 (100.00\\%)")
    _from_main("Rule + ML hybrid", "Rule + ML hybrid", "n/a (no query decision)")

    # Uncertainty-only active learning: diagnostics-only arm, at the same 25%
    # budget as the proposed method, for a directly matched comparison.
    unc = budget[(budget["method"] == "Uncertainty-only AL (strict-causal)") & (budget["budget"] == 0.25)].iloc[0]
    rows.append({
        "method": "Uncertainty-only active learning",
        "f1_score": float(unc["f1"]),
        "macro_f1": float(unc["macro_f1"]),
        "precision": float(unc["precision"]),
        "recall": float(unc["recall"]),
        "false_positive_rate": float(unc["fpr"]),
        "mcc": float(unc["mcc"]),
        "label_query_percentage": float(unc["label_query_percentage"]),
        "labels_used_display": f"{int(unc['total_labels_revealed']):,} of 40,000 (25.00\\%)",
        "label_budget_fraction": 0.25,
    })

    _from_main("Drift-triggered AL hybrid (strict-causal)", PROPOSED_METHOD_DISPLAY)
    _from_main("RL-guided hybrid (strict-causal)", "RL-guided hybrid (strict-causal)")

    df = pd.DataFrame(rows)
    order_index = {name: i for i, name in enumerate(ROW_ORDER)}
    df["_order"] = df["method"].map(order_index)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return df


def render_latex_table(df: pd.DataFrame) -> str:
    lines = []
    lines.append("% AUTO-GENERATED -- do not hand-edit.")
    lines.append("% Source: scripts/generate_strict_causal_thesis_table.py")
    lines.append(f"% Source CSV: results_ciciot2023_strict_causal/tables/{SOURCE_CSV_OUT.name}")
    lines.append("% Underlying pipeline outputs: results_ciciot2023_strict_causal/tables/strict_causal_main_results.csv")
    lines.append("%                              results_ciciot2023_strict_causal/diagnostics/strict_causal_budget_comparison.csv")
    lines.append("% Regenerate with: python scripts/generate_strict_causal_thesis_table.py")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Main strict-causal result on the balanced CICIoT2023 evaluation stream "
        r"(40,000 samples). The proposed method and the uncertainty-only active-learning "
        r"baseline are evaluated at a 25\% label budget; full-feedback methods use 100\% of "
        r"labels by definition and are upper-bound references, not budgeted results. "
        r"Generated automatically from the strict-causal result CSVs (see source comment above).}"
    )
    lines.append(r"\label{tab:strict_causal_main_results}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & F1 & Macro-F1 & Precision & Recall & FPR & MCC & Labels used \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        method = r["method"]
        bold = method == PROPOSED_METHOD_DISPLAY
        def cell(v: str) -> str:
            return r"\textbf{" + v + "}" if bold else v
        method_cell = r"\textbf{" + method + "}" if bold else method
        fpr_str = f"{r['false_positive_rate']:.5f}" if r["false_positive_rate"] < 0.001 else _fmt(r["false_positive_rate"])
        row_cells = [
            method_cell,
            cell(_fmt(r["f1_score"])),
            cell(_fmt(r["macro_f1"])),
            cell(_fmt(r["precision"])),
            cell(_fmt(r["recall"])),
            cell(fpr_str),
            cell(_fmt(r["mcc"])),
            cell(r["labels_used_display"]),
        ]
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    df = build_source_dataframe()
    SOURCE_CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOURCE_CSV_OUT, index=False)
    print(f"Saved {SOURCE_CSV_OUT}")

    tex = render_latex_table(df)
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text(tex, encoding="utf-8")
    print(f"Saved {TEX_OUT}")


if __name__ == "__main__":
    main()
