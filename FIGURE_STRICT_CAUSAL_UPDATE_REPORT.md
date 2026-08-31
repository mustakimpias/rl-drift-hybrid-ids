# Figure & Table Strict-Causal Update Report

Scope: Chapter 5 (`thesis_latex/chapters/chapter05/chap05.tex`) of the Overleaf thesis project.
All new figures and the new main-results table are generated exclusively from
`results_ciciot2023_strict_causal/` (main experiment tables + diagnostics) — never from the
archived `results_ciciot2023_balanced_test/` pipeline. No archived result file was modified.

## 1. Figures identified as showing archived/oracle-feedback numbers

Six Chapter 5 figures were sourced from the archived pipeline and needed attention:
`static_vs_proposed_f1.png`, `static_vs_proposed_fpr.png`, `label_saving_vs_performance.png`,
`label_budget_vs_f1.png`, `label_budget_vs_fpr.png`, `rl_action_distribution.png`. (The
unseen-attack and cross-dataset figures were left out of scope — they are not part of the
6.0.9383/95.53%/4.47% headline story and are already captioned in the surrounding text as
pending re-verification from a prior revision pass.)

## 2. Figures replaced (all six — none needed removal)

Every flagged figure could be safely regenerated with real strict-causal data, so none were
removed or demoted to an appendix. All six are new files under
`thesis_latex/thesis_overleaf_assets/figures/strict_causal/` (PNG @ 300dpi + PDF), produced by
`scripts/generate_strict_causal_figures.py`:

| Old figure (archived pipeline) | New figure (strict-causal) | Exact data source |
|---|---|---|
| `main/static_vs_proposed_f1.png` | `strict_causal/static_vs_proposed_f1_strict_causal.png` | `results_ciciot2023_strict_causal/tables/strict_causal_main_results.csv` (rows: Static ML only, Drift-triggered AL hybrid) |
| `main/static_vs_proposed_fpr.png` | `strict_causal/static_vs_proposed_fpr_strict_causal.png` | same CSV, `false_positive_rate` column |
| `main/label_saving_vs_performance.png` | `strict_causal/label_saving_vs_performance_strict_causal.png` | same CSV, all 7 rows (`label_query_percentage` vs. `f1_score`) |
| `budget/label_budget_vs_f1.png` | `strict_causal/label_budget_vs_f1_strict_causal.png` | `results_ciciot2023_strict_causal/diagnostics/strict_causal_budget_comparison.csv` (methods: Drift-triggered AL hybrid, Uncertainty-only AL, RL-guided hybrid; all 5 budgets) |
| `budget/label_budget_vs_fpr.png` | `strict_causal/label_budget_vs_fpr_strict_causal.png` | same diagnostics CSV, `fpr` column |
| `rl/rl_action_distribution.png` | `strict_causal/rl_action_distribution_strict_causal.png` | `results_ciciot2023_strict_causal/diagnostics/strict_causal_rl_action_by_budget.csv` (stacked by budget, 4 actions) |

The old archived figure files were **not deleted** (they remain under
`thesis_overleaf_assets/figures/{main,budget,rl}/` and still back Table
`tab:archived_oracle_feedback_result`'s discussion in Section 5.2's diagnostic subsection); they
are simply no longer `\includegraphics`-referenced from Chapter 5's headline discussion.

## 3. Figures moved to diagnostic discussion

None needed moving — the `rl_action_distribution_strict_causal.png` figure was already
positioned inside the RL exploratory-extension discussion (Section "RL Reward Tuning and
Controller Behaviour (Exploratory Extension)", which feeds the diagnostic negative-result
section) and stays there; it is now simply the strict-causal version of the same figure rather
than a new relocation.

## 4. Chapter 5 text/caption changes accompanying the figure swaps

- `static_vs_proposed_f1`/`fpr`: captions rewritten to state both methods are evaluated under
  the strict-causal protocol; the "predates the strict-causal correction... manual follow-up"
  caveat sentences were removed since the figures are now corrected.
- `label_saving_vs_performance`: caption rewritten to describe the 7-method scatter and
  explicitly calls out the ~3× label-cost gap between the proposed method (25%) and the
  RL-guided controller's best case (74.06%).
- `label_budget_vs_f1`/`fpr`: the strict-causal figures are now introduced as the section's
  headline evidence; the archived `table_budget_results.tex` \input was relocated (not
  duplicated — the original stray `\input` before the figures was removed) to sit after the
  strict-causal figure as explicit historical/superseded context, with prose stating the
  archived RL-guided number (0.9383 at nominal 10%) is superseded by the new figure.
- `rl_action_distribution`: caption and surrounding paragraph rewritten around the new
  per-budget stacked chart; the archived single-operating-point numbers (34,116 steps, 1,387
  events, reward 84,274.8) are kept only as an explicitly-labelled historical reference point.
- Chapter 5's closing summary paragraph (`sec:results_summary`) no longer lists these six
  figures as a "manual follow-up" item; it now states they are pipeline-generated and points to
  this report for the full mapping.

## 5. Verification that no figure visually claims the retracted numbers

Checked directly against the rendered PNGs and their captions:

- No figure or caption states RL-guided hybrid F1 = 0.9383 as a strict-causal result — the only
  place 0.9383 still appears is Table `tab:archived_oracle_feedback_result`'s explicitly-labelled
  "Archived (oracle-feedback simulation)" row and the surrounding prose, not a figure.
- No figure or caption states a 95.53% label saving — the new `label_saving_vs_performance`
  figure and its caption state the real strict-causal figure (75% saving at the proposed
  method's 25% budget).
- No figure or caption states "4.47% label-efficient causal feedback" as an achieved result —
  the new budget-sweep and action-distribution figures show the real strict-causal query
  percentages (1/5/10/25/100% nominal) with no reference to 4.47%.

## 6. Main-results table: automated generation

Per the accompanying task, the Chapter 5 headline table is now generated, not hand-typed:

- `scripts/generate_strict_causal_thesis_table.py` reads
  `results_ciciot2023_strict_causal/tables/strict_causal_main_results.csv` (6 of 7 rows) and
  `results_ciciot2023_strict_causal/diagnostics/strict_causal_budget_comparison.csv` (the
  uncertainty-only active-learning row, a diagnostics-only arm at the matched 25% budget), and
  writes:
  - `results_ciciot2023_strict_causal/tables/table_strict_causal_main_results_source.csv` — the
    curated, ordered CSV that is the single traceable source for the table.
  - `thesis_latex/tables/table_strict_causal_main_results.tex` — the `\input`-able LaTeX table,
    with a header comment naming its exact source CSV and the regeneration command.
- Chapter 5 (`sec:balanced_ciciot_results`) now does `\input{tables/table_strict_causal_main_results.tex}`
  in place of the previously hand-typed table with the identical label `tab:strict_causal_main_results`.
- No result value in this table was hand-typed; every number is `pandas`-read from the CSV and
  formatted by the script.

## 7. Regeneration commands

```bash
python scripts/generate_strict_causal_thesis_table.py
python scripts/generate_strict_causal_figures.py
```

Both are idempotent and safe to re-run after any change to
`results_ciciot2023_strict_causal/` — they only ever write into `thesis_latex/tables/`,
`thesis_latex/thesis_overleaf_assets/figures/strict_causal/`, and
`results_ciciot2023_strict_causal/tables/table_strict_causal_main_results_source.csv` (a new,
non-archived file), never into any archived results directory.

## 8. Verification performed

- All six new PNGs were rendered and visually inspected; one label-collision issue (the
  "Rule + ML hybrid" and "Drift-triggered AL hybrid" annotations overlapping in the scatter
  plot) was found and fixed by adjusting per-point label offsets before finalizing.
- Palette: categorical colors used in fixed slot order throughout (blue = proposed method,
  orange = RL-guided/exploratory, aqua = uncertainty-only baseline, violet/gray = full-feedback
  or static baselines, yellow = Rule+ML hybrid), consistent with the dataviz skill's
  fixed-hue-order rule; every chart also carries a legend or direct labels so identity is never
  color-alone.
- Structural check on the full LaTeX project after all edits: brace balance and
  `\begin`/`\end` pairing OK on every `.tex` file; **zero** dangling `\ref`/`\eqref` targets and
  **zero** duplicate `\label`s across `chapters/`, `frontmatter/`, `tables/`,
  `thesis_overleaf_assets/tables/latex/`, and `Main.tex`.
- No local pdfLaTeX distribution is available on this machine (see
  `STRICT_CAUSAL_LATEX_REVISION_REPORT.md` §6); an actual Overleaf/pdfLaTeX compile is still the
  outstanding verification step, same as the prior revision pass.
