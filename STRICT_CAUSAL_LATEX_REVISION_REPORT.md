# Strict-Causal LaTeX Revision Report

Source project: the Overleaf export `pias_thesis_overleaf_editable_fast_toc_fixed.zip` (found in
`C:\Users\HP\Downloads\`), extracted to `rl_drift_hybrid_ids/thesis_latex/` for editing (this
directory is not part of the git repository proper — the thesis text lives in a separate
Overleaf project, not in `rl_drift_hybrid_ids`). All numbers used below come from
`STRICT_CAUSAL_FINAL_CLAIMS.md`; none were invented for this pass. No archived results directory
and no code file were touched in this task — only the thesis LaTeX source.

## 1. Files edited

- `frontmatter/abstract.tex` — full rewrite.
- `chapters/chapter01/chap01.tex` — research questions, research contributions, workflow figure caption, thesis-organization paragraph.
- `chapters/chapter03/chap03.tex` — section intro, framework overview section (retitled), architecture figure caption, causal-protocol workflow table (fixed a real leak-shaped bug in step 6's wording), new strict-causal pseudocode block, active-learning/RL-controller section framing, ablation list, chapter summary.
- `chapters/chapter04/chap04.tex` — intro, experimental-design item 1, new strict-causal label-accounting table alongside the archived flow-count table, compared-methods list, validity-considerations paragraph, chapter summary.
- `chapters/chapter05/chap05.tex` — largest revision: new strict-causal main-results table and discussion, new "RL-Guided Controller: Diagnostic Negative Result" section with root-cause analysis, budget-wise section caveats, reward-tuning section (independence caveat strengthened), unseen-attack and cross-dataset section caveats, related-work table cell fixes, overall-discussion rewrite, threats-to-validity rewrite, chapter summary rewrite.
- `chapters/chapter06/chap06.tex` — intro, balanced-CICIoT2023 findings, RL-controller-behaviour subsection (retitled and rewritten as a diagnosed negative result), cross-dataset subsection caveat, **realigned "Answers to the Research Questions" to all six RQs from the revised Chapter 1** (previously only four answers existed for what were five RQs — a pre-existing misalignment, corrected as part of this pass), main-contributions section (rewritten, six contributions instead of five), practical-implications paragraph and the `extbf` bug fix, limitations (RL item), future-work (RL redesign direction, expanded to the four concrete changes requested), final remarks.

Not edited (checked, no changes needed): `chapters/chapter02/chap02.tex` (literature review — only generic, non-headline mentions of RL/defensibility), `chapters/appendices/app0A.tex` and `app0B.tex` (AutoFIM appendix and reproducibility appendix — already accurately describe the archive without overclaiming), `frontmatter/title.tex`, `certification.tex`, `acknowledgment.tex`, `nomenclature.tex` (no risky content; the official title was left unchanged as instructed).

## 2. Old claims removed

- "RL-guided hybrid improves F1 from 0.8849 to 0.9383 under label-efficient causal feedback" — removed as a standalone claim everywhere; retained only inside explicit oracle-feedback/diagnostic framing (Chapter 5's Table~tab:archived_oracle_feedback_result and surrounding text).
- The "95.53% label saving" headline (including the broken `\tab extbf{...}` paragraph in Chapter 6 that stated it) — removed and replaced with the verified 75% figure for the proposed method.
- "RL-guided hybrid IDS: the proposed method" (Chapter 3 ablation list, Chapter 4 compared-methods list, Chapter 1 contributions, Chapter 6 contributions) — removed everywhere; RL is now labelled "(exploratory extension)" throughout.
- Implicit claims that the archived run's causal protocol was already correctly implemented (Chapter 3's workflow table literally instructed "update the drift detector using the post-prediction error signal available in offline evaluation" for every step) — corrected to state the actual, gated behaviour.
- The reward-tuning sweep's implicit framing as an acceptable independent test (Chapter 5, Section "RL Reward Tuning") — replaced with an explicit "this sweep is not an independent evaluation" statement.
- Two instances of "the proposed method" being applied to RL-guided cross-dataset/unseen-attack numbers (Chapter 5) — corrected to "the RL-guided hybrid (earlier, non-strict-causal pipeline)."
- The RQ/contribution misalignment already flagged by the earlier external review (Chapter 6 answered 4 questions for what were 5 RQs in Chapter 1) — corrected by realigning to Chapter 1's (now 6) RQs.

## 3. New claims inserted

- The strict-causal main result: F1 = 0.9379, macro-F1 = 0.9413, precision = 0.9990, recall = 0.8838, FPR = 0.00086, MCC = 0.8889, at a 25% label budget (10,000/40,000 labels), for the newly named main method, the **Strict-Causal Drift-Triggered Active Learning Hybrid IDS**.
- The RL-guided controller's diagnosed negative result: F1 = 0.7161 at a matched 25% budget; F1 = 0.9356 only at ~74.06% realized label spend; root causes (state-discretization collapse to 1–2 dominant states out of 162 possible, a non-ranking per-sample query mechanism, reward sparsity as a secondary factor, threshold adjustment ruled out by ablation with ≤0.24 F1-point effect).
- The exact reconstruction of the archived "4.4675%" figure: (400 warm-up + 1387 controller queries) / 40,000 total stream samples, with the caveat that the drift detector and reward computation used 100% of labels regardless.
- A new strict-causal label-accounting table (Chapter 4): 40,000 total stream samples, 400 warm-up labels, 34,116 ML/active-learning decision steps, 5,484 rule-layer-handled samples, 10,000 labels revealed at the 25% budget.
- An explicit strict-causal protocol pseudocode block (Chapter 3), implemented as a captioned `tabbing` block inside a `figure` float (no `algorithm`/`algorithmic` package is loaded in `Main.tex`, so this is not a numbered "Algorithm" float — see §6).
- A rewritten Chapter 3 workflow-table step 6, correctly distinguishing supervised (queried-only) from unsupervised (always-on) drift updates.
- Six realigned research-question answers in Chapter 6, matching the six (previously five) research questions in Chapter 1.

## 4. Tables/figures needing manual replacement

These are the honest limits of a text-only revision pass — no plotting or table-regeneration was performed, per the task's scope (no code changes, no new experiment runs):

- **Figures generated from the archived, oracle-feedback pipeline** (all in Chapter 5, all now captioned as archived/pending regeneration): `static_vs_proposed_f1.png`, `static_vs_proposed_fpr.png`, `label_saving_vs_performance.png`, `label_budget_vs_f1.png`, `label_budget_vs_fpr.png`, `rl_action_distribution.png`. Regenerating these from `results_ciciot2023_strict_causal/tables/*.csv` and `results_ciciot2023_strict_causal/diagnostics/*.csv` would let Chapter 5 show the corrected numbers visually, not just in text.
- **`table_main_balanced_results.tex`, `table_budget_results.tex`, `table_rl_reward_tuning_best.tex`** (all `\input` from `thesis_overleaf_assets/tables/latex/`) — these files themselves were **not modified**; they still contain archived-pipeline numbers and are now explicitly captioned/discussed as archived. The new strict-causal main-results table in Chapter 5 (`tab:strict_causal_main_results`) was **hand-authored directly in `chap05.tex`** from the vetted numbers in `STRICT_CAUSAL_FINAL_CLAIMS.md`, not regenerated by the project's own CSV-to-LaTeX table pipeline. For full traceability, a follow-up should add a proper `table_strict_causal_main_results.tex` generator to the Python pipeline (mirroring the existing `table_main_balanced_results.tex` generation) so the thesis table is produced the same way as every other pipeline-generated table.
- **Unseen-attack and cross-dataset figures/tables** (`unseen_attack_comparison.png`, the four `cross_dataset_*` figures, `table_unseen_attack_results.tex`, `table_cross_dataset_results.tex`) — left entirely as-is (correctly, since no strict-causal re-run of these scenarios exists yet); text around them now states they are pending re-verification.

## 5. Unresolved thesis risks

- **Reward-sensitivity tuning is still not independent of the evaluation stream** (reviewer concern #6) — explicitly flagged in the revised text (Chapter 5, Section "RL Reward Tuning") but not fixed; this was out of scope for this task.
- **Unseen-attack-category and cross-dataset results have not been re-run under the strict-causal protocol.** They may or may not hold up the way the balanced in-distribution result did; the text now says so everywhere they are cited, but the underlying numbers are unverified under the corrected protocol.
- **The drift-event-count discrepancy (11 vs. 18) from the original external review is unrelated to this task and remains unaddressed.**
- **Six figures still visually depict the archived, oracle-feedback numbers** (see §4) — the prose is corrected, but a reader skimming figures only would see the old story.
- **No local PDF compile could be performed** (see §6) — a static syntax check was substituted; an actual Overleaf/pdfLaTeX compile is still needed before submission.
- The new pseudocode in Chapter 3 is a captioned `tabbing` figure, not a numbered algorithm float, since no algorithm package is loaded; this is a stylistic, not a correctness, gap.
- `chapters/appendices/app0B.tex` describes only the archived reproducibility record; it could be extended to also point at the new `results_ciciot2023_strict_causal/` artifacts, but this was left untouched to keep the change scope focused on the reframing itself.

## 6. Whether the PDF compiles locally

**No LaTeX distribution (pdfLaTeX, latexmk, MiKTeX, or TeX Live) is installed on this machine** — checked via both the bash `PATH` and Windows `Get-Command`/common install directories; none were found. This project is normally compiled on Overleaf (see `OVERLEAF_COMPILE_NOTES.md` in the project), not locally, which is consistent with this environment having no TeX toolchain.

In place of an actual compile, every `.tex` file in the project (edited and unedited) was run through a structural static check: brace-balance verification and `\begin{...}`/`\end{...}` pair-count verification. **All 14 files, including all six edited chapters and the rewritten abstract, passed both checks with no unbalanced braces or mismatched environments.** This does not guarantee a clean compile (it cannot catch, for example, an undefined command, a bad cross-reference, or a package-specific syntax error), but it rules out the most common source of a broken build introduced by manual editing.

**Recommendation:** the revised project has been repackaged as a zip at
`C:\Users\HP\Downloads\pias_thesis_strict_causal_revision.zip` (same structure as the original
Overleaf export, stale `.aux`/`.toc`/`.out`/`.lof`/`.lot`/`.bbl`/`.ilg`/`.ind` build artifacts
removed so Overleaf regenerates them cleanly, `_private_defense_notes/` excluded as in the
original). Upload this to Overleaf as a new project (or overwrite the existing one) and Recompile
to get a real pdfLaTeX build and catch anything this static check could not.
