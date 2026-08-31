# Strict-Causal Protocol Audit Report

Branch: `strict-causal-protocol`. Archived results are untouched. New outputs live in
`results_ciciot2023_strict_causal/`, produced by `scripts/run_strict_causal_experiment.py`
from the exact same processed data and seed (42) as the archived
`results_ciciot2023_balanced_test/` run.

## 1. What was wrong, precisely

The reviewer's concern #2 (label leakage in the causal protocol) is **confirmed**. In the
archived `hybrid_ids.run_rl_guided_hybrid` (left unmodified — see it for the original code),
for every stream sample:

- The classifier update was correctly gated on `queried` (no leak there).
- The drift detector's error-indicator update, the reward/Q-table update, and the
  `recent_f1`/`recent_fpr` rolling-window update (used as part of the RL controller's own
  **state**) all used the true label **unconditionally**, regardless of whether that sample's
  label had actually been queried.

Net effect: the reported "4.47% of labels queried" describes only the classifier's label cost.
The drift detector and the RL controller's decision-making machinery saw 100% of labels the
entire time. The archived F1=0.9383 result is not a genuine budget-constrained result.

## 2. What the fix does

Two new, additive functions implement a genuinely causal protocol. The archived functions are
untouched, so old results remain reproducible:

- `hybrid_ids.run_rl_guided_hybrid_strict_causal` (new) — the RL-guided hybrid arm.
- `hybrid_ids.run_drift_triggered_al_hybrid_strict_causal` (new), backed by
  `stream_utils.run_budgeted_stream(strict_causal=True)` (new parameter, default `False` so the
  archived call sites are unaffected) — the drift-triggered active-learning baseline.

Rules enforced (see `run_rl_guided_hybrid_strict_causal`'s docstring for the full statement):

1. Evaluator computes final metrics from every true label, but only after the run — the
   algorithm itself never sees this.
2. `y_t` for an unqueried, non-warm-up sample is never read by anything except the
   post-hoc evaluator.
3. For unqueried samples: no supervised drift update, no supervised reward, no
   `recent_f1`/`recent_fpr` update from the hidden label, no Q-table update *using* the hidden
   label, no classifier update.
4. Classifier update remains allowed only when queried (this was already correct).
5. Reward/Q-table updates use the true label only when it was revealed this step.
6. When no label is revealed, a documented zero/neutral proxy reward
   (`RLController.compute_proxy_reward()`, always `0.0`, never touches `true_label`) is used
   instead, and a normal Q-learning transition is still recorded with it — so Q-learning
   bookkeeping (visitation, epsilon decay) stays intact across every step.
7. Drift detection for unqueried samples uses only an unsupervised signal: prediction
   uncertainty, fed into a second, independent `DriftDetector` instance that runs on every
   step regardless of query decisions.
8. Supervised drift updates (fed the true-label error indicator) run only on queried or
   warm-up steps.

The "drift recently" flag consulted by the RL state and by `ACTION_UPDATE_IF_DRIFT` is the OR
of the supervised and unsupervised detectors' recency windows, so drift-awareness doesn't
silently collapse to "only aware when we happened to query."

Per-instance audit columns (`stream_index`, `routed_to_rule_layer`, `routed_to_ml_layer`,
`y_true_available_to_algorithm`, `queried`, `warmup_label`, `action`, `classifier_updated`,
`q_table_updated`, `supervised_reward_used`, `unsupervised_or_zero_reward_used`,
`supervised_drift_updated`, `unsupervised_drift_updated`, `recent_f1_fpr_updated`,
`prediction`, `final_metric_label_available_to_evaluator`) are emitted for every sample and
saved (a representative sample) to `strict_causal_protocol_trace_sample.csv`.

## 3. Invariant tests (`tests/test_strict_causal.py`, 8 tests, all passing)

Directly checks the rules above on real trace output from a run:

- `test_unqueried_nonwarmup_rows_never_see_the_label` — for every row where `queried=False`
  and `warmup_label=False`: `y_true_available_to_algorithm`, `classifier_updated`,
  `supervised_reward_used`, `supervised_drift_updated`, `recent_f1_fpr_updated` are all `False`.
- `test_q_table_updates_on_unqueried_rows_use_only_proxy_reward` — Q-table updates *do* still
  happen on those rows (rule 6), but always with `unsupervised_or_zero_reward_used=True` and
  never with `supervised_reward_used=True`.
- `test_queried_rows_do_use_the_label` — the mirror-image sanity check, so the above isn't
  vacuously true.
- `test_evaluator_always_has_the_final_label` — the evaluator-side flag is unconditionally
  `True`.
- `test_label_query_percentage_matches_trace_counts` — the reported percentage reconciles
  exactly with the trace.
- Two further tests exercise `run_budgeted_stream(strict_causal=True)` and
  `run_drift_triggered_al_hybrid_strict_causal` directly.

Existing suite (`test_sanity.py`, `test_cross_dataset.py`, 21 tests) still passes unmodified.
**Refactor-safety check**: before adding new code, the duplicated split-construction logic in
`run_hybrid_comparison`/`run_rl_reward_tuning` was extracted into
`build_rule_train_stream_split()`. Re-running the archived `run_rl_hybrid_experiment.py`
end-to-end after this extraction reproduced `rl_hybrid_metrics.csv` **bit-for-bit** (verified
by diffing against the archived table; total numeric diff = 0.0) — the refactor changed
nothing about the archived numbers.

## 4. Results: archived (oracle-feedback) vs. strict-causal

Both tables below share the same 9,992-row rule/static training prefix and 40,000-row
evaluation stream (same seed=42, same split function).

| Method | Archived F1 | Archived FPR | Archived label-query % | Strict-causal F1 | Strict-causal FPR | Strict-causal label-query % |
|---|---|---|---|---|---|---|
| Static ML only | 0.8849 | 0.0648 | 0% | 0.8849 (identical — offline, unaffected) | 0.0648 | 0% |
| Adaptive ML only | 0.9833 | 0.0255 | 100% | 0.9833 (identical — full-feedback by definition) | 0.0255 | 100% |
| Drift-aware adaptive ML | 0.9993 | 0.00075 | 100% | 0.9993 (identical) | 0.00075 | 100% |
| Rule + ML hybrid | 0.9407 | 0.0223 | 0%\* | 0.9407 (identical) | 0.0223 | 0%\* |
| Drift-triggered AL hybrid | 0.9379 @ 25% budget | 0.00086 | 25% | **0.9379 @ 25% budget (essentially reproduced)** | 0.00086 | 25% |
| **RL-guided hybrid** | **0.9383 @ nominal 10% budget, 4.4675% realized** | 0.0375 | **4.4675%** | **0.7161 @ nominal 10% budget, 10.00% realized** | 0.00303 | 10.00% |

\* "0%" here means no active *query* decision — it trains on 100% of rule-unmatched traffic,
a separate, already-disclosed quantity (`ml_layer_coverage`), not a label budget.

The four full-feedback / offline arms (Static ML, Adaptive ML, Drift-aware adaptive ML, Rule+ML
hybrid) are **numerically unaffected** by the fix — they never had a per-step query-gating
concept in the first place, so this confirms the fix is scoped correctly and didn't disturb
anything it shouldn't.

The **drift-triggered active-learning baseline is essentially reproduced**: at its 25%-budget
headline, F1 is unchanged to 4 decimal places (0.937882 archived vs. 0.937882 strict-causal),
because its query policy is uncertainty/drift-triggered top-k selection, not reward-driven —
the only change for this arm is that its own drift-detector bookkeeping for unqueried samples
now uses the unsupervised uncertainty signal, which in this particular run happened not to
change which samples were selected.

**The RL-guided hybrid does not survive the fix at its reported operating point.** At a
comparable (in fact slightly *larger*) 10% nominal budget, strict-causal F1 is **0.7161**, not
0.9383 — a drop of ~22 F1 points. Full budget-sweep detail:

| Nominal budget | RL-guided F1 (strict) | RL-guided realized query % | Drift-triggered AL F1 (strict) | Drift-triggered AL query % |
|---|---|---|---|---|
| 0.01 | 0.7104 | 1.00% | 0.7104 | 1.00% |
| 0.05 | 0.7160 | 5.00% | 0.7094 | 5.00% |
| 0.10 | 0.7161 | 10.00% | **0.8256** | 10.00% |
| 0.25 | 0.7161 | 25.00% | **0.9379** | 25.00% |
| 1.00 | **0.9356** | 74.06% (RL under-spends its own budget) | 0.9353 | 100.00% |

Two findings stand out:

1. **RL-guided F1 is flat (~0.71–0.72) from a 1% to a 25% label budget.** Extra budget in this
   range buys the strict-causal RL controller almost nothing — a strong sign that, once its
   reward signal is honest (mostly the `0.0` proxy, since queries are relatively rare per
   step), the tabular Q-learning policy is not learning an effective query-targeting strategy.
2. **The simpler, non-RL drift-triggered/uncertainty active-learning baseline dominates the
   RL-guided arm at every matched budget from 10% to 25%**, and ties it at 1%. RL-guided only
   catches up once it is allowed to spend the vast majority of the stream's labels (nominal
   budget 1.0 → 74.06% realized), at which point it matches (not exceeds) the baseline.
3. At nominal budget 1.0, RL-guided under-spent its own "unlimited" budget (74.06% vs. the
   drift-triggered arm's 100%) while matching its F1 — the *only* place in this sweep where the
   RL controller shows a genuine, honestly-earned label-efficiency advantage, and it is a
   ~26% saving, not the ~95% saving (4.47% vs. 100%) originally claimed.

## 5. The 4.4675% figure, reconstructed exactly

Read live from the archived `rl_hybrid_metrics.csv` / `rl_controller_summary.csv` (not
hardcoded) by the new script:

```
4.4675% = 100 * (400 warm-up + 1387 controller-driven queried labels) / 40000 total stream samples
```

`400` is the unconditional warm-up seed set (`compute_warmup_size`), `1387` is
`RLController.query_count` (steps where `compute_reward` was called with `queried_label=True`),
and `40000` is the full evaluation stream **including** the warm-up rows — not the 39,994 or
34,116 the reviewer guessed, and not previously stated explicitly anywhere in the archived
outputs. The 1387 vs. 1276 (`action_query_and_update`) gap is the 111 extra queries triggered
by `ACTION_UPDATE_IF_DRIFT` when a drift flag was active.

## 6. Which thesis claims are still valid

- The four full-feedback/offline ablation rows (Static ML, Adaptive ML, Drift-aware adaptive
  ML, Rule+ML hybrid) and their numbers — unaffected by this fix, safe to keep as-is.
- The drift-triggered active-learning baseline's reported numbers — reproduced almost exactly
  under strict causal accounting, safe to keep as-is.
- The general qualitative lesson that adaptation helps under drift but can raise false-alarm
  burden — still directionally supported by the full-feedback arms.
- The existence and severity of the label-leakage issue itself, and the fact that fixing it is
  tractable without destabilizing the rest of the pipeline (see the bit-identical refactor
  check in §3).

## 7. Which thesis claims must be softened or removed

- **Remove or clearly re-label** "RL-guided hybrid improves F1 from 0.8849 to 0.9383 while
  querying only 4.47% of labels." This is an oracle-feedback simulation result, not a
  budget-constrained one. If kept at all, it must be explicitly captioned as an *upper-bound /
  non-causal simulation*, with the strict-causal number (F1≈0.716 at a comparable/larger 10%
  budget) reported alongside it.
- **Revise the core comparative claim that the RL-guided hybrid is the strongest proposed
  method.** Under strict-causal accounting, the simpler drift-triggered active-learning
  baseline matches or clearly beats it at every practical budget (5–25%); RL-guided only
  reaches parity near a ~74–100% label rate. The thesis should either (a) reframe RL-guided as
  a competitive-but-not-dominant alternative, (b) investigate why the Q-learning policy fails
  to exploit smaller budgets once its reward signal is honest (e.g., reward sparsity, state
  discretization, exploration schedule) and report that as a limitation/future-work item, or
  (c) both.
- **Any claim that the RL controller "learns an efficient query-allocation policy"** needs
  direct evidence under strict-causal accounting; the flat F1-vs-budget curve above does not
  currently support it.
- **State explicitly, wherever 4.47% is quoted,** that it is
  `(warm-up + controller queries) / full stream including warm-up`, with the exact
  reconstruction in §5, so the figure is independently checkable.
- The drift-event-count discrepancy (11 vs. 18) is a separate reporting-aggregation issue
  (unrelated to this fix — the "18" is a row-count artifact, not a sum of drift events) and
  still needs its own correction in the thesis text; not addressed by this branch.

## 8. Files produced

- `results_ciciot2023_strict_causal/tables/strict_causal_main_results.csv` — Table 5.1
  equivalent (one row per method; budgeted arms use their best-F1 swept budget).
- `results_ciciot2023_strict_causal/tables/strict_causal_budget_sweep.csv` — every swept
  budget for both budgeted arms.
- `results_ciciot2023_strict_causal/tables/strict_causal_label_accounting.csv` — the full
  counter breakdown from task E (warm-up/query/update-if-drift/classifier-update counts,
  routing counts, supervised vs. unsupervised drift counts), per budget.
- `results_ciciot2023_strict_causal/tables/strict_causal_protocol_trace_sample.csv` — full
  per-instance audit trace (all "label revealed" rows + a systematic 1-in-10 sample of "label
  hidden" rows) for the headline RL-guided run.
- `results_ciciot2023_strict_causal/logs/rl_drift_ids.log` — full run log.
- `tests/test_strict_causal.py` — invariant tests (passing).

No file under `results_ciciot2023_balanced_test/` (or any other archived results directory)
was modified.
