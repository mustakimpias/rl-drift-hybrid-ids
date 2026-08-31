# Strict-Causal Final Claims (safe to write in the thesis)

Every claim below is directly traceable to a file under `results_ciciot2023_strict_causal/`
(main experiment or diagnostics). Nothing here is inferred, rounded beyond the source CSV's
own precision, or extrapolated. Use these as the vetted basis for the thesis rewrite — do not
add numbers not listed here without regenerating from the same pipeline.

## Numerical claims (stream: 40,000 samples, 9,992-row rule/static training prefix, seed=42)

1. Static ML only (offline RandomForest, no adaptation): F1 = 0.8849, macro-F1 = 0.8899,
   precision = 0.9288, recall = 0.8451, FPR = 0.0648, MCC = 0.7834. No label-query concept
   (fully offline). Source: `strict_causal_main_results.csv`.

2. Adaptive ML only (full-feedback upper bound — learns from every sample's true label):
   F1 = 0.9833, macro-F1 = 0.9832, precision = 0.9750, recall = 0.9918, FPR = 0.0255,
   MCC = 0.9665, 100% of labels used (40,000/40,000), 6 drift events detected. Source: same.

3. Drift-aware adaptive ML (full-feedback upper bound, resets on detected drift):
   F1 = 0.9993, macro-F1 = 0.9993, precision = 0.9993, recall = 0.9994, FPR = 0.0008,
   MCC = 0.9986, 100% of labels used, 2 drift events detected. Source: same.

4. Rule + ML hybrid (rule layer routes known/confident traffic; ML layer trains on every
   rule-unmatched sample, no label budget): F1 = 0.9407, macro-F1 = 0.9427, precision = 0.9760,
   recall = 0.9079, FPR = 0.0223, MCC = 0.8877. This arm has no query *decision* (0% is not a
   budget being spent efficiently, it means "no active querying, but 86.29% of the stream still
   reaches the ML layer and is fully labeled" — cite `ml_layer_coverage`, not
   `label_query_percentage`, if using this arm as an efficiency comparison). Source: same.

5. Uncertainty-only active learning, strict-causal (query the most-uncertain samples in each
   batch up to budget; no drift-awareness, no RL): at a 25% label budget, F1 = 0.9379,
   macro-F1 = 0.9413, precision = 0.9990, recall = 0.8838, FPR = 0.00086, MCC = 0.8889;
   10,000 of 40,000 labels revealed; 3 drift events detected (monitoring only, does not act on
   drift). At a 10% budget: F1 = 0.9191. At a 5% budget: F1 = 0.9136. Source:
   `diagnostics/strict_causal_budget_comparison.csv`.

6. Drift-triggered active-learning hybrid, strict-causal (uncertainty-ranked querying,
   full-batch querying inside a drift-recency window): at a 25% label budget, F1 = 0.9379,
   macro-F1 = 0.9413, precision = 0.9990, recall = 0.8838, FPR = 0.00086, MCC = 0.8889;
   10,000 of 40,000 labels revealed (39,600-sample evaluated stream, i.e. excluding the 400
   warm-up samples); 3 drift events detected. This is the best-F1 budget found for this method
   across the swept range [1%, 5%, 10%, 25%, 100%]. At a 10% budget: F1 = 0.8256. Source:
   `strict_causal_main_results.csv` and `diagnostics/strict_causal_budget_comparison.csv`.

7. RL-guided hybrid, strict-causal (tabular Q-learning controller decides query/update/
   threshold actions): best-F1 result occurs at a *nominal* 100% label budget, where only
   74.06% of labels were actually realized/spent (29,624 of 40,000) — F1 = 0.9356,
   macro-F1 = 0.9391, precision = 0.9965, recall = 0.8817, FPR = 0.0031, MCC = 0.8845;
   2 supervised drift events, 9 unsupervised drift events detected. At matched, realistic
   budgets its performance is materially worse than every other budgeted method tested:
   F1 = 0.7104 (1%), 0.7160 (5%), 0.7161 (10%), 0.7161 (25%). Source: `strict_causal_main_results.csv`
   and `diagnostics/strict_causal_budget_comparison.csv`.

8. At a matched 25% label budget, uncertainty-only AL and the drift-triggered AL hybrid both
   reach F1 = 0.9379, while the RL-guided hybrid reaches only F1 = 0.7161 — a difference of
   ~22 F1 points at the identical label cost. Source: `diagnostics/strict_causal_budget_comparison.csv`.

9. The archived (pre-correction) RL-guided hybrid headline of F1 = 0.9383 at "4.47% of labels
   queried" was computed with the drift detector, reward signal, Q-table, and rolling
   performance state all using every sample's true label regardless of whether it was queried.
   The exact reconstruction of that percentage is
   `(400 warm-up + 1387 controller-driven queries) / 40,000 total stream samples = 4.4675%`.
   Source: `STRICT_CAUSAL_AUDIT_REPORT.md` §5, computed live from the archived
   `rl_hybrid_metrics.csv` / `rl_controller_summary.csv`.

10. Diagnostic root cause for the RL-guided hybrid's collapse under strict-causal accounting:
    only 13–19 of 162 possible discretized RL states are ever visited at any tested budget, and
    a single state accounts for 75–92% of all decision steps at every budget except the
    near-full-feedback one. The RL controller's query decision is an independent per-sample
    gate on a 3-bin-discretized uncertainty feature, not a within-batch ranking by continuous
    uncertainty score (which both AL baselines use), which plausibly explains why it does not
    concentrate its label budget on the most informative samples the way the AL baselines do.
    A direct ablation disabling the controller's threshold-adjustment action changed F1 by at
    most 0.24 points at any budget, ruling out threshold adjustment as a cause. Source:
    `STRICT_CAUSAL_RL_FAILURE_DIAGNOSTICS.md`.

## Qualitative claims

- Full-feedback adaptive models (Adaptive ML only, Drift-aware adaptive ML) are legitimate
  upper bounds on what continual learning from 100% of labels can achieve on this stream, and
  are unaffected by the strict-causal correction.
- Under a genuinely causal (strict) label-budget protocol — where the algorithm cannot use a
  sample's true label unless that label was actually queried or part of the initial warm-up
  seed set — active learning with uncertainty-based sample selection remains an effective,
  label-efficient strategy for this intrusion-detection stream, reaching F1 ≈ 0.94 at a 25%
  label budget.
- A tabular Q-learning controller for label-query decisions can fail to outperform much
  simpler uncertainty-based active learning once its reward, drift, and state signals are
  restricted to genuinely available (queried-only) information — in this study, its
  discretized state space collapsed almost entirely into one or two dominant states, and its
  per-sample query gate did not target the most uncertain samples as effectively as an explicit
  top-k ranking policy.
- Strict causal evaluation protocol (hiding unqueried labels from every adaptive component —
  classifier, drift detector, reward, and state features alike) materially changes reported
  results for RL-based label-query controllers and should be treated as a required, not
  optional, methodological standard for adaptive/active-learning IDS evaluation claiming a
  label-budget benefit.
- The archived (oracle-feedback) RL-guided hybrid results remain valid as an upper-bound
  simulation of what the same architecture could achieve if it had unrestricted access to
  labels for its internal bookkeeping, but must not be presented as a label-budget-constrained
  result.

## Explicit exclusions (do not restate as fact without further work)

- Do not claim RL-guided hybrid is the strongest or recommended method among those evaluated.
- Do not claim a 95%+ label saving as an achieved, complete-system result — the closest honest
  RL-guided efficiency claim is a 25.94% reduction relative to full labeling (74.06% used vs.
  100%) at its own best operating point, not versus a 4.47% figure.
- Do not claim the archived run's drift detector, reward, or state features operated under a
  label budget — they used every label unconditionally (§5 of the audit report).
- Do not claim reward-configuration tuning (`run_rl_reward_tuning`) was performed independently
  of the evaluation stream — this is a separate, still-open issue (reviewer concern #6),
  unaddressed by this branch.
