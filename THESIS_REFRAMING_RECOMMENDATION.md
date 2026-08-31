# Thesis Reframing Recommendation

Based on `STRICT_CAUSAL_AUDIT_REPORT.md` (the leakage fix and its headline effect) and
`STRICT_CAUSAL_RL_FAILURE_DIAGNOSTICS.md` (why the RL-guided arm collapses). This is a decision
memo, not an implementation — no code beyond the two diagnostic branches already on
`strict-causal-protocol`, and no thesis text, has been changed.

## 1. Can strict-causal RL-guided hybrid still be defended as the main proposed method?

No, not in its current form. Under a genuinely causal label budget, it is dominated by a
simpler, non-RL uncertainty-based active-learning policy at every practical budget (5–25%),
by a wide margin (e.g. 0.716 vs. 0.919 F1 at a 10% budget) and only reaches parity once given
roughly three-quarters of the entire stream's labels. The diagnostics trace this to the RL
controller's own query mechanism — an undifferentiated per-sample gate from a collapsed,
nearly one-state discretization — not to a bug, and not primarily to reward sparsity or the
drift signal. Defending it as *the* contribution as currently designed is not supportable by
the evidence in hand.

## 2. If not, which strict-causal method should become the main method?

**Uncertainty-only active learning** (drift-unaware, strict-causal), or the
**drift-triggered active-learning hybrid** (which reaches the same headline F1 at 25% and is
already the thesis's existing secondary baseline) are the two candidates with real evidence
behind them:

- Uncertainty-only AL: 0.9136 F1 at 5%, 0.9191 at 10%, 0.9379 at 25% — strictly better than
  drift-triggered AL at 5% and 10%, tied with it at 25% and 100%.
- Drift-triggered AL hybrid: matches uncertainty-only at 25% and 100%, but is worse at 5–10%.

Given the thesis's stated interest in drift-awareness as a contribution, the **drift-triggered
active-learning hybrid** is the more defensible "main method" going forward: it already exists
as a named ablation, its strict-causal numbers reproduce almost exactly against the archived
ones (see the audit report), and it is competitive with the best deterministic baseline at the
budgets that matter. It is a smaller pivot than promoting a method that isn't currently a named
contribution. Uncertainty-only AL should be reported alongside it as the strongest
budget-efficiency baseline found in this investigation, since it currently *beats* the
drift-triggered hybrid at low-to-moderate budgets — that comparison itself is a legitimate,
citable finding.

## 3. Should RL be moved to exploratory ablation / negative result / future work?

Yes. Recommended framing: RL-guided hybrid becomes an **explicitly negative/exploratory
result** reported alongside the diagnostics that explain it, rather than the headline method.
This is scientifically stronger than quietly dropping it — the diagnostics report constitutes a
real, well-evidenced contribution in its own right (a documented failure mode of tabular
Q-learning for label-query control under a coarse discretized state space, with a working
ablation methodology to isolate the cause). Concretely:

- Present the RL-guided hybrid's design and archived (oracle-feedback) results as originally
  planned, but immediately follow with the strict-causal correction and the failure diagnosis.
- Frame the state-discretization collapse and the "per-sample gate vs. top-k ranking" finding
  as the thesis's diagnostic contribution regarding *why* naively applying tabular RL to
  label-query control underperforms simpler baselines here.
- Explicitly list "a finer state representation, a continuous or ranking-based query policy,
  or a contextual-bandit reformulation might close this gap" as future work — do not claim
  this branch attempted or ruled out those fixes; it did not (per instructions, no tuning was
  done in this pass).

## 4. Which archived claims must be removed or softened?

- **Remove**: "RL-guided hybrid improves F1 from 0.8849 to 0.9383 while querying only 4.47% of
  labels" as a real budget-constrained result. Keep it only if explicitly re-labeled as an
  oracle-feedback simulation upper bound, shown side-by-side with the strict-causal number
  (0.716 at a comparable/larger 10% budget).
- **Remove/soften**: any statement that RL-guided hybrid is the strongest or recommended
  method among those evaluated. It is not, under honest accounting, at any practical budget.
- **Soften**: claims that the RL controller "learns an efficient query-allocation policy" —
  the diagnostics show it learns *a* policy, but one that does not target informative samples
  as effectively as simple uncertainty ranking.
- **Correct**: every place 4.47% is quoted, replace with or footnote the exact reconstruction:
  `(400 warm-up + 1387 controller queries) / 40000 total stream samples`.
- **Flag separately** (not part of the RL story): the warm-up-consumes-the-entire-budget
  artifact at the 1% budget point affects *every* method's 1%-budget row in every sweep in this
  project, not just RL — the thesis's 1%-budget numbers for all methods should be captioned
  accordingly or that budget point reconsidered.
- **Unrelated but still open** (from the original review, not addressed by this branch): the
  drift-event-count discrepancy (11 vs. 18) and the train/test split-protocol issue (#1) and
  reward-tuning-on-the-eval-stream issue (#6) — explicitly out of scope for this investigation
  per your instructions, still need their own fixes before submission.

## 5. Which strict-causal numbers are safe to report?

Safe as-is (verified, reproducible, not touched by the leak):

- Static ML only: F1 0.8849
- Adaptive ML only: F1 0.9833 (full-feedback upper bound)
- Drift-aware adaptive ML: F1 0.9993 (full-feedback upper bound)
- Rule + ML hybrid: F1 0.9407
- Drift-triggered AL hybrid (strict-causal): 0.7104 (1%) / 0.7094 (5%) / 0.8256 (10%) / 0.9379
  (25%) / 0.9353 (100%)
- Uncertainty-only AL (strict-causal, new finding from this investigation): 0.7104 (1%) /
  0.9136 (5%) / 0.9191 (10%) / 0.9379 (25%) / 0.9353 (100%)
- RL-guided hybrid (strict-causal): 0.7104 (1%) / 0.7160 (5%) / 0.7161 (10%) / 0.7161 (25%) /
  0.9356 (nominal 100%, 74.06% realized) — safe to report as-is *as a negative result*, not as
  the headline.

All of the above are direct outputs of `results_ciciot2023_strict_causal/tables/` and
`results_ciciot2023_strict_causal/diagnostics/` — traceable to CSVs, not hand-computed.

## 6. Most honest MSc thesis contribution after the strict-causal correction

A hybrid rule-filtered IDS pipeline with drift-aware, budget-constrained active learning
(the drift-triggered AL hybrid) is a legitimate, verified, label-efficient method that reaches
F1≈0.94 at a 25% label budget under a genuinely causal protocol — that is a real contribution.
Layered on top of it, a rigorous demonstration that a naive tabular-RL controller for
label-query decisions, once a full-oracle-feedback leak is corrected, fails to outperform much
simpler uncertainty-based querying — traced to a specific, evidenced mechanism (state
discretization collapse plus a per-sample-gate query design that doesn't rank by
informativeness) rather than left as an unexplained negative result — is itself a defensible,
methodologically careful contribution for an MSc thesis: it demonstrates the value of strict
causal evaluation protocols in adaptive-IDS research and documents a concrete failure mode
future work can build on. This combination (a working drift-aware AL method + a rigorously
diagnosed RL negative result) is more defensible than the original "RL-guided hybrid is best"
claim, and does not require inventing new experiments beyond what's already run.
