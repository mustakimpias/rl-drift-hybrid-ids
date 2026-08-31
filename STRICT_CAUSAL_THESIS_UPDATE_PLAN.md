# Strict-Causal Thesis Update Plan

Companion to `STRICT_CAUSAL_FINAL_CLAIMS.md` (the vetted claim list this plan draws from) and
the underlying `STRICT_CAUSAL_AUDIT_REPORT.md` / `STRICT_CAUSAL_RL_FAILURE_DIAGNOSTICS.md`. This
is a plan and replacement-text draft only — no `.tex` file has been edited. All numbers are
copied from `STRICT_CAUSAL_FINAL_CLAIMS.md`; none are invented for this document.

## 1. New recommended thesis framing

**Main proposed method:** Strict-Causal Drift-Triggered Active Learning Hybrid IDS — the rule
+ ML hybrid architecture with a genuinely causal, budget-constrained, uncertainty-ranked,
drift-triggered query policy. Under the corrected protocol it reaches F1 = 0.9379 at a 25%
label budget, matching the strongest deterministic baseline found (uncertainty-only AL) and
clearly ahead of the RL-guided controller at every practical budget.

**Exploratory/diagnostic component:** the RL-guided controller, retained in the thesis as a
negative-result ablation studied under the same strict-causal limited-feedback protocol. Its
value to the thesis is the diagnosis, not the performance: a documented, evidenced account of
why coarse tabular Q-learning underperforms simpler querying here (state-discretization
collapse + a non-ranking per-sample query gate — see `STRICT_CAUSAL_RL_FAILURE_DIAGNOSTICS.md`).

## 2. Safe strict-causal result numbers (extracted, not recomputed here)

| Method | F1 | macro-F1 | Precision | Recall | FPR | MCC | Label-query % | Labels revealed | Stream size | Drift count |
|---|---|---|---|---|---|---|---|---|---|---|
| Static ML only | 0.8849 | 0.8899 | 0.9288 | 0.8451 | 0.0648 | 0.7834 | n/a (offline) | 0 | 40,000 | 0 |
| Adaptive ML only (full-feedback upper bound) | 0.9833 | 0.9832 | 0.9750 | 0.9918 | 0.0255 | 0.9665 | 100% | 40,000 | 40,000 | 6 |
| Drift-aware adaptive ML (full-feedback upper bound) | 0.9993 | 0.9993 | 0.9993 | 0.9994 | 0.0008 | 0.9986 | 100% | 40,000 | 40,000 | 2 |
| Rule + ML hybrid | 0.9407 | 0.9427 | 0.9760 | 0.9079 | 0.0223 | 0.8877 | n/a (no query decision; 86.29% ML-layer coverage) | — | 40,000 | 0 |
| Uncertainty-only active learning (strict-causal, @25% budget) | 0.9379 | 0.9413 | 0.9990 | 0.8838 | 0.00086 | 0.8889 | 25% | 10,000 | 40,000 | 3 |
| Drift-triggered active learning hybrid (strict-causal, @25% budget) — **new main method** | 0.9379 | 0.9413 | 0.9990 | 0.8838 | 0.00086 | 0.8889 | 25% | 10,000 | 40,000 | 3 |
| RL-guided hybrid (strict-causal, best case @nominal 100%/74.06% realized) | 0.9356 | 0.9391 | 0.9965 | 0.8817 | 0.0031 | 0.8845 | 74.06% | 29,624 | 40,000 | 2 (supervised) / 9 (unsupervised) |

**Matched-budget honesty check** (all three budgeted methods at the identical 25% budget):
Uncertainty-only AL F1 = 0.9379, Drift-triggered AL hybrid F1 = 0.9379, **RL-guided hybrid
F1 = 0.7161** — a ~22-point gap at equal label cost. This table, not the "each at its own best
budget" table above, is the one that should anchor any comparative claim about RL vs. the
proposed method; the thesis must not compare RL's best-case (near-full-feedback) number against
the others' 25%-budget numbers without stating the budget mismatch explicitly.

## 3. New main result claim

> The strict-causal drift-triggered active-learning hybrid is the strongest defensible method
> under a genuinely limited-feedback protocol, reaching F1 = 0.9379 (macro-F1 = 0.9413,
> FPR = 0.00086) while querying only 25% of stream labels — a result that holds under a
> protocol where the algorithm has provably no access to any unqueried sample's true label,
> anywhere in its pipeline (classifier, drift detector, reward, or state).

This claim is reportable as the thesis's main result. It should be presented alongside the
observation that a simpler, drift-unaware uncertainty-only policy reaches the identical
number at the identical budget — i.e., in this dataset/configuration, drift-awareness did not
add measurable benefit over pure uncertainty sampling at the reported operating point, though
it may still be motivated on other grounds (interpretability of *why* a sample was queried,
responsiveness during an actual drift episode — see the before/after query-rate analysis in
the diagnostics report).

## 4. Claims that must be removed or softened

- "RL-guided hybrid improves F1 from 0.8849 to 0.9383 under label-efficient causal feedback" —
  **remove**. The 0.9383 number was never computed under a causal protocol; the strict-causal
  re-implementation of the identical method reaches at most F1 = 0.7161 at a comparable (10%)
  or even larger (up to 25%) budget, and only approaches 0.9383 near full-feedback (74.06%
  realized spend).
- "RL achieves 95.53% label saving as a complete adaptive system" — **remove**. 95.53% is
  `100 - 4.4675`, and 4.4675% was never a real causal label cost for the full system (see claim
  9 in `STRICT_CAUSAL_FINAL_CLAIMS.md`) — the drift detector and RL state/reward machinery used
  100% of labels regardless. The only honest RL-guided label-saving figure is 25.94%
  (100 − 74.06) at its own best operating point, which is not competitive with the 75%
  (100 − 25) saving the drift-triggered hybrid achieves at equal or better F1.
- "RL is the main proposed method" — **remove**; reframe per §1/§3 above.
- "Unqueried labels were hidden in the archived run" — **remove/correct**: state plainly that
  they were *not* hidden in the archived run (that was the bug), and that this correction is
  precisely what this revision addresses.
- "Reward tuning is independent [of the evaluation stream]" — **remove/flag as unresolved**:
  this is reviewer concern #6, not addressed by the strict-causal correction; do not claim it
  is resolved. If the sentence must stay for now, mark it explicitly as a known open limitation.

## 5. Claims that can still remain

- Adaptive methods can help under drift/dataset shift (supported by the full-feedback upper
  bound arms: Adaptive ML only and Drift-aware adaptive ML both clearly outperform Static ML
  only).
- Full-feedback adaptive models are legitimate upper bounds on achievable performance, not
  themselves label-budget-constrained results.
- Label-efficient active learning can reduce annotation cost without materially harming
  detection performance (supported: 25% budget reaches macro-F1 = 0.9413, essentially matching
  what full labeling of the ML-routed portion achieves).
- Strict causal feedback is essential for fair evaluation of any adaptive IDS component that
  claims a label-budget benefit — this thesis's own before/after comparison is direct evidence
  for that methodological claim.
- Coarse tabular RL can fail under sparse causal rewards and a collapsed state representation —
  directly evidenced and diagnosed in this work, not merely asserted.

## 6. Chapter-by-chapter revision instructions

**Chapter 1 (Introduction):**
- Update the contributions list: lead with the strict-causal drift-triggered active-learning
  hybrid as the primary method contribution; add the strict-causal evaluation protocol itself
  (hiding unqueried labels from every adaptive component, not just the classifier) as a
  methodological contribution; add the RL-controller failure diagnosis as a secondary,
  exploratory contribution.
- Update research questions: any RQ framed as "does the RL-guided hybrid outperform X" should
  be reframed as "does a strict-causal drift-triggered active-learning hybrid outperform
  simpler baselines under a genuinely limited-feedback protocol," with a companion RQ on
  whether an RL controller adds value over deterministic querying under that same protocol
  (answer: not in the tested configuration — a legitimate negative-result RQ/answer pair).
- Remove RL as the headline claim anywhere in the chapter; it should appear only as one of
  several methods investigated, with its role (exploratory/diagnostic) stated up front so the
  reader isn't surprised by the negative result in Chapter 5/6.

**Chapter 3 (Methodology/Proposed Framework):**
- Rename the proposed method from "RL-guided hybrid IDS" (or equivalent) to the strict-causal
  drift-triggered active-learning hybrid throughout.
- Move the RL controller's description to a clearly labeled optional/exploratory extension
  section, presented after the main method, with a one-paragraph framing that it is evaluated
  as an alternative query-policy ablation, not the primary mechanism.
- Add explicit strict-causal protocol pseudocode. Suggested structure (mirrors
  `run_rl_guided_hybrid_strict_causal` / `run_budgeted_stream(strict_causal=True)`):
  ```
  for each incoming sample x_t:
      compute prediction and uncertainty from x_t only (no label)
      compute drift flag from detector state as of t-1 (no label at t)
      select query decision from state (uncertainty, drift flag, budget remaining)
      reveal y_t to the algorithm  IF AND ONLY IF  query decision == True
      if revealed:
          update classifier with (x_t, y_t)
          update supervised drift detector with error indicator
          update rolling performance state with (y_t, prediction)
          compute reward from (y_t, prediction, query decision)
      else:
          update unsupervised drift detector with uncertainty signal only
          use a fixed neutral proxy reward (never derived from y_t)
      update policy/Q-table with (state, action, reward, next_state)
      # evaluator-only, never fed back into the algorithm above:
      log (prediction, y_t) for final metric computation after the full run
  ```
- State explicitly that classifier, drift detector, reward, and state-feature updates are ALL
  gated on the same query decision — this is the exact fix over the archived implementation,
  and should be called out as such (a specific implementation contract, not just prose).

**Chapter 4 (Implementation/Experimental Setup):**
- Describe the strict-causal implementation: two independent drift detectors (supervised,
  gated on queried/warm-up labels only; unsupervised, fed prediction uncertainty on every
  step), a documented zero/neutral proxy reward for unqueried steps, and rolling
  performance-state features computed only from the queried subsequence.
- Add a label-accounting subsection/table with exactly the columns already computed in
  `strict_causal_label_accounting.csv` / `strict_causal_query_accounting_by_budget.csv`:
  total stream samples, warm-up labels, controller queries (split by
  query-and-update vs. update-if-drift), total labels revealed, and the resulting percentage —
  so a reader can independently recompute any reported label-query figure from first principles.
- Describe the archived (oracle-feedback) run explicitly as a **limitation of the original
  implementation**, presented either in a dedicated "Protocol Correction" subsection or moved
  to an appendix: what leaked (drift detector, reward, state features), how it was detected,
  and how it was fixed (the strict-causal re-implementation). Do not present it silently
  alongside corrected numbers without this framing.

**Chapter 5 (Results):**
- Replace the headline result table with the strict-causal table from §2 above (or the
  matched-25%-budget table, which is the more defensible comparison for the RL row).
- Present the archived RL-guided result only as a diagnostic/oracle-feedback comparison row,
  clearly captioned (e.g. "oracle-feedback simulation — not a label-budget-constrained
  result") and placed next to, not merged into, the strict-causal numbers.
- Add the RL failure diagnostics as their own subsection: state-visitation collapse (13–19 of
  162 states visited, one state capturing 75–92% of steps), the reward-density table, the
  action-distribution-by-budget table, and the threshold-adjustment ablation (Δ ≤ 0.24 F1
  points at every budget, ruling that mechanism out). These are the thesis's evidence for the
  negative-result claim, not filler — they should be given real space, not a footnote.

**Chapter 6 (Conclusion):**
- Revise the conclusion to lead with the strict-causal drift-triggered active-learning hybrid
  as the delivered contribution, at its actual, verified operating point (25% budget, F1 =
  0.9379).
- State the honest contribution explicitly: a working, label-efficient, drift-aware active
  learning IDS evaluated under a genuinely causal protocol, plus a rigorously diagnosed
  negative result for a natural RL-based alternative.
- Move "improve the RL controller's label efficiency" (finer state representation, a
  ranking/contextual-bandit reformulation of the query policy, reward-shaping for sparser
  feedback) to future work — explicitly, not as an implied next step buried in limitations.

## 7. Exact replacement text (drop-in drafts — adapt citations/numbering to your document)

**Abstract:**

> This thesis presents a hybrid intrusion detection system that combines a simulated
> signature-matching layer with a drift-aware, budget-constrained active learning policy,
> evaluated under a strict causal limited-feedback protocol in which the system has no access
> to any sample's true label unless that label was explicitly queried. Under this protocol,
> the proposed drift-triggered active learning hybrid reaches an F1-score of 0.9379 while
> querying only 25% of stream labels on the CICIoT2023 dataset, matching the strongest
> uncertainty-based baseline evaluated and substantially outperforming a reinforcement-learning
> (RL) based query controller evaluated under the identical protocol. A central finding of this
> work is methodological: an earlier implementation of the RL-guided controller, evaluated
> without strict causal guarantees, appeared to reach F1 = 0.9383 at a 4.47% label cost: closer
> analysis showed its drift detector, reward signal, and internal state features had
> unconditional access to every sample's true label regardless of whether that label was
> queried. Once corrected, the same RL controller's performance drops to F1 ≈ 0.72 at
> comparable label budgets. This thesis diagnoses the cause — a near-total collapse of the
> controller's discretized state space and a per-sample query mechanism that does not rank
> samples by informativeness — and presents it as a negative result with direct implications
> for evaluating adaptive, budget-constrained intrusion detection systems.

**Chapter 1 contribution paragraph:**

> This thesis makes three contributions. First, a hybrid intrusion detection architecture that
> routes confidently-classified traffic through a simulated signature layer and applies a
> drift-triggered, uncertainty-ranked active learning policy to the remainder, achieving
> F1 = 0.9379 at a 25% label budget under a genuinely causal feedback protocol. Second, a
> strict-causal evaluation protocol for adaptive IDS components, under which no part of the
> system — classifier, drift detector, reward computation, or internal state — may use a
> sample's true label unless that label was actually queried, together with a demonstration
> that this distinction materially changes reported results for at least one natural design
> (an RL-based query controller). Third, a diagnosis of why a tabular reinforcement-learning
> controller for label-query decisions underperforms simpler active-learning baselines once
> evaluated under this protocol, tracing the failure to state-space discretization collapse and
> a non-ranking per-sample query mechanism rather than to an implementation defect.

**Chapter 3 proposed framework introduction:**

> The proposed framework is a strict-causal drift-triggered active learning hybrid: a shallow,
> confidence-thresholded decision tree stands in for a signature-matching layer, routing
> high-confidence traffic directly to a verdict; all other traffic is scored by an online
> classifier and offered to a budget-constrained query policy that ranks candidate samples by
> predictive uncertainty within each processing batch and additionally queries the full batch
> when a recent concept-drift signal is active. Every component of this pipeline is subject to
> a strict causality constraint: a sample's true label becomes available to the classifier,
> the drift detector, and any rolling performance statistic only at the moment that sample is
> actually queried, never before and never for samples that are not queried. As an exploratory
> extension, Section [X] additionally evaluates whether a tabular reinforcement-learning
> controller can improve on this fixed query policy by learning when to query, update, or
> adjust its decision threshold; that extension is evaluated under the identical causal
> constraint and is reported as a negative result (Section [Y]).

**Chapter 5 main result opening paragraph:**

> Table [N] reports the main comparison under the strict-causal protocol described in Section
> [X]: no method in this table had access to any sample's true label except at the moment that
> label was queried. The proposed drift-triggered active learning hybrid reaches F1 = 0.9379
> (macro-F1 = 0.9413, FPR = 0.00086) at a 25% label budget, matching an uncertainty-only active
> learning baseline evaluated under the same protocol and budget. For comparison, Table [N+1]
> reports the RL-guided controller evaluated under the identical strict-causal protocol: at the
> same 25% budget it reaches only F1 = 0.7161, a result substantially below every other
> budgeted method tested; it approaches the proposed method's performance (F1 = 0.9356) only
> when allowed to spend 74.06% of the stream's labels. Table [N+2], included for completeness
> and clearly marked as an oracle-feedback simulation, reproduces the originally reported
> RL-guided result (F1 = 0.9383 at a nominal 4.47% label cost); Section [Z] establishes that
> this earlier result depended on the drift detector, reward computation, and state features
> having unconditional access to every sample's true label, and should not be read as a
> label-budget-constrained result.

**Chapter 6 conclusion paragraph:**

> This thesis set out to build an adaptive intrusion detection system that reduces labeling
> cost without sacrificing detection performance, and to evaluate whether a reinforcement
> learning controller could improve on simpler active-learning query policies for this purpose.
> The strict-causal drift-triggered active learning hybrid answers the first question directly:
> it reaches F1 = 0.9379 at a 25% label budget under a protocol that provably withholds every
> unqueried sample's label from every part of the system. The second question is answered
> negatively, and the negative result is itself a contribution: once evaluated under the same
> causal protocol, the RL-guided controller underperforms the proposed method and a simpler
> uncertainty-only baseline by roughly 22 F1 points at matched label budgets, a gap traced to a
> severely collapsed discretized state representation and a per-sample query mechanism that
> does not rank candidates by informativeness. This finding also demonstrates a broader
> methodological point: an earlier, non-causal evaluation of the same RL controller reported a
> substantially stronger result (F1 = 0.9383 at 4.47% of labels) purely because its internal
> drift detection and reward computation had unrestricted access to labels the system would not
> actually have under a real label budget — underscoring that strict causal evaluation is not
> an optional refinement but a precondition for any adaptive IDS claiming a label-efficiency
> benefit. Improving the RL controller's state representation and query mechanism so that it
> can exploit a label budget as efficiently as explicit uncertainty ranking is left to future
> work.

## 8. Short defense explanation

**Why did the thesis change after the strict-causal audit?** An independent review identified
that the RL-guided controller's drift detector, reward computation, and internal state
features used every sample's true label unconditionally, regardless of whether that label had
been queried under the stated label budget — meaning the headline 4.47%-label-cost result was
never actually evaluated under the label-budget constraint it claimed. Re-implementing the
identical architecture so that every one of those components is genuinely gated on the query
decision changed the measured result substantially, which is exactly what should happen when a
real leak is fixed.

**Why is the negative RL result still valuable?** Because it is diagnosed, not just reported.
The thesis doesn't merely say "RL performed worse" — it identifies the specific mechanism (a
state space that collapses into one or two dominant discretized states, and a per-sample query
gate that doesn't rank candidates by uncertainty the way the successful baselines do), rules
out a plausible alternative explanation by direct ablation (threshold adjustment), and shows
exactly where the gap closes (once the label budget is abundant enough that targeting
inefficiency stops mattering). That is a legitimate, defensible empirical contribution in its
own right, and considerably stronger evidence than an unexplained negative result would be.

**Why does drift-triggered active learning become the main method?** Under the corrected,
causally valid protocol, it is the best-supported method that was already a named contribution
in the thesis design: it reaches the same top performance as the strongest baseline found
(uncertainty-only AL) at a practical 25% label budget, and its drift-awareness gives it a
principled mechanism (not present in the plain uncertainty baseline) for responding to
concept drift, even though the two ended up tied at the reported operating point on this
particular dataset.

**What would a future RL design need?** Based on the diagnosis: (1) a finer or
continuous-valued state representation (or a function-approximation-based controller) so the
policy doesn't collapse into one or two states; (2) a query mechanism that ranks candidates
within a batch by uncertainty (or another informativeness score) rather than gating each
sample independently, so it can actually compete with top-k active learning; and (3) a denser
or shaped reward signal for the many steps where no label is queried, so Q-values differentiate
faster under a small label budget. None of these were implemented or tested in this branch —
they are explicitly future work, not claims of an attempted or partial fix.
