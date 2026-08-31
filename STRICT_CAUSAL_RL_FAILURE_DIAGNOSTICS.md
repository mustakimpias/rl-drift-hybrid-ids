# Strict-Causal RL-Guided Hybrid: Failure Diagnostics

Investigates why `run_rl_guided_hybrid_strict_causal` collapses to F1≈0.71–0.72 across the
1–25% label-budget range once the label leak documented in `STRICT_CAUSAL_AUDIT_REPORT.md` is
removed, while the drift-triggered and uncertainty-only active-learning baselines scale
normally with budget. All numbers below come from
`scripts/run_strict_causal_diagnostics.py`, run on the identical 9,992/40,000 split and seed=42
as the main strict-causal experiment. Outputs: `results_ciciot2023_strict_causal/diagnostics/`
(5 CSVs + `diagnostics_summary.json`). No archived results or thesis text were touched.

## B. Trace analysis

All figures below are for the evaluation stream: **40,000 total stream samples**, of which
**400 are warm-up labels** (constant across every budget — `compute_warmup_size` with
`warmup_fraction=0.01`, `warmup_min=100`), **34,116 are RL/ML-routed decision steps**, and
**5,484 are rule-layer-handled steps** (the rule layer's coverage is fixed by the split, so
this never changes with budget).

| Budget | Total labels revealed | Query % (full stream) | Query % (of 34,116 RL steps) |
|---|---|---|---|
| 0.01 | 400 | 1.00% | 0.00% (all consumed by warm-up) |
| 0.05 | 2,000 | 5.00% | 4.69% |
| 0.10 | 4,000 | 10.00% | 10.55% |
| 0.25 | 10,000 | 25.00% | 28.13% |
| 1.00 | 29,624 | 74.06% | 85.66% |

*Eligible query opportunities* (steps where `ACTION_QUERY_AND_UPDATE` was still unmasked, i.e.
budget not yet exhausted) is a third denominator this run did not separately instrument — the
data below establishes that at 0.05/0.10/0.25 the *entire* remaining budget was spent before
the stream ended (total revealed = budget total exactly), so eligible opportunities is smaller
than the full 34,116 RL steps for part of the run; at 1.00, budget was never exhausted
(74.06% < 100%), so it stays eligible for the whole stream. Getting an exact per-step count
would need another full rerun with extra logging — flagged here as a gap rather than guessed.

**Reward / Q-table accounting** (`strict_causal_rl_reward_density.csv`):

| Budget | RL steps | Supervised rewards | Proxy/zero rewards | Q-updates | Reward density |
|---|---|---|---|---|---|
| 0.01 | 34,116 | 0 | 34,116 | 34,116 | 0.000 |
| 0.05 | 34,116 | 1,600 | 32,516 | 34,116 | 0.047 |
| 0.10 | 34,116 | 3,600 | 30,516 | 34,116 | 0.106 |
| 0.25 | 34,116 | 9,600 | 24,516 | 34,116 | 0.281 |
| 1.00 | 34,116 | 29,224 | 4,892 | 34,116 | 0.857 |

**Classifier updates** = supervised rewards (classifier update is gated identically to
queried, so these columns match `total_labels_revealed - warmup_labels`). **Supervised drift
updates** = the same count as supervised rewards (both gated on `queried`). **Unsupervised
drift updates** = 34,116 at every budget (every RL-routed step, unconditionally).

**RL action distribution by budget** (`strict_causal_rl_action_by_budget.csv`):

| Budget | no_query | query_and_update | update_if_drift | adjust_threshold |
|---|---|---|---|---|
| 0.01 | 93.45% | 0.00% | 3.24% | 3.32% |
| 0.05 | 62.94% | 4.24% | 26.26% | 6.56% |
| 0.10 | 57.29% | 10.12% | 4.27% | 28.31% |
| 0.25 | 43.94% | 27.55% | 5.04% | 23.47% |
| 1.00 | 8.03% | 84.61% | 4.14% | 3.23% |

`update_if_drift` is selected often (26.3% at budget 0.05) but only a small fraction of those
selections actually produce a query — of the 155 real "update_if_drift" queries at budget 0.05
(`strict_causal_query_accounting_by_budget.csv`), that's only 1.7% of the ~8,960 times the
action was selected; the rest are no-ops because the drift flag wasn't active. At low/moderate
budgets this action is effectively "no_query in disguise" most of the time.

**Threshold values over time** (`decision_threshold` column of the trace):

| Budget | Min | Max | Mean | Final | % steps above 0.5 |
|---|---|---|---|---|---|
| 0.01 | 0.35 | 0.75 | 0.506 | 0.5 | 17.3% |
| 0.05 | 0.35 | 0.90 | 0.519 | 0.5 | 20.0% |
| 0.10 | 0.35 | 0.90 | 0.602 | 0.5 | 37.6% |
| 0.25 | 0.35 | 0.90 | 0.584 | 0.5 | 32.4% |
| 1.00 | 0.35 | 0.90 | 0.512 | 0.5 | 14.1% |

**Recent-F1/FPR state availability** (fraction of ML-routed steps where the feature is at its
still-untouched neutral default, vs. simply unchanged from the last queried sample —
"stale" this step):

| Budget | % steps at untouched default (never queried yet) | % steps stale this step (not updated) |
|---|---|---|
| 0.01 | 100.0% | 100.0% |
| 0.05 | 0.05% | 95.3% |
| 0.10 | 0.05% | 89.4% |
| 0.25 | 0.05% | 71.9% |
| 1.00 | 31.8%\* | 14.3% |

\*Higher at budget 1.0 because `recent_f1==1.0` also occurs legitimately whenever the queried
window has zero false negatives, not only when never-queried — this column over-counts
slightly there; the "stale this step" column is the reliable one.

**State visitation** (`strict_causal_state_visitation.csv`; state space is
3×2×3×3×3 = 162 possible discretized states):

| Budget | Unique states visited | Top-1 state share | Entropy (bits) |
|---|---|---|---|
| 0.01 | 4 / 162 | 79.3% | 0.78 |
| 0.05 | 15 / 162 | 91.7% | 0.65 |
| 0.10 | 14 / 162 | 87.9% | 0.81 |
| 0.25 | 13 / 162 | 75.4% | 1.25 |
| 1.00 | 19 / 162 | 41.3% | 2.12 |

At budget 0.10, state `(uncertainty_bin=0, drift_flag=0, recent_f1_bin=0, recent_fpr_bin=0,
budget_bin=0)` alone accounts for **29,986 of 34,116 decision steps (87.9%)**. Its Q-values:
`no_query=3.599`, **`query_and_update=3.999` (highest)**, `update_if_drift=3.599`,
`adjust_threshold=3.599` — the greedy policy in this state does prefer querying, by a modest
margin, but three of four actions are numerically tied, and `uncertainty_bin=0` means this is
the **least** uncertain third of samples, i.e. the state the controller spends 88% of its time
in is "the model already looks confident" — the opposite of where an efficient active-learning
policy should concentrate its budget.

## C. Hypothesis testing

### Hypothesis 1 — Reward sparsity: **confirmed as a contributing factor**

Reward density (supervised rewards / RL decision steps) is 0.0 at 1%, 4.7% at 5%, 10.6% at
10%, 28.1% at 25%, and 85.7% at 100%. At every budget below 100%, the large majority of
Q-table transitions are trained on the fixed `0.0` proxy reward, not a real signal — the
Q-values that do differentiate actions (see the dominant-state Q-values above) are learned
from a thin trickle of real feedback. This plausibly slows and weakens policy learning, but by
itself doesn't explain the *flatness* between 5%, 10%, and 25% (reward density roughly
doubles/quadruples between these while F1 doesn't move) — it's an amplifier, not the whole
story.

### Hypothesis 2 — State discretization collapse: **confirmed, severe**

Only 13–19 of the 162 possible discretized states are ever visited (8–12% of the space) at any
budget, and a single state captures 75–92% of all decision steps at every budget except 1.0
(41%). The controller effectively cannot distinguish most of the stream from a single
"business as usual" context — three of its four actions are Q-value-tied in that dominant
state at budget 0.10. This directly explains why more label budget doesn't translate into a
more differentiated, more effective policy: there's almost no state signal left to condition on.

### Hypothesis 3 — Action/query mismatch: **confirmed, primary mechanism**

The clearest evidence is the head-to-head comparison at matched budgets
(`strict_causal_budget_comparison.csv`):

| Budget | RL-guided (strict-causal) F1 | Uncertainty-only AL (strict-causal) F1 | Drift-triggered AL (strict-causal) F1 |
|---|---|---|---|
| 0.05 | 0.7160 | **0.9136** | 0.7094 |
| 0.10 | 0.7161 | **0.9191** | 0.8256 |
| 0.25 | 0.7161 | **0.9379** | 0.9379 |

Both AL baselines select queries via `select_topk` on a **continuous** uncertainty score
within each batch — they always spend their budget on the most uncertain samples available
right now. The RL controller instead makes an **independent per-sample decision from a
3-bin-discretized uncertainty feature**: within a bin, all samples look identical to it, so
querying is not concentrated on the genuinely hardest cases the way top-k selection is. Given
that its dominant state (88% of steps) is specifically the *low*-uncertainty bin, much of its
query budget under this design is not spent where it would help most. This — not reward
sparsity or drift-signal defects — best explains why identical or larger label counts under RL
buy far less F1 than under either baseline, and why performance is flat across the 5-25% range:
regardless of how much total budget is available, this per-sample gate never triggers a
targeted push into the hardest-decision region until nearly the whole budget is available.

At budget 1.0, `query_and_update` is finally selected 84.6% of the time and F1 recovers to
0.9356 — consistent with this theory: once the label supply is abundant enough that the
targeting inefficiency stops mattering, RL catches up to (but does not exceed) the baselines.

### Hypothesis 4 — Drift signal mismatch: **partially confirmed, not the primary driver**

The supervised (label-gated) detector never fires at all at budgets 0.01/0.05/0.10/0.25 (0
events) — a direct consequence of reward/label sparsity — and only fires twice at budget 1.0.
The unsupervised (always-on) detector fires 4–10 times per budget. Where it does fire, the
signal looks meaningful: at budgets 0.10 and 0.25, mean per-sample error rate in a 500-step
window **before** an unsupervised-flagged event is 0.220, dropping to 0.091 **after** — the
controller's post-drift query-rate does rise correspondingly (0.45→0.54 at budget 0.10,
0.45→0.55 at budget 0.25), so the drift-response mechanism is doing something real. This is
not the bottleneck; it's a small, mostly-working slice of behavior overshadowed by the dominant
no-drift regime described under Hypothesis 2/3.

### Hypothesis 5 — Threshold-adjustment harm: **refuted**

Ablation (`disable_threshold_adjustment=True`, same seed/budget, `ACTION_ADJUST_THRESHOLD`
still selectable but its effect suppressed):

| Budget | F1 (normal) | F1 (threshold-adjustment disabled) | Δ |
|---|---|---|---|
| 0.01 | 0.7104 | 0.7104 | 0.0000 |
| 0.05 | 0.7160 | 0.7161 | +0.0001 |
| 0.10 | 0.7161 | 0.7161 | +0.0000 |
| 0.25 | 0.7161 | 0.7162 | +0.0001 |
| 1.00 | 0.9356 | 0.9380 | +0.0024 |

Differences are noise-level at every budget (largest is 0.24 F1 points, at full budget, in the
direction of the ablation being marginally *better*). Threshold adjustment is not causing or
masking the collapse.

### Hypothesis 6 — RL adds no benefit over a deterministic policy: **confirmed**

Full comparison at matched budgets, same stream, no tuning on the test stream (uncertainty
threshold for the "simple rule" baseline is `configs/default.yaml`'s existing
`active_learning.uncertainty_margin_threshold: 0.15`, used as-is):

| Budget | RL-guided (strict-causal) | Drift-triggered AL | Uncertainty-only AL | Simple rule (uncertainty-or-drift) |
|---|---|---|---|---|
| 0.01 | 0.7104 | 0.7104 | 0.7104 | 0.7104 |
| 0.05 | 0.7160 | 0.7094 | **0.9136** | 0.7104 |
| 0.10 | 0.7161 | 0.8256 | **0.9191** | 0.7104 |
| 0.25 | 0.7161 | 0.9379 | 0.9379 | 0.7104 |
| 1.00 | 0.9356 | 0.9353 | 0.9353 | 0.7104 |

Uncertainty-only active learning — a much simpler, non-RL, non-drift-aware policy — matches or
beats the RL-guided hybrid at every budget, and clearly beats it at 5% and 10%. The "simple
rule" diagnostic baseline is uninformative here: with the config's existing (not tuned)
uncertainty threshold of 0.15, it almost never queries beyond the warm-up set at any nominal
budget (query % pinned at ~1.0–1.005% regardless of the nominal budget) — this is a separate,
minor observation (that threshold value doesn't fire often on this model's calibrated
uncertainty scores) rather than a hypothesis-6 result either way; it neither supports nor
refutes RL's value, it's just not a useful comparison point as configured.

## A budget-independent artifact, separate from the RL-specific findings

At the 1% budget, **every single method tested (RL, drift-triggered AL, uncertainty-only AL,
disabled-threshold RL, and the simple rule) produces the identical F1 = 0.7104**, because
`compute_warmup_size`'s `budget_cap` argument caps the warm-up seed set at the full nominal
budget (400 of 400 available labels), leaving **zero** remaining budget for any subsequent
query decision by any method. This means the 1% budget point in every sweep in this project
(including the main strict-causal results) tests only the warm-up-trained model, not any
method's query policy — a shared implementation detail worth flagging in the thesis
methodology text (not touched by this branch), independent of the RL-specific findings above.

## Root cause summary

The flat F1 across 5–25% budgets is a **design/architecture limitation of the RL controller's
query mechanism**, not an implementation bug and not primarily reward sparsity or a broken
drift signal:

1. The state space collapses almost entirely into 1–2 dominant discretized states (Hypothesis
   2), driven by coarse 3-bin discretization of continuous signals.
2. Within the dominant state, the controller makes an undifferentiated per-sample query
   decision rather than ranking samples by uncertainty within a batch the way both AL
   baselines do (Hypothesis 3) — this is the primary mechanism, directly demonstrated by
   uncertainty-only AL beating RL by ~20 F1 points at the same 5–10% budgets.
3. Reward sparsity (Hypothesis 1) and the drift signal's rarity (Hypothesis 4) are real but
   secondary — they slow/weaken policy learning rather than cause the ceiling effect.
4. Threshold adjustment (Hypothesis 5) is empirically ruled out by direct ablation.
5. RL only closes the gap once budget is so large (74%+ realized spend) that per-sample
   targeting inefficiency stops mattering — confirming this is about *how* labels are chosen,
   not *whether* the classifier can learn from them.
