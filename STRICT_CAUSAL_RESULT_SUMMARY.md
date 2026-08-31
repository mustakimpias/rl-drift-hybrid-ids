# Strict-Causal Results — Summary

Full detail and methodology: `STRICT_CAUSAL_AUDIT_REPORT.md`. This is the short version.

## One-line result

Once the RL-guided hybrid arm is rebuilt so it genuinely cannot see an unqueried sample's true
label anywhere (drift detector, reward, Q-table, or its own rolling-performance state), its F1
at a comparable label budget drops from the archived **0.9383** to **0.7161** — and a simpler,
non-RL active-learning baseline (drift-triggered uncertainty sampling) beats it at every
practical budget from 10% to 25%.

## Before / after (same 40,000-row evaluation stream, same seed)

| Method | Archived (oracle-feedback) F1 | Strict-causal F1 | Verdict |
|---|---|---|---|
| Static ML only | 0.8849 | 0.8849 | unaffected (offline, no leak possible) |
| Adaptive ML only | 0.9833 | 0.9833 | unaffected (declared full-feedback) |
| Drift-aware adaptive ML | 0.9993 | 0.9993 | unaffected (declared full-feedback) |
| Rule + ML hybrid | 0.9407 | 0.9407 | unaffected |
| Drift-triggered AL hybrid | 0.9379 @ 25% | 0.9379 @ 25% | reproduced almost exactly |
| **RL-guided hybrid** | **0.9383 @ "4.47%"** | **0.7161 @ 10%** (matched budget) | **does not survive the fix** |

RL-guided only reaches parity with the archived number (0.9356) when allowed to spend ~74% of
the stream's labels — not the ~4.5% originally reported.

## What this means for the thesis

**Keep as-is:** every full-feedback/offline ablation row, and the drift-triggered
active-learning baseline's numbers — none of these depended on the leaked pathway.

**Must change:**
1. Remove or clearly caption the "0.9383 F1 at 4.47% labels" headline as an oracle-feedback
   upper-bound, not a real budget-constrained result.
2. Soften or rework the thesis's central claim that the RL-guided controller is the strongest
   proposed method — under honest accounting it is dominated by the simpler active-learning
   baseline across the realistic 5–25% budget range, and only ties it near full labeling.
3. State the exact 4.47% reconstruction wherever it's quoted:
   `(400 warm-up + 1387 queries) / 40000 total stream samples`.

**Open question worth investigating (not yet answered by this run):** why does the strict-causal
RL controller's F1 stay flat from a 1% to a 25% budget instead of improving with more labels the
way the non-RL baseline does? The audit report's §4 has the full budget-sweep numbers; likely
culprits are reward sparsity (real reward only on queried steps, `0.0` proxy elsewhere) and the
tabular state discretization, but this branch only diagnoses the drop — it does not yet attempt
a fix to the RL controller's learning dynamics.

## Where things live

- Branch: `strict-causal-protocol` (new; `main` untouched).
- New results: `results_ciciot2023_strict_causal/` (archived `results_ciciot2023_balanced_test/`
  untouched).
- New code: `run_rl_guided_hybrid_strict_causal` and
  `run_drift_triggered_al_hybrid_strict_causal` in `src/hybrid_ids.py`,
  `strict_causal=True` option in `src/stream_utils.py::run_budgeted_stream`,
  `RLController.compute_proxy_reward()` in `src/rl_controller.py`,
  new script `scripts/run_strict_causal_experiment.py`.
- New tests: `tests/test_strict_causal.py` (8 tests, all passing); full existing suite (21
  tests) still passes.
- Thesis text has **not** been edited (per instructions) — this branch is code + results +
  audit only.
