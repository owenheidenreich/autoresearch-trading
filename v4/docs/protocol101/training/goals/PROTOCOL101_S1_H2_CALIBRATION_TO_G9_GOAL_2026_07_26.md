# Goal: Protocol101 H2 Policy-5 Calibration Repair Through G9

> **Historical execution note (2026-07-26):** This goal was written and
> executed under the original hard-G8 contract. Its artifacts remain immutable
> historical evidence, but its hard-G8 stop is superseded for future routing
> by `PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`. Do not rerun this
> H2-only repair. The next authorized sequence is model-free H0-H3
> reaggregation, independent selection, and only then seed-45 G9 if earned.

## Goal ID

`S1-H2-CALIBRATION-TO-G9-GRAPH`

## Objective

Run one bounded, role-separated graph that repairs only the G8 confidence
calibration of H2 policy 5, independently verifies G1-G8, and proceeds to one
fresh-seed G9 run only if G1-G8 passes.

The graph must stop after the independent G9 verdict. It must not begin final
fit, open the protected holdout, build learned exits, or touch any runtime.

## Read First

- `v4/docs/PROTOCOL101_GOAL_SIZED_GATED_TRAINING_SYSTEM_2026_07_25.md`
- `v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/PROTOCOL101_TRADER_CHARTER.md`
- `v4/docs/PROTOCOL101_STAGE1_AUTORESEARCH_GRAPH_2026_07_25.md`

Use:

```text
~/.autoresearch-trading/runtime-venv/bin/python
```

## Authorization

This goal authorizes:

- offline calibration diagnosis and implementation;
- a single preregistered calibration-repair experiment for H2 policy 5;
- reuse and rescoring of the existing H2 policy-5 seed 42/43/44 model
  artifacts;
- fold-local calibration-method selection using training-tail calibration
  sessions only;
- independent G1-G8 aggregation and audit;
- if and only if G1-G8 is independently accepted, five-fold H2 policy-5
  training and threshold selection using fresh seed `45`;
- independent G9 audit of G1, G2, and G4.

It does not authorize:

- protected-holdout, recorder, parity-confirmation, or sealed-day access;
- seed 46 or any replacement G9 seed;
- feature, label, hypothesis, policy, fold, threshold-search, guard, fill,
  fee, simulator, gate, or trading-contract changes;
- new entry-model hyperparameters or another H0-H3 search;
- learned exits or Stage-2 design/execution;
- broker/API calls, paper-submit, paid downloads, promotion/default changes,
  runtime flags, launchd changes, or real-money paths.

## Starting Checkpoint

Require and hash-verify:

- H2 producing packet:
  `v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h2_attempt001/`
- H2 gate packet:
  `v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h2_attempt001_gates/`
- H2 independent audit:
  `v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h2_attempt001_audit/`
- H0-H3 comparison:
  `v4/audit/autoresearch/protocol101_scoped_canonical_stage1_cross_hypothesis_selection/`
- fixed-exit attribution:
  `v4/audit/autoresearch/protocol101_stage1_fixed_exit_binding_attribution_attempt001/`
- completed graph:
  `v4/audit/autoresearch/protocol101_stage1_autoresearch_graph/`

The accepted starting facts are:

- H2 policy 5 passes G1-G7 for seeds 42, 43, and 44;
- H2 policy 5 fails only G8;
- its weighted validation ECE values are approximately
  `0.114034`, `0.105864`, and `0.107810`, above the frozen `0.10` bar;
- fixed exits are not established as the Stage-1-wide binding problem;
- no candidate has run G9;
- seed `45` has not been used in this Stage-1 candidate search.

If any starting artifact, source binding, or stated fact is contradictory,
stop with `starting_checkpoint_invalid` and report the exact conflict.

## Frozen Scientific Contract

Freeze these before computing any repaired validation ECE:

- contract: `protocol101-scoped-canonical-stage1-v1`;
- hypothesis: H2 only;
- features: the 12 certified non-VIX context features plus internal delta and
  gamma;
- exit policy: policy 5 only;
- existing repair seeds: 42, 43, 44;
- G9 seed: 45, spend once;
- accepted 301-session registry and governed 271-session CV scope;
- five chronological expanding-window folds and one-session embargo;
- protected 2025-05-16 through 2025-06-30 holdout exclusion;
- bounded HGB model family and all existing H2 model hyperparameters;
- payoff / return-on-premium training target, never win probability;
- existing threshold, epsilon, selection dead-band, slot-margin, fallback,
  intersection-guard, and abstention semantics;
- seven-policy labels and policy-5 fixed-exit semantics;
- pessimistic fill rung, `$3.00` fee with `$2.60/$4.00` reporting;
- serial simulator v4, one account, one contract, and 5% daily stop;
- exact signed G1-G9 definitions, including G8 ECE bins and `<=0.10` limit.

Calibration must remain a readout from the payoff score to realized win
probability. It must not alter the payoff model target or become model alpha.

## Bounded Calibration Repair

The only authorized scientific change is the confidence-map fitting rule.

Preregister exactly this two-method menu before recomputing repaired
validation ECE:

1. `training_tail_isotonic_v1`: the current isotonic payoff-score-to-realized-
   win map.
2. `training_tail_platt_v1`: an L2-regularized logistic sigmoid mapping the
   same payoff score to the same realized-win outcome, using fixed
   `C=1.0`, `solver=lbfgs`, `max_iter=1000`, no class weighting, and a
   deterministic seed.

For each outer fold and model seed:

1. Use only that fold's existing training-tail calibration sessions.
2. Compare the two methods through chronological inner forward splits within
   those calibration sessions; never shuffle sessions.
3. Select the method with the lowest observation-weighted inner ECE using the
   unchanged 10-bin ECE definition.
4. Break an exact tie in favor of the existing isotonic method.
5. Refit the selected method on all of that fold's training-tail calibration
   rows.
6. Apply it once to that fold's untouched outer-validation rows.

No outer-validation ECE, PnL, gate result, or protected value may choose the
method or any method parameter. Do not add another calibrator, regularization
value, bin count, score transform, feature, or search batch after results are
visible.

If an inner split is one-class, use the existing deterministic training-tail
base-rate fallback for that split and report it. If the complete method
comparison cannot be performed causally, emit
`calibration_repair_insufficient_training_tail` and stop before G9.

## Durable Graph

Implement or extend a persistent graph with immutable state, events, logs,
source hashes, bounded mechanical retries, and resumable nodes:

```text
CAL-PREREGISTER
  -> CAL-MACHINERY-SMOKE
  -> CAL-RUN
  -> CAL-G1-G8-GATE
  -> CAL-G1-G8-INDEPENDENT-AUDIT
  -> C1-G9-MACHINERY
  -> C2-G9-RUN-SEED45
  -> C3-G9-INDEPENDENT-AUDIT
```

Use output root:

```text
v4/audit/autoresearch/protocol101_h2_policy5_calibration_to_g9_graph/
```

The graph may proceed past `CAL-G1-G8-INDEPENDENT-AUDIT` only when that node
freezes `accepted_eligible_G1_G8`. Otherwise it must stop.

Each node may retry a transient or mechanical non-substantive blocker at most
three times. Before repair, preserve and mark affected evidence void. A retry
must not change the frozen scientific contract. Scientific gate failure is a
completed result, not a retry condition.

## Calibration Outputs

Write immutable packets:

```text
v4/audit/autoresearch/protocol101_h2_policy5_calibration_repair_design/
v4/audit/autoresearch/protocol101_h2_policy5_calibration_repair_attempt001/
v4/audit/autoresearch/protocol101_h2_policy5_calibration_repair_attempt001_gates/
v4/audit/autoresearch/protocol101_h2_policy5_calibration_repair_attempt001_audit/
```

Required evidence:

- preregistration and source hashes written before repaired validation
  results;
- per fold/seed inner split membership and method scores;
- selected method and fitted confidence-map state;
- validation confidence/outcome rows and ECE;
- method/fallback counts;
- exact original-versus-repair hashes for model weights, epsilon, threshold,
  selected slots, actions, entry intents, trades, PnL, and G1-G7;
- gate table with numeric G1-G8 results;
- independent recomputation, freeze, report, and side-effect audit.

The producing calibration task may not accept its own result. The independent
auditor must recompute ECE and G1-G8 from source artifacts without trusting the
producer's verdict.

## G1-G8 Pass Condition

Pass only when:

- model, epsilon, threshold, selected slot, action, trade, PnL, and G1-G7
  evidence remains identical to the accepted H2 policy-5 seeds 42/43/44
  evidence;
- every seed's observation-weighted outer-validation ECE is `<=0.10`;
- G1-G8 all pass under the exact signed aggregator;
- the independent auditor finds zero implementation defects, unexplained
  numeric differences, or insufficient-evidence blockers;
- no forbidden side effect occurred.

If G8 still fails, emit `accepted_calibration_repair_failed_G8`, stop, and do
not try another calibration method or run G9.

If any economic or selection evidence changes, mark the attempt
`void_not_calibration_only`, diagnose the defect, and do not use its result.

## C1-G9 Machinery

Only after an accepted G1-G8 freeze:

- freeze the exact H2 policy-5 configuration, calibration selection rule,
  source hashes, corpus, folds, and seed 45;
- build/smoke the one-seed five-fold runner if required;
- prove seed 45 is absent from all H0-H3 selection evidence;
- prove G9 reads no protected, recorder, parity, or confirmation data;
- independently verify the machinery before spending seed 45.

The smoke may use synthetic/disposable inputs only. It may not produce a G9
economic result.

## C2-G9 Run

Run exactly:

- H2;
- policy 5;
- seed 45;
- all five frozen folds;
- the exact accepted calibration rule and all other frozen H2 settings.

Model fitting and fold-local threshold/calibration fitting are authorized only
for this seed-45 confirmation run.

Do not inspect intermediate economic results to alter or restart the run.
Seed 45 is spent once when its first non-void scientific result is produced.
A scientific failure may not be retried with seed 45 or replaced by another
seed.

Write:

```text
v4/audit/autoresearch/protocol101_h2_policy5_g9_seed45_attempt001/
```

## C3-G9 Independent Audit

The G9 producer may not accept its own result.

A read-only independent auditor must verify hashes, exclusions, all five
folds, serial replay, and the exact signed G9 gates:

- G1 profitability;
- G2 matched-null significance;
- G4 drawdown/survival.

G9 passes only if seed 45 independently satisfies all three. Report G8 for
seed 45 as a diagnostic, but do not silently add it to or remove anything from
the signed G9 rule.

Write:

```text
v4/audit/autoresearch/protocol101_h2_policy5_g9_seed45_attempt001_audit/
```

Allowed terminal decisions:

- `g9_pass_candidate_eligible_for_final_fit`;
- `g9_fail_candidate_burned`;
- `g9_void_implementation_or_artifact_defect`;
- `g9_blocked_insufficient_evidence`.

On a scientific G9 failure, burn this candidate configuration. Do not try seed
46, retune calibration, return to H0/H1/H3, or open the holdout.

## Mechanical Verification

Use the runtime interpreter to:

- compile every changed Python module;
- run focused unit tests for calibration, graph recovery, G1-G8 aggregation,
  seed exclusion, G9 execution, independent audits, and simulator invariants;
- run disposable smoke tests before evidence production;
- verify every preregistration/freeze/source/output SHA-256;
- verify graph state is terminal and no worker remains;
- verify all protected-data and side-effect flags.

Do not claim completion from a console message alone. Required artifacts,
hashes, independent audits, and terminal graph state must exist.

## Final Report

Report plainly:

- which calibration method was selected in each fold/seed and why;
- original versus repaired ECE for seeds 42/43/44;
- confirmation that G1-G7 and every entry/trade outcome were unchanged;
- independent G1-G8 verdict;
- whether G9 ran;
- seed-45 G1/G2/G4 numbers and independent G9 verdict if it ran;
- exact artifact paths and hashes;
- all repairs/retries/void packets;
- tests and side-effect audit;
- exact next allowed goal.

Highest allowed claim after G1-G8 but before G9:

> H2 policy 5 is an offline candidate eligible for fresh-seed confirmation.

Highest allowed claim after a G9 pass:

> H2 policy 5 is a G9-confirmed offline candidate eligible for the separately
> governed final-fit phase.

Do not claim holdout, runtime-transfer, paper-trade, or real-money readiness.

## Stop Line

Stop after the independent G9 verdict, or immediately after any earlier
scientific failure. Do not begin final fit, request or open the protected
holdout, build learned exits, contact IBKR, submit paper orders, or change any
promotion/runtime/real-money state.
