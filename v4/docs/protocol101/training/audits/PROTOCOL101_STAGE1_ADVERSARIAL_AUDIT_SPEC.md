# Protocol101 Stage-1 Adversarial Audit Specification

Status: FROZEN SPECIFICATION FOR GOAL A  
Date: 2026-07-26

## Purpose

This specification governs a read-only adversarial audit of the frozen
Protocol101 H0-H3 Stage-1 evidence.

The audit must answer one practical question:

> Is a model economically bad, is the experiment invalid, or is potentially
> useful economic evidence being rejected by a governance rule?

These are different outcomes. The audit may not collapse them into one
`PASS`/`FAIL` label.

This specification does not authorize model selection, G9, protected-holdout
access, candidate training, strategy redesign, or paper trading.

## Governing References

Read these before creating the audit preregistration:

- `v4/docs/PROTOCOL101_TRADER_CHARTER.md`
- `v4/docs/PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
- `v4/docs/PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
- `v4/docs/PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md`
- `v4/docs/PROTOCOL101_G4_HOLDOUT_REVISION_2026_07_19.md`
- `v4/docs/PROTOCOL101_G8_CALIBRATION_REVISION_2026_07_26.md`
- `v4/docs/PROTOCOL101_TRAINING_REGIMEN_AND_INTEGRITY_CONTROL_2026_07_26.md`
- `v4/docs/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`
- `v4/docs/PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md`
- frozen H0-H3 RUN/GATE/AUDIT artifacts;
- `v4/audit/autoresearch/protocol101_stage1_gate_validity_and_integrity_audit_attempt001/`.

## Frozen Current State

- H0-H3 comprise 28 hypothesis-policy rows and 420 fitted fold/seed units.
- All existing fitted models, thresholds, epsilons, folds, policies, trades,
  and economic results are immutable evidence.
- G8 is mandatory reporting but is not an eligibility gate under the signed
  2026-07-26 revision.
- The previous cross-hypothesis selection used pre-amendment G8 semantics and
  is historical, not a current selection result.
- H2 policy 5 is provisional. It is not selected until a later model-free
  global reaggregation and independent selection accept it.
- Seed 45 is unspent.
- The protected holdout is unopened.

## Three Required Verdicts

Every one of the 28 hypothesis-policy rows must receive three separate
verdicts.

### 1. Experiment Validity

Allowed values:

- `TRUSTWORTHY_WITH_STATED_LIMITS`
- `INVALID_EVIDENCE`
- `INSUFFICIENT_EVIDENCE`

This verdict covers feature timing, leakage, CV roles, simulator/accounting
semantics, artifact integrity, selection contamination, and effective sample
size.

### 2. Economic Evidence

Allowed values:

- `NO_EVIDENCE_OF_SIGNAL`
- `EVIDENCE_CONSISTENT_WITH_SIGNAL`
- `INCONCLUSIVE`

No row may be called confirmed genuine signal before fresh-seed confirmation,
the protected holdout, and candidate-specific transfer.

### 3. Governance Eligibility

Allowed values:

- `ELIGIBLE_UNDER_CURRENT_G1_G7_G8_REPORTED`
- `INELIGIBLE_UNDER_CURRENT_RULES`
- `NOT_EVALUABLE`

G8 must be reported but excluded from eligibility. G9 must be
`NOT_RUN_GOVERNED`.

## Units Of Analysis

The audit must keep these units distinct:

| Unit | Meaning |
|---|---|
| Raw row | One contract at one minute |
| Decision unit | The complete eligible ladder at one completed minute |
| Trade opportunity | One serially executable entry decision |
| Statistical unit | One trading session |
| Regime unit | A governed era/session cluster |

Report raw row counts, decision-minute counts, trade counts, and unique session
counts separately. Use session-clustered or session-weighted uncertainty.

Verify whether minutes with more eligible contracts receive greater training
weight. If so, classify it as intended, harmless, or a defect.

## Observable Trading Problem

The regimen must represent this decision:

1. Observe only information available at a completed minute.
2. Examine the eligible SPXW 0DTE call/put ladder.
3. Score competing contracts.
4. Abstain unless an opportunity clears the frozen action rules.
5. Select at most one contract.
6. Enter one contract with conservative ask accounting.
7. Occupy the account until the fixed Stage-1 exit policy closes the trade.
8. Exit with conservative bid accounting.
9. Preserve fees, affordability, daily loss controls, forced-flat behavior,
   and account continuity.

Audit these as separate model qualities:

- target regression;
- within-minute contract ranking;
- abstention;
- score-to-action policy conversion;
- serial account utility.

## Read-Only Boundary

Treat all existing feature, label, training, simulator, gate, selection, and
governance source code as read-only.

Audit-only scripts, tests, models, and outputs may be created only beneath:

```text
v4/audit/autoresearch/
  protocol101_stage1_training_regimen_adversarial_audit_attempt001/
```

Existing audit utilities may be copied there and repaired locally. If governed
project code requires a change, record a defect and proposed repair. Do not
make the repair in Goal A.

At audit start, freeze:

- `git status --porcelain=v2`;
- hashes of every governing document;
- hashes of every consumed source artifact;
- hashes of governed code inspected.

At completion, prove that no path outside the authorized audit output directory
was newly changed by Goal A. Pre-existing repository changes must be recorded
and left untouched.

## Checkpoint A: Evidence Freeze And Truth Map

Create an inventory of:

- signed documents and amendments;
- H0-H3 preregistrations;
- 420 fitted units and model hashes;
- fold/session manifests;
- null and heuristic artifacts;
- RUN/GATE/AUDIT packets;
- historical selection packets;
- G8 repair/audit packets;
- side-effect and protected-data flags.

Reconcile historical status and current governed status without editing the
historical artifacts.

Required primary artifact:

```text
row_failure_waterfall.csv
```

It must contain one row for each hypothesis-policy combination and at least:

```text
hypothesis
policy_index
policy_name
historical_status
current_governed_status
unique_sessions
decision_minutes
trade_count
raw_outer_validation_pnl
fee_adjusted_pnl
profitable_fold_count
matched_null_z
heuristic_delta_pnl
worst_seed_pnl
worst_seed_null_z
era_verdict
stress_verdict
max_drawdown
calmar
minimum_equity
trades_per_day
zero_trade_days
g8_ece
first_failing_gate
all_failing_gates
economically_positive_before_gates
experiment_validity_verdict
economic_evidence_verdict
governance_eligibility_verdict
primary_failure_attribution
plain_english_diagnosis
```

`primary_failure_attribution` must be exactly one of:

- `MODEL_ECONOMICS`
- `EXPERIMENT_INVALIDITY`
- `GOVERNANCE_RULE`
- `INSUFFICIENT_EVIDENCE`
- `NONE`

No cross-hypothesis ranking or winner selection is allowed.

## Checkpoint B: Trading Semantics

Independently verify from governed code and frozen artifacts:

- completed-minute feature timing;
- exact model-facing feature allowlists;
- contract identity and ladder construction;
- wait/call/put actions;
- deterministic slot selection;
- candidate and intersection guards;
- one contract and one open position;
- account continuity across trades and sessions;
- premium-plus-fee affordability;
- ask entry and bid exit;
- $3 fee and signed sensitivities;
- 5% session-starting-equity daily stop;
- forced flat;
- no labels, future paths, MFE/MAE, or outcomes in model alpha;
- zero protected/recorder/parity/confirmation sessions in model fitting,
  calibration, or threshold selection.

Trace representative frozen decisions end to end:

- profitable trade;
- losing trade;
- abstention;
- guard-blocked or unaffordable opportunity;
- daily-stop event, if present;
- forced-flat event, if present.

If a requested event does not exist, report `NOT_OBSERVED`; do not manufacture
one.

Describe observable decision behavior and feature associations. Do not claim
feature attribution proves what the model causally learned.

## Checkpoint C: Machine-Learning Validity

Verify:

- five chronological expanding outer folds;
- one-session embargo;
- no session shuffling;
- disjoint fit, training-tail calibration, and outer-validation roles;
- fold-local threshold and epsilon fitting;
- no outer-validation value in training or tuning;
- exact HGB bounds, hyperparameters, seeds, and target;
- payoff/return-on-premium target, never win probability;
- all seven exit policies as alternative Stage-1 labels;
- effective sample size and within-session dependence;
- session-weighted and session-clustered uncertainty;
- campaign multiplicity across 28 hypothesis-policy rows;
- repeated-CV process-overfitting risk;
- whether target labels overweight cheap-premium lottery tickets;
- whether fixed exits conflate entry quality with exit-policy quality.

For each row, decompose economics by:

- entry premium bucket;
- dollar PnL and return on premium;
- delta and moneyness;
- call/put side;
- time of day;
- audit-only liquidity/guard bucket;
- exit policy;
- predicted-score decile;
- duration and account-occupancy bucket.

Raw quote/liquidity fields used for this decomposition remain audit-only.

## Checkpoint D: Preregistered Falsification Smokes

Write `diagnostic_preregistration.json` before running a diagnostic. It must
freeze exact inputs, seeds, randomization unit, repetitions, metrics, expected
direction, pass criterion, invalidating result, and maximum repair count.

Diagnostics may not be changed after their economic output is inspected.

Disposable fits must:

- use only governed non-holdout sessions;
- live beneath the audit directory;
- carry `NON_CANDIDATE=true`;
- never enter a research registry, selection packet, promotion, or later
  candidate evidence.

### D1. Label-Permutation Control

- Diagnostic model: exact H2-shaped bounded HGB and policy 5, used only as a
  pipeline canary.
- Folds/hyperparameters: same frozen Stage-1 definitions.
- Randomization: shuffle complete 30-minute decision blocks within each
  session; move the whole ladder's labels together.
- Repetitions: 20.
- Seeds: integers 8600 through 8619.
- Primary metrics: pooled fee-adjusted PnL, G1, and G2.
- Expected result: median pooled PnL `<= 0`, median matched-null z `< 1.0`,
  and no more than 1 of 20 repetitions jointly passes G1 and G2.
- Invalidating result: 2 or more repetitions jointly pass G1/G2, or median
  z `>= 1.0`, absent a preregistered mathematical explanation.

If unequal session/block geometry prevents the exact permutation, mark the
diagnostic `BLOCKED_SPECIFICATION_MISMATCH`. Do not invent a substitute.

### D2. Time-Shift Sensitivity

- Apply frozen models without refitting.
- Shift authorized context features backward by 1, 5, and 15 completed minutes
  within session; never cross a session boundary.
- Report target, ranking, action, and serial-PnL changes.
- This is diagnostic because causal market features may be autocorrelated.
- Invalidating result: none by magnitude alone. A result becomes a P1 concern
  only if timestamps or supposedly shifted rows did not actually change, or if
  impossible future data is discovered.

### D3. Future-Feature Rejection Positive Control

- Add a disposable field named
  `AUDIT_ONLY_FUTURE_15M_REALIZED_OPTION_RETURN`.
- Attempt to pass it through the exact model-facing contract/preflight.
- Expected result: fail closed before fitting.
- Pass criterion: explicit unauthorized/future-feature rejection.
- Invalidating result: governed fitting begins or an artifact is accepted.
- Do not fit a model with this field.

### D4. Random-Score Null Reproduction

- Use the exact frozen matched-random null implementation and corpus.
- Repetitions and seed come from the frozen null artifact; do not substitute.
- Independently reproduce its count, mean, standard deviation, and relevant
  quantiles.
- Numeric tolerance: exact integer counts and absolute floating difference
  `<= 1e-9` when deterministic serialization permits; otherwise document the
  smallest justified machine tolerance before comparison.

### D5. Fixed-Heuristic Reproduction

- Reproduce the signed VWAP-side plus best-single-shape baseline from primary
  inputs.
- Expected policy: 5, with the signed deterministic tie-break.
- Expected pooled fee-adjusted PnL: `$4,592`.
- Pass criterion: exact trade identities and PnL to cent precision.

### D6. Source-Identity Control

- Use only already-authorized, unsealed, temporally paired parity samples.
- Never access sealed recorder evidence.
- Pair by session, completed minute, and static ladder slot.
- Use the already signed scoped feature set and frozen source-discriminator
  family.
- Report leave-one-day-out AUC per fold and pooled.
- Signed diagnostic ceiling: pooled AUC `<= 0.55`.
- If no sufficient authorized paired sample exists, report
  `BLOCKED_INSUFFICIENT_AUTHORIZED_EVIDENCE`.

### D7. Synthetic Simulator Reference

Create a hand-checkable synthetic packet containing:

- two sessions;
- one profitable trade;
- one losing trade;
- one unaffordable candidate;
- one daily-stop block;
- one forced-flat exit;
- one final pending trade requiring session realization.

Freeze expected actions, cash, fees, fills, PnL, equity, and block reasons
before execution. Require exact reproduction to cent precision.

### D8. Friction Monotonicity

- Hold a frozen trade sequence and fills timestamps fixed.
- Recompute PnL at the signed fee/fill ladder.
- Worse friction must never improve trade, session, or pooled PnL.
- Dynamic replay with changed affordability is a separate descriptive
  diagnostic and is not subject to strict monotonicity.

### D9. Within-Timestamp Ordering

- Permute contract/file row order only within the same decision timestamp.
- Run 20 fixed permutations with seeds 8700 through 8719.
- Require identical action and selected contract under the signed deterministic
  tie-break.
- Do not permute chronological trade order.

### D10. Duplication And Overlap

On disposable copies only, inject:

- duplicate contract rows;
- duplicate decision rows;
- duplicate sessions;
- an overlapping trade proposal.

Each must fail closed or be deterministically deduplicated with an explicit
auditable reason. No injected copy may enter governed evidence.

## Checkpoint E: Gate Audit

Per-row gate recomputation from frozen evidence is allowed.

Globally specified counterfactual application of a gate definition to each
individual row is allowed only when clearly labeled diagnostic.

Forbidden:

- cross-hypothesis pooling;
- winner ranking;
- global eligibility reaggregation;
- candidate selection.

For G1-G9, record:

- trader-charter purpose;
- hard/report/routing/confirmation role;
- exact inputs and formula;
- independent reproduced value where applicable;
- feasibility under oracle/null controls;
- redundancy or contradiction;
- reward-hacking surface;
- current recommendation.

G9 is specification-only. All result-dependent G9 fields must be
`NOT_RUN_GOVERNED`.

Do not recommend loosening a gate because a preferred row failed. Any proposed
change must be justified by charter alignment, statistical validity,
oracle/null feasibility, or a demonstrated implementation defect, and it must
apply globally.

## Cross-Implementation Verification

Goal A's producer may create a second metric implementation that:

- reads primary frozen artifacts directly;
- ignores producer verdict fields;
- does not import producer audit modules;
- recomputes a preregistered sample;
- compares with frozen tolerances.

Call this `cross_implementation_verification`, not an independent audit.

Actual independent acceptance requires a fresh Codex thread or separate Goal
that receives the frozen evidence manifest before the producer narrative.

## Defect Severity

- `P0`: direct leakage, protected-data use, impossible fills, false
  accounting, overlapping headline trades, or future information in decisions.
- `P1`: invalid CV, selection contamination, reward-hackable hard gate,
  material simulator mismatch, stale benchmark, or unaddressed multiplicity.
- `P2`: unclear reporting, weak diagnostic coverage, or non-material issue.

## Evidence Labels

Every material statement must be labeled:

- `REPRODUCED`
- `CONSISTENT_WITH_EVIDENCE`
- `INFERRED`
- `BLOCKED`
- `NOT_TESTABLE`

## Required Output

```text
v4/audit/autoresearch/
  protocol101_stage1_training_regimen_adversarial_audit_attempt001/
    preregistration.json
    diagnostic_preregistration.json
    source_inventory.json
    row_failure_waterfall.csv
    row_failure_waterfall.json
    trading_semantics_audit.json
    ml_validity_audit.json
    reward_hacking_smokes.json
    gate_validity_matrix.json
    defects.json
    progress.json
    commands.jsonl
    hashes.sha256
    summary.json
    report.md
    cross_implementation_verification/
      preregistration.json
      summary.json
      report.md
```

## Terminal Routes

- `regimen_valid_ready_for_model_free_reaggregation`
- `regimen_repairable_owner_amendment_required`
- `regimen_invalid_redesign_required`
- `regimen_blocked_insufficient_evidence`

The first route requires:

- zero unresolved P0/P1 defects;
- expected preregistered falsification behavior;
- reproduced trading semantics;
- exact frozen truth-map reconciliation;
- reproducible applicable gates;
- no required experiment repair.

Missing evidence never becomes a pass. Continue unaffected audit work, mark the
claim `BLOCKED` or `NOT_TESTABLE`, and use the blocked route only when missing
evidence prevents a defensible overall decision.

## Stop Boundary

Do not:

- retrain or retune H0-H3;
- reaggregate across hypotheses;
- rank or select a winner;
- spend seed 45;
- calculate a G9 result;
- open the protected holdout;
- access sealed recorder evidence;
- change a signed feature, strategy, simulator, gate, or selection contract;
- contact IBKR or a broker;
- submit paper orders;
- change promotion, runtime, launchd, paid-data, or real-money state.

Highest allowed claim:

> Protocol101 Stage-1 training-regimen adversarial audit complete.

