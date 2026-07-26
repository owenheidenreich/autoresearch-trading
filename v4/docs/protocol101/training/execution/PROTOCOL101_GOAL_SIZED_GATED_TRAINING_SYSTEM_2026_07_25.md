# Protocol101 Goal-Sized Gated Training System

Status: ACTIVE OPERATIONAL CONTROL  
Effective: 2026-07-25

## Purpose

This document controls how Protocol101 moves from the first real H0 training
run to IBKR paper trading without building on unverified work.

It does not change the signed feature contract, trader charter, H0-H3 design,
G1-G9 thresholds, simulator, folds, or holdout rules. It divides that approved
work into small Codex goals with explicit stopping points.

The central rule is:

> One goal owns one gate outcome and never starts the next gate. An
> evidence-producing goal may diagnose, repair, and retry non-substantive
> plumbing failures needed to reach its own outcome, but it may not change the
> frozen scientific or trading contract.

## Authority And Precedence

If documents disagree, use this order:

1. `PROTOCOL101_TRADER_CHARTER.md`
2. `PROTOCOL101_SCOPED_SYNCHRONIZATION_DECISION_2026_07_25.md`
3. `PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md` plus the signed G4
   and G8 revisions
4. `PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md`
5. this goal-sized execution system
6. `PROTOCOL101_STAGE1_TO_LIVE_EXECUTION_PLAN.md`
7. older plans and historical packets

The signed documents control substance. This document controls sequencing.

## Current Starting Checkpoint

Gate `F0_FOUNDATION_FREEZE` is complete.

Machine evidence:

- final machinery status:
  `protocol101_scoped_canonical_stage1_machinery_readiness/summary.json`
- required state: `ready_for_separate_owner_approved_H0`
- blockers: none
- research training performed: false
- runner freeze:
  `protocol101_scoped_canonical_stage1_plumbing_smoke_validation/runner_freeze.json`

Frozen at F0:

- contract `protocol101-scoped-canonical-stage1-v1`;
- exact 17-feature allowlist and quarantine boundary;
- H0, H1, H2, and H3 definitions;
- 301-session accepted registry and 271-session fold scope;
- five chronological expanding folds and one-session embargo;
- seven menu-v2 exit policies and three initial seeds;
- bounded HGB family and payoff/return-on-premium target;
- 1x measured-noise primary evaluation and 0x/0.5x/2x diagnostics;
- selection dead-band, slot-margin rule, and deterministic fallback;
- pessimistic fill rung and $2.60/$3.00/$4.00 fee reports;
- serial simulator v4, one account, one contract, and 5% daily stop;
- exact null, fixed heuristic, and G1-G9 rules;
- runner, contract, simulator, validator, and gate-aggregator hashes.

Changing any F0 item requires a separate owner-approved amendment and a new F0
smoke/freeze. It cannot be changed inside a training goal.

## The Goal Contract

Every future Codex goal prompt must contain these sections:

1. `Goal ID`: one stable identifier from this document.
2. `Objective`: one sentence and one deliverable class.
3. `Starting checkpoint`: exact accepted freeze/hash that authorizes the goal.
4. `Allowed work`: the only files, computation, or evidence the goal may touch.
5. `Forbidden work`: especially the next gate, protected data, broker actions,
   paid downloads, promotion, and runtime changes.
6. `Frozen inputs`: exact contract, corpus, folds, seeds, policies, fees,
   simulator, references, and source hashes.
7. `Required outputs`: exact artifact directory and schemas.
8. `Mechanical verification`: commands and invariant checks.
9. `Pass condition`: a binary artifact-backed definition.
10. `Failure routing`: what happens for blocker, bug, no signal, or real signal.
11. `Stop line`: an explicit instruction to stop without beginning the next
    goal.

A completion message is not a gate. The required artifact and its verification
must exist.

## Goal-Size Rules

- One evidence-producing run per goal.
- One hypothesis per training goal.
- No training goal may interpret or promote its own result.
- No verification goal may fit a model, select a threshold, or repair code.
- A RUN goal may perform a bounded blocker-recovery loop before producing
  evidence or after voiding affected partial evidence.
- Mechanical recovery is limited to paths, indexing, readiness adapters,
  deterministic resume behavior, artifact writers/verifiers, imports,
  environment wiring, and equivalent operational defects.
- Mechanical recovery must be logged, narrowly tested, and re-frozen before
  evidence production resumes.
- A repair may not change features, labels, data membership, folds, embargo,
  seeds, policy semantics, model family, objective, calibration semantics,
  noise assumptions, guards, fills, fees, simulator economics, gates, or
  protected-data rules.
- No repair goal may preserve an affected result as valid evidence.
- No attribution goal may change code or rerun training.
- No goal may read the protected holdout unless its exact goal ID is the
  owner-authorized one-shot holdout execution.
- No goal may contact IBKR unless its exact gate explicitly authorizes no-order
  shadow or paper execution.
- No goal may continue automatically to another ID in this document.
- When a goal ends, its result returns to the owner/coordinator before the next
  prompt is drafted.

## Blocker Recovery Loop

A goal must not mark itself blocked merely because its first attempt encounters
a mechanical problem.

For every blocker:

1. classify it as `transient`, `mechanical_non_substantive`,
   `scientific_contract`, `protected_action`, or `external_owner_required`;
2. write the diagnosis and evidence before changing anything;
3. retry transient failures with bounded backoff;
4. repair `mechanical_non_substantive` failures narrowly, run focused tests,
   regenerate any affected smoke/readiness/freeze hashes, and retry the same
   goal;
5. if partial result artifacts were affected, preserve them, mark them void,
   and use a new immutable attempt directory or a preregistration-preserving
   resume path;
6. ask the owner and wait when the solution would change the scientific
   contract, access protected data, spend money, contact a broker, or alter
   paper/real-money state.

An owner question is a live waiting state, not permission to mark the goal
complete or abandon it. A goal may report `blocked` only when the blocker is
genuinely external, the owner or external state is required, and the repeated
blocked-audit threshold has been met.

Recovery ends when the current gate's original pass condition is proven. It
does not authorize the next gate or any interpretation of model performance.

## Freeze Standard

Every accepted gate writes an immutable freeze packet containing:

- gate ID and status;
- input artifact paths and SHA-256 hashes;
- source-code hashes or commit identifier;
- corpus, registry, fold, feature, model, policy, seed, and simulator hashes;
- preregistration hash where applicable;
- output artifact hashes;
- side-effect audit;
- known limitations;
- exact next allowed goal IDs;
- superseded or void attempt IDs.

A freeze is append-only. If a frozen file must change, create a new version,
mark the old freeze superseded, and repeat every downstream gate that depended
on it.

## Verification Roles

Each evidence gate has two checks:

- `M` - mechanical verification: schemas, hashes, row counts, invariants,
  side-effect flags, and frozen rules.
- `I` - independent verification: a fresh read-only Codex task or other checker
  that did not produce the evidence tests claims against artifacts.

Owner approval `O` is additionally required for:

- real H0-H3 training authorization;
- any contract or gate amendment;
- G9 candidate selection;
- protected-holdout opening;
- Stage-2 execution;
- live shadow activation;
- paper promotion and paper-submit;
- any real-money review.

The evidence-producing task cannot mark its own result accepted.

## Master Gate Map

| Gate | Evidence-producing goal | Acceptance goal | Frozen output |
|---|---|---|---|
| F0 Foundation | Complete | Complete | Signed contract + runner freeze |
| H0 | `S1-H0-RUN` | `S1-H0-GATE`, then `S1-H0-AUDIT` | Accepted H0 verdict |
| H1 | `S1-H1-RUN` | `S1-H1-GATE`, then `S1-H1-AUDIT` | Accepted H1 verdict |
| H2 | `S1-H2-RUN` | `S1-H2-GATE`, then `S1-H2-AUDIT` | Accepted H2 verdict |
| H3 | `S1-H3-RUN` | `S1-H3-GATE`, then `S1-H3-AUDIT` | Accepted H3 verdict |
| Stage-1 selection | none | `S1-ALL-SELECT` | One candidate or an explicit stop/branch |
| Fresh-seed confirmation | `C2-G9-RUN` | `C3-G9-AUDIT` | G9 pass or candidate burn |
| Final candidate fit | `C4-FINAL-FIT` | `C5-FINAL-FIT-AUDIT` | Immutable pre-holdout candidate |
| Protected holdout | `C7-HOLDOUT-RUN` | `C8-HOLDOUT-AUDIT` | One-shot pass or candidate burn |
| Research candidate | none | `C9-RESEARCH-CANDIDATE-FREEZE` | Historical candidate freeze |
| Runtime transfer | `R3-PAIRED-TRANSFER-RUN` | `R4-PAIRED-TRANSFER-AUDIT` | Paired transfer verdict |
| Live shadow | `R6-LIVE-SHADOW-COLLECTION` | `R7-LIVE-SHADOW-AUDIT` | No-order live verdict |
| IBKR paper | `P3-PAPER-EVIDENCE-COLLECTION` | `P4-PAPER-AUDIT` | Paper verdict |
| Real money | none | `L1-REAL-MONEY-REVIEW` | Separate owner decision |

No row unlocks until the prior row's acceptance freeze exists. Conditional
machinery, repair, attribution, and authorization goals sit between rows when
required.

## Stage 1: H0-H3 Entry Search

All four hypotheses run in order: H0, H1, H2, H3. Interim success does not skip
the remaining hypotheses.

### Goal Chain Per Hypothesis

For `HX` in H0, H1, H2, H3:

Artifact roots are concrete:

| Hypothesis | Primary run directory | Gate directory |
|---|---|---|
| H0 | `protocol101_scoped_canonical_stage1_h0_attempt001` | `protocol101_scoped_canonical_stage1_h0_attempt001_gates` |
| H1 | `protocol101_scoped_canonical_stage1_h1_attempt001` | `protocol101_scoped_canonical_stage1_h1_attempt001_gates` |
| H2 | `protocol101_scoped_canonical_stage1_h2_attempt001` | `protocol101_scoped_canonical_stage1_h2_attempt001_gates` |
| H3 | `protocol101_scoped_canonical_stage1_h3_attempt001` | `protocol101_scoped_canonical_stage1_h3_attempt001_gates` |

#### `S1-HX-RUN` - Produce Model Evidence

Objective: preregister and execute exactly one full hypothesis batch.

Allowed:

- seven policies x three seeds x five folds;
- fold-local fitting, calibration, threshold selection, 1x-noise validation,
  and strict serial replay;
- resumable completion of the same immutable batch.

Required output:

```text
v4/audit/autoresearch/protocol101_scoped_canonical_stage1_hx_attempt001/
  preregistration.json
  runner_plan.json
  progress.json
  units/.../summary.json
  units/.../model.pkl
  summary.json
```

Pass to the next goal only when:

- preregistration predates result artifacts;
- all 105 policy/seed/fold units exist;
- every unit and model hash is recorded;
- batch status is
  `unit_execution_complete_pending_preregistered_gate_aggregation`;
- no protected, recorder, parity, or confirmation data was read;
- no broker, paper, promotion, paid-data, runtime, or real-money action occurred.

Stop line: do not aggregate G1-G8, interpret edge, run another hypothesis, run
G9, or read the holdout. Follow the blocker-recovery loop for mechanical
failures, but stop for owner input before any substantive contract change.

#### `S1-HX-GATE` - Compute Mechanical Gates

Objective: compute G1-G7 eligibility and the required report-only G8
diagnostic without fitting anything.

Allowed:

- read the frozen HX batch, exact null, fixed heuristic, and era manifest;
- verify all hashes;
- reconstruct chronological out-of-fold account continuity;
- compute G1-G7, report G8, compute fee/fill/noise diagnostics, and produce
  mechanical routing.

Required output:

```text
v4/audit/autoresearch/
  protocol101_scoped_canonical_stage1_hx_attempt001_gates/
    summary.json
    gate_results.json
    report.md
```

Pass to the next goal only when:

- aggregator blockers are empty;
- all 105 unit hashes and preregistration hash match;
- the exact fold/registry/reference/simulator hashes match F0;
- every G1-G8 number is present per policy and seed, with G8 excluded from
  eligibility under the signed 2026-07-26 revision;
- G9 remains false and unexecuted;
- no fit or threshold selection occurred.

Stop line: report the mechanical route only. Follow the blocker-recovery loop
for non-substantive aggregation plumbing failures, but do not change gate
definitions or frozen evidence. Do not accept the verdict, run attribution,
start another hypothesis, or act on a candidate selection.

#### `S1-HX-AUDIT` - Independently Accept, Void, Or Escalate

Objective: independently verify the HX run and gate packet, then freeze one
accepted routing verdict.

Allowed:

- read-only artifact inspection;
- recompute representative folds, trades, PnL, drawdown, z-scores, frequency,
  ECE, fee sensitivity, and side-effect claims;
- compare the result to signed G1-G9 language;
- write an independent verification and freeze packet.

Allowed verdicts:

- `accepted_eligible_G1_G7_G8_reported`;
- `accepted_real_entry_signal_fixed_exit_or_other_gate_failure`;
- `accepted_no_real_signal`;
- `void_implementation_or_artifact_defect`;
- `blocked_insufficient_evidence`.

Pass only when an independent freeze records all checked hashes and one verdict.

Stop line: do not fix, rerun, start HX+1, run attribution, or select a candidate.

### Conditional Goals Inside Stage 1

#### `S1-HX-FIX-N`

Use only after `void_implementation_or_artifact_defect`.

Objective: repair one precisely attributed machinery defect and reestablish F0.

Rules:

- the affected attempt is permanently marked void;
- no economic result from it may be reused;
- repair, tests, disposable smoke, independent validation, and a new runner
  freeze occur before a new run goal;
- if more than one independent defect exists, use separate fix goals.

The next evidence run uses a new attempt ID.

#### `S1-HX-ATTRIBUTION`

Use only after an accepted packet has real signal but fails another gate.

Objective: explain the failure from existing artifacts only.

Allowed outputs:

- entry-signal vs fixed-exit attribution;
- era, seed, concentration, calibration, frequency, fill, or drawdown diagnosis;
- a routing recommendation.

Forbidden:

- training, tuning, code changes, gate changes, or a new feature.

#### `S1-HX-CONSERVATIVE-DESIGN` And `S1-HX-CONSERVATIVE-RUN`

The one nearby conservative batch is never automatic.

It requires:

1. accepted HX attribution;
2. a separate preregistered design goal that freezes the one permitted nearby
   batch before results;
3. owner approval;
4. a separate run goal;
5. the same GATE and AUDIT chain.

There is no third batch and no open-ended search.

### `S1-ALL-SELECT` - Cross-Hypothesis Selection

This goal is allowed only after accepted AUDIT freezes exist for H0-H3 and any
approved conservative batch.

Objective: compare frozen eligible packets and select at most one Stage-1
candidate configuration.

Rules:

- eligibility requires all G1-G7; G8 must be present and reported but is not
  an eligibility gate under the signed 2026-07-26 revision;
- choose among eligible candidates by the signed plain fee-adjusted strict
  serial PnL rule and frozen deterministic tie-breaks;
- no new model fit, threshold, feature, or utility;
- if no candidate is eligible, do not manufacture one.

Routes:

- `selected_for_G9`;
- `stage2_design_may_be_drafted` when accepted entry signal and path evidence
  support that branch;
- `stage1_no_accepted_edge_stop_and_redesign`;
- `selection_blocked_by_inconsistent_packets`.

The selection freeze identifies the exact hypothesis, policy, configuration,
source packet hashes, and next allowed goal.

## Candidate Hardening

### `C1-G9-MACHINERY`

Run only if the selected configuration needs new execution machinery.

Objective: build and smoke the fresh-seed confirmation path without running the
candidate confirmation.

The machinery must be independently frozen before `C2-G9-RUN`.

### `C2-G9-RUN`

Objective: execute the selected configuration with one never-used seed.

Pass: fresh seed independently satisfies G1, G2, and G4 exactly as signed.

Failure burns the candidate. No replacement seed may be tried.

### `C3-G9-AUDIT`

Objective: independently verify G9 and freeze pass or burn.

No fitting or holdout access.

### `C4-FINAL-FIT`

Objective: fit one final candidate on the full allowed non-holdout development
scope using the selected frozen configuration and training-tail calibration.

The weights, threshold, confidence map, epsilon, contract, guards, source code,
and candidate manifest freeze before the holdout is opened.

This goal must not read or score the protected holdout.

### `C5-FINAL-FIT-AUDIT`

Objective: independently verify the final-fit manifest, data exclusions, model
hashes, and reproducibility.

Route only a green immutable candidate to holdout authorization.

### `C6-HOLDOUT-AUTHORIZATION`

Objective: produce a read-only one-shot holdout readiness packet and request
the owner's explicit token.

It must prove the candidate was frozen first and that no holdout result has
been read.

### `C7-HOLDOUT-RUN`

Objective: open and score the protected 2025-05-16 through 2025-06-30 block
exactly once for the frozen candidate.

No tuning or repair is allowed. The result is spend-once evidence.

### `C8-HOLDOUT-AUDIT`

Objective: independently verify the one-shot result and freeze pass, audit
trigger, or permanent candidate failure.

A failed holdout cannot route back to a threshold, feature, or model adjustment
using that holdout.

### `C9-RESEARCH-CANDIDATE-FREEZE`

Objective: bind the passing G1-G9, final-fit, and holdout evidence into one
research-candidate manifest.

This freeze still does not change the paper default.

## Stage-2 Branch

Stage-2 is conditional, not part of an entry-training goal.

If Stage-1 finds accepted entry signal but fixed exits are the documented
constraint:

1. `E1-STAGE2-DESIGN` drafts the objective, labels, causal state, baselines,
   nulls, gates, and entry-model freeze.
2. Owner signs that design.
3. `E2-STAGE2-MACHINERY` builds and smokes only.
4. Stage-2 evidence then uses the same RUN -> GATE -> AUDIT pattern.
5. A composed entry/exit candidate returns to candidate hardening.

No Stage-2 execution is implied by a Stage-1 routing packet.

## Runtime Transfer And Live Shadow

### `R1-RUNTIME-ADAPTER`

Objective: implement the frozen candidate in the no-order runtime path.

No live activation and no broker call.

### `R2-RUNTIME-ADAPTER-AUDIT`

Objective: verify exact feature, selection, guard, account-state, model, and
schema identity between frozen research and runtime.

### `R3-PAIRED-TRANSFER-RUN`

Objective: run the frozen candidate on allowed paired historical and IBKR
evidence with no orders.

### `R4-PAIRED-TRANSFER-AUDIT`

Objective: independently gate action agreement, selected-slot agreement,
clean-window resets, score drift, guard behavior, and disagreements.

### `R5-LIVE-SHADOW-AUTHORIZATION`

Objective: owner-approved activation plan for a no-order live shadow only.

### `R6-LIVE-SHADOW-COLLECTION`

Objective: collect at least the preregistered number and regime mix of live
no-order sessions. It produces evidence but no final verdict.

### `R7-LIVE-SHADOW-AUDIT`

Objective: compare settled live decisions to same-input replay and freeze the
shadow verdict.

Failure routes to runtime repair, never directly to model retuning.

## IBKR Paper Trading

### `P1-PAPER-PROMOTION-PACKET`

Objective: assemble historical, holdout, transfer, shadow, risk, rollback, and
known-weakness evidence for an owner paper-only decision.

No runtime flag changes or orders.

### `P2-PAPER-ACTIVATION`

Objective: after explicit owner approval, activate only the guarded IBKR paper
path for the frozen one-contract candidate.

Real money remains false. Kill switch, 5% daily stop, affordability, forced
flat, and logging must be verified before the first order.

### `P3-PAPER-EVIDENCE-COLLECTION`

Objective: collect the preregistered minimum paper sessions and complete
decision/order/fill/exit logs.

### `P4-PAPER-AUDIT`

Objective: independently reconcile decisions, fills, slippage, safety,
same-day replay, PnL band, drawdown, concentration, and charter metrics.

Routes:

- continue paper;
- restrict or roll back;
- paper-validated candidate eligible for real-money review.

### `L1-REAL-MONEY-REVIEW`

This is a separate owner decision, not a continuation of paper trading.
Default route is defer. Any approval requires its own limits, capital,
rollback, runtime flag, and promotion packet.

## Bug And Failure Law

When a goal fails:

- data/artifact missing: `blocked`, repair the missing prerequisite only;
- implementation defect: mark evidence attempt `void`, repair, re-smoke,
  independently refreeze, then rerun under a new attempt ID;
- economic gate failure: accept the result, do attribution separately, never
  loosen the gate;
- parity/runtime failure: repair runtime or contract transfer, not historical
  PnL;
- holdout failure: burn candidate, never tune to the holdout;
- paper failure: disable/restrict first, diagnose second.

No failed evidence disappears. It remains in the registry with its reason.

## Status Discipline

After every accepted AUDIT or FREEZE goal:

1. update `PROTOCOL101_STATUS.md`;
2. append the experiment/freeze ledger;
3. record exact next allowed goal IDs;
4. state the highest allowed claim;
5. state which actions remain unauthorized.

Only accepted verification freezes can move the project checkpoint.

## Immediate Next Goal

The next and only currently authorized goal shape is:

```text
S1-H0-RUN - preregister and execute the full H0 offline batch.
```

The evidence command for that goal is:

```bash
~/.autoresearch-trading/runtime-venv/bin/python \
  -m v4.scripts.run_protocol101_scoped_stage1_hgb_runner \
  --mode train-hypothesis \
  --hypothesis H0 \
  --out-dir v4/audit/autoresearch/protocol101_scoped_canonical_stage1_h0_attempt001 \
  --owner-approved-offline-training
```

Do not add `--force` to a first execution. Resumption must first prove the
existing preregistration and frozen hashes are unchanged.

That prompt may authorize offline H0 model fitting and threshold selection. It
must stop at
`unit_execution_complete_pending_preregistered_gate_aggregation`.

It must not aggregate G1-G8, interpret whether H0 has edge, start H1, run G9,
read the holdout, contact IBKR, submit paper orders, change the paper default,
or touch real-money paths.
