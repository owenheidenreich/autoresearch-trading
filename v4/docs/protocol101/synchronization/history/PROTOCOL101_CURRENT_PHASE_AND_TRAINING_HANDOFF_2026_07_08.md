# Protocol101 Current Phase And Training Handoff

Generated: 2026-07-08

## Read This First

This document is the handoff from the historical/live synchronization thread
into the offline training thread.  It exists because the proof artifacts are
spread across audit folders, and future agents were getting pulled back into
old `protocol101-live-v1` synchronization work.

The synchronization phase is closed unless new evidence contradicts the packet
below.  The active work is now the fair-contract training phase.

## Current Phase

```text
phase: Protocol101 fair-contract training readiness
sync status: closed
training status: 15-month 5-fold offline training dry-run ready, owner approval required
paper-submit: blocked
promotion/default changes: blocked
real-money trading: blocked
```

The project is no longer trying to prove that the old historical stack was
trustworthy.  It is trying to train a new paper-trade-ready SPXW 0DTE long
options candidate under the certified fair game.

## Certified Training/Live Contract

The only approved contract for the next training phase is:

```text
protocol101-live-v2-microstructure-masked
```

The required model-facing feature transform is:

```text
mask_vendor_sensitive_option_quote_greek_microstructure
```

This means:

- model scoring must not use vendor-sensitive option quote/Greek
  microstructure as brittle alpha;
- raw bid, ask, and mid still remain available for tradability,
  affordability, labels, fills, PnL, and audit reconstruction;
- the v2 mask is the fair baseline control, not the final ceiling;
- Greeks, IV, OI, volume, bid/ask, spread, and quote sizes may be recovered
  later only through a preregistered parity-plus-uplift feature ladder;
- historical replay and future live/IBKR replay must use the same contract,
  feature ordering, feature transform, account rules, and decision semantics;
- `protocol101-live-v1` is now historical context, not the active training
  contract.

## Primary Certification Evidence

Sync certification:

```text
v4/docs/PROTOCOL101_PARITY_CERTIFICATION_V2_MICROSTRUCTURE_MASKED_2026_07_07.md
```

Training phase definition:

```text
v4/docs/PROTOCOL101_FAIR_CONTRACT_TRAINING_PHASE_2026_07_08.md
```

Current steering plan:

```text
v4/docs/PROTOCOL101_FAIR_CONTRACT_TRAINING_AND_FEATURE_RECOVERY_PLAN_2026_07_08.md
```

Governed v2 15-month full acceptance:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_acceptance/summary.json
```

Latest full-verifier state:

```text
status: fail
schema_version: Protocol101OwnedRawAcceptanceRegistryV3_5
verifier_version: 35
session_count: 311
pass_count: 301
fail_count: 1
report_only_count: 9
```

The full verifier remains failed on purpose because `2025-10-22` has a real
index-context placeability failure and nine other sessions are report-only.  Do
not weaken verifier thresholds to make this pass.

Training-scope pass-only acceptance:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_scope_acceptance/summary.json
```

Latest training-scope accepted state:

```text
status: pass
schema_version: Protocol101OwnedRawAcceptanceRegistryV3_5TrainingScopeV1
session_count: 301
pass_count: 301
fail_count: 0
report_only_count: 0
registry_hash: 6c656cfb2caeaab1d03e78eee39a164f7d169709abeaec5b0c490f1b1e29f0fe
```

Excluded from the 15-month training scope:

```text
2025-04-07 report_only low_tradable_liquidity
2025-04-08 report_only low_tradable_liquidity
2025-04-09 report_only low_tradable_liquidity
2025-04-10 report_only low_tradable_liquidity
2025-04-11 report_only low_tradable_liquidity
2025-07-03 report_only early_close_not_close_aware
2025-07-30 report_only missing_index_context
2025-10-22 fail context_lag_exact_one_minute
2025-11-28 report_only early_close_not_close_aware
2025-12-24 report_only early_close_not_close_aware
```

Latest runner dry-run:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_15mo_training_runner/runner_plan.json
```

Latest dry-run state:

```text
status: dry_run_ready
decision: manifest_and_split_ready_no_training_executed
blockers: []
selected_feature_contract: protocol101-live-v2-microstructure-masked
model_scoring_feature_transform: mask_vendor_sensitive_option_quote_greek_microstructure
split_sessions: 249 train, 10 validation, 10 diagnostic_test
expanding_folds: 5 chronological expanding-window folds, 1-session embargo
model_training_executed: false
threshold_selection_executed: false
broker_endpoint_called: false
paper_submit_allowed: false
```

Important: the single train/validation/diagnostic split remains in the runner
plan as a readiness/plumbing split.  The model-training path now uses the
governed five-fold chronological expanding-window scaffold instead, with a
one-session embargo per fold.  Actual training remains blocked unless the owner
supplies explicit model-training and threshold-selection approval flags plus an
approval note.

Full-corpus null/canary artifact:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_null_canary_15mo_policy1/summary.json
```

Latest null/canary state:

```text
status: pass
decision: null_canary_plumbing_passed_no_training
blockers: []
sensitive_max_abs: 0.0
row_count_min/max: 359 / 359
first_decision_et_values: 09:32
last_decision_et_values: 15:30
```

## Important Timing Convention

The v2 15-month training corpus currently uses:

```text
first decision: 09:32 ET
last entry decision: 15:30 ET
rows per full session: 359
context lag: source_context_time = decision_time - 1 minute
```

This is different from older paired-diff reports that displayed 360 paired rows
for recorder comparisons.  Do not ignore this.  Before any future live
shadow/paper phase, the live runtime/recorder replay must be confirmed to use
the same first-decision convention as the trained candidate.  If live evaluates
at 09:31 while training begins at 09:32, the project has recreated a small
train/live game mismatch.

For the offline training phase, the 09:32 / 359-row convention is accepted only
because the verifier/null-canary path accepted it for the v2 training scope with
exact one-minute context lag and no future context.

## Current Corpus Role

The active v2 training scope is the 301-session pass-only 15-month corpus from
2025-01-02 through 2026-03-31, excluding report-only/fail sessions and excluding
protected holdout sessions from train/validation/diagnostic splits.

The split currently uses 249 train sessions, 10 validation sessions, 10
diagnostic-test sessions, two one-session embargo gaps, and 30 protected holdout
sessions excluded from model-selection splits.

Confirmation/recorder sessions remain confirmation-only.  Do not train or tune
on IBKR recorder days.

## What Has Not Happened

No model training has happened under the current v2 phase.

No threshold tuning has happened under the current v2 phase.

No paper-submit, broker call, paid-data download, promotion/default update,
runtime flag edit, launchd change, or real-money path change is authorized by
the current artifacts.

## Immediate Next Objective

The next conversation should not return to synchronization debugging unless a
new contradiction appears.

The next objective is to define and then owner-approve the first offline
autoresearch/training run:

- trader style;
- model family;
- labels and horizons;
- split policy, using the governed 5-fold chronological expanding-window
  validation scaffold with one-session embargo;
- primary metric;
- drawdown/risk gates;
- trade-frequency gates;
- overfitting controls;
- stopping criteria;
- experiment registry behavior.

After that approval, run offline training only through the governed v2 runner.

After the masked v2 baseline, do not unmask features because headline PnL is
weak. Use the feature recovery ladder in the steering plan: each feature group
must first pass paired live-vs-historical parity and then prove out-of-sample
uplift under the same gates.

## Guardrails For The Next Agent

Use:

```text
~/.autoresearch-trading/runtime-venv/bin/python
```

Do not use the repo-local `.venv` for this phase unless it is deliberately
revalidated; previous work found `.venv` materialization/import issues.

Forbidden unless separately owner-approved:

- model training execution;
- threshold tuning;
- paper-submit;
- broker/API calls;
- paid data downloads;
- promotion/default changes;
- runtime flag edits;
- launchd/automation changes;
- real-money path changes.

Allowed without further approval:

- read artifacts and source;
- update research docs;
- run dry-run/preflight checks;
- run focused local tests;
- inspect the v2 runner plan and governance blockers.

## Plain-English Summary

The old problem was: historical training and IBKR/live replay were not playing
the same game.  That synchronization phase produced a certified fair contract.
The new problem is: train a model that can make money under that fair contract.

Future agents should not chase the old `protocol101-live-v1` sync trail.  They
should start from `protocol101-live-v2-microstructure-masked`, respect the
09:32 / 359-row convention until changed by an explicit tested repair, and
begin the offline training phase only after the owner approves the concrete
training objective and gates.  The mask should be treated as the control model
surface, not as an admission that option microstructure contains no edge.
