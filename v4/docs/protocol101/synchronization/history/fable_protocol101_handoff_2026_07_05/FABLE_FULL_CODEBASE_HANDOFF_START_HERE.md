# Protocol101 Full-Codebase Handoff For Fable

Generated: 2026-07-05

This is a fresh-start handoff for a new Fable instance with access to the full repository:

```text
/Users/gduby/Documents/autoresearch-trading
```

You should assume you have no memory of the prior 12 Fable rounds. This document gives the project context, what has already been audited, what remains unresolved, and what we need from you before the project moves into model hill climbing.

## One-Sentence Mission

Audit whether Protocol101 now has a clean, causal, live-reproducible data/simulator/fold foundation that is safe to use for future model hill climbing, without leakage, fake edge, invalid replay semantics, or contaminated validation splits.

In plain English:

```text
Before we try to train a better model, prove the game is fair.
```

## What This Is Not

This is not a request to:

- train a new model;
- tune thresholds;
- promote a paper default;
- place IBKR orders;
- change real-money paths;
- download paid data;
- optimize an equity curve directly.

The purpose of this Fable conversation is methodology/audit hardening. Hill climbing begins only after this audit work says the runway is clean.

## Project Background

The project is a v4 SPXW 0DTE options-trading system. The long-term goal is an AI options trader that can eventually trade real money, but only after passing data, validation, paper-trading, and real-money review gates.

The current official default is:

```text
PAPER_DEFAULT_PROTOCOL101
```

Current paper-default registry:

```text
v4/promotion/PAPER_TRADING_DEFAULT.json
```

The working concern that launched this audit was:

```text
Historical Protocol101 replay and IBKR live/paper behavior were not playing the same game.
```

The historical system showed trades and P&L that did not reproduce cleanly when the model was evaluated through live-reproducible/IBKR-like inputs. That forced the project to stop trusting headline `equity.html` results until the causal data contract, replay simulator, labels, and live-vs-historical semantics were audited.

## Current High-Level Finding

There were multiple overlapping mismatches, not one:

| Comparison | Meaning |
|---|---|
| Older official result | Event-policy simulation using older/precomputed semantics. Not yet a valid live benchmark. |
| Same-runner legacy contract | Current replay machinery with historical feature contract. Better benchmark, but still not the final live game. |
| Live-reproducible contract | Current replay with only features/semantics that should exist live. More realistic, but materially weaker performance. |
| IBKR recorder/captured feed | Used to test whether offline replay can reproduce live-observed candidate/feature/decision behavior. |

The working project direction is:

```text
Keep strict one-account serial replay.
Keep live-reproducibility as law.
Repair causal feature semantics only when provable.
Accept lost edge only when it depended on unavailable/stale/non-causal information.
If frozen Protocol101 cannot perform under the fair contract, retrain later on the fair contract.
```

## Present End Goal Of The Fable Conversation

Fable is no longer being used to debug IBKR launchd, monitors, or daily paper sessions. Fable is being used as a skeptical methodology auditor.

The Fable conversation is successful when the project can say:

1. Every usable session has explicit data provenance.
2. Every usable session has a governance role: train, test, diagnostics-only, report-only, confirmation-only, or blocked.
3. No fold can place a session unless code-level predicates pass.
4. Runtime features are causal and live-reproducible.
5. Labels/PnL are recomputable from raw quote paths.
6. The simulator semantics are pinned and tested.
7. Early-close and special-session edge cases cannot silently contaminate folds.
8. The validation/fold scaffold cannot accidentally use disallowed, unaccepted, stale, or old-verifier data.
9. Null baselines and mutation tests show the pipeline can reject bad data and fake edge.
10. The remaining task becomes “run experiments,” not “argue whether the data/simulator/folds are valid.”

When those are true, Fable is no longer needed for foundation review, and the project can move into a separate goal/hill-climbing mode.

## Definition Of Ready For Hill Climbing

Hill climbing should stay blocked until these conditions are met:

```text
Data acceptance:
  - Full owned raw pre-program era accepted or explicitly rejected by code.
  - At minimum, full October 2024 acceptance passes under verifier v2+ before first diagnostics folds.
  - Eventually Oct 2024 -> Jun 2025 should be batched chronologically.

Labels:
  - Raw CBBO spot recompute verifies labels_net_pnl.
  - Fee model is pinned.
  - All-zero label placeholder failure mode is impossible to admit.

Context:
  - Decision timestamps are calendar-correct.
  - Index context lag is exactly causal.
  - No future context or leading open backfill is admitted.
  - Derived SPX/VIX context columns are either independently reconstructed or explicitly scoped.

Simulator:
  - Strict one-account serial replay is the official validation simulator.
  - Forced-flat semantics are pinned.
  - Position/cash/account state cannot use impossible overlaps or unaffordable trades.

Folds:
  - Era role policy is explicit.
  - Fold placement predicate joins era permission, accepted processed rows, acceptance status, verifier version, and special-session exclusions.
  - Confirmation/live recorder sessions cannot become train/test data.
  - Early-close sessions are report-only or otherwise governed.

Validation:
  - Future model search uses accepted data only.
  - Chronological/purged/embargoed validation is used where relevant.
  - Null baselines and mutation tests exist.
  - Protected holdout policy is explicit.
```

## Current Code And Artifact Status

### Simulator

Primary file:

```text
v4/model/protocol101_serial_simulator.py
```

Current intended semantics:

- v2 simulator semantics.
- Strict one-account serial replay.
- Separate trade-identity hash versus simulator-semantics hash.
- Forced-flat action represented explicitly.
- Forced-flat synthetic exit timestamp in UTC.
- No stale alias that lets old simulator records masquerade as new ones.

Relevant tests:

```text
v4/tests/test_protocol101_serial_simulator.py
v4/tests/test_protocol101_fair_contract_selected_candidate_replay_gate.py
```

### Replay Gate

Primary file:

```text
v4/scripts/run_protocol101_fair_contract_selected_candidate_replay_gate.py
```

Purpose:

- Gate selected candidate results through strict replay semantics.
- Ensure old/hash-aliased simulator outputs do not pass as current.

### Era Manifest

Primary file:

```text
v4/scripts/build_protocol101_session_era_manifest.py
```

Current artifact:

```text
v4/audit/autoresearch/protocol101_session_era_manifest/summary.json
v4/audit/autoresearch/protocol101_session_era_manifest/report.md
```

Latest known result:

```text
status: pass
session_count: 429
manifest_hash: 6767427de34e6ef1f1558ab7854c5f7b599068ed05302629bd5aec083793ce16
```

Era rules currently used:

```text
pre_program_oct2024_jun2025: 2024-10-01 through 2025-06-30
owned_jul_dec2025: 2025-07-01 through 2025-12-31
q1_2026_development: 2026-01-01 through 2026-03-31
post_q1_gap_apr_may2026: 2026-04-01 through 2026-05-31
confirmation_jun_jul2026: 2026-06-01 through 2026-07-31
unassigned_requires_decision: fail-closed default
```

Known count summary from prior run:

```text
confirmation_jun_jul2026: 9
owned_jul_dec2025: 131
post_q1_gap_apr_may2026: 35
pre_program_oct2024_jun2025: 191
q1_2026_development: 63
```

Important distinction:

```text
The era manifest is facts, not permission.
```

It records what sessions exist and where they fall in time. It does not by itself authorize fold placement.

### Era Role Policy

Primary file:

```text
v4/scripts/build_protocol101_era_role_policy.py
```

Current artifact:

```text
v4/audit/autoresearch/protocol101_era_role_policy/summary.json
v4/audit/autoresearch/protocol101_era_role_policy/report.md
```

Latest known result:

```text
status: pass
policy_hash: 6f51e65f5b9d271f14c83f6752d30390dca1797d5444c64a710bb9b2214fac7d
```

Key policy concepts:

- `diagnostics_only` may appear in diagnostics-tier fold test windows for gates, samplers, nulls, and uplift; never model-tier train/test.
- `test` grants model-tier test windows.
- `confirmation_one_shot` applies to recorder/live-parity days only, not train/tune.
- `report_only` is context only.
- `unassigned_requires_decision` permits no roles.

Promotion guard:

```text
If pooled criteria pass but pre_program_oct2024_jun2025 test folds are systematically negative,
status should be regime_bound_requires_owner_review, not pass.
```

Placeholder eras exist for possible future purchases:

```text
extension_2024h1
extension_2023
```

These placeholders do not approve data use by themselves.

### Acceptance Verifier

Primary file:

```text
v4/scripts/run_protocol101_owned_raw_acceptance_verifier.py
```

Current schema:

```text
Protocol101OwnedRawAcceptanceRegistryV2
verifier_version = 2
```

Current fold-placement predicate:

```text
Protocol101FoldPlacementPredicateV1

placeable =
  era_permits_role
  AND canonical_processed_rows_exist
  AND acceptance_status_pass
  AND verifier_version >= 2
  AND not_early_close_fold_session
```

Current verifier checks include:

- raw product files present for all four Databento products;
- Parquet/DBN row-count cross-check;
- SPX/VIX index products present;
- processed rows exist;
- expected decision-minute count using early-close-aware calendar;
- ladder shape sanity check;
- tradable-minute share;
- mean tradable candidate count;
- near-ATM tradable share;
- finite labels;
- nonzero labels;
- positive and negative label values present;
- raw CBBO label spot recompute;
- `feature_contract_version == protocol101-live-v1`;
- decision timestamps match expected calendar bounds;
- exact one-minute index-context lag;
- no future context;
- no leading open backfill;
- early-close flag;
- data-plane-only evidence markers.

Current v2 acceptance artifact:

```text
v4/audit/autoresearch/protocol101_owned_raw_acceptance_2024_10_v2_partial/summary.json
v4/audit/autoresearch/protocol101_owned_raw_acceptance_2024_10_v2_partial/report.md
```

Latest known result:

```text
status: pass
schema_version: Protocol101OwnedRawAcceptanceRegistryV2
verifier_version: 2
session_count: 14
pass_count: 14
fail_count: 0
registry_hash: cf9107b369dc4673c60ef61d9f54cc430582b31ec23ead8e89a3a76e09c50aee
```

Accepted current subset:

```text
2024-10-01
2024-10-02
2024-10-03
2024-10-04
2024-10-07
2024-10-08
2024-10-09
2024-10-10
2024-10-11
2024-10-14
2024-10-15
2024-10-16
2024-10-17
2024-10-18
```

Current v2 raw-label spot recompute result:

```text
70 / 70 matched
fee_model: gross_no_fees
observed reasons: stop_hit, target_hit, time_exit
forced_flat_capped: not yet sampled in this subset
```

Important caveat:

```text
Full October 2024 is not accepted yet.
```

The build reached 2024-10-18 and then stalled while attempting 2024-10-21. The process was stopped deliberately so no silent build would remain running.

### Acceptance Verifier Tests

Primary test:

```text
v4/tests/test_protocol101_owned_raw_acceptance_verifier.py
```

Important mutation-style tests currently cover:

- old verifier version rejected;
- all-zero labels exposed/rejected;
- empty candidate population rejected;
- zero-lag/future context rejected;
- Databento raw-symbol mapping checked.

Latest broader local test slice:

```text
87 passed
```

## Current Data Build State

The live-v1 October output directories are:

```text
v4/normalized_protocol101_owned_raw_acceptance_2024_10_live_v1/
data/processed/spxw_0dte_neural_protocol101_owned_raw_acceptance_2024_10_live_v1/
```

Processed October sessions currently written through 2024-10-18.

The attempted command for the second half of October was:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m v4.scripts.build_databento_neural_dataset \
  --start-date 2024-10-15 \
  --end-date 2024-10-31 \
  --raw-root data/raw \
  --normalized-dir v4/normalized_protocol101_owned_raw_acceptance_2024_10_live_v1 \
  --processed-dir data/processed/spxw_0dte_neural_protocol101_owned_raw_acceptance_2024_10_live_v1 \
  --context-mode official \
  --official-spx-dir data/vendor/thetadata/index/spx_1m \
  --official-vix-dir data/vendor/thetadata/index/vix_1m \
  --feature-contract protocol101-live-v1 \
  --compute-live-policy-labels \
  --summary-out v4/audit/autoresearch/protocol101_owned_raw_acceptance_2024_10_v2_full/build_summary_2024_10_15_to_31.json
```

It completed:

```text
2024-10-15
2024-10-16
2024-10-17
2024-10-18
```

Then became idle on 2024-10-21 with no CPU for several minutes.

## Main Remaining Problems We Need Fable To Help With

### 1. Is verifier v2 strict enough?

Please audit whether `run_protocol101_owned_raw_acceptance_verifier.py` now correctly fixes the two major v1 weaknesses:

```text
full_ladder_share was only a shape check
labels_present could pass all-zero placeholder labels
```

Also audit whether the current v2 checks are still too broad, too narrow, or vulnerable to passing bad sessions.

### 2. Raw-label spot recompute branch coverage

Current v2 checks five deterministic tuples per session. Across 14 sessions:

```text
70/70 matched
fee_model = gross_no_fees
observed = stop_hit, target_hit, time_exit
missing = forced_flat_capped
```

We need a better sampling rule that forces coverage of:

```text
policy 0/1/2
near/far offset
stop_hit
target_hit
time_exit
forced_flat_capped
```

Question:

```text
Should forced-flat-capped be required per month, per batch, or per accepted era?
```

### 3. Raw SPX/VIX context reconstruction

Current v2 checks processed-row causality:

```text
decision timestamp bounds
source_context_time = decision_time - 1 minute
no future context
no leading open backfill
```

It does not yet fully recompute derived market context columns from raw SPX/VIX files:

```text
VWAP
momentum
OMAR
range
derived window features
```

Question:

```text
What minimum independent context reconstruction is required before admitting sessions to folds?
```

### 4. Full October build stalled on 2024-10-21

The current builder can stall a whole batch. We likely need a resumable per-session or per-month batch runner with:

- per-session timeout;
- per-session stdout/stderr logs;
- skip/resume support;
- explicit status JSON;
- acceptance run after each month;
- no silent hanging process.

Question:

```text
What is the simplest robust batch orchestration design that does not mutate model semantics?
```

### 5. Candidate-population thresholds

Current v2 floors are broad sanity checks:

```text
min_mean_tradable_candidates = 10
min_near_atm_tradable_share = 0.50
min_tradable_minute_share = 0.50
```

These catch obviously broken data but are not calibrated from the larger corpus yet.

Question:

```text
Should calibration wait until full Oct 2024 passes, or should it use already processed Jul 2025-Mar 2026 now?
```

### 6. Early-close policy

The current cheap solution:

```text
early_close_session = true
fold placement excludes early-close sessions
```

Question:

```text
Is report-only early-close handling sufficient for this phase, or should early-close semantics be made close-aware before folds?
```

### 7. When can Fable declare foundation ready?

We need a clear end condition:

```text
At what exact point should Fable say: stop asking me, begin model hill climbing?
```

My current proposed answer:

```text
After full October v2 acceptance passes, then chronological Oct 2024-Jun 2025 processing/acceptance passes or explicitly rejects sessions, then first diagnostics folds/nulls can run. If those fold/null/gate mechanics hold, Fable's foundation audit is complete.
```

Please refine or correct this.

## Suggested Read Order In Full Codebase

Start with governance/current-state:

```text
AGENTS.md
docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md
v4/promotion/PAPER_TRADING_DEFAULT.json
v4/docs/HISTORICAL-VS-LIVE-IBKR-TRADING-COMPARISON.md
v4/docs/PROTOCOL101_LIVE_FEATURE_CONTRACT_V1.md
```

Then current audited code:

```text
v4/scripts/run_protocol101_owned_raw_acceptance_verifier.py
v4/tests/test_protocol101_owned_raw_acceptance_verifier.py
v4/scripts/build_protocol101_session_era_manifest.py
v4/tests/test_protocol101_session_era_manifest.py
v4/scripts/build_protocol101_era_role_policy.py
v4/tests/test_protocol101_era_role_policy.py
v4/model/protocol101_serial_simulator.py
v4/tests/test_protocol101_serial_simulator.py
v4/scripts/run_protocol101_fair_contract_selected_candidate_replay_gate.py
v4/tests/test_protocol101_fair_contract_selected_candidate_replay_gate.py
v4/dataset/spxw_0dte_neural.py
v4/scripts/build_databento_neural_dataset.py
```

Then artifacts:

```text
v4/audit/autoresearch/protocol101_session_era_manifest/report.md
v4/audit/autoresearch/protocol101_session_era_manifest/summary.json
v4/audit/autoresearch/protocol101_era_role_policy/report.md
v4/audit/autoresearch/protocol101_era_role_policy/summary.json
v4/audit/autoresearch/protocol101_owned_raw_acceptance_2024_10_v2_partial/report.md
v4/audit/autoresearch/protocol101_owned_raw_acceptance_2024_10_v2_partial/summary.json
```

Then prior condensed Fable packages if useful:

```text
v4/docs/fable_protocol101_handoff_2026_07_05/round_11_owned_raw_acceptance_registry/00_message_to_fable.md
v4/docs/fable_protocol101_handoff_2026_07_05/round_12_acceptance_verifier_v2/00_message_to_fable.md
```

## Commands That Should Be Safe For Audit

Read-only or local/offline commands:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m pytest \
  v4/tests/test_protocol101_owned_raw_acceptance_verifier.py \
  v4/tests/test_protocol101_era_role_policy.py \
  v4/tests/test_protocol101_session_era_manifest.py \
  v4/tests/test_protocol101_serial_simulator.py \
  v4/tests/test_protocol101_fair_contract_selected_candidate_replay_gate.py
```

Re-run current v2 acceptance on completed October subset:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m v4.scripts.run_protocol101_owned_raw_acceptance_verifier \
  --start-date 2024-10-01 \
  --end-date 2024-10-18 \
  --out-dir v4/audit/autoresearch/protocol101_owned_raw_acceptance_2024_10_v2_partial \
  --role diagnostics_only
```

Do not run broker/live/paper-submit/paid-download/model-training commands unless the owner explicitly authorizes them.

## Non-Negotiable Guardrails

Do not recommend bypassing these:

- no lookahead;
- no future/path-derived features in runtime inputs;
- no random train/test split across same timestamp/session;
- no hidden use of Q1 as untouched holdout;
- no restoring old P&L by reintroducing unavailable fields;
- no declaring a session fold-placeable unless era role, processed rows, acceptance status, verifier version, and special-session policy all pass;
- no using IBKR paper fills as ground-truth execution quality;
- no paper-submit or real-money changes in this foundation phase.

## What We Want From Fable Now

Please respond with:

1. Your audit of verifier v2.
2. Any remaining verifier failure modes that are still too weak.
3. A recommended design for resumable Oct 2024-Jun 2025 batch acceptance.
4. A precise rule for label spot-check branch coverage, especially forced-flat-capped.
5. A precise rule for raw SPX/VIX derived-context reconstruction.
6. A final checklist for “foundation ready; hill climbing may begin.”

Be skeptical. Do not optimize for making the current implementation look good. The point is to prevent future hill climbing from being built on fake edge.

