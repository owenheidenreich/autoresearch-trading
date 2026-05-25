# CEO Dashboard

Last updated: 2026-05-24

## 1. Current Operational Default

- Name: `PAPER_DEFAULT_PROTOCOL101`
- Scope: guarded IBKR paper runtime only
- Real money authorized: `false`

## 2. Current Safety Posture

- Posture: `execution-and-parity falsification`
- Replay profitability remains a hypothesis until execution and parity assumptions are tested.

## 3. Active Iteration

- None.

## 4. Latest Completed Iteration

- Iteration: `ITER-003_replay_live_feature_parity`
- Assumption: `A004`
- Title: Replay/live feature parity
- Status: `completed`

## 5. Decisions Required

- Resolve open P0 assumptions before any promotion, threshold, runtime, or model-capacity work.
- Replay/live feature parity status: `replay metrics not yet usable`.
- Existing logs cannot prove Protocol051-to-Protocol101 live feature construction matches replay.
- A004 remains open and blocking.

## 6. P0 Assumptions

- `A001` quote age truth - `open` - next: `quote_age_observability_packet`
- `A002` ask-entry/bid-exit replay is executable - `open` - next: `execution_realism_fill_table`
- `A003` paper fill/cancel observations are sufficient - `open` - next: `paper_observation_sufficiency_audit`
- `A004` live Protocol051/101 features equal replay features - `open` - next: `protocol051_101_feature_parity_harness`
- `A005` lifecycle live state equals replay/training sequence state - `open` - next: `lifecycle_sequence_parity_harness`
- `A006` every decision can be reconstructed from logs - `open` - next: `decision_reconstruction_log_audit`
- `A011` validation windows are not being over-mined - `open` - next: `validation_provenance_ledger`
- `A013` one-minute CBBO is sufficient for 0DTE execution - `open` - next: `high_resolution_timing_fragility_audit`

## 7. Blocked Actions

- no challenger promotion
- no threshold tuning
- no new model training
- no protected holdout scoring for exploration
- no assuming replay profitability proves live edge
- no assuming deterministic ask-entry/bid-exit proves fillability
- no runtime flag mutation as part of research
- no paid data downloads without approval

## 8. Newly Confirmed Evidence

- Existing inspected logs do not contain paired replay/live feature evidence.
- Existing inspected logs do not contain enough Protocol101 logits or replay logits for logit parity.
- Existing candidate-set logs are not sufficient to compare live candidate membership against replay candidate membership.

## 9. Newly Falsified Assumptions

- The claim that current logs already prove live Protocol051-to-Protocol101 feature parity is falsified for the inspected corpus.

## 10. Next Recommended Codex Prompt

`ITER-004_quote_age_observability_patch_rfc`: write a CEO-decision RFC for a logging-only patch that captures raw quote timestamps, per-candidate quote ages, decision reconstruction fields, and feature/logit hashes without changing trading behavior.

## Dashboard Rule

This dashboard is not primary evidence. Evidence lives in iteration artifacts, verifier reports, logs, manifests, and decision memos.
