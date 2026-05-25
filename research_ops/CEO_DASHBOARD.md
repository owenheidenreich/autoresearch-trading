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

- Iteration: `ITER-001_quote_age_truth`
- Assumption: `A001`
- Title: Quote age truth
- Status: `completed`

## 5. Decisions Required

- Resolve open P0 assumptions before any promotion, threshold, runtime, or model-capacity work.
- Quote age truth status: `unknown`.
- Paper-submit trust is affected: existing logs do not prove that quote freshness guards are operating on measured live quote ages.
- A001 remains open and blocking.
- Do not promote challengers, tune thresholds, train new models for promotion, assume replay profitability proves live edge, or rely on paper-submit freshness until true quote timestamp logging is proven.

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

- Existing inspected paper/shadow JSONL logs do not contain enough raw quote timestamp fields to prove quote-age truth.
- The existing log corpus inspected by this iteration contains `0` broker endpoint rows.
- The inspected log corpus contains only `1` persisted `quote_age_ms` row, and that row is unreconstructable because it lacks a raw quote timestamp.
- Current existing logs cannot support a paper-submit quote-age-truth pass.

## 9. Newly Falsified Assumptions

- The claim that existing logs are already sufficient to prove live quote age truth is falsified for the inspected log corpus.
- The claim that persisted `quote_age_ms` alone is sufficient freshness evidence is falsified by the diagnostic design and artifact result.

## 10. Next Recommended Codex Prompt

`ITER-002_quote_age_observability_packet`: design a CEO-approved logging-only observability change that records raw quote timestamp source, receive timestamp, decision timestamp, persisted `quote_age_ms`, recomputed quote age, and guard result for every selected and rejected candidate without changing trading behavior.

## Dashboard Rule

This dashboard is not primary evidence. Evidence lives in iteration artifacts, verifier reports, logs, manifests, and decision memos.
