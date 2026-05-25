# Decision Memo

Iteration ID: `ITER-001_quote_age_truth`
Assumption ID: `A001`
Title: Quote age truth

## Decision

- Quote age truth status: `unknown`.
- Paper-submit trust is affected: existing logs do not prove that quote freshness guards are operating on measured live quote ages.
- A001 remains open and blocking.
- Do not promote challengers, tune thresholds, train new models for promotion, assume replay profitability proves live edge, or rely on paper-submit freshness until true quote timestamp logging is proven.

## Context

This iteration tested whether existing Protocol101 paper/shadow logs can prove that `quote_age_ms` is a real measured age rather than a placeholder value. The audit identified quote age truth as a P0 dependency for execution realism, replay/live parity, decision reconstruction, and paper-submit trust.

## Evidence Reviewed

- `research_ops/iterations/ITER-001_quote_age_truth/01_cartography.md`
- `research_ops/iterations/ITER-001_quote_age_truth/02_rfc.md`
- `research_ops/iterations/ITER-001_quote_age_truth/03_implementation_summary.md`
- `research_ops/iterations/ITER-001_quote_age_truth/04_verifier_report.md`
- `research_ops/iterations/ITER-001_quote_age_truth/artifacts/quote_age_rows.csv`
- `research_ops/iterations/ITER-001_quote_age_truth/artifacts/quote_age_summary.json`
- `research_ops/iterations/ITER-001_quote_age_truth/artifacts/quote_age_summary.md`

## Decision Details

- Research question: Are live quote ages real, or are freshness guards passing because runtime supplies placeholder ages?
- Result: `unknown`.
- Parsed rows: `6648`.
- Files read: `15`.
- Broker endpoint rows: `0`.
- Persisted quote-age rows: `1`.
- Trustworthy rows: `0`.
- The only persisted `quote_age_ms` row lacked a raw quote timestamp and was classified `unreconstructable`.
- Existing persistent paper-submit logs did not contain reconstructable raw quote timestamp and persisted age evidence.

## Assumptions Accepted Or Rejected

- Accepted: existing inspected logs are insufficient to prove quote age truth.
- Rejected: persisted `quote_age_ms` alone is sufficient freshness evidence.
- Rejected: existing logs already support a paper-submit freshness pass.
- Not rejected: the possibility that current runtime code can compute real age when raw IBKR timestamps are present. This remains unproven in the inspected logs.

## Follow-Up Actions

- Next iteration should be an observability RFC for quote timestamp capture and candidate-level quote-age logging.
- Any runtime logging change requires CEO approval because it touches live/paper observability surfaces.
- Future verifier must prove that selected and rejected candidates log raw quote timestamp, receive timestamp, decision timestamp, persisted age, recomputed age, and guard outcome.

## Question Asked

Can existing logs prove that live quote ages are real measured ages rather than placeholders?

## Assumption Tested

`A001 quote age truth`

## Result

`unknown`, blocking.

## Evidence

- Diagnostic verdict: `unknown`.
- `6648` parsed rows.
- `0` broker endpoint rows.
- `1` persisted quote-age row.
- `0` trustworthy quote-age rows.
- `6648` rows had `trust_status=unknown`.

## Verifier Objections

- Protocol160 source is not present in the clean transition branch, so implementation-level verification of the persistent paper runtime path is incomplete here.
- The diagnostic used local source-workspace logs outside the clean transition branch as read-only evidence.
- Many `missing` rows are structural log rows without option quote payloads; the strongest evidence is absence of reconstructable quote-age fields, not the raw missing count alone.

## What This Confirms

- Existing inspected logs cannot prove quote age truth.
- Paper-submit freshness trust remains unresolved.
- Missing raw quote timestamp is correctly treated as unknown, not pass.

## What This Falsifies

- Existing logs are sufficient to prove quote age truth.
- A numeric persisted `quote_age_ms` should be treated as sufficient evidence without raw timestamp reconstruction.

## What Remains Unknown

- Whether current Protocol158/160 runtime supplies real IBKR quote timestamps during actual paper-submit decisions.
- Whether raw IBKR timestamp fields are consistently available and causally aligned with decision timestamps.
- Whether selected and rejected candidates can be reconstructed with true quote ages.
- Whether quote age truth holds under actual broker endpoint/fill/cancel observations.

## Actions Now Allowed

- Read-only refinement of quote-age diagnostics.
- CEO-approved RFC for logging true quote timestamp, receive timestamp, decision timestamp, and recomputed age.
- Read-only comparison of future logs against this diagnostic.

## Actions Still Blocked

- Challenger promotion.
- Threshold tuning for deployment or promotion.
- New model training for promotion.
- Protected holdout scoring for exploration.
- Assuming replay profitability proves live edge.
- Assuming deterministic ask-entry/bid-exit proves fillability.
- Runtime flag mutation as part of research.
- Paid-data downloads without approval.
- Treating paper-submit quote freshness as trusted.

## Next Recommended Iteration

`ITER-002_quote_age_observability_packet`: design a CEO-approved logging-only observability change that records raw quote timestamp source, receive timestamp, decision timestamp, persisted `quote_age_ms`, recomputed quote age, and guard result for every selected and rejected candidate without changing trading behavior.

## CEO Decision Required

- No CEO decision is required to accept this read-only diagnostic result.
- A CEO decision is required before any future iteration modifies paper runtime logging or Protocol158/160 observability surfaces.
