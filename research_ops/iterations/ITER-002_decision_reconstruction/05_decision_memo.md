# Decision Memo

Iteration ID: `ITER-002_decision_reconstruction`
Assumption ID: `A006`
Title: Decision reconstruction

## Decision

- Decision reconstruction status: `schema patch required`.
- Logs are not sufficient to reconstruct every decision from durable evidence.
- A006 remains open and blocking.

## Context

The improvement plan says missing feature, quote, timestamp, candidate, or artifact references falsify decision reconstruction. This iteration tested the existing local paper/shadow JSONL corpus against that standard.

## Evidence Reviewed

| Artifact | Location | Weight |
|---|---|---|
| Matrix | `artifacts/decision_reconstruction_matrix.csv` | Primary |
| Missing field report | `artifacts/missing_fields_report.md` | Primary |
| Schema gap report | `artifacts/log_schema_gap_report.md` | Primary |
| Verifier report | `04_verifier_report.md` | Primary |

## Decision Details

```text
Decision: schema patch required
Does this change PAPER_DEFAULT_PROTOCOL101: no
Does this authorize model training: no
Does this authorize broker/data/runtime action: no
Evidence reviewed: existing JSONL logs only
Risks accepted: none
Reversal condition: future logs contain reconstructable candidate, feature, quote, model, guard, account, artifact, and order evidence per decision
Owner: research_ops
```

## Assumptions Accepted Or Rejected

| Assumption ID | Treatment | Rationale |
|---|---|---|
| A006 | Remains open | 0 of 3495 decision rows were sufficient under the diagnostic criteria. |

## Follow-Up Actions

- Design a logging schema patch RFC after quote-age observability is approved.
- Require per-decision artifact references, feature hashes/rows, candidate sets, quote timestamps, model logits, guard inputs, account state, and order/fill outcomes.

## Actions Still Blocked

- Challenger promotion.
- Threshold tuning.
- New model training for promotion.
- Protected holdout scoring for exploration.
- Assuming replay profitability proves live edge.
- Treating current logs as sufficient audit evidence.
