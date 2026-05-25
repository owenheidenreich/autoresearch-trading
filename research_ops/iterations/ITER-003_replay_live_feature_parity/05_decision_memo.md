# Decision Memo

Iteration ID: `ITER-003_replay_live_feature_parity`
Assumption ID: `A004`
Title: Replay/live feature parity

## Decision

- Replay/live feature parity status: `replay metrics not yet usable`.
- Existing logs cannot prove Protocol051-to-Protocol101 live feature construction matches replay.
- A004 remains open and blocking.

## Context

Protocol101 depends on Protocol051 surface scores and Protocol101 candidate feature rows. If live candidate sets, scaled features, or logits differ from replay, model artifact replay metrics are not sufficient deployment evidence.

## Evidence Reviewed

| Artifact | Location | Weight |
|---|---|---|
| Feature report | `artifacts/feature_parity_report.md` | Primary |
| Feature diff | `artifacts/feature_diff.csv` | Primary |
| Logit diff | `artifacts/logit_diff.csv` | Primary |
| Candidate-set diff | `artifacts/candidate_set_diff.csv` | Primary |
| Verifier report | `04_verifier_report.md` | Primary |

## Decision Details

```text
Decision: replay metrics not yet usable
Does this change PAPER_DEFAULT_PROTOCOL101: no
Does this authorize model training: no
Does this authorize broker/data/runtime action: no
Evidence reviewed: existing JSONL logs only
Risks accepted: none
Reversal condition: paired replay/live market-state packets match candidate sets, feature rows/hashes, scaled tensors, logits, and selected actions
Owner: research_ops
```

## Assumptions Accepted Or Rejected

| Assumption ID | Treatment | Rationale |
|---|---|---|
| A004 | Remains open | Existing logs contain no comparable paired replay/live feature, logit, or candidate-set evidence. |

## Follow-Up Actions

- Design a replay/live parity harness RFC after quote-age and reconstruction logging are fixed.
- The harness should compare candidate set membership, Protocol051 scores, Protocol101 feature rows, scaled tensor hashes, logits, and selected action for the same timestamp/contracts.

## Next Recommended Iteration

`ITER-004_quote_age_observability_patch_rfc`: write a CEO-decision RFC for a logging-only patch that captures raw quote timestamps, per-candidate quote ages, decision reconstruction fields, and feature/logit hashes without changing trading behavior.

## Actions Still Blocked

- Using replay metrics as deployment evidence.
- Challenger promotion.
- Threshold tuning.
- New model training for promotion.
- Protected holdout scoring for exploration.
- Assuming replay profitability proves live edge.
