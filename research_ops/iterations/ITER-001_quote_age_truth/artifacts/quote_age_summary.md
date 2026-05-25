# Quote Age Truth Diagnostic Summary

Generated at: `2026-05-25T01:45:51.264995+00:00`

## Verdict

- Verdict: `unknown`
- Reason: available logs do not contain enough broker/paper-submit trustworthy evidence
- Max age: `1500.0` ms
- Tolerance: `250.0` ms

## Counts

- Total rows: `6648`
- Files read: `15`
- Quote evidence rows: `1`
- Persisted quote-age rows: `1`
- Broker endpoint rows: `0`
- Broker problem rows: `0`
- Paper-submit quote rows: `0`
- Paper-submit problem rows: `0`
- Placeholder ratio among persisted: `0.0`
- Trustworthy ratio among persisted: `0.0`

## Classification Counts

- `missing`: 6647
- `unreconstructable`: 1

## Trust Status Counts

- `unknown`: 6648

## Inputs Read

- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-14/protocol142_executor_smoke.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-15/manual_dry_run.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-15/manual_fail_closed_rehearsal.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-15/protocol155_integrated_dry_run.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-15/protocol155_readiness_dry_run.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-19/protocol101_intent-shadow_bridge_verify_2026-05-19.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-19/protocol101_no-order-shadow_2026-05-19.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-19/protocol101_no-order-shadow_entrybridge_verify_2026-05-19.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-19/protocol101_no-order-shadow_plumbing_verify_2026-05-19.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-19/protocol101_paper_dryrun_alignment_2026-05-19.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-20/protocol101_no-order-shadow_2026-05-20.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-20/protocol101_persistent_cadence_blocker_smoke_2026-05-20.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-20/protocol101_persistent_cadence_smoke_2026-05-20.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-20/protocol101_persistent_smoke_2026-05-20.jsonl`
- `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.jsonl`

## Missing Inputs

- None.

## Artifacts

- CSV: `/Users/gduby/Documents/autoresearch-trading-research-ops-transition/research_ops/iterations/ITER-001_quote_age_truth/artifacts/quote_age_rows.csv`
- JSON: `/Users/gduby/Documents/autoresearch-trading-research-ops-transition/research_ops/iterations/ITER-001_quote_age_truth/artifacts/quote_age_summary.json`
- Markdown: `/Users/gduby/Documents/autoresearch-trading-research-ops-transition/research_ops/iterations/ITER-001_quote_age_truth/artifacts/quote_age_summary.md`

## Interpretation

- `trustworthy` means logged age matched recomputed age from raw timestamps. It does not prove fillability.
- `placeholder` means a persisted age, especially zero, lacked raw timestamp support or contradicted recomputation.
- Missing raw quote timestamp is never treated as pass.
- `unknown` remains blocking evidence for paper-submit trust until live paper-submit rows carry reconstructable timestamps.
