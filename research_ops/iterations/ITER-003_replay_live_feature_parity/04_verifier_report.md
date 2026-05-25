# Verifier Report

Iteration ID: `ITER-003_replay_live_feature_parity`
Assumption ID: `A004`
Title: Replay/live feature parity

## Verification Question

Does the implementation answer whether existing logs can prove replay/live feature parity without expanding scope into model loading or replay execution?

## Evidence Reviewed

| Evidence | Location | Notes |
|---|---|---|
| RFC | `02_rfc.md` | Defines paired evidence requirement. |
| Diagnostic | `research_ops/diagnostics/feature_parity.py` | Stdlib-only log evidence parser. |
| Feature report | `artifacts/feature_parity_report.md` | States decision and counts. |
| CSVs | `feature_diff.csv`, `logit_diff.csv`, `candidate_set_diff.csv` | Comparison availability rows. |
| Tests | `tests/test_feature_parity_diagnostic.py` | Covers not-comparable cases and CLI artifacts. |

## Results

- Decision: `replay metrics not yet usable`.
- Feature comparison rows: `13592`, all `not_comparable`.
- Logit rows: `1103`, all `not_comparable`.
- Candidate-set rows: `906`, all `not_comparable`.

## Boundary Checks

- No v4 files modified.
- No broker APIs called.
- No paid-data scripts run.
- No models loaded or trained.
- No thresholds tuned.
- No runtime flags or launchd assets changed.

## Newly Confirmed Evidence

- Existing inspected logs do not contain paired replay/live feature evidence.
- Existing inspected logs do not contain enough Protocol101 logits or replay logits for logit parity.
- Existing candidate-set logs are not sufficient to compare live candidate membership against replay candidate membership.

## Newly Falsified Assumptions

- The claim that current logs already prove live Protocol051-to-Protocol101 feature parity is falsified for the inspected corpus.

## Risks And Objections

- This is an evidence-availability diagnostic, not a numeric parity harness.
- It does not prove features differ; it proves logs cannot establish equality.
- A real parity harness will require a CEO-approved design for paired replay/live market-state packets.

## Verdict

`partially_supported`

The implementation supports the decision `replay metrics not yet usable`. A004 remains open and blocking.
