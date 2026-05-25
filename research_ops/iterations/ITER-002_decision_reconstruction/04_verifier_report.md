# Verifier Report

Iteration ID: `ITER-002_decision_reconstruction`
Assumption ID: `A006`
Title: Decision reconstruction

## Verification Question

Does the implementation answer whether decisions can be reconstructed from logs alone without mutating runtime state or executing broker/model code?

## Evidence Reviewed

| Evidence | Location | Notes |
|---|---|---|
| RFC | `02_rfc.md` | Defines required reconstruction categories before results. |
| Diagnostic | `research_ops/diagnostics/decision_reconstruction.py` | Stdlib-only log parser. |
| Matrix | `artifacts/decision_reconstruction_matrix.csv` | Event-level coverage and missing fields. |
| Reports | `artifacts/missing_fields_report.md`, `artifacts/log_schema_gap_report.md` | Aggregate decision and schema gap. |
| Tests | `tests/test_decision_reconstruction_diagnostic.py` | Covers sparse row and CLI artifacts. |

## Results

- Verdict: `schema patch required`.
- Parsed rows: `6663`.
- Decision rows: `3495`.
- Sufficient decision rows: `0`.
- Insufficient decision rows: `3495`.
- Broker endpoint rows: `0`.

## Boundary Checks

- No v4 files modified.
- No broker APIs called.
- No paid-data scripts run.
- No models loaded or trained.
- No runtime flags, launchd files, or paper-submit behavior changed.

## Newly Confirmed Evidence

- Existing inspected logs cannot reconstruct every live/paper/no-order decision from logs alone.
- Full candidate features, model logits, artifact references, quote timestamps, and guard inputs are missing at decision-level coverage.
- The inspected log corpus contains `0` broker endpoint rows.

## Newly Falsified Assumptions

- The claim that current logs are sufficient for independent decision reconstruction is falsified for the inspected corpus.

## Risks And Objections

- The diagnostic is conservative and event-level; it does not attempt to infer hidden state.
- Some rows are structural events rather than actual model decisions, but all 3495 decision-class events were insufficient under the RFC.
- Protocol160 source is not present in the clean branch, so source-level runtime verification remains incomplete here.

## Verdict

`partially_supported`

The implementation supports the decision `schema patch required`. A006 remains open and blocking.
