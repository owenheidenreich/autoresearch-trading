# Implementation Summary

Iteration ID: `ITER-002_decision_reconstruction`
Assumption ID: `A006`
Title: Decision reconstruction

## Change Summary

Implemented a stdlib-only read-only diagnostic that parses Protocol101 JSONL logs and reports whether each decision event has enough durable evidence for independent reconstruction.

## Files Created

- `research_ops/diagnostics/decision_reconstruction.py`
- `tests/test_decision_reconstruction_diagnostic.py`
- `research_ops/iterations/ITER-002_decision_reconstruction/artifacts/decision_reconstruction_matrix.csv`
- `research_ops/iterations/ITER-002_decision_reconstruction/artifacts/decision_reconstruction_summary.json`
- `research_ops/iterations/ITER-002_decision_reconstruction/artifacts/missing_fields_report.md`
- `research_ops/iterations/ITER-002_decision_reconstruction/artifacts/log_schema_gap_report.md`

## Files Modified

- `research_ops/iterations/ITER-002_decision_reconstruction/00_request.md`
- `research_ops/iterations/ITER-002_decision_reconstruction/01_cartography.md`
- `research_ops/iterations/ITER-002_decision_reconstruction/02_rfc.md`
- `research_ops/iterations/ITER-002_decision_reconstruction/03_implementation_summary.md`
- `research_ops/iterations/ITER-002_decision_reconstruction/manifest.yaml`

## Commands Run

- `python3 -m research_ops.diagnostics.decision_reconstruction --log-root /Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading --out-dir research_ops/iterations/ITER-002_decision_reconstruction/artifacts`
- `python3 -m pytest tests/test_quote_age_truth_diagnostic.py tests/test_decision_reconstruction_diagnostic.py tests/test_feature_parity_diagnostic.py tests/test_research_ops_scripts.py`

## Tests

Focused run passed: `20 passed`.

## Artifacts Generated

- `decision_reconstruction_matrix.csv`
- `decision_reconstruction_summary.json`
- `missing_fields_report.md`
- `log_schema_gap_report.md`

## Diagnostic Result

- Decision: `schema patch required`.
- Total rows parsed: `6663`.
- Decision rows: `3495`.
- Sufficient decision rows: `0`.
- Insufficient decision rows: `3495`.
- Broker endpoint rows: `0`.

Dominant missing fields:

- `artifact_refs`: `3398`
- `candidate_count`: `2492`
- `option_quote`: `2588`
- `quote_age`: `2588`
- `quote_timestamp`: `2589`
- `full_candidate_features`: `906`

## Limitations

- This is a log sufficiency diagnostic, not a replay engine.
- It does not import or execute trading runtime code.
- It cannot infer missing model logits, features, quote timestamps, or artifact references.
