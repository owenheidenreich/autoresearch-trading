# Implementation Summary

Iteration ID: `ITER-003_replay_live_feature_parity`
Assumption ID: `A004`
Title: Replay/live feature parity

## Change Summary

Implemented a stdlib-only read-only diagnostic that checks whether existing logs contain paired replay/live feature, candidate, and logit evidence.

## Files Created

- `research_ops/diagnostics/feature_parity.py`
- `tests/test_feature_parity_diagnostic.py`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/artifacts/feature_parity_report.md`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/artifacts/feature_diff.csv`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/artifacts/logit_diff.csv`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/artifacts/candidate_set_diff.csv`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/artifacts/feature_parity_summary.json`

## Files Modified

- `research_ops/iterations/ITER-003_replay_live_feature_parity/00_request.md`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/01_cartography.md`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/02_rfc.md`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/03_implementation_summary.md`
- `research_ops/iterations/ITER-003_replay_live_feature_parity/manifest.yaml`

## Commands Run

- `python3 -m research_ops.diagnostics.feature_parity --log-root /Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading --out-dir research_ops/iterations/ITER-003_replay_live_feature_parity/artifacts`
- `python3 -m pytest tests/test_quote_age_truth_diagnostic.py tests/test_decision_reconstruction_diagnostic.py tests/test_feature_parity_diagnostic.py tests/test_research_ops_scripts.py`

## Tests

Focused run passed: `20 passed`.

## Artifacts Generated

- `feature_parity_report.md`
- `feature_diff.csv`
- `logit_diff.csv`
- `candidate_set_diff.csv`
- `feature_parity_summary.json`

## Diagnostic Result

- Decision: `replay metrics not yet usable`.
- Feature comparison rows: `13592`, all `not_comparable`.
- Logit comparison rows: `1103`, all `not_comparable`.
- Candidate-set comparison rows: `906`, all `not_comparable`.

## Limitations

- This diagnostic does not load models or compute numeric tensor diffs.
- It proves evidence absence, not numeric feature mismatch.
- A future approved parity harness must create paired replay/live rows for the same timestamp and contracts.
