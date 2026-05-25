# Cartography Report

Iteration ID: `ITER-003_replay_live_feature_parity`
Assumption ID: `A004`
Title: Replay/live feature parity

## Question

Does live Protocol051-to-Protocol101 feature construction match replay for the same market state?

## Files And Functions Involved

| Surface | Evidence | Parity Relevance |
|---|---|---|
| `/Users/gduby/Documents/autoresearch-trading/v4/live/protocol051_surface_edge.py::score_surface_decisions` | Scores `SurfaceDecision` objects with the frozen surface model. | Protocol101 depends on upstream surface scores and edge. |
| `/Users/gduby/Documents/autoresearch-trading/v4/live/protocol101_entry.py::protocol101_candidate_frame_from_surface` | Converts surface decisions and scores into Protocol101 candidate feature rows. | Main live feature construction surface. |
| `/Users/gduby/Documents/autoresearch-trading/v4/live/protocol101_entry.py::predict_protocol101_entry` | Scales feature rows and emits wait/candidate logits and selected action. | Parity requires scaled tensor/logit equivalence. |
| `/Users/gduby/Documents/autoresearch-trading/v4/live/protocol101_live_entry.py::build_live_surface_row` | Builds live normalized-like option ladder and market window from IBKR quotes. | Potential source of replay/live semantic drift. |
| `/Users/gduby/Documents/autoresearch-trading/v4/live/protocol101_live_entry.py::selected_contract_payload` | Includes selected quote/timestamp fields when present. | Helps selected-contract audit but not full candidate parity. |
| `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py::run_one_entry_decision` | Logs candidate set, model decision, and risk gate. | Existing logs do not include full tensors or replay references. |
| `docs/CURRENT_TRADING_BOT_IMPROVEMENT_QUESTIONS.md` | Says live/replay parity is critical because Protocol101 depends on Protocol051 surface scores and live feature differences can change logits. | Binding research question. |

## Inputs And Outputs

Inputs:

- Existing paper/shadow JSONL logs.
- Read-only source workspace code references for Protocol051/101 live construction.

Outputs:

- `feature_diff.csv`
- `logit_diff.csv`
- `candidate_set_diff.csv`
- `feature_parity_report.md`

## Artifacts Used

- Existing local paper-trading logs.
- No model binaries, scaler files, training outputs, paid-data files, or broker APIs.

## Mutable State

No runtime state was touched. The diagnostic writes only under this iteration.

## Broker/Data Risk

Running true parity would require loading artifacts and constructing replay/live paired rows. This iteration does not do that. It checks whether existing logs already contain enough paired evidence.

## Stale Docs

The clean branch lacks the live Protocol051/101 source files referenced by the Stage 1 inventory. Source workspace code was read-only context.

## Tests That Cover This Path

- `tests/test_feature_parity_diagnostic.py`
- `tests/test_research_ops_scripts.py` protected import scan

## Tests Missing

- Synthetic paired replay/live feature packet test.
- Future model-free hash comparison test for feature rows and candidate sets.
- Future model-loaded verifier test for logits, only if explicitly approved.

## Observability Gaps

- Existing logs do not contain paired replay reference rows.
- Existing logs do not contain full live feature tensors or hashes.
- Existing logs do not contain full Protocol051 score vectors and Protocol101 candidate logits.
- Candidate samples are not enough to verify parity.

## Exact Implementation Constraints For The RFC

- Use a read-only log evidence diagnostic.
- Do not load models.
- Do not import trading runtime.
- Do not score holdouts.
- Produce explicit `not_comparable` rows when evidence is absent.
