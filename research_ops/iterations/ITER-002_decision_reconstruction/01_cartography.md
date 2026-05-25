# Cartography Report

Iteration ID: `ITER-002_decision_reconstruction`
Assumption ID: `A006`
Title: Decision reconstruction

## Question

Can an independent reviewer reconstruct every live, paper, and no-order Protocol101 decision from logs alone?

## Files And Functions Involved

| Surface | Evidence | Reconstruction Relevance |
|---|---|---|
| `v4/live/paper_trade_log.py::make_trade_log_event` | Paper log contract writes event type, timestamp, session, run id, mode, selected contract, order, account, market snapshot, model decision, and risk gate. | Good base schema, but does not require full candidate features, raw logits, artifact references, or raw quote timestamps. |
| `v4/live/paper_trade_log.py::flatten_trade_event` | Flattens paper JSONL to CSV with `quote_age_ms`, model action/score/threshold, risk reason, and timing fields. | CSV cannot reconstruct decisions when JSONL lacks raw feature tensors and candidate sets. |
| `v4/live/protocol101_shadow_schema.py::validate_shadow_event` | Requires no-order shadow fields and blocks broker intent fields. | Safer for no-order mode, but still validates presence rather than independent reproducibility. |
| `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py::run_one_entry_decision` | Appends `candidate_set`, `model_decision`, `risk_gate`, and account state events. | Logs candidate count/sample and model action, but not enough to rerun Protocol051/101 exactly. |
| `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py::append_event` | Shared paper/shadow event writer used by Protocol160. | Operationally relevant, but source exists in the local source workspace, not the clean transition branch. |
| `docs/CURRENT_TRADING_BOT_IMPROVEMENT_QUESTIONS.md` | Says missing feature, quote, timestamp, candidate, or artifact references falsify reconstruction. | Binding research question. |

## Inputs And Outputs

Inputs:

- Existing JSONL logs under `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading`.
- Stage 1 source-of-truth docs and baseline inventory.

Outputs:

- Event-level reconstruction matrix.
- Missing-field report.
- Schema gap report.

## Artifacts Used

- `research_ops/iterations/ITER-001_quote_age_truth/artifacts/quote_age_rows.csv`
- Existing local paper-trading JSONL logs.
- No broker APIs, paid data, model loading, training, threshold tuning, or runtime mutation.

## Mutable State

- Paper logs are operational evidence and were read only.
- `v4/runtime/**`, `v4/ops/**`, `v4/live/ibkr_paper_executor.py`, and `v4/live/ibkr_paper_guard.py` remain protected.

## Broker/Data Risk

The diagnostic must not execute Protocol158/160, import IBKR code, or call `placeOrder`. It only parses JSONL.

## Stale Docs

The clean transition branch references Protocol160 as current paper runtime, but Protocol160 source is absent from this branch. Local source workspace code was treated as read-only explanatory evidence, not branch truth.

## Tests That Cover This Path

- `tests/test_decision_reconstruction_diagnostic.py`
- `tests/test_research_ops_scripts.py` protected import scan

## Tests Missing

- A future test that runs against a deliberately complete synthetic decision packet with all model logits, feature hashes, and guard inputs.
- A future schema-level test that enforces candidate/feature/logit fields at logging time.

## Observability Gaps

- Full candidate features are not logged.
- Rejected candidate logits are not logged.
- Runtime artifact references are not logged per decision.
- Raw quote timestamps and recomputed quote age are not logged per decision.
- Guard inputs are not logged with enough fidelity to rerun each gate.
- Paper-submit logs contain no broker endpoint rows in the inspected corpus.

## Exact Implementation Constraints For The RFC

- Implement stdlib-only log parsing.
- Read existing JSONL logs only.
- Write artifacts only under this iteration.
- Do not import `v4`, broker clients, model libraries, or paid-data clients.
- Classify rows as sufficient only when independent reconstruction fields are present.
