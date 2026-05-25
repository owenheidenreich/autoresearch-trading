# Cartography Report

Iteration ID: `ITER-001_quote_age_truth`
Assumption ID: `A001`
Title: Quote age truth

## Question

How is option quote age created, normalized, passed into Protocol101 paper decisions, validated by guards, and preserved in logs? Can current logs prove that `quote_age_ms` is a measured live quote age instead of a placeholder?

## Sources Read

| Source | Why read | Binding/stale/unknown |
|---|---|---|
| `research_ops/ASSUMPTION_REGISTRY.csv` | A001 states quote age truth is P0 and blocks replay/live trust. | Binding research-ops state. |
| `research_ops/CURRENT_STATE.yaml` | Confirms current default is guarded IBKR paper Protocol101 and next phase is execution/parity falsification. | Binding research-ops state. |
| `research_ops/bootstrap/V4_BASELINE_INVENTORY.md` | Baseline says Protocol101 paper is the frozen control and quote freshness remains unresolved. | Binding Stage 1 inventory, but some runtime files it cites are not present in this clean branch. |
| `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md` | Describes operational defaults, logs, quote freshness caveat, and Protocol158/160 paper path. | Source-of-truth document, but explicitly contains drift around Protocol158 quote age. |
| `docs/CURRENT_TRADING_BOT_IMPROVEMENT_QUESTIONS.md` | Defines the quote-age diagnostic as a priority read-only audit. | Current research question source. |
| `v4/schema/normalized.py` | Defines normalized `event_time`, `receive_time`, and nullable `quote_age_ms`. | Code truth in clean branch. |
| `v4/ingest/optionsdx.py` | Shows historical OptionsDX ingest sets `receive_time` equal to `event_time` and leaves `quote_age_ms` null. | Code truth in clean branch. |
| `v4/scripts/run_protocol128_paper_risk_gate.py` | Shows replay/risk-gate payload can convert missing `quote_gap_seconds` to `quote_age_ms=0`. | Code truth in clean branch for audit/replay tooling. |
| `v4/live/protocol101_risk_gate.py` | Shows Protocol101 risk gate only validates numeric `quote_age_ms <= 1500`. | Code truth in clean branch. |
| `v4/live/ibkr_paper_guard.py` | Shows final paper-order guard only validates numeric `quote_age_ms <= 1500`. | Protected code truth in clean branch; read-only. |
| `v4/live/ibkr_paper_executor.py` | Shows guard validation happens before any broker endpoint and logs the quote/context payload. | Protected code truth in clean branch; read-only. |
| `v4/live/paper_trade_log.py` | Shows paper JSONL schema and CSV flattening preserve `quote_age_ms` but not raw quote timestamps as required flattened fields. | Code truth in clean branch. |
| `v4/live/protocol101_shadow_schema.py` | Shows no-order shadow schema requires option `timestamp` and `quote_age_ms`. | Code truth in clean branch for shadow stream. |
| `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py` | Read-only source workspace evidence for Protocol158 because the clean branch docs reference it but the file is absent here. | Operationally relevant but not committed in this transition branch. Treat as external local evidence. |
| `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading/**/*.jsonl` | Existing paper/shadow logs available in the source workspace for read-only diagnostic evidence. | Local operational evidence, not tracked in this clean branch. |

## Files And Functions Involved

| Layer | File/function | Quote-age behavior |
|---|---|---|
| Normalized schema | `v4/schema/normalized.py::NORMALIZED_SCHEMA` | Defines `event_time`, nullable `receive_time`, `timestamp_source`, and nullable `quote_age_ms`. This is a schema slot, not proof of measured age. |
| Historical OptionsDX ingest | `v4/ingest/optionsdx.py::_row_to_records` | Parses `QUOTE_UNIXTIME` as `event_time`, sets `receive_time` equal to `event_time`, and emits `quote_age_ms: None`. OptionsDX therefore cannot prove live quote freshness. |
| Historical/replay risk payload | `v4/scripts/run_protocol128_paper_risk_gate.py::quote_payload` | Converts `quote_gap_seconds` to milliseconds when present, but writes `quote_age_ms=0` when `quote_gap_seconds` is missing. This is a known placeholder pattern. |
| Entry risk gate | `v4/live/protocol101_risk_gate.py::evaluate_entry_risk_gate` | Reads `quote.get("quote_age_ms")`; blocks only if missing or greater than `max_option_quote_age_ms=1500`. It does not require a raw quote timestamp. |
| Paper-order guard | `v4/live/ibkr_paper_guard.py::validate_order_intent` | Reads `quote.get("quote_age_ms")`; blocks `missing_quote_age` or `stale_option_quote`. A numeric zero passes. |
| Paper executor | `v4/live/ibkr_paper_executor.py::execute_guarded_paper_order` | Calls the guard before broker submission and preserves the raw `quote` and `context` in result payloads. It does not recompute quote age. |
| Paper log builder | `v4/live/paper_trade_log.py::make_trade_log_event` and `executor_result_event` | Writes `market_snapshot.option_nbbo` into JSONL. If raw timestamp fields are present in the quote dict they can survive JSONL, but the schema does not require them. |
| Paper CSV flattener | `v4/live/paper_trade_log.py::flatten_trade_event` | Exports `quote_age_ms` but does not flatten `quote_timestamp`, `received_timestamp`, or `decision_timestamp`. CSV logs are not sufficient for quote-age reconstruction. |
| Shadow schema | `v4/live/protocol101_shadow_schema.py::validate_shadow_event` | Requires `market_snapshot.option_nbbo.quote_age_ms` and `market_snapshot.option_nbbo.timestamp`, but only checks type/presence, not that age equals decision minus quote timestamp. |
| Protocol158 local source evidence | `/Users/gduby/Documents/autoresearch-trading/v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py::quote_freshness_from_ticker` | Current local source has helpers to coerce IBKR ticker timestamps and compute `quote_age_ms`, `received_timestamp`, and `decision_timestamp`. This partially supersedes the older doc claim that the bridge always sets zero. |
| Protocol160 | `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py` | Referenced by docs/inventory as the persistent paper default, but absent from this clean transition branch. Baseline says Protocol160 passes selected quote age to the executor; this iteration cannot verify that code inside this branch. |

## Inputs And Outputs

### Inputs

- Historical normalized rows: `event_time`, `receive_time`, nullable `quote_age_ms`.
- Runtime quote payloads: `market_snapshot.option_nbbo` inside paper/shadow JSONL.
- Runtime decision timestamps: top-level `timestamp`, optional `timing.decision_emitted_at`, optional `market_snapshot.option_nbbo.decision_timestamp`.
- Runtime quote timestamps, when present: possible fields include `market_snapshot.option_nbbo.timestamp`, `quote_timestamp`, `quote_timestamp_ms`, `time`, `timeStamp`, `rtTime`, `received_timestamp`, and `received_timestamp_ms`.
- Guard config: 1500 ms max quote age in both `Protocol101RiskConfig` and `PaperOrderGuardConfig`.

### Outputs

- Risk-gate decision: pass/fail plus reasons such as `missing_quote_age` and `stale_option_quote`.
- Paper JSONL events under `v4/logs/paper_trading/YYYY-MM-DD/*.jsonl`.
- Paper CSV exports that include `quote_age_ms` but omit most raw timestamp fields.
- Shadow JSONL events that require a quote timestamp and quote age but do not validate arithmetic consistency.

## Artifacts Used

- Stage 1 baseline inventory: `research_ops/bootstrap/V4_BASELINE_INVENTORY.md`.
- Source-of-truth docs: `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md` and `docs/CURRENT_TRADING_BOT_IMPROVEMENT_QUESTIONS.md`.
- Source workspace paper logs under `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading`.
- No broker APIs, market-data APIs, paid-data scripts, model artifacts, or runtime flags were used.

## Mutable State

The following are mutable or operationally sensitive and must remain read-only or untouched during this iteration:

- `v4/runtime/protocol101_paper_order_enablement.json`
- `v4/runtime/protocol101_live_index_context.jsonl`
- `v4/logs/paper_trading/**`
- `v4/ops/launchd/**`
- `v4/ops/ibkr/run_protocol101_paper_session.sh`
- `v4/live/ibkr_paper_executor.py`
- `v4/live/ibkr_paper_guard.py`
- Any Protocol101/051/066/081 model, scaler, or manifest artifact

The diagnostic may read copied or existing logs but must write only under `research_ops/iterations/ITER-001_quote_age_truth/artifacts/`.

## Broker/Data Risk

- `v4/live/ibkr_paper_executor.py` can call `ib.placeOrder` after guard/permission success when `dry_run=False`. It must not be imported or executed by this iteration.
- Protocol158/160 paper paths are broker-connected in paper-submit mode according to docs and inventory. They must not be run.
- Paid-data scripts and raw/normalized market-data directories are irrelevant to the read-only log audit and must not be accessed except as static source references already committed in Git.
- Existing paper logs may contain sensitive operational evidence. Diagnostic artifacts should summarize filenames and row-level timestamp fields, not account IDs or raw credentials. Paper log validation already forbids raw `account_id`, but the diagnostic should not assume that is always true.

## Stale Or Contradictory Docs

| Claim | Evidence | Interpretation |
|---|---|---|
| Source-of-truth says Protocol158 `live_option_quotes()` sets `quote_age_ms=0`. | `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md` marks this as a current weakness. | Partially stale relative to local source workspace Protocol158 helpers that compute age when IBKR ticker timestamps exist. Still directionally valid because logs have not proven those fields are present and arithmetic-consistent. |
| Stage 1 inventory cites Protocol160 code paths and paper logs. | `research_ops/bootstrap/V4_BASELINE_INVENTORY.md` references `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`. | The clean transition branch does not contain that file. For Stage 7, Protocol160 must be treated as documented operational context, not branch-verifiable code. |
| Shadow schema makes quote timestamp mandatory. | `v4/live/protocol101_shadow_schema.py` requires option `timestamp`. | This is stronger than the paper guard, but it does not prove quote-age arithmetic or paper-submit parity. |
| Paper CSV is a paper evidence source. | `v4/live/paper_trade_log.py::flatten_trade_event` exports `quote_age_ms`. | CSV is not enough for true-age audit because raw quote timestamp fields are not flattened. JSONL is the primary source. |

## Tests That Cover This Path

- Existing research-ops tests cover governance scripts and forbidden path validation: `tests/test_research_ops_scripts.py`.
- Existing code-level guard tests may exist in the full source workspace, but the clean transition branch currently exposes only the research-ops test file.
- No existing test in this branch validates quote-age reconstruction from paper JSONL.
- No existing test in this branch proves `quote_age_ms=0` with missing raw timestamp is treated as unknown instead of pass.

## Tests Missing

- Parser tests for quote timestamp candidates in paper/shadow JSONL.
- Classifier tests for `trustworthy`, `missing`, `placeholder`, `stale`, and `unreconstructable`.
- Regression test that `quote_age_ms=0` plus missing raw quote timestamp is never classified as trustworthy.
- Regression test that missing raw quote timestamp remains unknown even when guard-persisted age is within the threshold.
- CLI test that the diagnostic reads JSONL and writes CSV/markdown artifacts without importing `v4` or broker modules.

## Observability Gaps

1. Paper-order guard can pass a measured-looking zero without raw quote timestamp evidence.
2. Paper JSONL can preserve raw quote timestamp fields if runtime supplies them, but the schema does not require them.
3. Paper CSV loses timestamp fields needed for true-age reconstruction.
4. Shadow logs require an option timestamp but do not verify age arithmetic.
5. Protocol160 is operationally documented but not present in this clean branch, so code-level verification of persistent paper quote handling is incomplete here.
6. Existing logs may be insufficient to reconstruct candidate-level quote age for rejected candidates if they log only selected contract snapshots.
7. `received_timestamp`, `decision_timestamp`, and IBKR raw quote timestamp are not guaranteed fields in all runtime events.
8. Historical OptionsDX rows do not carry live receive-time semantics and should not be used to validate live paper freshness.

## Exact Implementation Constraints For The RFC

- Implement a read-only diagnostic that imports only Python standard-library modules and research-ops-local code.
- Do not import `v4`, `ib_insync`, broker modules, paid-data modules, model libraries, or model artifacts.
- Read only existing JSONL logs from paths provided on the command line.
- Write only to `research_ops/iterations/ITER-001_quote_age_truth/artifacts/`.
- Recompute age only when both a quote timestamp and a decision/received/top-level event timestamp are parseable.
- Classify missing raw quote timestamp as unknown evidence, never as pass.
- Classify `quote_age_ms=0` without a raw quote timestamp as `placeholder`.
- Report rows with missing persisted age as `missing`.
- Report rows with age above 1500 ms as `stale`.
- Report rows with parse errors, negative impossible ages, or mismatches as `unreconstructable`.
- Include source file and line number in output CSV for reproducibility.
- Avoid printing or copying `.env`, credentials, account IDs, or raw paid-data content.
