# Experiment RFC

Iteration ID: `ITER-001_quote_age_truth`
Assumption ID: `A001`
Title: Quote age truth

## 1. Research Question

Are live/paper quote ages in existing Protocol101 logs real measured ages, or are quote freshness guards passing because runtime supplies placeholder values such as `quote_age_ms=0` without raw quote timestamp evidence?

## 2. Null Hypothesis

The existing logs do not prove quote age truth. A row is not trusted unless the diagnostic can recompute quote age from a raw quote timestamp and a decision/observation timestamp, and the recomputed age agrees with persisted `quote_age_ms` within tolerance.

## 3. Required Inputs

- Read-only JSONL paper/shadow logs supplied by CLI, expected roots:
  - `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading`
  - optional in-branch `v4/logs/paper_trading` if present
- The cartography report for field candidates:
  - `research_ops/iterations/ITER-001_quote_age_truth/01_cartography.md`
- No broker APIs.
- No paid data.
- No model artifacts.
- No runtime flags.

## 4. Required Outputs

All diagnostic outputs must be written under:

`research_ops/iterations/ITER-001_quote_age_truth/artifacts/`

Required files:

- `quote_age_rows.csv`: one row per parsed JSONL event with quote-age fields and classification.
- `quote_age_summary.md`: human-readable counts, findings, and interpretation limits.
- `quote_age_summary.json`: machine-readable summary counts.

## 5. Formulas

Let:

- `persisted_age_ms` = numeric `market_snapshot.option_nbbo.quote_age_ms` or another explicit event-level `quote_age_ms`.
- `quote_ts` = first parseable raw quote timestamp field from `market_snapshot.option_nbbo`.
- `decision_ts` = first parseable decision timestamp from `market_snapshot.option_nbbo.decision_timestamp`, `timing.decision_emitted_at`, `timing.intended_entry_time`, or top-level `timestamp`.
- `received_ts` = first parseable receive/observation timestamp from `market_snapshot.option_nbbo.received_timestamp` or equivalent.
- `reference_ts` = `decision_ts` if present, else `received_ts`, else top-level event `timestamp`.
- `recomputed_age_ms = max(0, (reference_ts - quote_ts) * 1000)` if the raw difference is not less than `-1000 ms`.
- `age_delta_ms = persisted_age_ms - recomputed_age_ms`.

Default thresholds:

- `max_age_ms = 1500`
- `tolerance_ms = 250`

The 250 ms tolerance is not a trading claim. It absorbs serialization and clock-source granularity differences for this first diagnostic only.

## 6. Columns

`quote_age_rows.csv` must include:

- `source_file`
- `line_number`
- `event_type`
- `timestamp`
- `run_id`
- `mode`
- `selected_action`
- `broker_order_endpoint_called`
- `quote_age_path`
- `quote_timestamp_path`
- `reference_timestamp_path`
- `received_timestamp_path`
- `persisted_quote_age_ms`
- `quote_timestamp`
- `reference_timestamp`
- `received_timestamp`
- `recomputed_quote_age_ms`
- `age_delta_ms`
- `has_quote_timestamp`
- `has_reference_timestamp`
- `has_received_timestamp`
- `classification`
- `trust_status`
- `reason`

## 7. Pass/Fail Criteria

Use row classifications:

- `trustworthy`: persisted age exists; raw quote timestamp exists; reference timestamp exists; recomputed age is within `tolerance_ms`; neither persisted nor recomputed age is greater than `max_age_ms`.
- `stale`: persisted or recomputed age is greater than `max_age_ms`.
- `placeholder`: persisted `quote_age_ms` is exactly zero or near-zero and the row lacks a raw quote timestamp, or recomputation shows nonzero age while persisted age is zero.
- `missing`: no persisted `quote_age_ms` exists.
- `unreconstructable`: timestamps are malformed, reference timestamp is unavailable, raw quote timestamp is missing while persisted age is nonzero, quote timestamp is implausibly after reference timestamp, or recomputed age mismatches persisted age.

Use aggregate verdict:

- `pass`: at least 95% of rows with persisted quote age are `trustworthy`, all broker endpoint rows are `trustworthy`, and `placeholder`, `missing`, `stale`, and `unreconstructable` rows are explainable non-trading smoke/logging rows.
- `fail`: any broker endpoint row is `placeholder`, `missing`, `stale`, or `unreconstructable`; or more than 5% of persisted quote-age rows are `placeholder`; or persistent paper-submit rows rely on placeholder age.
- `unknown`: no broker endpoint rows exist and available logs are mostly missing/unreconstructable/placeholder, or the diagnostic cannot observe raw quote timestamps for paper-submit decisions.

Important rule: missing raw quote timestamp is never a pass. It is `placeholder` when persisted age is zero and `unreconstructable` otherwise.

## 8. Tests Required

Add tests that prove:

- A row with quote timestamp, reference timestamp, and matching persisted age is `trustworthy`.
- A row with `quote_age_ms=0` and no raw quote timestamp is `placeholder` and `trust_status=unknown`.
- A row with no persisted `quote_age_ms` is `missing`.
- A row above `max_age_ms` is `stale`.
- A row with missing raw quote timestamp and nonzero persisted age is `unreconstructable`.
- The CLI writes CSV, markdown, and JSON artifacts from a temporary JSONL log.
- The diagnostic does not import `v4`, `ib_insync`, paid-data clients, `torch`, `pandas`, or `numpy`.

## 9. Implementation Plan

Create:

- `research_ops/diagnostics/__init__.py`
- `research_ops/diagnostics/quote_age_truth.py`
- `tests/test_quote_age_truth_diagnostic.py`

Modify:

- `tests/test_research_ops_scripts.py` to include research-ops diagnostics in the protected import scan.

Execution:

1. Implement stdlib-only timestamp parsing for ISO strings and epoch seconds/milliseconds/microseconds/nanoseconds.
2. Implement path-based field extraction for option quote, timing, and top-level timestamp candidates.
3. Implement row classification exactly as specified above.
4. Implement CLI:
   - `--log-root` repeatable
   - `--log-file` repeatable
   - `--out-dir`
   - `--max-age-ms`
   - `--tolerance-ms`
5. Run against existing local source logs in read-only mode:
   - `/Users/gduby/Documents/autoresearch-trading/v4/logs/paper_trading`
6. Write artifacts only under the iteration artifact directory.
7. Run tests and iteration validation.

## 10. Interpretation Guide

- `trustworthy` means the row supports quote-age arithmetic consistency. It does not prove the quote was executable, fillable, or latency-free.
- `placeholder` means the row is direct evidence that freshness could pass without raw timestamp truth.
- `missing` means the row cannot be used for quote-age trust.
- `stale` means the row would violate the current guard threshold if passed into the guard with that age.
- `unreconstructable` means available logs are insufficient or internally inconsistent.
- Aggregate `unknown` is still a blocker for paper-submit trust if paper-submit rows lack raw timestamp evidence.

## 11. Limitations

- Existing logs may not include Protocol160 source-level context because the clean transition branch lacks Protocol160.
- Existing logs may be smoke/no-order/paper-dryrun logs rather than live broker endpoint rows.
- The diagnostic audits logged evidence only. It cannot prove IBKR's true market-data timestamp semantics.
- CSV exports are not sufficient when JSONL lacks raw timestamp fields.
- This diagnostic does not modify logging. If logs are insufficient, a later CEO-approved observability iteration is required.
- This diagnostic does not score protected holdouts, train models, tune thresholds, promote challengers, mutate runtime flags, or call brokers.
