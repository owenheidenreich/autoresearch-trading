# Protocol101 Parity Pause Point - 2026-06-16

## Recorder-First Superseding Update - 2026-06-26

The June 23-26 packet failed before Protocol101 or the recorder could collect evidence. Gateway startup occurred, but the Python session repeatedly raised `OSError: [Errno 11] Resource deadlock avoided` while importing source from the iCloud File Provider-backed research repository. The old retry wrapper also allowed `set -e` to terminate a failed attempt before its retry loop could continue. Process-alive monitoring therefore reported a misleadingly healthy setup while no JSONL evidence grew.

The replacement is recorder-first:

- Market-hours capture runs from an immutable hash-named bundle under `~/.autoresearch-trading`, never from the repository in `Documents`.
- The recorder imports only the standard library and `ib_insync`; it has no model or order-execution imports.
- Raw `IBKRMarketCaptureV1` events are append-only and survive recorder restarts.
- Health checks require a growing JSONL, fresh heartbeat, live SPX/VIX, a complete SPXW ladder, and no write/subscription errors.
- Protocol101 runs only after close over the captured stream.
- Monday June 29 is development data. June 30 through July 2 run only if Monday capture integrity, opening context, trace extraction, and exact same-input replay all pass.
- July 3 has no job because the options market is closed.

Operational reference: `v4/docs/PROTOCOL101_RECORDER_FIRST_PARITY.md`.

## Historical Pause Note For June 22 Resume

The problem the owner identified was that Protocol101 was not behaving the same in IBKR live paper trading as it did in Databento/ThetaData historical replay. Historical live-style replay for the early June dates produced entry signals and trades, while the captured IBKR paper sessions did not. That proved the project had not yet earned the claim that live and historical were playing the same decision game.

The investigation found two important layers:

1. The older June 4/5/8/9 IBKR logs were useful for coarse forensic comparison, but they were not replay-grade. They did not contain full `Protocol101DecisionTraceV1` candidate universes, token feature vectors, feature hashes, score hashes, and per-candidate model outputs. Because of that, they could show that live/historical behavior diverged, but they could not support exact same-input replay.
2. The repaired June 12 logging path does contain replay-ready decision traces. The June 12 captured session now extracts `312` decision traces, and same-input replay passes with `312/312` exact matches. That means the trace plumbing is ready for the next clean live session, even though June 12 itself remains partial/diagnostic because startup/context issues affected the day.

What has been fixed so far:

- Historical and live feature-contract work was started so both sides use the same live-reproducible game.
- Live decision logging now emits `Protocol101DecisionTraceV1` fields needed for replay-level comparison.
- Offline readiness tooling was added to distinguish old top-token logs from replay-grade logs.
- A captured-trace extractor was added so IBKR paper JSONL can be converted into paired-diff-ready live trace JSONL.
- The readiness checker was repaired to treat pre-score administrative blocks correctly. Rows blocked before scoring, such as `outside_time_bucket` and `insufficient_live_index_context`, do not need model token vectors because the model was intentionally not invoked. Scored candidate rows still must include full token features and hashes.
- June 12 captured logs now pass the trace-readiness gate, and same-input replay passes.

What still needs to be checked:

- A clean full-day IBKR session after market-data eligibility is restored.
- That the session starts with correct opening context, including OMAR from the 09:30 ET market-open candle.
- That every live candidate-set row is trace-ready from market open through the historical trading cutoff.
- That live candidate universes, feature hashes, scores, threshold distances, block reasons, and account/risk states can be paired to the matching historical replay.
- That the matching Databento/ThetaData historical day, built under the same live feature contract, either matches live decisions or produces bounded, explained, non-systematic differences.
- That any actual trade intents, submissions, fills, cancels, rejects, exits, and forced-flat behavior can be reconstructed from logs.

When IBKR live market data is restored, the next step is not hill climbing. The next step is one clean prospective parity day:

1. Re-enable only the authorized Protocol101 IBKR paper collection stack.
2. Run one repaired full-day paper session.
3. Run captured trace readiness on that live log.
4. Extract live decision traces.
5. Confirm same-input replay is exact.
6. Download/build the matching historical Databento/ThetaData day only if explicitly approved and available.
7. Run Protocol101 historical replay under the same live feature contract.
8. Run paired live-vs-historical diff.
9. Classify any mismatches as candidate-universe drift, feature drift, score drift, action drift, risk/account drift, timestamp/source drift, or execution-only drift.
10. Only after one clean paired day should the project schedule a new multi-day evidence packet.

Until then, live collection remains paused through the expected June 22 resume window, and no model training, threshold tuning, paid-data broadening, promotion/default changes, or real-money paths should be touched.

## Current State

IBKR live market-data collection is paused until account equity/cash settlement restores market-data eligibility. Do not restart the IBKR paper-submit collection stack until that external condition is fixed and owner-authorized.

The offline trace bridge is ready:

- `v4/scripts/run_protocol101_captured_trace_readiness.py`
- `v4/scripts/run_protocol101_extract_captured_decision_traces.py`
- `v4/tests/test_protocol101_captured_trace_tools.py`

June 12 captured live paper logs now pass the replay-readiness gate:

- `v4/audit/autoresearch/protocol101_captured_trace_readiness_2026_06_12/summary.json`
- `v4/audit/autoresearch/protocol101_extract_captured_decision_traces_2026_06_12/summary.json`

June 12 extracted `312` replay-ready decision traces and same-input replay passed with `312` exact matches.

The archived June 4/5/8/9 live logs still fail replay readiness because they predate full `Protocol101DecisionTraceV1` logging. They remain useful for coarse forensic comparison, but not exact same-input replay.

## What Was Repaired Offline

The readiness checker now distinguishes:

- scored candidate rows, which must include full token features, token feature hashes, candidate universe hash, feature hash, and score hash;
- pre-score administrative block rows, such as `outside_time_bucket` and `insufficient_live_index_context`, which may carry a raw quote universe without model token vectors because the model was intentionally not invoked.

This does not change trading behavior. It only prevents replay-readiness tooling from rejecting valid pre-score blocked decision rows.

## Safe Verification Commands

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  v4/tests/test_protocol101_captured_trace_tools.py \
  v4/tests/test_protocol101_decision_trace_and_diff.py

PYTHONPATH=. .venv/bin/python -m v4.scripts.run_daily_paper_autopilot --print-selection
```

## Resume Criteria

Resume live collection only after all are true:

- IBKR market-data snapshot/live eligibility is restored.
- Owner explicitly authorizes broker connectivity and paper-submit collection.
- Protocol101 remains selected in `v4/promotion/PAPER_TRADING_DEFAULT.json`.
- Real-money trading remains disabled.
- IBKR paper autostart/session/monitor/shutdown labels are intentionally reloaded.

## Next Live Step

Run one repaired full-day IBKR paper session first. After the session:

1. Run captured trace readiness for that date.
2. Extract captured decision traces.
3. Confirm same-input replay is exact.
4. Only then download/build the matching historical Databento/ThetaData day if explicitly approved and available.
5. Run paired live-vs-historical diff.

Do not train, tune thresholds, broaden paid downloads, or change the paper default during this gate.
