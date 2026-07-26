# Protocol101 Recorder-First Parity Operations

> Synchronization selection and baseline lineage are tracked in
> `v4/docs/PROTOCOL101_SYNCHRONIZATION_RESOLUTION.md`.

## Objective

Capture one complete, immutable IBKR paper-market stream and make market hours necessary only for acquisition. Protocol101 runs after close over that captured stream. The resulting candidate universes, feature vectors, model scores, thresholds, actions, and block reasons can then be compared with a matching Databento/ThetaData replay at identical decision timestamps.

This phase does not train a model, tune thresholds, submit paper orders, or enable real-money trading.

## Packet

The packet covers:

```text
2026-06-29  development capture; always scheduled
2026-06-30  confirmation capture; requires Monday gate
2026-07-01  confirmation capture; requires Monday gate
2026-07-02  confirmation capture; requires Monday gate
2026-07-03  no job; observed Cboe options-market holiday
```

The launchd label prefix is:

```text
com.autoresearch.protocol101.parityrecorder
```

The immutable runtime is deployed under:

```text
~/.autoresearch-trading/runtime-bundles/protocol101-parity-v1/<bundle-hash>/
```

No scheduled market-hours process reads from the repository in `Documents`.

## Schedule

```text
05:30 PT  start paper IB Gateway
05:40 PT  API/live-entitlement preflight
05:45 PT  start recorder, client ID 159
05:50 PT  recorder health
06:00 PT  recorder health
06:15 PT  live feed and full-ladder health
06:28 PT  final pre-open health
13:05 PT  stop recorder and Gateway
13:10 PT  finalize manifest and checksums
13:15 PT  integrity audit and exact same-input replay
```

## Raw Capture Contract

`IBKRMarketCaptureV1` is append-only. Every row has a monotonic sequence, producer instance ID, connection epoch, session, event type, source timestamp, receipt timestamp, and payload. Restarts append to the existing file with a new producer instance. A process lock prevents concurrent writers.

Artifacts:

```text
~/.autoresearch-trading/live_runtime/ibkr_capture/<session>/<capture-id>/
  market_events.jsonl
  capture_state.json
  capture_manifest.json
  checksums.sha256
  preflight.json
  health_*.json
```

The recorder captures SPX, VIX, the SPXW 0DTE definition ladder, event-driven option updates, source and receipt timestamps, market-data type, bid/ask/sizes/last, available IBKR Greeks, errors, disconnects, reconnects, and one full ladder checkpoint for each completed regular-session minute.

The recorder imports no model, torch, training, or order-execution code. It cannot call an order endpoint.

## Offline Replay

The canonical adapter applies this causal clock:

```text
09:30:00-09:30:59 ET observations -> completed 09:30 minute
completed 09:30 minute            -> 09:31 decision
decision input                    -> latest received observation at or before decision
last entry decision               -> 15:30 ET
```

Derived artifacts:

```text
ibkr_capture_quality.json
ibkr_canonical_minutes.parquet
ibkr_protocol101_traces.jsonl
same_input_replay_summary.json
same_input_replay_report.md
```

The raw JSONL is never repaired or replaced. Any future adapter correction creates new derived artifacts from the same immutable source.

## Monday Gate

Tuesday through Thursday start only when:

```text
~/.autoresearch-trading/live_runtime/ibkr_capture/collection_gate_2026-06-29.json
```

is `Protocol101CollectionGateV1` and all four fields pass:

```text
capture_integrity=pass
opening_context=pass
trace_extraction=pass
same_input_replay=pass
```

Monday capture acceptance requires 390 unique checkpoints from 09:30 through 15:59 ET, a present 09:30 opening minute, live rather than delayed feeds, valid JSON, monotonic sequences, quote/contract evidence, exact repeated replay hashes, and zero order events.

## Commands

Build and install the exact-date packet:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.deploy_protocol101_recorder_parity --install-launchd
```

Inspect labels:

```bash
launchctl list | rg 'com.autoresearch.protocol101.parityrecorder'
```

Inspect the current deployment:

```bash
readlink ~/.autoresearch-trading/runtime-bundles/protocol101-parity-v1/current
```

Paid Databento/ThetaData acquisition is a separate, explicit approval after a usable IBKR capture exists.
