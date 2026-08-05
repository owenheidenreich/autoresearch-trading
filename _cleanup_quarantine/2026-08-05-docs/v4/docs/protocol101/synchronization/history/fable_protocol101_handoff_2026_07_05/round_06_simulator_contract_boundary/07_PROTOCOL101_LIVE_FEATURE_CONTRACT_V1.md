# Protocol101 Live Feature Contract V1

## Purpose

`Protocol101LiveFeatureContractV1` defines the model-facing information state that must be reproducible in both historical replay and IBKR live paper trading. It exists so Protocol101 is not trained or validated on one game and then asked to trade another.

The contract version is:

```text
protocol101-live-v1
```

## Contract Rules

- A decision at `T` may only use option quotes and SPX/VIX context available at or before `T`.
- Every row records `decision_time`, `source_quote_time`, `source_context_time`, `max_quote_age_ms`, and `feature_contract_version`.
- The SPXW 0DTE universe uses the same ATM-centered 5-point strike ladder across live and historical builders.
- Tradability uses the same quote-age, bid/ask, mid, spread, spread-fraction, size, and required-Greek checks.
- Greek repair uses the same repaired-Greek routine and the same bid/ask/mid, rate, dividend, expiry, and underlying inputs.
- Fields that IBKR live cannot reproduce reliably, including `option_ohlcv_volume` and `stat_open_interest`, are zeroed in live-style historical rows rather than leaked from Databento history.
- Future labels, best exits, future fills, and path-derived values are never part of runtime features.
- Candidate, feature, and score hashes are emitted so paired replay can diff the actual decision game.

## Builder Usage

Historical Databento/ThetaData rows are built in live-style mode with:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.build_databento_neural_dataset \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD> \
  --processed-dir <live-style-processed-dir> \
  --context-mode official \
  --official-spx-dir data/vendor/thetadata/index/spx_1m \
  --official-vix-dir data/vendor/thetadata/index/vix_1m \
  --feature-contract protocol101-live-v1
```

The legacy historical builder remains the default when `--feature-contract` is not supplied.

## Pre-Live Gate

Before repaired `paper-submit` sessions count as synchronization evidence, run a live-style historical replay and then the pre-live sanity gate:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_protocol101_pre_live_historical_sanity_gate \
  --protocol161-summary <protocol161-out-dir>/summary.json \
  --chart-summary <chart-out-dir>/summary.json
```

Protocol158 and Protocol160 block `paper-submit` by default unless the gate summary exists and reports:

```json
{
  "status": "pass",
  "paper_submit_allowed_by_gate": true
}
```

Shadow and dry-run modes may still be used for local validation.

## Paired Replay

After a repaired live session and matching historical replay exist, compare them with:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_protocol101_paired_live_historical_diff \
  --live-traces <live-jsonl> \
  --historical-traces <historical-decision-traces.jsonl> \
  --out-dir <paired-diff-out-dir>
```

Same-input replay must be exact. Cross-vendor replay must match non-threshold-adjacent decisions or classify the mismatch as bounded, explainable, and non-systematic.

## Serial Simulator Contract

Feature parity alone is not enough; the replay account state must also be live-reproducible. Future fair-contract folds and selected-candidate replay gates use:

```text
protocol101_serial_simulator_v2
```

The simulator contract is documented in:

```text
v4/docs/PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2.md
```

Required state semantics:

- daily-loss basis: `raw_realized_net_pnl`
- cash/affordability basis: `raw_realized_net_pnl`
- stress application: `metrics_only`

The stress haircut is a validation/reporting concept. It must not trigger daily-loss stops, affordability blocks, cooldown, lifecycle state, or any other in-simulation state transition. Historical artifacts created before this contract remain archived under their original semantics and must not be rerun as new evidence without a new registry entry.
