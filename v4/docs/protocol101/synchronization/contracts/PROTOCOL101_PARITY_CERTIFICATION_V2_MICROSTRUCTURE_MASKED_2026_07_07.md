# Protocol101 V2 Microstructure-Masked Parity Certification

Generated: 2026-07-07

## Decision

Status: `hill_climbing_unblocked_for_protocol101_live_v2_microstructure_masked`

This packet certifies the data/feature game, not a profitable model.  The
frozen attempt107 model is coherent under the final contract but loses money on
the Q1 2026 sanity pass.  The correct next step is model hill climbing/training
on the certified fair contract.  Paper-submit, promotion/default changes, and
real-money trading remain blocked.

## Final Feature Contract

Selected contract:

```text
protocol101-live-v2-microstructure-masked
```

Implementation:

- `v4/live/protocol101_feature_contract.py`
- `v4/dataset/spxw_0dte_neural.py`
- `v4/live/protocol101_live_entry.py`
- `v4/scripts/build_databento_neural_dataset.py`
- `v4/scripts/run_protocol101_fair_contract_ibkr_capture_replay.py`
- `v4/scripts/run_protocol101_fair_contract_dataset_replay.py`

Model-scoring transform:

```text
mask_vendor_sensitive_option_quote_greek_microstructure
```

Masked before model scoring:

```text
bid
ask
mid
spread
spread_frac
bid_size
ask_size
option_ohlcv_volume
stat_open_interest
iv
delta
gamma
theta
breakeven_distance
```

Preserved outside model scoring:

- Raw bid, ask, and mid remain available for tradability, affordability, fills, labels, P&L, and audit.
- Raw quote/source metadata remains in decision traces.
- Greeks may be computed and logged, but v2 does not allow missing repaired Greeks to silently remove a candidate from the model-facing universe.

Why this is the final contract:

- The original raw/current contract had action and selected-contract mismatches on paired IBKR/Databento days.
- Diagnostic masking removed those mismatches.
- This implementation turns that diagnostic into a named contract and applies it through historical replay and IBKR recorder replay.

## Code Repairs

1. Added `Protocol101LiveFeatureContractV2MicrostructureMasked`.
2. Added a contract-level model-scoring transform for vendor-sensitive option quote and Greek microstructure.
3. Added `feature_contract_requires_model_scoring_greeks`.
4. Updated the historical builder so v2 does not require finite Greeks for candidate inclusion when Greeks are not model-scored.
5. Updated live-row construction so recorder/live row builders can use the same feature contract and candidate gate.
6. Updated IBKR capture replay and historical dataset replay to force the v2 model-scoring transform.
7. Updated paired diff classification so candidate-universe drift during identical hard no-entry account/risk blocks is reported as `candidate_universe_drift_non_actionable`, not a decision failure.
8. Added an offline historical sanity exporter:

```text
v4/scripts/run_protocol101_fair_contract_historical_sanity_report.py
```

## Recorder-Day Evidence

Paired recorder days:

```text
2026-06-30
2026-07-01
2026-07-02
```

Historical build artifact:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_2026_06_30_07_02_build/summary.json
```

Build totals:

```text
definition_rows: 58638
cbbo_rows: 842060
ohlcv_rows: 104728
statistics_rows: 10690
normalized_rows: 842060
derived_spx_rows: 1173
derived_vix_rows: 1118
neural_rows: 1080
```

Replay artifacts:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_ibkr_capture_replay_2026_06_30/
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_ibkr_capture_replay_2026_07_01/
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_ibkr_capture_replay_2026_07_02/
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_historical_replay_2026_06_30/
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_historical_replay_2026_07_01/
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_historical_replay_2026_07_02/
```

All six replay jobs passed same-input exactness.

Cross-vendor paired diff artifacts:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_paired_diff_2026_06_30/
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_paired_diff_2026_07_01/
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_paired_diff_2026_07_02/
```

Cross-vendor result:

| Session | Rows | Status | Decision Failures | Action Mismatches | Selected-Contract Mismatches |
|---|---:|---|---:|---:|---:|
| 2026-06-30 | 360 | `pass_with_review` | 0 | 0 | 0 |
| 2026-07-01 | 360 | `pass_with_review` | 0 | 0 | 0 |
| 2026-07-02 | 360 | `pass_with_review` | 0 | 0 | 0 |

Residual review categories:

- `source_timestamp_drift`: expected cross-vendor timestamp/source differences.
- `feature_drift`: remaining non-action-changing drift, mainly market/index context and raw-vendor audit fields.
- `score_drift`: bounded score-hash differences after masking.
- `candidate_universe_variation`: overlap above the hard drift threshold.
- `candidate_universe_drift_non_actionable`: two rows only, both during identical hard no-entry blocks (`cooldown` or `session_trade_cap`).

There are no unclassified action mismatches and no selected-contract mismatches.

## Q1 Historical Sanity Evidence

Q1 v2 build artifact:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_q1_2026_build/summary.json
```

Build scope:

```text
start_date: 2026-01-02
end_date: 2026-03-31
sessions_considered: 63
sessions_built: 61
sessions_skipped: 2026-01-19, 2026-02-16
neural_rows: 21899
normalized_rows: 11140845
```

Replay artifact:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_q1_2026_historical_replay/
```

Replay result:

```text
same_input_exact: true
decision_rows: 21899
candidate_rows: 636139
selected_entries: 179
feature_transform: mask_vendor_sensitive_option_quote_greek_microstructure
threshold: -16.8679
max_trades_per_session: 3
broker_endpoint_called: false
paper_submit_allowed: false
model_training_executed_here: false
threshold_tuning_executed_here: false
```

Historical sanity artifact:

```text
v4/audit/autoresearch/protocol101_live_v2_microstructure_masked_greekgate_q1_2026_historical_sanity/
```

Required outputs:

```text
equity.html
trades.csv
summary.json
report.md
```

Sanity metrics:

```text
status: pass
starting_equity: 10000.0
ending_equity: 7210.0
total_pnl: -2790.0
trades: 179
wins: 77
losses: 102
win_rate: 0.4301675977653631
profit_factor: 0.9472490073737947
max_drawdown: -8870.0
premium_deployed: 463700.0
return_on_premium: -0.006016821220616778
sessions_with_trades: 60
min_cash: 6220.0
```

Interpretation:

- The final contract is coherent enough to train against: it produces rows, candidates, decisions, traces, selected entries, and equity/trade artifacts without replay crashes, missing features, zero-trade collapse, impossible cash ruin, broker calls, or tuning.
- The frozen attempt107 model is not good enough under the final fair contract.  This does not block hill climbing; it is exactly why hill climbing should now optimize on the certified contract rather than on the older unfair or vendor-sensitive game.

## Verification Commands Run

Focused tests:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m pytest -q \
  v4/tests/test_protocol101_feature_contract.py \
  v4/tests/test_protocol101_live_entry.py \
  v4/tests/test_supervised_pilot.py \
  v4/tests/test_protocol101_synchronization.py

PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m pytest -q \
  v4/tests/test_protocol101_decision_trace_and_diff.py \
  v4/tests/test_protocol101_feature_contract.py \
  v4/tests/test_protocol101_live_entry.py \
  v4/tests/test_supervised_pilot.py
```

Results:

```text
72 passed
57 passed
```

Compile checks:

```bash
PYTHONPATH=. ~/.autoresearch-trading/runtime-venv/bin/python -m py_compile \
  v4/live/protocol101_feature_contract.py \
  v4/dataset/spxw_0dte_neural.py \
  v4/live/protocol101_live_entry.py \
  v4/scripts/run_protocol101_fair_contract_ibkr_capture_replay.py \
  v4/scripts/run_protocol101_fair_contract_dataset_replay.py \
  v4/scripts/build_databento_neural_dataset.py \
  v4/scripts/run_protocol101_fair_contract_historical_sanity_report.py
```

## Remaining Risks

- Cross-vendor features are not byte-identical.  The accepted gate is decision parity plus explicit drift classification, not raw-price identity.
- The active paper-submit scripts still default to their existing live contract unless a future promotion/runtime packet explicitly wires a v2-trained candidate to v2.  Do not run paper-submit until that promotion/runtime packet exists.
- The frozen attempt107 model loses money under v2 Q1 sanity.  Hill climbing is unblocked, but paper trading is not.
- Residual source timestamp drift is expected because IBKR recorder observations and Databento/Theta historical rows have different source timestamps.
- Candidate-universe drift must continue to fail if it occurs while a trade is actually possible.  The non-actionable classification only applies when both sides have the same hard no-entry block.

## Final Statement

Future model training may now optimize the same model-facing feature game that
IBKR recorder replay can reproduce:

```text
protocol101-live-v2-microstructure-masked
```

Hill climbing is unblocked for this final fair contract.  Paper-submit,
promotion/default changes, and real-money trading remain blocked until a
separately trained candidate passes the project promotion and paper-readiness
gates.
