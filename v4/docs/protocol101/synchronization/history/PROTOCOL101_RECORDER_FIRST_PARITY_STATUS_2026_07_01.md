# Protocol101 Recorder-First Parity Status - 2026-07-01

## Objective

Prove that frozen Protocol101 can consume a complete IBKR capture offline and make economically equivalent decisions to the matching Databento/ThetaData historical reconstruction without training, threshold tuning, paper orders, or a default-model change.

## Verified Evidence

### IBKR captures

- June 30: full 390/390 regular-session checkpoints, no broker orders, raw JSONL preserved.
- July 1: full 390/390 regular-session checkpoints, no broker orders, raw JSONL preserved.
- Both captures produce 360 Protocol101 entry-decision rows from 09:31 through 15:30 ET.
- Repeated same-input replay is exact for both days.
- July 1 offline replay: 360 waits, zero entries, maximum edge `18.2996`.

July 1 capture root:

```text
~/.autoresearch-trading/live_runtime/ibkr_capture/2026-07-01/protocol101-recorder-2026-07-01/
```

### June 30 historical suite

- Databento OPRA.PILLAR definition, cbbo-1m, ohlcv-1m, and statistics downloaded successfully.
- Estimated Databento spend: `$0.6072`, below the approved `$2.00` cap.
- ThetaData SPX: 391 rows.
- ThetaData VIX: 350 rows.
- No other date was downloaded.

## Defects Found And Repaired

1. Captured quotes did not populate `quote_age_ms`; the live filter treated every quote as infinitely stale.
2. `LiveIndexState` retained only 50,000 ticks and silently dropped the opening minute on a 50,749-update session, corrupting OMAR and structure features.
3. Captured replay used IBKR vendor Greeks while historical replay used repaired Greeks. Both now use the shared repair calculation; vendor Greeks remain raw audit fields.
4. Historical index context used the current minute's completed OHLC close, which is future information at the decision boundary. Option CBBO remains at `T`, while index context now ends at `T-1 minute`.
5. Replay used checkpoints written after the minute boundary. It now reconstructs the latest IBKR option state received at or before each boundary from immutable events; checkpoints are fallback only.
6. The 30-minute context gate required 31 candles. It now requires 30 rows spanning 29 elapsed minutes.
7. Live early-session ATR was forced to zero while historical ATR used available completed bars. Both now use the completed bars available so far.
8. Historical Greek repair ignored the completed-context SPX override. The shared feature contract now supplies the explicit underlying to both paths.
9. Candidate count, risk-gate shape, token feature shape, and score shape were inconsistent across trace producers. They now use normalized semantics.
10. The paired diff treated every cross-vendor hash difference as a hard failure. Same-input remains exact; cross-vendor bounded value drift is review evidence, while action, contract, feature-contract, lifecycle, unbounded candidate, and risk divergence remain hard failures.
11. Offline replay scanned the 3 GB capture twice and repeatedly rebuilt dataframes from all index ticks. It now scans once and compacts index ticks to minute OHLC extrema for repeated model passes.

## June 30 Result

The final causal comparison uses:

```text
option CBBO: interval-end observation at decision T
index context: completed bars through T-1 minute
IBKR options: latest event received at or before T
decision window: 09:31-15:30 ET
```

Results:

- Paired decisions: `360`.
- Action mismatches: `0`.
- Live entries: `0`.
- Historical entries: `0`.
- Exact candidate sets: `336/360`.
- Mean candidate identity overlap: `99.5233%`.
- Minimum candidate identity overlap: `85.7143%`.
- Minutes below the 80% overlap gate: `0`.
- Median absolute flat-score drift: `0.0217`.
- Median absolute max-edge drift: `0.0928`.
- 95th-percentile absolute max-edge drift: `5.9738`.
- Live maximum edge: `21.6802`.
- Historical maximum edge: `21.7256`.
- Minutes at or above the frozen edge gate of 25: `0` on both streams.
- Paired result: `pass_with_review`, with zero hard decision failures.

Primary artifacts:

```text
v4/audit/autoresearch/protocol101_2026_06_30_cross_vendor_feature_audit_completed_context/
v4/audit/autoresearch/protocol101_2026_06_30_ibkr_vs_historical_paired_diff_completed_context/
v4/audit/autoresearch/protocol101_2026_06_30_parity_historical_replay_completed_context/
```

## Remaining Reviews

1. June 30 is a no-trade day. It proves wait-action parity but not entry selection, lifecycle parity, or fills.
2. ThetaData has no completed 09:30 VIX bar on June 30. The first historical row records neutral missing VIX while IBKR has a stale-but-causal pre-open value. This occurs inside the blocked first-30-minute bucket.
3. Historical entry traces do not carry an explicit account-state object. The paired report records `account_state_unavailable`; it does not assume equality.
4. Raw bid/ask sizes remain vendor-sensitive, although the final score comparison is close on most minutes.
5. A 0.14-point SPX vendor difference at 10:54 ET crossed the binary `last10_break_state` boundary and created the largest allowed-window edge outlier. Both decisions remained well below entry. This is classified as a threshold-adjacent feature discontinuity, not tuned away.
6. Execution, paper fills, slippage, and lifecycle behavior remain untested because neither stream entered.

## Next Gate

1. Preserve June 30 as development evidence.
2. Keep July 1 as an untouched captured confirmation day.
3. The existing recorder-only launchd packet remains date-scoped for July 2. It may collect one additional raw confirmation day; it cannot run live Protocol101 inference or call an order endpoint.
4. After explicit paid-data approval, download the July 1 matching Databento/ThetaData suite and run the same completed-context paired comparison.
5. If July 1 also passes, rebuild and replay the locally available Q1-2026 suite under the repaired feature contract and run the historical equity sanity gate.
6. Do not resume paper-submit until the wider historical sanity artifacts are acceptable.
7. A future parity day must contain entry opportunities before entry-contract and lifecycle parity can be claimed.

## Controls Preserved

- Protocol101 remains the selected paper default.
- Real-money trading remains disabled.
- No model training occurred.
- No threshold tuning occurred.
- No promotion/default change occurred.
- No broker order endpoint was called.
- Obsolete June 30 download and paused legacy heartbeat automations were removed after completion.

## June 30 Forensic Follow-Up

Matching `wait` actions are only coarse evidence. A second forensic pass therefore compared candidate identity, candidate ranking, model-standardized feature drift, max-edge tails, and frozen-model counterfactuals.

Additional defects found and repaired:

1. Historical market windows could read prior-session rows into missing pre-open slots. Live-style historical windows are now session-scoped.
2. Live missing pre-open slots mixed `NaN` prices with zero OMAR/range/momentum values. Both paths now represent a missing leading minute as a fully missing feature row before model sanitization.
3. Historical one-minute context lag also shifted decision-time bucket flags. Market observations remain lagged, while 10:00, 11:30, and 13:30 bucket semantics now use the actual decision time.
4. The live contract fell back to IBKR vendor Greeks when shared Greek repair failed, while historical rejected the same contract. Vendor Greeks are now audit-only; both paths require successful shared repair.
5. The cross-vendor audit now reports time-bucket drift, candidate rank/top-k agreement, model-standardized feature drift, edge-gate sensitivity, and largest outlier minutes.
6. A frozen-model score-attribution runner now swaps one paired feature family at a time without training or changing thresholds.

Post-repair June 30 metrics:

- Paired decisions: `360`.
- Action mismatches: `0`.
- Exact candidate sets: `348/360`, improved from `336/360`.
- Mean candidate identity overlap: `99.8407%`, improved from `99.5233%`.
- Candidate-set variation minutes: `12`, reduced from `24`.
- Median candidate-rank correlation: `0.9978`.
- Median top-3 identity overlap: `1.0`.
- Top-contract identity match: `88.61%`; most disagreements were among low/negative-edge contracts.
- Median absolute max-edge drift: `0.0755`, improved from `0.0928`.
- Mean absolute max-edge drift: `0.2577`, improved from `0.8055`.
- 95th-percentile absolute max-edge drift: `0.8023`, improved from `5.9738`.
- Live/historical maximum edge: `21.6802` / `21.7256`, on the same 7445 call at 11:02 ET.
- Frozen edge-25 crossing disagreements in allowed entry buckets: `0`.
- Allowed minutes within five points of the edge gate: `1` on each feed.

The remaining material outlier is 10:54 ET:

- Live max edge: `5.2087` on the 7495 put.
- Historical max edge: `-23.6206` on the 7440 call.
- A `0.14`-point SPX vendor difference crossed the binary `last10_break_state` boundary.
- This changed `pattern_last10_breakout`, `pattern_compression_breakout`, and `pattern_count_norm`.
- Frozen-model counterfactual replacement of only those discrete pattern fields closed `28.4318` of the `28.8293` edge-point gap.

Typical frozen-model sensitivity after repair:

- Historical quote/spread fields substituted into IBKR: p95 edge effect `0.6113`.
- Historical size/liquidity fields substituted into IBKR: p95 edge effect `0.3444`.
- Historical repaired-Greek/decay fields substituted into IBKR: p95 edge effect `0.1404`.
- Historical scalar context substituted into IBKR: p95 edge effect `0.0668`.
- All historical option-token fields substituted into IBKR: p95 edge effect `0.7355`.

Artifacts:

```text
v4/audit/autoresearch/protocol101_2026_06_30_cross_vendor_forensic_repaired_v3/
v4/audit/autoresearch/protocol101_2026_06_30_cross_vendor_score_attribution_repaired_v3/
v4/audit/autoresearch/protocol101_2026_06_30_parity_historical_replay_repaired_v2/
```

Interpretation:

- June 30 now supports broad wait-action, candidate-ranking, and ordinary-score parity.
- June 30 does not prove entry, selected-contract, lifecycle, execution, or fill parity.
- The binary breakout-pattern discontinuity remains a real model-input fragility. It should not be tuned against this one day. Any buffered/continuous replacement must be preregistered, rebuilt historically, and pass the historical sanity gate before prospective use.
- Raw quote sizes remain vendor-sensitive. Their measured edge effect is usually bounded but must be checked on an entry-opportunity day.
