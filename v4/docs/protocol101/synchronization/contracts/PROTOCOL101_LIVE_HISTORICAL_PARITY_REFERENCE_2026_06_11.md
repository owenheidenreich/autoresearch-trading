# Protocol101 Live/Historical Parity Reference - 2026-06-11

## Objective

The current parity objective is to prove that Protocol101 live IBKR paper trading and historical Databento/ThetaData replay are playing the same decision game before any hill climbing, threshold tuning, or paper-default promotion. The evidence question is:

```text
Given the same decision minute, contract universe, feature contract, model artifacts,
risk/account state, and execution assumptions, does live paper runtime produce the
same candidate, score, enter/wait, and lifecycle behavior as historical replay?
```

As of this reference sheet, the answer is still no. The live runtime has been operationally complete on several days, but the old archived live logs and repaired historical replay still disagree on surface edge and entry behavior.

## Current Artifacts

- Archived live root: `/Users/gduby/.autoresearch-trading/archive/ibkr_live_trading_sessions_2026-06-04_05_08_09_10_20260611T155134Z/live_runtime_paper_trading`
- Corrected completed-minute historical replay: `v4/audit/autoresearch/v4_aplus_hypothesis_161_june2026_fiveday_historical_replay_completed_minute_lag1`
- Corrected serial lifecycle replay: `v4/audit/autoresearch/v4_aplus_hypothesis_162_june2026_fiveday_serial_lifecycle_replay_completed_minute_lag1`
- Minute-level anomaly scan: `v4/audit/autoresearch/protocol101_live_historical_parity_anomaly_scan_2026_06_11`

## Repairs Already Made

### Live Feature Parity Repairs

Files:

- `v4/live/protocol101_live_entry.py`
- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`
- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`

Repairs:

- Live index context now preserves enough rows for full-session feature construction.
- Live opening structure features use opening-minute semantics closer to historical replay.
- Live minute grid starts from the 09:30 ET open and avoids non-causal leading backfill.
- Live opening-gap logic can load prior-session close context.
- Live strike ladder uses the historical-style SPXW 0DTE ladder: +/- 50 points around ATM at 5-point steps.
- Live candidate/tradability mask now follows historical `NeuralDatasetConfig` defaults:
  - quote age <= 90 seconds
  - mid between 0.50 and 35.00
  - absolute spread <= 0.50
  - spread fraction <= 0.25
  - minimum bid/ask size checks
  - required IV/delta/gamma/theta
- Live logs now include full decision-trace fields for future parity diffs:
  - `candidate_universe`
  - `candidate_universe_hash`
  - `features`
  - `feature_hash`
  - `model_scores`
  - `score_hash`
  - selected contract, risk gate, and block reasons

### Historical Replay Timing Repairs

File:

- `v4/scripts/run_protocol161_may2026_historical_replay.py`

Repairs:

- Added `--completed-minute-lag-minutes`.
- In completed-minute mode, replay logs the effective live decision at `T` while scoring only the source row/features from the completed historical minute.
- Fixed a mixed-time bug where option/window features came from `T-1` but market-structure features were accidentally recomputed at `T`.
- Historical decision traces now include:
  - `source_decision_time`
  - `source_decision_ts`
  - `completed_minute_lag_minutes`
  - `timestamp_alignment_mode`
- Added `--scope-index-context-to-sessions` so small parity runs do not scan hundreds of unrelated SPX/VIX files.
- Added stale-prior-context protection with `--max-prior-context-gap-days`; the scoped replay no longer silently uses a weeks-old prior close when the actual prior session file is absent.

### Live Greek Repair Fallback Alignment

File:

- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`

Repair:

- `live_option_quotes()` now passes `ask` and `bid` into `_compute_live_greeks()`, matching the historical Greek repair path more closely. Previously live used mid only, while historical could repair from mid, ask, then bid.

This is a parity cleanup, not proof that Greeks are now fully aligned. Future full decision traces are still required to compare every token's IV/delta/gamma/theta directly.

## Corrected Replay Results

Completed-minute Protocol161 replay over June 4, 5, 8, and 9:

| Session | Historical Decision Rows | Historical Entry Signals | Historical Edge Minutes |
|---|---:|---:|---:|
| 2026-06-04 | 360 | 0 | 0 |
| 2026-06-05 | 360 | 8 | 14 |
| 2026-06-08 | 360 | 1 | 6 |
| 2026-06-09 | 360 | 29 | 29 |

Serial Protocol162 replay from those entries:

- Candidate signals: 38
- Serial trades: 7
- Serial total PnL: 5250.00
- Serial win rate: 57.1%
- Serial max loser: -1870.00
- 31 independent entry signals were skipped due to an already-open serial position.

These numbers are evidence of historical behavior only. They are not trusted live-equivalent PnL while the live/historical score mismatch remains unresolved.

## Live-Vs-Historical Findings

The archived live sessions were not sparse. Live had roughly full-session candidate evaluation and fresh option quotes:

- Live candidate rows were about 381-385 rows/day.
- Median live option quote count was usually 41-42.
- Median live option quote age at joined replay minutes was about 56-146 ms.
- Median live SPX age was about 449-518 ms.
- Median live VIX age was about 14 seconds.

The remaining mismatch is not broker execution and not missing order submission. It happens before orders:

- Historical replay produced 38 entry signals across June 5, 8, and 9.
- Live produced 0 above-min-edge minutes across the same joined decision minutes.
- At every historical entry minute, live's filter reason was `below_min_edge`.

Minute anomaly scan:

| Session | Historical Edge Minutes | Live Edge Minutes | Historical Entry Signals | Historical Entries With Live Edge Candidate | Median Abs Max-Edge Delta |
|---|---:|---:|---:|---:|---:|
| 2026-06-04 | 0 | 0 | 0 | 0 | 12.684 |
| 2026-06-05 | 14 | 0 | 8 | 0 | 14.892 |
| 2026-06-08 | 6 | 0 | 1 | 0 | 16.261 |
| 2026-06-09 | 29 | 0 | 29 | 0 | 24.175 |

This means the live runtime did evaluate the market, but it evaluated the surface as much weaker than historical replay.

## Current Best Root-Cause Hypotheses

### 1. Surface Score / Feature Drift

The largest unresolved issue is not startup, not order permissions, and not sparse live data. It is score/feature drift before the Protocol101 entry gate.

Example from June 9 around 10:11 ET:

- Historical completed-minute replay selected put-side candidates with edges above 30.
- Live had fresh quotes and enough context, but its best surface tokens remained below the 25-point min-edge gate.
- The same neighborhood showed materially different option economics between historical and live views.

This requires future full-trace candidate diffs using the repaired live logs.

### 2. Option Quote / Greek / Timestamp Coupling

Historical Databento CBBO one-minute rows and official SPX/VIX context may not be internally aligned the same way live IBKR snapshots are aligned. Even after the completed-minute repair, the processed historical row can still combine:

- option quote fields from a one-minute CBBO convention,
- official index context from a one-minute bar convention,
- repaired Greeks based on the historical quote/context pair.

Live IBKR uses near-real-time option quotes and near-real-time SPX/VIX snapshots. If historical options are effectively stale or differently labeled relative to index context, the repaired Greeks and surface scores can be dramatically different.

### 3. Missing Prior Context For June 4

The official context directory does not contain `2026-06-03` SPX/VIX files. The first scoped replay initially fell back to `2026-05-20` as the prior available session, which is too stale for opening-gap features. This has been repaired so stale prior files are skipped and recorded instead of silently used.

June 4 had no historical entry signals, so this did not explain the trade mismatch, but it is still a data-completeness issue.

### 4. Old Live Logs Lack Full Token Feature Traces

The archived June 4/5/8/9 logs were captured before full candidate/feature/score trace logging was added. They contain useful diagnostics and top-token samples, but not the full model-facing feature vector for every candidate. Future live sessions should be used for exact `Protocol101DecisionTraceV1` diffs.

### 5. Local `.venv` Torch Import Stall

The repo `.venv` can hang on `import torch`. The live runtime venv at `/Users/gduby/.autoresearch-trading/runtime-venv/bin/python` imports torch cleanly and was used for the corrected replay runs. For parity tooling, prefer the runtime venv until the repo `.venv` is repaired.

## Current Answer: Change Historical Or Change Live?

Do not train directly on raw IBKR live feed alone. There is not enough IBKR history, and a few live weeks cannot cover enough 0DTE regimes.

Also do not force live IBKR to impersonate an unrealistic historical row if the historical row is non-causal or internally misaligned.

The right target is a single live-reproducible feature contract:

```text
Historical raw data -> live-style causal feature builder -> model inputs
Live IBKR snapshots -> same live-style causal feature builder -> model inputs
```

In practice, this means:

- Keep Databento/ThetaData for broad historical coverage.
- Transform historical data into the same information state available to live at decision time.
- Use completed-minute or quote-timestamp rules explicitly.
- Use the same candidate/tradability mask.
- Use the same Greek repair method and inputs.
- Use the same opening/context semantics.
- Use IBKR live sessions as calibration and parity evidence, not the main training corpus.

The model should see the live-style feature contract in both training and live trading. That usually means changing how historical data is converted into model rows, while also making live compute exactly the same contract. It does not mean trusting the old historical rows blindly, and it does not mean discarding historical data in favor of raw IBKR logs.

## Next Recommended Investigation

Before declaring the live/historical feed irreconcilable, run a repaired future live day with full decision traces, then download/replay the same day and diff:

1. candidate IDs by minute,
2. bid/ask/mid/spread by contract,
3. SPX/VIX context by timestamp,
4. IV/delta/gamma/theta by contract,
5. scalar feature hashes,
6. token feature hashes,
7. surface flat/action scores,
8. Protocol101 logits and selected action.

If the next full-trace day still shows historical edge where live shows none, the next repair should be in the historical data builder: rebuild historical decision rows from a strictly live-style causal snapshot contract, then rerun Protocol101 on that rebuilt dataset.

## Synchronization Phase Implementation

The project now has an explicit live-style feature contract and pre-live sanity gate.

Files added:

- `v4/live/protocol101_feature_contract.py`
- `v4/docs/PROTOCOL101_LIVE_FEATURE_CONTRACT_V1.md`
- `v4/scripts/run_protocol101_paired_live_historical_diff.py`
- `v4/scripts/run_protocol101_pre_live_historical_sanity_gate.py`

Files updated:

- `v4/dataset/spxw_0dte_neural.py`
- `v4/scripts/build_databento_neural_dataset.py`
- `v4/live/protocol101_live_entry.py`
- `v4/scripts/run_protocol158_protocol101_live_entry_paper_bridge.py`
- `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`
- `v4/scripts/run_protocol161_may2026_historical_replay.py`
- `v4/live/protocol101_decision_trace.py`
- `v4/live/protocol101_paired_replay_diff.py`

New invariant:

```text
IBKR paper-submit synchronization evidence is blocked until a transformed
historical Protocol101 pass produces a passing pre-live sanity gate summary.
```

The first validation sequence should be:

1. Rebuild June 4/5/8/9 or Q1-2026 rows with `--feature-contract protocol101-live-v1`.
2. Rerun Protocol161 and Protocol162 on the transformed rows.
3. Export `equity.html`, `trades.html`, `trades.csv`, `summary.json`, and `report.md`.
4. Run `run_protocol101_pre_live_historical_sanity_gate.py`.
5. Only if the gate passes, collect a repaired IBKR live full-trace session.
6. Download/replay the matching date and run `run_protocol101_paired_live_historical_diff.py`.
