# Hardened Plan To Reach Hill Climbing

> Current resolution status, baseline corrections, and data-plane selection
> gates are documented in
> `v4/docs/PROTOCOL101_SYNCHRONIZATION_RESOLUTION.md`.

## Summary

The goal is to reach model hill climbing only after the project proves it is optimizing the same game the bot can play live. The path is:

1. Build deterministic decision-replay and cross-vendor live-vs-historical parity tooling.
2. Capture paired IBKR no-order sessions, then replay the same timestamps from Databento.
3. Separate decision trust from execution trust and generalization trust.
4. Backfill the missing modern SPXW daily-expiration era only after parity is stable.
5. Start hill climbing under purged validation, experiment registry control, uncertainty reporting, and a hardened Balanced Profitability Score.

Protocol101 remains the baseline and paper default until a separate promotion packet explicitly changes it.

## Synchronization Implementation - 2026-06-11

The next phase is synchronization, not hill climbing. The implemented direction is:

```text
Databento/ThetaData raw history -> protocol101-live-v1 feature contract -> Protocol101 replay
IBKR live snapshots -> protocol101-live-v1 feature contract -> Protocol101 runtime
```

New synchronization controls:

- `v4/live/protocol101_feature_contract.py` defines `Protocol101LiveFeatureContractV1`.
- `build_databento_neural_dataset` accepts `--feature-contract protocol101-live-v1`.
- Live and historical rows record `feature_contract_version`, `source_quote_time`, `source_context_time`, and quote freshness metadata.
- Protocol161 emits feature-contract and source-timestamp fields into `decision_traces.jsonl`.
- `run_protocol101_pre_live_historical_sanity_gate.py` creates the pre-live pass/fail artifact.
- Protocol158 and Protocol160 block `paper-submit` unless the pre-live historical sanity gate passes, unless explicitly skipped for a controlled local test.
- `run_protocol101_paired_live_historical_diff.py` compares live and historical traces offline.

Operational rule:

```text
Do not resume repaired IBKR paper-submit evidence collection until the live-style historical sanity gate passes.
```

## Status Update - 2026-06-11

The initial inspection that motivated this plan has now passed. The project set out to compare IBKR live paper sessions against same-date historical replay because Protocol101 appeared to trade historically while taking no trades live. That concern is now proven real.

Corrected completed-minute historical replay over the complete archived live sessions produced historical Protocol101 entry signals while IBKR live did not:

| Session | Corrected Historical Entry Signals | IBKR Live Above-Edge Minutes | IBKR Live Trades |
|---|---:|---:|---:|
| 2026-06-04 | 0 | 0 | 0 |
| 2026-06-05 | 8 | 0 | 0 |
| 2026-06-08 | 1 | 0 | 0 |
| 2026-06-09 | 29 | 0 | 0 |

This proves that the historical replay and IBKR live paper runtime are not yet playing the same model-facing game. The mismatch happens before broker execution: at every historical entry minute, live had market data and evaluated the model, but the live surface gate remained `below_min_edge`.

The project should now move from broad evidence collection into synchronization work. The objective is no longer merely to collect five sessions. The objective is to define and enforce one live-reproducible feature contract used by both paths:

```text
Historical raw data -> live-style causal feature builder -> Protocol101 model inputs
IBKR live snapshots -> same live-style causal feature builder -> Protocol101 model inputs
```

This does not mean training on raw IBKR logs alone. IBKR live logs are too small to cover enough SPXW 0DTE regimes. Historical Databento/ThetaData remains the broad training source, but historical rows must be transformed into the same causal information state the live IBKR runtime can actually observe.

Current reference artifacts:

- `v4/docs/PROTOCOL101_LIVE_HISTORICAL_PARITY_REFERENCE_2026_06_11.md`
- `v4/audit/autoresearch/v4_aplus_hypothesis_161_june2026_fiveday_historical_replay_completed_minute_lag1`
- `v4/audit/autoresearch/v4_aplus_hypothesis_162_june2026_fiveday_serial_lifecycle_replay_completed_minute_lag1`
- `v4/audit/autoresearch/protocol101_live_historical_parity_anomaly_scan_2026_06_11`

Until this synchronization layer is implemented and proven, `equity.html` and historical replay PnL should be treated as useful research evidence, not trusted live-equivalent PnL.

## Key Changes

### Parity Standards

Implement two parity bars:

- **Same-input deterministic replay:** exact match required. The same captured live JSONL must reproduce identical candidates, features, scores, actions, risk gates, and lifecycle decisions.
- **Cross-vendor IBKR-vs-Databento replay:** exact match required for non-threshold-adjacent decisions. Threshold-adjacent mismatches are allowed only if bounded, explained, logged, and non-systematic.

A decision is threshold-adjacent if any of these are true:

- Protocol101 entry margin is within `5%` of the frozen entry threshold.
- Protocol051 surface edge is within `5%` of the candidate gate.
- Selected-vs-runner-up logit gap is within `5%` of the entry threshold.

Five paired sessions are an engineering smoke gate only. Passing five days proves the harness and reconstruction work; it does not prove profitability, regime coverage, or execution calibration.

### Decision Trace And Diff Interfaces

Add `Protocol101DecisionTraceV1` for both live and historical replay records.

Required fields:

- session, decision timestamp, source, received timestamp, raw quote timestamp, quote age
- SPX/VIX context and freshness
- full SPXW contract ladder and candidate set
- candidate hash, feature hash, raw Protocol051 scores, raw Protocol101 logits
- enter/wait decision, selected contract, selected action reason
- lifecycle hold/exit decision when holding
- account/risk state, affordability, open position count, block reasons
- broker flags: broker endpoint called, live orders enabled

Add `Protocol101PairedReplayDiffV1` with mismatch classes:

- contract universe mismatch
- quote/value drift
- feature calculation mismatch
- score/logit mismatch
- action mismatch
- lifecycle mismatch
- account/risk mismatch
- execution-only mismatch

Cross-vendor pass thresholds:

- `100%` match for non-threshold-adjacent scheduled decisions
- `100%` match for actual entry-intent decisions unless mismatch is explicitly vendor-threshold-adjacent and economically equivalent
- no more than `1%` threshold-adjacent behavioral mismatches
- no systematic side, time-bucket, premium, or moneyness bias in mismatches

### Synchronize The Game

Before paired replay, align or freeze these rules:

- SPXW same-day PM-settled contracts only
- ATM ± `$50`, `$5` strikes, calls and puts
- same stale-quote policy for parity replay
- same bid/ask/spread/size filters before scoring
- same affordability and one-open-position rules
- same internal Greek repair logic historically and live
- same timestamp policy, with Databento replay forced to IBKR decision timestamps
- same volume/open-interest policy: either live-reproducible fields or deterministic zeroing in both paths

Known parity risks must be fixed or explicitly documented before hill climbing: quote freshness mismatch, live/historical Greek repair mismatch, live missing volume/OI, and lifecycle sequence-vs-live-row mismatch.

### Live Evidence Targets

Decision-parity evidence:

- Phase 1: `5` paired no-order IBKR/Databento sessions.
- Phase 2: expand to `10` paired no-order sessions or until at least `3` matched entry-intent events are observed.
- If no entry-intent events occur, label the result `abstention_parity_only` and keep collecting.

Execution evidence:

- Phase 1 smoke: `30` order outcomes and `10` complete round trips.
- Phase 2 calibration: continue until fill/slippage estimates are stable by side, time bucket, premium bucket, spread bucket, quote age, moneyness, and volatility regime.
- IBKR paper fills are evidence, not truth. Historical equity remains stress-modeled until fill/slippage stability is proven.

Execution reports must include fill rate, cancel rate, rejection rate, latency, slippage versus decision quote, and profitability under `$0.10` and `$0.25` per-side stress.

### Data Backfill

After paired parity passes, request paid approval for the missing modern daily-SPXW era.

Backfill target:

- start: `2022-05-11`, after Cboe completed Tuesday/Thursday SPXW expiration expansion
- end: current available date, excluding future protected holdout
- products: SPXW definitions, `cbbo-1m`, `ohlcv-1m`, statistics, SPX/VIX 1m context
- high-resolution audit slices for parity failures, execution calibration days, and threshold-adjacent decisions

Raw downloads must be preserved and stamped with source, schema, date range, symbols, parameters, and approval metadata.

## Hill-Climb Controls

### Validation Rules

Use chronological session-level splits only.

Rules:

- no random row/contract splits
- no contracts from the same timestamp split across train/validation/test
- purge all rows whose label horizon overlaps validation/test windows
- embargo at least `5` trading sessions between train and validation/test windows
- protected holdout is scored once only after model/search rules are frozen
- paired parity days are diagnostics, not model-selection data

Default broad-history folds after backfill:

- Fold A: train 2022-05-11 through 2023-12-31, validate 2024 H1, test 2024 H2
- Fold B: train through 2024-12-31, validate 2025 H1, test 2025 H2
- Fold C: train through 2025-12-31, validate 2026 Q1, exposed diagnostic test 2026 current
- Protected holdout: future unseen block collected after the scoring rules are frozen

### Experiment Registry

Add append-only `ExperimentRegistryEntryV1`.

Each hill-climb attempt must record:

- hypothesis, failure mode, owner, timestamp
- code version, data version, artifact versions
- feature set, label definition, action space
- train/validation/test windows and embargo settings
- hyperparameters, seeds, threshold rules
- score, uncertainty report, acceptance/rejection reason
- whether protected data was touched

Negative results must be stored. The number of attempted variants must be included in overfit-risk reporting.

### Balanced Profitability Score V2

Use hard gates before scoring.

Hard fail if:

- parity gate not passed
- leakage or future/path fields in runtime features
- unaffordable, overlapping, or not-flat-by-close headline trades
- protected holdout used for tuning
- negative under `$0.10` per-side slippage stress
- promotion candidate negative under `$0.25` per-side stress
- max drawdown worse than `1.25x` Protocol101 baseline
- trade count below `75%` of Protocol101 baseline
- top day > `35%` of positive PnL
- top 5 days > `65%` of positive PnL
- top 10 trades > `40%` of total PnL
- removing the best 10 trades makes total PnL non-positive

Score formula:

```text
BPS_V2 =
  30 * robust_net_pnl_per_day_ratio
+ 20 * return_on_premium_ratio
+ 15 * drawdown_efficiency_ratio
+ 10 * win_rate_ratio
+ 10 * expectancy_per_trade_ratio
+ 10 * trade_frequency_sufficiency_score
+  5 * regime_stability_score
- penalties
```

Definitions:

- `robust_net_pnl_per_day_ratio`: candidate winsorized mean daily net PnL divided by Protocol101 baseline.
- `drawdown_efficiency_ratio`: candidate PnL/max-drawdown efficiency divided by baseline.
- `trade_frequency_sufficiency_score`: rises to `1.0` at baseline trade count, caps there, and penalizes churn above `1.5x` baseline trades.
- `regime_stability_score`: fraction of required regime buckets with positive PnL and no worse than `80%` of baseline PnL/day.
- penalties cover slippage fragility, outlier concentration, excessive churn, worse tail loss, and live/historical drift.

Protocol101 baseline scores `100`. A candidate is hill-climb-interesting at `>=105`, promotion-interesting at `>=115`, and still must pass all hard gates.

### Uncertainty Report

Every candidate must report:

- day-block bootstrap confidence intervals for daily PnL and return on premium
- top-day, top-month, top-trade, side, time-bucket, and regime concentration
- score after removing best 1, 5, and 10 trades
- score after removing best 1 and 5 days
- slippage stress curve
- PBO/CSCV or equivalent family-selection risk
- Deflated Sharpe or equivalent multiple-testing-aware statistic when enough variants exist

## Scalp-Versus-Runner Path

Do not train a runner model first. Diagnose first.

Measure:

- MFE, MAE, exit efficiency, continuation after exit
- early-exit opportunity cost
- forced-flat opportunity cost
- winner giveback risk
- whether longer holding improves expected value after slippage and drawdown
- whether scalp and runner setups are separable from causal entry-time features

Only after the diagnostic proves an economically meaningful missed-runner problem should a lifecycle or runner classifier become a hill-climb hypothesis.

## Test Plan

Required tests:

- same-input captured-live replay is bitwise/deterministically identical
- Databento replay can be forced to live IBKR decision timestamps
- contract ladder parity for ATM ± `$50`
- Greek repair parity on identical quote inputs
- stale/missing/fresh quote cases produce identical pass/block reasons
- no-trade sessions are diffed, not ignored
- entry-intent sessions match selected contract and decision time
- lifecycle replay matches hold/exit/forced-flat decisions
- BPS_V2 gives Protocol101 baseline score `100`
- hard gates block candidates before score ranking
- experiment registry records both accepted and rejected trials
- existing Protocol101, data, inference, and fake paper-runtime tests continue to pass

## Assumptions

- Full historical backfill happens after parity, not before.
- Five paired days are a smoke test, not statistical validation.
- Same-input replay must be exact.
- Cross-vendor replay may tolerate bounded threshold-adjacent drift only when classified and non-systematic.
- Paper-submit, broker connectivity, paid downloads, training, tuning, and promotion remain owner-authorized actions.
- Hill climbing starts only after decision parity, validation controls, experiment registry, and protected-holdout rules are in place.
