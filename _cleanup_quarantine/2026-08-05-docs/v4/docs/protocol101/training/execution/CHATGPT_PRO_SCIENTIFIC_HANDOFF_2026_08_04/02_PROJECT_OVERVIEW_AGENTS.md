# AGENTS.md

Agent reference for `/Users/gduby/Documents/autoresearch-trading`.

Last updated: 2026-07-26

## 1. Project Overview

This project is a v4 SPXW 0DTE system for building an AI options trader.

The goal is simple to say and hard to earn: build an AI that trades options and makes money. The project gets there through a staged path: first the data foundation, then model training, then historical validation, then live paper trading, and only then real-money review. The current operating control is:

```text
PAPER_DEFAULT_PROTOCOL101
```

The current paper-default registry is:

```text
v4/promotion/PAPER_TRADING_DEFAULT.json
```

The current paper-default runtime entrypoint is:

```text
v4.scripts.run_protocol160_protocol101_persistent_paper_trader
```

The active spine is:

```text
market data
  -> normalized data
  -> causal datasets and labels
  -> surface scoring
  -> Protocol101 entry decision
  -> lifecycle and exit decision
  -> guarded paper execution
  -> logs, monitors, equity charts, trade charts, reports
```

## 2. End Goal

The end goal is an AI options trader that can trade SPXW 0DTE options profitably with real money.

The path to that goal is:

```text
data foundation
  -> model training
  -> historical validation
  -> live paper trading
  -> real-money trading review
```

Each stage has to earn the next one. Before the project can responsibly risk real capital, it must:

- Prove where its data came from.
- Build live-reproducible features.
- Train and validate model candidates without leakage.
- Compare candidates against the current paper default under the same trading game.
- Run the selected paper default through live paper trades.
- Reconstruct every decision, order, fill, exit, block, and failure from logs.
- Show that paper-trading behavior supports the historical edge instead of contradicting it.

Making money is the target, not an assumption. Real-money trading comes only after the data, model, validation, and paper-trading evidence justify a separate owner-approved review.

## 3. Mental Model

Think of the project as five connected layers:

| Layer | Role | Main locations |
|---|---|---|
| Data | Acquire, normalize, validate, and store market data. | `data/`, `v4/raw/`, `v4/normalized/`, `v4/ingest/`, `v4/checks/` |
| Research | Build datasets, train candidates, replay trades, and generate reports. | `v4/dataset/`, `v4/model/`, `v4/scripts/`, `v4/audit/autoresearch/` |
| Model stack | Convert market state into entry and exit decisions. | `v4/live/protocol051_surface_edge.py`, `v4/live/protocol101_entry.py`, `v4/live/protocol066_inference.py` |
| Paper runtime | Run the current paper default with account, quote, and order guards. | `v4/scripts/run_daily_paper_autopilot.py`, `v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py`, `v4/live/`, `v4/runtime/`, `v4/logs/` |
| Governance | Define the current default, promotion rules, validation gates, and safety boundaries. | `AGENTS.md`, `PROJECT_SECTION_AND_FEATURE_MAP.md`, `docs/`, `research_ops/`, `v4/docs/`, `v4/promotion/` |

## 4. Important Files

Start here:

```text
AGENTS.md
PROJECT_SECTION_AND_FEATURE_MAP.md
docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md
v4/README.md
v4/docs/protocol101/training/README.md
v4/docs/DATA_CONTRACT.md
v4/docs/PROJECT_SECTIONS_AND_HILL_CLIMB_GATES.md
v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md
v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md
v4/docs/NAMING_GUIDE.md
v4/docs/PROTOCOL101_DAILY_PAPER_TRADING.md
```

For Protocol101 training, begin with
`v4/docs/protocol101/training/README.md`. It separates current training
authority from synchronization history and records the next permitted gate.

Current default and runtime state:

```text
v4/promotion/PAPER_TRADING_DEFAULT.json
v4/runtime/protocol101_paper_order_enablement.json
v4/runtime/protocol101_live_paper_state.json
v4/runtime/protocol101_live_index_context.jsonl
```

Current model/runtime code:

```text
v4/live/paper_model_registry.py
v4/live/protocol051_surface_edge.py
v4/live/protocol101_entry.py
v4/live/protocol101_live_entry.py
v4/live/protocol066_inference.py
v4/live/ibkr_paper_guard.py
v4/live/ibkr_paper_executor.py
v4/live/paper_trade_log.py
v4/scripts/run_daily_paper_autopilot.py
v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py
v4/scripts/run_protocol157_protocol101_daily_ops_monitor.py
v4/scripts/export_protocol101_trade_charts.py
```

Operations and evidence:

```text
v4/ops/ibkr/
v4/ops/launchd/
v4/logs/paper_trading/
v4/audit/autoresearch/
```

## 5. End-To-End Pipeline

### Step 1: Data Acquisition

Purpose:

Collect option and index data with clear provenance.

Main files:

```text
v4/docs/SPXW_0DTE_DATA_DOWNLOADS.md
v4/ingest/databento_opra.py
v4/scripts/download_databento_*
v4/scripts/download_thetadata_index_bars.py
```

Inputs:

- Vendor option data.
- SPX context.
- VIX context.
- Contract definitions.
- Date windows.
- Approval/provenance metadata.

Gate to proceed:

- Paid-data use is explicitly approved.
- Source, date range, products, and purpose are recorded.
- Raw data is preserved.

### Step 2: Data Validation And Preparation

Purpose:

Convert raw data into normalized and model-ready rows without leakage.

Main files:

```text
v4/docs/DATA_CONTRACT.md
v4/checks/
v4/checks/paid_data_guard.py
v4/dataset/spxw_0dte_neural.py
v4/scripts/build_databento_neural_dataset.py
```

Outputs:

- Normalized market data.
- Processed decision rows.
- Data-quality reports.
- Provenance summaries.

Gate to proceed:

- Timestamps are causal.
- Feature rows are live-reproducible.
- Labels are separated from runtime features.
- Quote, contract, and context checks pass.
- Paid-data guard behavior is verified.

Safe local check:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  v4/tests/test_databento_opra.py \
  v4/tests/test_spxw_0dte_neural.py \
  v4/tests/test_checks.py \
  v4/tests/test_index_bars.py
```

### Step 3: Feature Construction And Model Decisions

Purpose:

Convert prepared rows into model inputs and decisions.

Main files:

```text
v4/model/
v4/live/protocol051_surface_edge.py
v4/live/protocol101_entry.py
v4/live/protocol101_live_entry.py
v4/live/protocol066_inference.py
```

Current decision flow:

```text
surface row
  -> Protocol051/054 surface score
  -> Protocol101 entry candidate frame
  -> Protocol101 enter/wait decision
  -> Protocol081/066 hold/exit decision after entry
```

Gate to proceed:

- Historical and live-style feature names match.
- No future/path/exit labels enter runtime features.
- Entry and lifecycle artifacts load from manifests.
- Inference tests pass.

Safe local check:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  v4/tests/test_protocol101_entry.py \
  v4/tests/test_protocol051_surface_edge.py \
  v4/tests/test_protocol066_inference.py
```

### Step 4: Training And Historical Validation

Purpose:

Train or preserve model candidates and compare them to the current paper default.

Main files:

```text
v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md
v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md
v4/scripts/run_protocol101_event_history_policy.py
v4/scripts/export_protocol101_trade_charts.py
v4/audit/autoresearch/
```

Gate to start training:

- A hypothesis is preregistered.
- Allowed data is listed.
- Training, validation, and protected test splits are explicit.
- Primary metric is chosen before results are known.
- Paper default remains unchanged unless a promotion packet later says otherwise.

Gate to claim improvement:

- Candidate is compared against `PAPER_DEFAULT_PROTOCOL101`.
- Comparison uses strict one-account serial replay.
- Starting equity and position limits match the current paper game.
- Entries use ask accounting.
- Exits use bid accounting.
- No overlapping headline trades.
- No unaffordable headline trades.
- All sessions are flat by close.
- Slippage stress, drawdown, concentration, churn, side/time behavior, and skipped opportunity cost are reported.

Chart export:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.export_protocol101_trade_charts \
  --out-dir <output-dir> \
  --skip-train-validation
```

Expected chart outputs:

```text
equity.html
trades.html
trades.csv
summary.json
report.md
```

### Step 5: Runtime Parity

Purpose:

Prove the model can play the same game live-style that it played historically.

Gate to proceed:

- Same candidate filters.
- Same feature names.
- Same feature calculations.
- Same account-state fields.
- Same action space.
- Same stale-quote rules.
- Same SPXW contract universe.
- Same affordability masks.
- Same output schema.
- Latency is logged.
- Quote freshness is logged.
- Blocked reasons are logged.
- Paper-submit uses the guarded IBKR paper path only after explicit owner authorization.

Local readiness commands:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_daily_paper_autopilot --print-selection
PYTHONPATH=. .venv/bin/python -m pytest \
  v4/tests/test_daily_paper_autopilot.py \
  v4/tests/test_protocol160_persistent_paper_trader.py
```

Current evidence gathering is not a no-order exercise. When the owner has
authorized the Protocol101 paper evidence run, let the current paper default run
through its normal guarded paper-submit path and use logs, monitor output, and
post-session reports to prove what happened.

### Step 6: Paper Runtime

Purpose:

Run the selected paper default through guarded paper execution and record what happens.

Main files:

```text
v4/promotion/PAPER_TRADING_DEFAULT.json
v4/docs/PROTOCOL101_DAILY_PAPER_TRADING.md
v4/scripts/run_daily_paper_autopilot.py
v4/scripts/run_protocol160_protocol101_persistent_paper_trader.py
v4/scripts/run_protocol157_protocol101_daily_ops_monitor.py
v4/live/ibkr_paper_guard.py
v4/live/ibkr_paper_executor.py
v4/live/paper_trade_log.py
v4/logs/paper_trading/
```

Paper guard requirements:

- Paper account only.
- Real-money trading is false.
- Explicit paper-order authorization is present.
- Fresh SPX/VIX context.
- Fresh SPXW quote.
- SPXW 0DTE contract universe.
- Quantity limit enforced.
- Max open-position limit enforced.
- Affordability enforced.
- Entry order is a buy limit.
- Exit order is a sell limit.
- Forced-flat behavior is observable.

Safe selection check:

```bash
PYTHONPATH=. .venv/bin/python -m v4.scripts.run_daily_paper_autopilot \
  --session-date 2026-05-26 \
  --print-selection
```

Safe local runtime tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  v4/tests/test_daily_paper_autopilot.py \
  v4/tests/test_paper_trade_log.py \
  v4/tests/test_protocol142_paper_executor.py \
  v4/tests/test_protocol157_daily_ops_monitor.py \
  v4/tests/test_protocol158_live_entry_paper_bridge.py \
  v4/tests/test_protocol160_persistent_paper_trader.py \
  v4/tests/test_tuesday_paper_fill_fake_e2e.py
```

Paper-submit, broker connectivity, live market-data capture, runtime flag edits, and launch scheduling changes require explicit owner authorization.

### Step 7: Monitoring And Reconstruction

Purpose:

Make the paper session understandable after it runs.

Main files:

```text
v4/scripts/run_protocol157_protocol101_daily_ops_monitor.py
v4/scripts/export_protocol101_trade_charts.py
v4/logs/paper_trading/
v4/audit/autoresearch/
```

Expected evidence:

- Paper JSONL log.
- Paper CSV log.
- Daily monitor report.
- Runtime state.
- Order/fill/cancel rows when paper-submit is authorized and guards pass.
- Equity and trade charts for historical review.

Daily review questions:

- Did the bot start?
- Did it receive fresh data?
- Did it evaluate the correct candidate universe?
- Did guards pass or block for clear reasons?
- Did it place only allowed paper orders?
- Were fills close to historical assumptions?
- Did it exit or flatten correctly?
- Can the session be reconstructed from logs?

### Step 8: Promotion

Purpose:

Decide whether a candidate can replace the current paper default.

Main files:

```text
v4/docs/HYPOTHESIS_TO_PROMOTION_PROCESS.md
v4/docs/MODEL_IMPROVEMENT_GUIDELINES.md
v4/docs/NAMING_GUIDE.md
v4/promotion/
v4/promotion/PAPER_TRADING_DEFAULT.json
```

Required stage sequence:

```text
Observation
  -> Hypothesis
  -> Experiment
  -> Validation
  -> Research Freeze
  -> Runtime Parity
  -> Paper Promotion
  -> Paper Operation
  -> Real-Money Review
```

Gate to change the paper default:

- Validation passed under the correct metric scope.
- Research freeze exists.
- Runtime parity passed.
- Known weaknesses are documented.
- Paper risk controls are defined.
- Rollback criteria are written.
- Owner explicitly approves replacement.
- Registry update is included in a promotion packet.

## 6. Current Known Status

Current local/offline validation confirms:

- Current registry selects Protocol101.
- Protocol101 entry inference loads.
- Surface scorer loads.
- Lifecycle inference loads.
- Paid-data guard tests pass.
- Data/check tests pass.
- Paper-runtime local/fake tests pass.
- Protocol101 chart export produces `equity.html` and `trades.html`.

Current readiness gates indicate:

```text
Section 1/2 foundation: ready
Formal validation controls: ready
Fill model evidence: keep stress replay
Untouched holdout: pending data collection
Live paper-submit evidence: owner-authorized for Protocol101 paper account only
New model hill climbing: blocked by foundation gates
Paper-default replacement: not authorized without promotion packet
```

Interpretation:

The active v4 Protocol101 pipeline is functional locally. The project can continue operating and validating the current paper default, while new training and paper-default replacement remain gated.

## 7. Command Safety

Generally safe:

```text
Read files.
Run registry print-selection.
Run targeted local pytest groups.
Export charts from existing artifacts.
Run readiness scripts that do not contact broker or paid-data endpoints.
```

Requires explicit owner authorization:

```text
Broker connectivity.
Live market-data capture.
Paper-submit sessions unless the current thread explicitly authorizes them.
Paid data downloads.
Model training.
Threshold tuning.
Promotion/default changes.
Runtime flag edits.
Launch scheduling changes.
```

## 8. Evidence Standard

When making a claim, ground it in at least one of:

- Current source file.
- Registry entry.
- Manifest.
- Test result.
- Paper log.
- Monitor report.
- Chart export.
- Governance document.

If evidence is missing, write `UNKNOWN`. If evidence conflicts, record the conflict.

## 9. New Session Checklist

1. Read `AGENTS.md`.
2. Read `docs/CURRENT_TRADING_BOT_SINGLE_SOURCE_OF_TRUTH.md`.
3. Read `v4/promotion/PAPER_TRADING_DEFAULT.json`.
4. Identify which pipeline section the request touches.
5. State the allowed mutation scope.
6. State the verification plan.
7. Work inside the relevant gate.

## 10. One-Paragraph Summary

This is a v4 SPXW 0DTE pipeline whose end goal is an AI options trader that makes money. The current control is `PAPER_DEFAULT_PROTOCOL101`, selected through `v4/promotion/PAPER_TRADING_DEFAULT.json`. The pipeline moves from market data to normalized datasets, model training and validation, surface scoring, Protocol101 entry decisions, lifecycle exits, live paper trading, and reconstruction reports. The active system is locally functional, and future progress is governed by explicit gates for data quality, model validation, runtime parity, paper operation, promotion, and real-money review.
