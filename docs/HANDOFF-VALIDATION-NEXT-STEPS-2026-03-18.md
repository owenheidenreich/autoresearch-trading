# Handoff: Validation Next Steps (Post-Upgrade)

Date: 2026-03-18  
Workspace: `/Users/gduby/Documents/Trinity/Trinity/autoresearch-trading`

## Goal
Validate the new foundation upgrades end-to-end before long training:
- Data/training sidecars and realism gates
- Replay 10x outputs (ledger + QA + battery gate)
- Live paper signal->execution parity evidence

This handoff is the exact execution checklist.

## What Was Upgraded
- Training/data sidecars for deeper OTM ladders and realism/risk labels.
- Loop reliability/promotion gates extended with realism metrics.
- Replay upgraded with:
  - live-like semantics controls,
  - canonical ledgers (`trades`, `bars`, `days`),
  - QA artifacts and anomaly logs,
  - multi-day battery runner.
- Live parity tooling added:
  - `tools/live_order_parity_report.py`.

## Pre-Validation Baseline
1. `60-feature` core contract remains locked.
2. Keep using current `training/best_train.py` + `training/best_model.pt` as baseline.
3. One-time note: the data fingerprint logic changed. First run after this upgrade may require explicit acknowledgement.

## Phase 1: Data + Loop Preflight
Run from repo root:

```bash
python3 tools/data_quality_report.py
set -a && source .env && set +a
cd training
python3 -u run_loop.py --dry-run --allow-data-fingerprint-change
```

Pass criteria:
- `data.pt` loads with `60 features` and acceptable NaN profile.
- Dry-run ends with `ALL CHECKS PASSED`.
- New run folder exists and `results/current_run.txt` points to it.

Fail criteria:
- Feature count mismatch.
- Prompt/contract/fingerprint guard fails and is not intentionally acknowledged.

## Phase 2: Short Training Validation (3 experiments)

### Local quick validation (optional)
```bash
cd training
python3 -u run_loop.py --hours 0.5 --max-experiments 3 --allow-data-fingerprint-change
```

### Akash/H100 validation (preferred)
From repo root:
```bash
./infra/deploy.sh boot
./infra/deploy.sh start --hours 0.5 --max-experiments 3 --allow-data-fingerprint-change
./infra/deploy.sh status
```

Pass criteria:
- Experiments are written to `results/run-*/experiments.v2.jsonl`.
- No contract drift/safety crash loops.
- `status.json` includes reliability fields and updates each experiment.

Recommended post-check:
```bash
python3 tools/ingest_evidence.py --results-root results --output-root results/analysis
```

## Phase 3: Replay Battery Validation (Release Gate)
From repo root:

```bash
python3 tools/replay_battery.py \
  --dates 2026-03-11,2026-03-12,2026-03-13 \
  --no-download \
  --model training/best_model.pt \
  --train-py training/best_train.py \
  --min-trade-prob 0.55 \
  --risk-mode live_like \
  --output-root results/analysis/replay-battery/validation
```

Pass criteria:
- Exit code `0`.
- `release_gate_passed: true` in `battery_summary.json`.
- Per-day artifacts exist:
  - `*_ledger_trades.csv/.parquet`
  - `*_ledger_bars.csv/.parquet`
  - `*_ledger_days.csv/.parquet`
  - `*_qa.json`
  - `*_qa_anomalies.jsonl`
- No critical QA anomalies.

Fail criteria:
- Exit code `2`.
- Any date fails replay QA or run execution.

## Phase 4: Live Paper Validation (Market Hours)

### 4.1 Entitlements
```bash
python3 tools/ib_entitlements.py --port 4002
```
Must return `"passed": true`.

### 4.2 Short controlled session
```bash
python3 tools/paper_live.py \
  --paper-auto \
  --port 4002 \
  --model training/best_model.pt \
  --train-py training/best_train.py \
  --max-minutes 10
```

### 4.3 Parity report
```bash
python3 tools/live_order_parity_report.py \
  --audit-path results/live/audit.jsonl \
  --out-json results/live/order_parity_report.json \
  --out-md results/live/order_parity_report.md
```

Pass criteria:
- Audit contains entry/risk/exit and broker lifecycle events (`ib_order_status`, `ib_exec_details` when orders occur).
- `order_parity_report.json` shows no mismatches for generated intents.

Fail criteria:
- Missing lifecycle linkage for generated intents.
- IB error events tied to placed intents.

## Final Go/No-Go Decision
Go to longer training + paper iteration cycle only if all are true:
1. Phase 1 preflight passes.
2. Phase 2 short loop runs cleanly (no systemic contract/safety failures).
3. Phase 3 replay battery passes release gate.
4. Phase 4 entitlement + parity checks pass for a live test window.

If any phase fails:
- Stop and fix the failing subsystem first.
- Re-run only that phase and downstream phases (do not skip forward).

## Artifacts to Attach in Next Handoff
- `results/current_run.txt`
- `results/run-*/experiments.v2.jsonl`
- `results/analysis/replay-battery/validation/battery_summary.json`
- `results/analysis/replay-battery/validation/battery_summary.md`
- `results/live/order_parity_report.json`
- `results/analysis/decision-digest-<date>.md`
