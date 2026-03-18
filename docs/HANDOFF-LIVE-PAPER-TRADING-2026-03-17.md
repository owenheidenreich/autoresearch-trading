# Handoff: Live IBKR Paper Trader Build + Validation

Date: 2026-03-17  
Workspace: `/Users/gduby/Documents/Trinity/Trinity/autoresearch-trading`

## 0) Foundation Cutover Alignment (2026-03-18)

This handoff remains useful for live-paper subsystem implementation details, but
the project foundation is now hard-cut to these canonical rules:

- Canonical training run root is `results/` (not `training/results/`).
- Active run pointer is `results/current_run.txt`.
- Canonical experiment log is run-folder `experiments.v2.jsonl` only.
- Training foundation phase is locked to 60-feature, two-head contract.
- `training/program.md` is strict contract-only.
- Exploratory ideas moved to `docs/IDEAS-BACKLOG.md` (not injected into loop prompt).

## 1) What Was Requested

Implement the new real-time IBKR paper-trading system with:

- Two-stage flow: historical pre-open bootstrap + live streaming session
- Model-driven trade intents (entry/size/stop/target/updates/exit)
- Mandatory OCO/bracket exits on every entry
- Hard monotonic ratchet rules:
  - Stop can only move up
  - Take-profit can only move up
- Kill switch and fail-closed behavior
- Entitlement checks for non-delayed required data
- Tests and runnable tools

## 2) What Was Implemented

### New live subsystem (`training/live/`)

- `contracts.py`
  - Added `LiveContextBundle`, `DecisionIntent`, `RiskUpdateIntent`, `ExecutionState`
  - Added `FeatureContractVersion` + `FEATURE_CONTRACT_VERSION = "ibkr_live_v1"`
  - Added bundle save/load helpers

- `context.py`
  - Added `refresh_context_bundle(...)`
  - Uses Polygon historical context (SPY + options history) and IBKR fill-ins (SPX/VIX)
  - Builds feature arrays + normalization context buffers and persists dated bundle
  - Added `load_latest_context_bundle(...)`

- `entitlements.py`
  - Added entitlement probe for `SPY`, `SPX`, `VIX`, `SPXW_ATM_CALL`
  - Validates live data mode + required fields
  - Returns fail/pass report consumed by service

- `features.py`
  - Added 5-second -> 1-minute aggregation
  - Appends live-only minute bars to context and recomputes normalized snapshots
  - Tracks completeness + staleness diagnostics

- `resolver.py`
  - Added all 6 action mappings to SPXW contracts (ATM/OTM ±5/±10)
  - Added option mid-price quoting helper

- `decision.py`
  - Loads model and emits entry intents + risk update intents
  - Generates stop/target and qty from model confidence + feature context
  - Includes ratchet update logic (up-only intent generation)
  - Validates model feature contract version

- `execution.py`
  - Added OCO/bracket entry path (parent + stop + target)
  - Enforces monotonic ratchet rules strictly
  - Supports dry-run/live, flatten-on-exit, kill switch, audit log

- `service.py`
  - End-to-end orchestration:
    - context load/refresh
    - entitlement gate
    - stream loop (5-sec ingest, minute decisions)
    - entry/update/exit calls to execution layer
    - EOD flatten
  - Paper account guard (`DU*`)
  - Fails closed when entitlements missing

### New tools

- `tools/paper_live.py`
  - Main CLI for context refresh and live session
  - Supports `--context-only`, `--dry-run`, `--paper-auto`, kill switch, time windows

- `tools/ib_entitlements.py`
  - Standalone entitlement probe CLI

### Tests added

- `tests/test_live_resolver.py`
- `tests/test_live_execution.py`
- `tests/test_live_contracts.py`

All pass.

### Documentation updates

- `README.md`
  - Added live paper-trader section + run commands
- `docs/IBKR-LIVE-CHECKLIST.md`
  - Added live-data prerequisites/runbook
- `docs/IDEAS-BACKLOG.md`
  - Human planning backlog for deferred feature ideas (not loop prompt input)

### Training checkpoint contract handling

- `training/live/decision.py`
  - Reads `feature_contract_version` from checkpoint config when present.
  - Defaults to `ibkr_live_v1` if field is missing and enforces strict match.

## 3) Validation Run History

## Unit / static checks

- `python3 -m pytest -q` -> `4 passed`
- `python3 -m py_compile ...` on new modules/tools -> passed
- `python3 tools/paper_live.py --help` -> passed

## IBKR connectivity checks

- `python3 tools/ib_probe.py --port 4002` -> passed connectivity:
  - ES bars: OK
  - SPX bars: OK
  - SPXW chain + today expiry: OK

## Entitlement checks (critical)

- `python3 tools/ib_entitlements.py --port 4002` -> **failed (as designed fail-closed)**
  - Account detected: `DUP440540` (paper account)
  - Missing live subscriptions / API entitlement errors:
    - `10089` (SPY ARCA top data requires additional subscription)
    - `354` (SPX, VIX, SPXW not subscribed)
    - `10090` (partial option data, subscription-independent ticks only)
  - Result: `passed: false`

## Context refresh run

First run failed due a bug in weekly resampling (see section 4).  
After patch:

- `python3 tools/paper_live.py --context-only --port 4002 --context-days 30` -> succeeded
- Bundle written:
  - `/Users/gduby/.cache/autoresearch-trading/live_context/context-2026-03-17.pt`

## Live dry-run smoke

- `python3 tools/paper_live.py --dry-run --no-context-refresh --port 4002 --model training/best_model.pt --start-time-et 00:00 --end-time-et 23:59 --max-minutes 2`
- Aborted at entitlement gate (expected with current subscriptions):
  - `RuntimeError: Entitlement probe failed; live session aborted`

No orders were placed.

## 4) Bugs Found and Fixed During Validation

1. `context_refresh` crash in `_compute_mtf_stats`:
- Error: `TypeError: Only valid with DatetimeIndex...`
- Fix: convert grouped daily index to datetime before weekly resample.
- File: `training/live/context.py`

2. CLI import path issues for new scripts:
- `ModuleNotFoundError` for package imports
- Fix: prepend repo root to `sys.path` in tool scripts.
- Files:
  - `tools/paper_live.py`
  - `tools/ib_entitlements.py`

3. Feature contract mismatch protection:
- Added enforcement that model checkpoint contract version matches live engine contract.
- Files:
  - `training/live/decision.py`
  - `training/live/service.py`

## 5) Current Blocker

Live paper session cannot proceed because required live market-data entitlements are missing in IBKR for API use.

Current fail conditions in probe:
- `SPY`: no live bid/ask/last
- `SPX`: no live bid/ask/last
- `VIX`: no live bid/ask/last
- `SPXW_ATM_CALL`: no live top-of-book + no Greeks

## 6) Exact Resume Checklist

1. In IBKR, enable required live subscriptions for API use:
- SPY real-time top-of-book
- SPX index real-time
- VIX index real-time
- SPXW options top-of-book and option-computation/Greeks path

2. Re-run entitlement probe:

```bash
python3 tools/ib_entitlements.py --port 4002
```

Expected: `"passed": true` and non-stale fields populated.

3. Refresh context bundle (pre-open):

```bash
set -a && source .env && set +a
python3 tools/paper_live.py --context-only --port 4002 --context-days 30
```

4. Live dry-run smoke:

```bash
python3 tools/paper_live.py --dry-run --no-context-refresh --port 4002 --model training/best_model.pt --max-minutes 5
```

5. Paper auto (only after dry-run looks good):

```bash
python3 tools/paper_live.py --paper-auto --no-context-refresh --port 4002 --model training/best_model.pt
```

## 7) Operational Artifacts

- Context bundle:
  - `/Users/gduby/.cache/autoresearch-trading/live_context/context-2026-03-17.pt`
- Audit log:
  - `/Users/gduby/Documents/Trinity/Trinity/autoresearch-trading/results/live/audit.jsonl`

Latest audit entries currently show:
- successful `context_refresh`
- failed `entitlement_probe`

## 8) Repo State Notes

The working tree may include unrelated in-progress foundation changes in addition to live-paper modules.
Do not assume this handoff reflects the entire current diff; verify against current `README.md` and `training/program.md`.

## 9) Files Added/Changed For This Work

Added:
- `training/__init__.py`
- `training/live/__init__.py`
- `training/live/contracts.py`
- `training/live/context.py`
- `training/live/entitlements.py`
- `training/live/resolver.py`
- `training/live/decision.py`
- `training/live/features.py`
- `training/live/execution.py`
- `training/live/service.py`
- `tools/paper_live.py`
- `tools/ib_entitlements.py`
- `tests/test_live_contracts.py`
- `tests/test_live_execution.py`
- `tests/test_live_resolver.py`
- `docs/IBKR-LIVE-CHECKLIST.md`

Modified:
- `README.md`
- `training/program.md`
- `training/train.py`
