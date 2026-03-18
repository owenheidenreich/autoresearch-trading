# autoresearch-trading

## Overview
This project builds an autoresearch-driven SPX options trader using a strict foundation-first loop:

`data -> training -> replay -> paper -> real`

The current objective is reliability and mechanical consistency, not feature expansion.

## Foundation Scope (Locked)
- Feature contract: **60 features**.
- Model contract: **two-head** output.
  - Gate: `[NO_TRADE, TRADE]`
  - Direction: `[CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]`
- Action semantics: **8 effective actions** (`DO_NOTHING`, 6 entries, `EXIT`).
- Keep policy: score improvement + reliability gates.
- 64+Charm migration is explicitly deferred to a separate coordinated phase.

## Canonical Runtime Layout
All active loop artifacts are run-folder scoped under `results/`.

- Active run pointer:
  - `results/current_run.txt`
- Run folder naming:
  - `results/run-YYYY-MM-DD-HHMMSS/`
- Required run artifacts:
  - `experiments.v2.jsonl`
  - `status.json`
  - `run_metadata.json`
  - `data_quality_report.json`
  - `artifacts/exp-<id>/...`

No root-level `experiments.jsonl` or `status.json` files are used.

## Promotion Ledger
Promotions are explicit and append-only.

- `results/promoted/current.txt` -> active promoted run name
- `results/promoted/history.jsonl` -> promoted event stream

`run_loop.py` uses **promoted history only** for LLM experiment context.

## Project Map

```text
training/
  prepare.py
  train.py
  best_train.py
  best_model.pt
  run_loop.py
  replay.py
  program.md
  live/

tools/
  monitor.py
  ingest_evidence.py
  data_quality_report.py

infra/
  deploy.sh

docs/
  IDEAS-BACKLOG.md
  0dte-domain-knowledge.md
  HANDOFF-LIVE-PAPER-TRADING-2026-03-17.md
```

## Prompt Control Plane
- Strict autonomous contract source:
  - `training/program.md`
- Human-only exploratory roadmap:
  - `docs/IDEAS-BACKLOG.md`

Only `training/program.md` is injected into the autonomous loop prompt.

## Minimal Runbook

### Local preflight
```bash
set -a && source .env && set +a
cd training
python3 -u run_loop.py --dry-run
```

### One short local smoke run
```bash
python3 -u run_loop.py --hours 0.1 --max-experiments 1 --time-budget 20
```

### Inspect active run artifacts
```bash
RUN_ID=$(cat ../results/current_run.txt)
ls -la ../results/$RUN_ID
ls -la ../results/$RUN_ID/artifacts
```

### Local monitor
```bash
python3 ../tools/monitor.py --local ../results
```

### Replay check
```bash
python3 replay.py --date 2026-03-17 --output replay-trades.csv
```

## Deployment / Sync Behavior
`infra/deploy.sh` status/sync/download resolve the active run from `/root/results/current_run.txt` and mirror it to local `results/current_run.txt`.

The script fails fast when the active run pointer is missing or invalid.

## Safety Notes
- Real-money trading is not enabled in this phase.
- Live paper-trading remains under construction and validation.
- Reliability evidence is mandatory before broadening strategy scope.
