# After-Hours Readiness (Market Closed)

Use this when markets are closed (for example, after 4:00 PM ET) to advance reliability before the next paper session.

## Goal

Validate everything that does not require live intraday ticks:

- Data/feature integrity
- Model/replay behavior
- Entitlement state and blockers
- Evidence ingestion and decision digest generation

## 1) Data + Contract Integrity

```bash
python3 tools/data_quality_report.py
```

Pass criteria:
- Fingerprint stable (or intentionally changed and accepted).
- `60 features`, expected date range, low/zero NaN.

## 2) IBKR Entitlement Probe (Blocker Discovery)

```bash
python3 tools/ib_entitlements.py --port 4002
```

Interpretation:
- `passed=true`: entitlement gate is ready for live paper session.
- `passed=false`: treat as release blocker for live paper execution.

Note: after hours, feeds may be stale, but missing subscription errors are still actionable.

## 3) Replay Battery (No Live Feed Required)

Run replay from cache only to validate decision logic + execution semantics:

```bash
cd training
python3 replay.py --date 2026-03-11 --no-download --model best_model.pt --train-py best_train.py --output ../results/analysis/nightly-replay/replay-2026-03-11.csv
python3 replay.py --date 2026-03-12 --no-download --model best_model.pt --train-py best_train.py --output ../results/analysis/nightly-replay/replay-2026-03-12.csv
python3 replay.py --date 2026-03-13 --no-download --model best_model.pt --train-py best_train.py --output ../results/analysis/nightly-replay/replay-2026-03-13.csv
```

Outputs:
- CSV trade logs in `results/analysis/nightly-replay/`
- JSON journals (`*_journal.json`) with session/trade details
- Canonical replay ledger files (`*_ledger_trades.*`, `*_ledger_bars.*`, `*_ledger_days.*`)
- Replay QA artifacts (`*_qa.json`, `*_qa_anomalies.jsonl`)

Optional release-gate battery:

```bash
python3 tools/replay_battery.py \
  --dates 2026-03-11,2026-03-12,2026-03-13 \
  --no-download \
  --model training/best_model.pt \
  --train-py training/best_train.py \
  --output-root results/analysis/replay-battery/nightly
```

## 4) Evidence Ingestion (Inner + Outer Loop)

```bash
python3 tools/ingest_evidence.py --results-root results --output-root results/analysis
```

Outputs:
- `results/analysis/evidence.parquet`
- `results/analysis/incidents.jsonl`
- `results/analysis/decision-digest-<date>.md`

## 5) Feature Parity Report (After Next Live Run)

Per-feature (all 60) availability report depends on `bar_snapshot` rows from a live paper session:

```bash
python3 tools/live_feature_parity_report.py \
  --audit-path results/live/audit.jsonl \
  --out-json results/live/feature_parity_report.json \
  --out-md results/live/feature_parity_report.md
```

If it errors with "No usable bar_snapshot rows", run one market-hours dry-run first.

## Morning Open Hand-off

Before market open:

1. Context refresh:
```bash
python3 tools/paper_live.py --context-only --port 4002
```
2. Optional dry-run:
```bash
python3 tools/paper_live.py --dry-run --port 4002 --no-context-refresh --max-minutes 5
```
3. Generate feature parity report from audit log:
```bash
python3 tools/live_feature_parity_report.py
```
