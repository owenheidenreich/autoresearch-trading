# Live Market-Hours Test Strategy (Paper Account)

Goal: run a controlled, observable real-time test that validates mechanics before full-day paper automation.

## Scope

- Account: paper (`DU*`)
- Mode: start in `--dry-run`, then optional `--paper-auto`
- Universe: current locked foundation (60-feature, two-head, long-only single-contract behavior from current model policy)

## Pre-Open Checklist (T-20 to T-10 minutes)

1. Account snapshot:
```bash
python3 tools/ib_account_snapshot.py --port 4002
```
2. Entitlements:
```bash
python3 tools/ib_entitlements.py --port 4002
```
3. Refresh context bundle:
```bash
python3 tools/paper_live.py --context-only --port 4002 --context-days 30
```

## Phase A: Dry-Run Live Mechanics (10 minutes)

Run a short dry-run inside regular market hours:

```bash
python3 tools/paper_live.py \
  --dry-run \
  --no-context-refresh \
  --port 4002 \
  --model training/best_model.pt \
  --train-py training/best_train.py \
  --start-time-et 09:30 \
  --end-time-et 09:40 \
  --max-minutes 10
```

Pass criteria:
- `entitlement_probe` passes.
- `bar_snapshot` events are present.
- `completeness` mostly >= 0.65 and no persistent extreme staleness.
- No runtime exceptions in session.

## Phase B: Feature Availability Evidence

Generate per-feature live parity from audit stream:

```bash
python3 tools/live_feature_parity_report.py \
  --audit-path results/live/audit.jsonl \
  --out-json results/live/feature_parity_report.json \
  --out-md results/live/feature_parity_report.md
```

Review:
- `core_spx_spy_time` should be near 100%.
- Option/Greeks buckets should be high enough to support model signal quality.
- Any missing/degraded features become explicit blockers or tuning inputs.

## Phase C: Limited Paper Auto (Optional, 15 minutes)

Only run if Phase A/B look healthy:

```bash
python3 tools/paper_live.py \
  --paper-auto \
  --no-context-refresh \
  --port 4002 \
  --model training/best_model.pt \
  --train-py training/best_train.py \
  --start-time-et 09:45 \
  --end-time-et 10:00 \
  --max-minutes 15 \
  --kill-switch results/live/KILL_SWITCH
```

Safety:
- Keep position size at default 1.
- Kill switch file can be toggled to stop trading flow quickly.

## Post-Run Analysis

```bash
python3 tools/ingest_evidence.py --results-root results --output-root results/analysis
```

Review:
- `results/analysis/decision-digest-<date>.md`
- `results/analysis/incidents.jsonl`
- `results/live/audit.jsonl`

## Abort Conditions (Immediate Stop)

- Entitlement probe fails.
- No `bar_snapshot` flow after open.
- Persistent feature completeness collapse.
- Repeated execution/risk-update errors.
- Any behavior inconsistent with model contract or risk controls.

