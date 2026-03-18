# IBKR Mock Training Guide (Yesterday Historical)

This workflow builds an isolated dataset from IBKR historical data and optionally runs a short mock train, without touching your primary `training/best_model.pt` or main cache.

## Command

```bash
python3 tools/run_ibkr_mock_training.py \
  --end-date 2026-03-17 \
  --calendar-days 20 \
  --time-budget 90 \
  --ib-port 4002
```

For dataset-only smoke:

```bash
python3 tools/run_ibkr_mock_training.py \
  --end-date 2026-03-17 \
  --calendar-days 20 \
  --skip-train \
  --ib-port 4002
```

## Outputs

- `results/mock-training/run-*/metadata.json`
- `results/mock-training/run-*/logs/01_prewarm.log`
- `results/mock-training/run-*/logs/02_prepare.log`
- `results/mock-training/run-*/logs/03_train.log` (if training enabled)
- isolated model artifact under `results/mock-training/run-*/sandbox/best_model.pt`

## Prerequisites

- IB Gateway/TWS paper session reachable on requested port.
- Historical data requests allowed from your current IP/session.
- Required market-data subscriptions enabled for your target symbols.

## Common Failure: Error 162 (Different IP)

If logs show:

`Historical Market Data Service error: Trading TWS session is connected from a different IP address`

then IBKR is blocking historical pulls for this session. Fix by:

1. Close duplicate TWS/Gateway sessions on other machines/IPs.
2. Re-login paper Gateway/TWS on the machine running the script.
3. Re-run the command.

## Notes

- This is for experimentation/diagnostics, not production promotion.
- Keep the foundation lock (60-feature two-head contract) unchanged.
- Use replay + live dry-run evidence to decide promotion; do not auto-promote mock-train outputs.

