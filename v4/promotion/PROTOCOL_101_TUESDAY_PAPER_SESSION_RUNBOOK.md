# Protocol 101 Tuesday Paper Session Runbook

Purpose: make the live/paper session auditable. The default morning job starts in `no-order-shadow` mode. IBKR paper orders are allowed only after live parity passes and the paper-order enablement gate says `ready_for_guarded_ibkr_paper_orders`.

## Morning Automation

- `06:28 Pacific`: `com.autoresearch.ibgateway.paper` starts IB Gateway paper mode.
- `06:29 Pacific`: `com.autoresearch.protocol101.paper-preflight` waits for the local API and checks live market data.
- `06:30 Pacific`: `com.autoresearch.protocol101.paper-session` starts Protocol101 analysis/logging.

## First Checks

```bash
launchctl print "gui/$UID/com.autoresearch.ibgateway.paper"
launchctl print "gui/$UID/com.autoresearch.protocol101.paper-preflight"
launchctl print "gui/$UID/com.autoresearch.protocol101.paper-session"
```

Expected: all three LaunchAgents are loaded. During the session, the paper-session agent should either be running or have written a dated report/log.

## Logs To Inspect

- Launchd stdout/stderr: `~/Library/Logs/autoresearch-trading/`
- Session JSONL/CSV: `v4/logs/paper_trading/YYYY-MM-DD/`
- Post-session analyzer: `v4/audit/autoresearch/v4_aplus_hypothesis_148_protocol101_post_session_analyzer/`
- Live visual dashboard: `v4/audit/autoresearch/v4_aplus_hypothesis_149_protocol101_live_log_visual/`

## After The Session Starts

```bash
python -m v4.scripts.run_protocol148_protocol101_post_session_analyzer
python -m v4.scripts.run_protocol149_protocol101_live_log_visual
```

Review the analyzer first. It should answer:

- Did the log validate?
- Did the API port open?
- Did IBKR connect?
- Did live SPX/VIX/SPXW data clear parity?
- Did Protocol101 emit decisions?
- Were paper orders blocked, submitted, or filled?
- What blocked any decision?

## Paper-Order Enablement

Paper trading is not real-money trading, but it still uses a broker endpoint. Only enable it after a clean no-order live-shadow session:

```bash
V4_ALLOW_IBKR_PAPER_ORDERS=YES python -m v4.scripts.run_protocol150_protocol101_paper_order_enablement_gate \
  --account-id DU_REDACTED \
  --enable-paper-orders \
  --acknowledge-paper-loss \
  --write-runtime-flag
```

The gate must pass before switching the morning session from `no-order-shadow` to `paper`. Keep max initial size at one contract until actual paper fills, slippage, cancels, and exits are reviewed.

## Stop Conditions

- Missing or stale SPX/VIX context.
- Missing, stale, crossed, locked, or zero SPXW NBBO.
- AM-settled `SPX` contract appears where `SPXW` is required.
- Session log validation fails.
- Any broker endpoint call appears before the paper-order gate passed.
- Orders or fills appear without a matching JSONL audit row.
