# Daily Paper Autopilot Operations

> **SUPERSEDED-BY for Path-D model-building status (2026-08-01):** [`protocol101/PATH_D_CURRENT_STATE.md`](protocol101/PATH_D_CURRENT_STATE.md); this runbook remains the legacy guarded paper-default runtime and does not authorize Path-D training, backtesting, or paper readiness.

Feature name: `daily paper autopilot`

The autopilot runs the current paper-approved default model from
`v4/promotion/PAPER_TRADING_DEFAULT.json`. Today that default is Protocol101.
Future paper-approved models should update that registry rather than requiring
the launchd schedule to be rewritten.

This is the operating plan for collecting live evidence every market day.

## Default Schedule

- `06:28 PT / 09:28 ET`: launch IB Gateway paper mode.
- `06:29 PT / 09:29 ET`: run IBKR API and entitlement preflight.
- `06:30 PT / 09:30 ET`: start the daily paper autopilot session.
- The session loop follows regular market hours and stops after `16:00 ET`.
- The scheduled session runs the selected paper-default model in guarded `paper-submit`
  mode by default. It records candidate sets, model decisions, risk gates,
  account state, submitted paper orders, fills/cancels, and lifecycle exits.
- The paper account is the validation surface for the trained policy: the bot
  should trade live paper exactly the way the frozen historical replay expects,
  then the logs tell us whether timing, fills, and exits are realistic.

## Daily Monitor

Run this after or during the session:

```bash
v4/ops/ibkr/run_protocol101_daily_monitor.sh --session YYYY-MM-DD
```

The monitor writes:

```text
v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/YYYY-MM-DD/<run-id>/daily_monitor.html
v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/latest_daily_monitor.html
```

`latest_daily_monitor.html` is the normal morning page to open. It auto-refreshes
every 30 seconds and only shows the current day/run with a bounded recent-event
timeline, so the dashboard stays light. The append-only JSONL/CSV logs remain
the source of truth for historical review.

It reports startup state, entitlement state, live shadow rows, paper order rows,
paper fills, reconstructed PnL, contracts bought/sold, current position, and
failure/blocker counts.

## Paper Trading Rules

The IBKR account reserve is not trading capital. The operational paper account
starts from `$10,000`.

Paper order submission remains guarded by all of the following:

- Paper account only, normally `DU...`.
- `V4_ALLOW_IBKR_PAPER_ORDERS=YES`.
- Persistent paper-order command flags from the launchd wrapper.
- Fresh SPX/VIX context and fresh SPXW NBBO.
- SPXW PM-settled contracts only.
- Max quantity `1` contract for the current phase.
- Max concurrency `1` open position for the current phase.
- No fixed premium cap and no fixed daily-loss cap in this phase; affordability
  and one-position accounting are still enforced.
- Entries use a buy limit at the current ask and cancel if unfilled after the
  configured timeout, currently `15` seconds.
- Exits use the frozen lifecycle model's exit decision. The runner also checks
  open SPXW paper positions each cycle so it can keep holding or submit a paper
  exit.
- No real-money order endpoint, ever.

The JSONL trade log is the source of truth. The daily monitor is only a viewing
layer over those logs.

## Manual Checks

Live entry bridge only, no order submission:

```bash
.venv/bin/python -m v4.scripts.run_protocol158_protocol101_live_entry_paper_bridge \
  --mode intent-shadow \
  --session-date YYYY-MM-DD \
  --run-id protocol101_intent-shadow_YYYY-MM-DD
```

Guarded dry-run, still no order submission:

```bash
V4_ALLOW_IBKR_PAPER_ORDERS=YES \
.venv/bin/python -m v4.scripts.run_protocol158_protocol101_live_entry_paper_bridge \
  --mode paper-dry-run \
  --enable-paper-orders \
  --acknowledge-paper-loss \
  --session-date YYYY-MM-DD
```

Guarded paper submission for a manual session:

```bash
V4_ALLOW_IBKR_PAPER_ORDERS=YES \
.venv/bin/python -m v4.scripts.run_protocol158_protocol101_live_entry_paper_bridge \
  --mode paper-submit \
  --enable-paper-orders \
  --acknowledge-paper-loss \
  --session-date YYYY-MM-DD
```

Use this manual command only for spot checks. The normal daily workflow is the
scheduled launchd session plus the daily monitor.
