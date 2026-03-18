# IBKR Live Paper Trading Checklist (No Delayed Data)

Use this checklist before running `tools/paper_live.py`.

Detailed subscription walkthrough:
- `docs/IBKR-SUBSCRIPTION-SETUP-GUIDE.md`

Live session test flow:
- `docs/LIVE-MARKET-HOURS-TEST-STRATEGY.md`

Isolated historical mock training:
- `docs/IBKR-MOCK-TRAINING-GUIDE.md`

## Foundation Alignment (Current)

- This checklist covers the live-paper subsystem only.
- Core training foundation is locked to 60-feature two-head contract.
- Canonical runtime outputs are under `results/` with active pointer `results/current_run.txt`.
- Live subsystem docs/plans do not override `training/program.md` contract rules.

## 1) TWS / Gateway API Settings

- Enable API connections.
- Allow local socket clients.
- Use paper port (`4002` for Gateway, `7497` for TWS paper).
- Keep read-only API **disabled** for paper automation.

## 2) Account / Session Guardrails

- Run paper sessions on a `DU*` account only.
- Do not run conflicting live + paper sessions that consume the same data entitlements in incompatible ways.
- Keep one primary API client for the paper bot.

## 3) Required Market Data Entitlements

The service probes these before trading and fails closed if unavailable:

- `SPXW` options top-of-book (bid/ask/last)
- SPXW option computation (Greeks/IV path)
- `SPX` index feed
- `VIX` index feed
- `SPY` real-time feed (volume proxy)

## 4) Delayed/Missing Data Protection

- Bot requests `reqMarketDataType(1)` and validates non-delayed state.
- Delayed tick signatures and stale feeds are logged.
- If required instruments do not have live entitlements, order placement is blocked.
- High active ticker counts are flagged to avoid market-data-line saturation.

## 5) Required Runtime Controls

- `kill switch` file path configured in production runs.
- `paper-auto` enabled only after dry-run validation.
- Audit log path writable (`results/live/audit.jsonl`).

## 6) Daily Runbook

1. Pre-open (~9:10 ET): `python3 tools/paper_live.py --context-only --port 4002`
2. Dry-run sanity check (optional): `python3 tools/paper_live.py --dry-run --port 4002`
3. Paper auto session: `python3 tools/paper_live.py --paper-auto --port 4002`
4. Verify audit stream and entitlement probe events in `results/live/audit.jsonl`
