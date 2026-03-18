# IBKR Subscription Setup Guide (for SPX/SPXW Paper Bot)

This guide maps the current entitlement failures to the subscriptions and settings needed for your live paper-trading stack.

## Latest Verification (2026-03-18 00:42 PT)

From:

```bash
python3 tools/ib_probe.py --port 4002
python3 tools/ib_entitlements.py --port 4002
```

Observed now:

- API socket connectivity is healthy (manual 60-second hold test stayed connected).
- `SPX` historical bars are available.
- `ES` request returned `Error 162` with `HMDS query returned no data` (this is a data/contract coverage issue, not an IP-conflict message).
- Entitlements still failing with:
  - `Error 354` on `SPX`, `VIX`, `SPXW` (`market data not subscribed`)
  - `Error 10089` on `SPY` (`additional API subscription required`)
  - `Error 10090` on `SPXW` (`partial option data not subscribed`)

Important:
- Gateway logs showing `Socket connection ... has closed. Reason: Connection terminated` for clients `99` and `91` are expected if short-lived probe scripts connect then exit.
- This specific log pattern alone does not mean IB Gateway crashed.

## Current Failure Snapshot

From `python3 tools/ib_entitlements.py --port 4002` on **2026-03-17 23:48 PT** (UTC `2026-03-18T06:48:43`):

- `Error 354`: market data not subscribed (`SPX`, `VIX`, `SPXW` option)
- `Error 10089`: additional API subscription required (`SPY ARCA/TOP/ALL`)
- `Error 10090`: partial option data not subscribed (`SPXW` option stream)
- Probe result: `passed=false` and all required symbols stale/no bid-ask/no last

## What Your Bot Specifically Needs

Your live service requests:

- `SPY` top-of-book (volume proxy)
- `SPX` index top-of-book
- `VIX` index top-of-book
- `SPXW` option top-of-book + Greeks/IV

Based on IBKR API documentation, this maps to:

1. `NYSE American, BATS, ARCA, IEX, and Regional Exchanges (Network B)` for `SPY` (ARCA-listed symbols like SPY).
2. `CBOE Streaming Market Indexes` for `SPX` (and typically `VIX`, both Cboe indices).
3. `OPRA (US Options Exchanges)` for `SPXW` options L1.
4. API enablement: `Market Data API Acknowledgement` must be enabled and signed.

Important IBKR note:
- Options Greeks require subscriptions for both derivative and underlying.
- Options/index subscriptions do not automatically include underlying index data.

## Website Navigation (Client Portal)

1. Go to Client Portal login: `https://ndcdyn.interactivebrokers.com/sso/Login`
2. Top-right profile menu (`Welcome {Your Name}`) -> `Settings`
3. Open `Market Data Subscriptions`
4. Click the configure cog and add/confirm:
   - `Network B` (or your preferred US equity bundle that includes ARCA data)
   - `CBOE Streaming Market Indexes`
   - `OPRA (US Options Exchanges)`
5. Continue through agreements/disclosures until confirmation.
6. On the same Market Data settings page, find `Market Data API Acknowledgement`:
   - Click cog -> select `Yes` to enable API market data
   - Review/sign the form
7. Log out of TWS/IB Gateway/Client Portal, then log back in.

## If You Are Unsure Which Package Covers a Symbol

Use IBKR `Market Data Assistant` to resolve exact package by symbol/exchange:

1. In Client Portal, open `Help` -> `Support Center`
2. Open `Market Data Assistant`
3. Search symbols:
   - `SPY` on `ARCA`
   - `SPX` on `CBOE`
   - `VIX` on `CBOE`
   - `SPXW` option on `SMART` / trading class `SPXW`

## Post-Subscription Verification

Run in this order:

```bash
python3 tools/ib_entitlements.py --port 4002
python3 tools/paper_live.py --dry-run --port 4002 --no-context-refresh --max-minutes 5
python3 tools/live_feature_parity_report.py
```

Expected:
- Entitlement probe returns `passed=true`
- Dry-run emits `bar_snapshot` events (not only entitlement failure)
- Feature parity report generates successfully from `results/live/audit.jsonl`

## Troubleshooting

- If market data still fails right after subscribing, wait and retry (exchange entitlement propagation can lag).
- Ensure account meets IBKR minimum equity + subscription costs.
- Confirm you are using the intended paper/live linked user pair for market data sharing.
- If unclear, open an IBKR ticket under `Account Services -> Market Data Subscription`.

## IP-Conflict Recovery (Only if you see "different IP address")

If you specifically see:
- `Error 162 ... Trading TWS session is connected from a different IP address`

then do this:

1. Fully log out of IBKR on all devices (TWS, Gateway, mobile, other servers).
2. Wait 60-90 seconds for session cleanup.
3. Start only one IB Gateway session on the machine you will run the bot from.
4. Keep API port fixed (`4002`) and use unique API `clientId` per script.
5. Re-run:

```bash
python3 tools/ib_probe.py --port 4002
python3 tools/ib_entitlements.py --port 4002
```
