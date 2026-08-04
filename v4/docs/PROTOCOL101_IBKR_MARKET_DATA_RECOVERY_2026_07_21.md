# Protocol101 IBKR Market-Data Recovery - 2026-07-21

## Incident

The recorder launchd jobs started on schedule, but the recorder did not produce
healthy ladder checkpoints during the opening portion of the session.

Two IBKR-side errors were observed:

- `10141`: the paper-trading API disclaimer had not yet been accepted.
- `10197`: SPX/VIX market data was denied while a competing live-data session
  held the shared entitlement.

The watchdog then restarted the recorder every 35 seconds. Each recorder and
Gateway-readiness connection used `ib_insync.IB.connect`, which synchronizes
account, position, order, and execution state even though the recorder only
needs market data. Rapid reconnects exhausted IBKR's account-summary request
limit and made recovery noisier.

## Repair

- Recorder and Gateway readiness now connect through the market-data-only
  client handshake and do not request account/order synchronization.
- Watchdog recovery terminates the launchd recorder job before cleaning up
  orphan children.
- Watchdog restart cooldown begins at 120 seconds and backs off to 300 seconds.
- The deployment source now preserves the installed startup schedule and retry
  settings instead of reverting to the older schedule on redeploy.
- A new immutable bundle was installed:
  `~/.autoresearch-trading/runtime-bundles/protocol101-parity-v1/b7ad868edf1a2f18`.

## Verified State

At approximately 08:02 PT, a fresh preflight passed with live SPX/VIX and 42
qualified SPXW contracts. The restarted recorder then passed all health checks
with 44 subscribed contracts, fresh heartbeats, fresh checkpoint progress,
zero subscription/write errors, and no broker order endpoint call.

The 2026-07-21 capture remains partial because healthy recording resumed after
the market opened. Preserve it for diagnostics, but do not treat it as a
complete sealed confirmation session.

## Operator Rule

Before relying on a recorder day, close other IBKR sessions that consume the
same live market-data entitlement and accept any paper/API disclaimer shown by
IB Gateway. Error `10197` cannot be repaired by local restart logic alone.
