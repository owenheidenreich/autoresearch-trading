# The project is a long-call/long-put SPXW 0DTE model only

**Owner ruling:** “its a long call or long put SPXW model only.”

This removes every proposed escape into a different payoff structure. MES, next-expiry options, debit
verticals, credit spreads, iron flies, short premium and multileg positions are historical analysis only.
They cannot be active strategies or satisfy the goal.

## Exact action space

The model simulates a trader with $10,000 who sees causal SPX context and the current full SPXW 0DTE
ladder. It may:

1. wait while flat;
2. buy to open one same-day SPXW call or put;
3. hold that long contract; or
4. sell to close it.

It may not write an option, trade a second leg, change expiry, use another traded product or route a
position to a different regime merely because the clock crossed 12:46.

Morning/afternoon entry and exit distinctions remain part of the modelling question. They should first be
represented by one shared time-aware model with role heads; four independent networks are warranted only
if out-of-sample evidence demonstrates stable regime specialization and the parameter budget supports it.

## What the project actually lacks

The owned causal full-ladder quote corpus has 251 sessions. The project separately owns **794** non-empty
SPXW 0DTE OHLCV sessions from 2022-06-01 through 2025-07-31. Those older bars contain last trades, not a
causal bid/ask ladder, and cannot be used for selectable contract economics because prior tests proved
that last-trade noise can manufacture false profit.

The clean same-game unlock is therefore the missing 794-session SPXW 0DTE definition plus CBBO-1m
backfill—not another strategy.

Using costs already recorded in 420 historical download sessions:

- mean CBBO-1m cost: **$0.02715/session**;
- mean definition cost: **$0.02900/session**;
- estimated 794-session total: **$44.59**; and
- proposed hard cap after exact preflight: **$75**.

No vendor was contacted and nothing was downloaded.

## Evidence

- owner ruling: `v5/governance/LONG_SPXW_0DTE_SINGLE_LEG_SCOPE_2026_08_14.md`
- active route: `v5/work/entry-exit-attribution/SPXW_0DTE_SINGLE_LEG_ROUTE_V1.md`
- cost receipt: `v5/work/entry-exit-attribution/SPXW_0DTE_CBBO_BACKFILL_ESTIMATE_V1.json`
- estimator: `v5/ops/estimate_spxw_0dte_cbbo_backfill.py`
- test: `v5/tests/test_spxw_0dte_cbbo_backfill.py`
