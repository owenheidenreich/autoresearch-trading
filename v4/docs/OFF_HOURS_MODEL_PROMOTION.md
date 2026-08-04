# Off-Hours Model Promotion

This project can make most of a model-promotion decision while markets are
closed, but it must separate three ideas:

- Historical edge: does the candidate beat the current paper default under the
  same one-account serial replay rules?
- Runtime parity: can the same code path build the candidate set, features,
  risk masks, and order-intent records without future information?
- Live feed parity: during market hours, do IBKR quotes and timestamps match the
  historical assumptions closely enough to trust paper execution?

Only the first two can be fully decided after hours. Live feed parity still needs
market hours.

## Off-Hours Promotion Gate

An after-hours replacement packet may mark a challenger as
`offline_replacement_ready`, but not `paper_default_enabled`, when all of these
are true:

1. The candidate beats `PAPER_DEFAULT_PROTOCOL101` under strict serial replay:
   one account, one open position max, one contract for the current paper phase,
   executable ask-entry/bid-exit pricing, affordability checks, and flat by close.
2. The candidate survives protected historical blocks without tuning directly to
   them: Q3 2025, Q4 2025, Q1 2026, March 2026, recent 2026, and any available
   external block.
3. The no-order runtime harness reproduces the same selected trades on a
   historical replay window.
4. The live-style feature builder reproduces the historical feature columns one
   event at a time.
5. The candidate's premium, moneyness, side, and time-of-day behavior is known
   and documented.
6. Paid-data and broker guardrails pass.
7. No paper/live default is changed without an explicit decision packet.

## Market-Hours Activation Gate

During live market hours, before enabling a new paper default:

1. Run the challenger no-order beside the existing paper default.
2. Confirm SPX, VIX, and SPXW quote freshness.
3. Confirm the live candidate surface matches the historical contract contract:
   SPXW PM-settled only, $5 strikes, ATM +/- $50, valid bid/ask, repaired Greeks,
   affordability, and one-position limits.
4. Compare live moneyness and premium distribution against the off-hours audit.
5. Validate JSONL records for candidate set, model decision, risk gate,
   intended order, account state, and blocked reason.
6. Only then create a separate `paper_default_enabled` decision.

## Current Meaning

`CHALLENGER_FULL_ACTION_SURFACE_EDGE_HISTORY_V1` has historical edge and several
offline runtime checks, but it is not automatically the paper default. It can be
moved to `offline_replacement_ready` after a replacement packet verifies the
off-hours gate. It still needs market-hours no-order parity before paper orders
should use it.
