# Pre-fit design — compact shared SPXW 0DTE lifecycle

**Status: DESIGN ONLY. NO FIT, ECONOMICS OR GATE REOPENING.**

## Question

Can one shared, time-aware representation express the owner's complete long-call/long-put game while
remaining inside the worst-case conservative parameter budget projected for the same-day CBBO backfill?

## Frozen action and observation law

- Traded position: one long SPXW 0DTE call or put only.
- Context: completed SPX candles and the complete contemporaneous SPXW 0DTE two-sided ladder.
- Flat actions: `WAIT` or buy one eligible contract.
- Holding actions: `HOLD` or sell the same contract.
- The opening regime owns the exit; the position vector carries origin regime across 12:46.
- 10:00, 13:30 and morning/afternoon location are inputs, never forced trades.

## Architecture family: exactly one member

The member uses:

- eight last-completed-candle geometry fields;
- all five causal clock fields;
- mean and maximum summaries of five fields over the **whole visible ladder**;
- two account fields;
- one shared three-value state representation;
- a common per-contract base plus state-by-side and state-by-moneyness interactions;
- a trained `WAIT` value; and
- one shared `HOLD`/`SELL` head using eight causal held-position fields, including origin regime.

There are no morning/afternoon specialist networks. Time and origin are observed variables inside one
shared model. Separate specialists remain unsupported unless this shared form first shows stable
chronological regime specialization.

## Acceptance and failure law

- The parameter count must come from the built module, never this document.
- It must be at most **122**, the worst-case conservative budget in the observed-completeness backfill
  scenario. A larger count fails the design before any fit.
- Masked future candles and invisible ladder nodes may not change any score.
- Whole-ladder context outside the entry-action mask must remain observable.
- Chart state must be capable of reversing call/put ordering.
- Exit scores must depend on held-position state and preserve opening-regime ownership.
- Building this module grants no fit authorization. Training remains refused until a future declaration,
  acquired evidence and applicable owner gate all agree.

No economic outcome is inspected by this design test.
