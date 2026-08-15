# Active route — long single-leg SPXW 0DTE model

**Status: RESEARCH LAW FROZEN; DATA PREFLIGHT/DOWNLOAD REQUIRES OWNER AUTHORIZATION.**

## The trader being simulated

The simulator starts each session with $10,000 and the information visible at that moment:

- the completed SPX candle history available causally;
- the entire currently quoted SPXW 0DTE call/put ladder, including bid, ask, size and causal Greeks; and
- its cash, clock, current position and time remaining.

While flat it may `WAIT` or buy to open one SPXW 0DTE call or put. While holding it may `HOLD` or sell to
close that exact contract. It cannot write options, add legs, change expiration, hold overnight or trade
another instrument.

## Model law

- Treat each trading day as one sequential episode; never mix future minutes into the current state.
- Learn `BUY THIS CONTRACT NOW` relative to `WAIT`, not raw future ITM depth alone.
- Train the exit only on entries actually produced out of fold by the matching entry policy.
- Explicitly encode morning `09:30–12:45` and afternoon `12:46–16:00` time regimes. A position remains
  owned by its opening regime until close.
- Treat 10:00 “Magic Time,” 13:30 and other clock locations as causal features and diagnostics, never
  forced trades.
- Compare one shared time-aware model with entry/exit role heads before considering independent models.
  The data, not the prompt, decides whether four heads materially help.
- Keep one open position and the declared small daily trade cap; entry at ask, exit at bid, measured fees.
- Price midpoint gross first. A non-positive result stops before executable economics or operating-point
  search.
- Any surviving candidate must beat outcome-blind composition-matched and shuffled-label controls, have
  corrected lower bounds above zero, be positive in at least four of five chronological folds, and obey
  the $10,000 risk limit.

## Same-game data unlock

The project owns 794 non-empty SPXW 0DTE OHLCV sessions from 2022-06-01 through 2025-07-31 in addition to
the 251-session quote corpus, but the older sessions lack causal full-ladder CBBO. Last-trade bars cannot
substitute because prior work proved that their pricing noise creates false edge.

Acquire for those **794 sessions only**:

- OPRA `definition` for exact same-day SPXW symbol identity; and
- OPRA `cbbo-1m` for only the contracts expiring that session.

No longer expiration, spread, short-premium data, ES, MES, futures, ETF, broker, paper or live action is
included. Write into a new manifest-backed external directory and never overwrite the 251-session corpus.

Recorded historical costs estimate:

- CBBO-1m: **$21.56**;
- definitions: **$23.03**;
- combined: **$44.59**; and
- hard cap: **$75**.

An exact read-only vendor preflight must run first. Stop before purchase if its total exceeds $75.

## Sequence after acquisition

1. Verify hashes, same-day expiry, session coverage and SPXW/SPX-only provenance.
2. Rebuild the full-ladder causal episode corpus without opening the protected post-cutoff sessions.
3. Re-measure effective sample size; the pre-acquisition projection is 122–216 conservative parameters.
4. Admit only the built 120-parameter `compact_shared_lifecycle` member if it still fits the post-QC
   conservative budget. The architecture and complete research law are frozen in
   `COMPACT_SHARED_LIFECYCLE_DESIGN_V1.md` and `COMPACT_SHARED_LIFECYCLE_PROTOCOL_V1.md`.
5. Fit entry chronologically, generate nested out-of-fold entry trajectories, freeze entry weights, then
   train the shared HOLD/SELL head only on those trajectories.
6. Fit the identical shuffled-label path and evaluate behind the midpoint firewall with the declared
   composition-matched control.
7. Close the branch on any kill failure. Do not rescue it with another threshold, seed, horizon, side,
   time window or architecture.

## Decision requested

Authorize a read-only Databento cost preflight and, only if the exact total is at most **$75**, download of
the 794-session same-day SPXW definition and CBBO-1m backfill above. This authorizes no order, paper trade,
promotion or real-money action.
