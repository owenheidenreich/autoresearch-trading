# Rejected route — next-expiry SPXW debit vertical is outside owner scope

**Status: REJECTED BY OWNER; DO NOT PREFLIGHT, DOWNLOAD OR EXECUTE.**

On 2026-08-14 Owen Heidenreich clarified: **“its a long call or long put SPXW model only.”** The active
game is a single long SPXW 0DTE call or put. This packet is preserved only to show what was considered;
its decision request is withdrawn. No vendor was contacted and no data was downloaded.

## Scope

- Traded instrument: SPXW options only.
- Context: SPXW chain state or SPX index only. No ES, MES, futures or ETF feature.
- Expiration: the nearest listed SPXW expiration strictly after the decision session.
- Position: long one 10-point debit vertical in the selected direction.
- Hold: at most 120 minutes; no overnight position.
- Account: $10,000; entry debit plus two measured round-trip fees must be at most $500.

## Acquisition boundary

- Sessions: 2022-06-01 through 2026-07-31, expected 1,045 trading sessions.
- Schemas: OPRA `definition` and `cbbo-1m` only.
- Symbol law: resolve the first expiry after each session from that session's definition data, then request
  only those exact SPXW contract symbols.
- Destination: a new manifest-backed directory under `/Volumes/AR_TRADING_DATA`; never overwrite the
  owned 0DTE corpus.
- Exact preflight first; stop before download if the total exceeds **$100**.
- No post-2026-08-05 session, paper/live order, broker contact, promotion or runtime change.

The owned-data extrapolation is **$59.81** total ($29.50 CBBO plus $30.31 definitions). It is not a fresh
vendor quote.

## Research sequence after acquisition

1. Verify hashes, expiry identity, quote coverage and the SPXW/SPX-only feature boundary.
2. Before P&L, require exact 10-point entry pairs within the $500 cap across at least four chronological
   folds. A failed coverage check ends the route without economics.
3. Measure effective sample size. Do not assume 1,045 sessions admit any particular model.
4. Predeclare a single compact enter-versus-wait architecture at approximately 29–50 built parameters,
   one 120-minute horizon, one 10-point structure and the complete carried multiplicity family.
5. Price midpoint gross first. A value at or below zero stops before touch economics.
6. If positive, require ask/bid combo execution plus two measured fees, outcome-blind composition-matched
   and independently shuffled controls, corrected lower bounds above zero, four of five positive folds,
   and $500 risk compliance.

No width, horizon, side, seed, threshold or time-window retry follows a negative result.

## Withdrawn decision request

Do not authorize or execute this packet. Any future data request must acquire same-day SPXW full-ladder
quotes for the single-leg long-call/long-put model.
