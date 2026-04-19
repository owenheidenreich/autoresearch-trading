# Strategy Card — Opening-Structure Reversion

## Hypothesis

When the opening auction overextends away from fair value and fails to keep
going, the best 0DTE opportunity is the **reclaim / reject back toward VWAP
and opening structure**, expressed with SPX/SPXW same-day long premium.

This is a **reversion** thesis, not a breakout thesis:

- buy calls after failed downside extension and reclaim
- buy puts after failed upside extension and reject
- use higher-delta contracts because the hold is short and theta is expensive

## Instrument and trade object

- Underlying: `SPX` / `SPXW`
- Expiry: `0DTE` only
- Position type: long calls or long puts only
- No spreads
- No hedge conversions
- No futures execution

## Session window

- Earliest evaluation: `09:45 ET` (after the first 15 minutes exist)
- Latest new entry: `11:30 ET`
- No entries during lunch or later-session windows in the first baseline

## Setup definition

### Bull call setup

The market has sold off or stretched lower, then fails to continue and
reclaims fair value.

Required conditions:

1. Price was meaningfully below VWAP earlier in the morning.
2. Current bar reclaims back above VWAP after that extension.
3. Price is back inside or above the first-15-minute value area, not still
   accepting lower.
4. The reclaim bar has positive conviction.
5. The option is still worth expressing through long premium after spread and
   theta.

### Bear put setup

The market has rallied or stretched higher, then fails to continue and rejects
back below fair value.

Required conditions:

1. Price was meaningfully above VWAP earlier in the morning.
2. Current bar rejects back below VWAP after that extension.
3. Price is back inside or below the first-15-minute value area, not still
   accepting higher.
4. The rejection bar has negative conviction.
5. The option is still worth expressing through long premium after spread and
   theta.

## Mechanical baseline specification

### Spot-side trigger

Use raw spot/session state, not the normalized `X` tensor.

Bull call trigger:

- `bar_of_day` between `15` and `120`
- price had been at least `10 bps` below session VWAP within the prior `10`
  bars
- current bar closes back above VWAP
- `first15_acceptance >= 0`
- `bar_delta > 0`

Bear put trigger:

- `bar_of_day` between `15` and `120`
- price had been at least `10 bps` above session VWAP within the prior `10`
  bars
- current bar closes back below VWAP
- `first15_acceptance <= 0`
- `bar_delta < 0`

### Contract selection

- Calls for bull setup, puts for bear setup
- choose the nearest executable contract in absolute delta band
  `0.45-0.55`
- if no contract exists in the band, skip
- reject entries when `option_spread_pct` exceeds the configured quality cap
  (baseline: `20%`)

### Exit family

Use a spot-driven exit engine for the first test.

Bull call exits:

- hard stop: first close back below VWAP after entry
- profit target: first touch of the first-15-minute high
- time stop: `30` minutes after entry or `11:30 ET`, whichever comes first

Bear put exits:

- hard stop: first close back above VWAP after entry
- profit target: first touch of the first-15-minute low
- time stop: `30` minutes after entry or `11:30 ET`, whichever comes first

### Why this contract shape

- higher-delta 0DTE long premium matches the short-horizon thesis better than
  far-OTM convexity
- gamma is welcome if the reversion accelerates
- theta is controlled by keeping the trade early and the hold short

## What would falsify the hypothesis

The baseline is falsified if, on a clean out-of-sample evaluation:

1. target-hit rate does not exceed stop-hit rate,
2. net option return after costs is non-positive on the primary delta bucket,
3. and a simple randomized or time-matched control performs as well or better.

A weak or noisy result does **not** imply "buying SPX 0DTE is dead." It only
falsifies this opening-reversion expression.

## What is intentionally out of scope

- continuation breakouts
- event-volatility expansion
- Pickles-style ES / NQ / breadth confluence
- adaptive ML entry ranking
- later-day power-hour setups

Those can be tested later, but they are not part of this first mechanical
baseline.

### Note on IB-derived continuation features

The live feature contract contains `ib_break`, `ib_extension_pct`, and
`breakout_confirmation` (see
[feature_schema.md](feature_schema.md)). They stay outside the v1 trigger for
two reasons:

1. They describe a **continuation** thesis (break-and-follow-through), which
   is the opposite of the failed-extension-and-reclaim mechanism being
   tested here.
2. They are **point-in-time unsafe before `bar_of_day >= 30`** because the
   initial-balance high/low is broadcast day-wide. Any future continuation
   baseline that uses these features must enforce the `bar_of_day >= 30`
   guard explicitly.

Excluding them from v1 is both a thesis-coherence decision and a safety
decision.
