# Strategy Card — Opening-Structure Reversion

## Status — revised 2026-04-19 after V1 / V2 falsification sequence

The hand-coded V1A / V1B mechanical triggers built around "reclaim back
through VWAP" were **falsified** on held-out folds. V2 (a learned
RandomForest scorer over the 12-core shortlist) then **passed**
falsification with +2.28pp edge over a same-side same-day random-bar
control, and V2 stress-test ablations revealed the learned model's edge
is NOT driven by the reclaim event itself — the load-bearing features
are overnight displacement (`opening_gap_pct`), first-15 settlement
(`first15_close_position`), and first-15 acceptance
(`first15_acceptance`). `vwap_reclaim_state` is actively slightly
harmful to the learned baseline.

The reversion FAMILY survives; the central-event description has
shifted. This card has been rewritten accordingly.

## Hypothesis (post V1/V2)

When the opening auction displaces away from the prior-day close and
fails to extend that move into the first 15 minutes of RTH, the session
tends to accept back into the opening-range structure and offer a
short-horizon 0DTE long-premium opportunity on the rejecting side.

The predictive core of "which bar is a good entry" is:

1. **Overnight / opening displacement** (`opening_gap_pct`).
2. **First-15-minute settlement** (`first15_close_position`) — where the
   opening auction's 15-minute close sat within its own range.
3. **First-15 acceptance** (`first15_acceptance`) — whether current
   price is being accepted back inside the opening 15 minutes, neutral,
   or still pushing outside.

The reclaim/reject event itself (`vwap_reclaim_state`) is **not** part
of the predictive core. The original thesis writeup centered on that
event; the V2 stress-test ablations refute it as a load-bearing signal
for this dataset.

This is a **reversion** thesis in the sense that the trade benefits
when price accepts back into opening structure rather than continuing
the pre-open displacement — but the operational definition is opening
STRUCTURE-based, not a reclaim-through-VWAP event.

- buy calls on days with a downside gap / low-settled first-15 that the
  session does not push lower
- buy puts on days with an upside gap / high-settled first-15 that the
  session does not push higher
- use higher-delta contracts because the hold is short and theta is
  expensive

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

### Historical: V1A / V1B hand-coded trigger (FALSIFIED)

The original spot-side trigger below was falsified by the V1A / V1B
runs. It is kept here for reference only; do not use it for new
baselines. See `v2/analysis/mechanical_baseline_opening_reversion.py`
and `v2/analysis/mechanical_baseline_v1b_opening_reversion.py` for the
exact implementations that were tested.

Bull call trigger (falsified):

- `bar_of_day` between `15` and `120`
- price had been at least `10 bps` below session VWAP within the prior `10`
  bars
- current bar closes back above VWAP
- `first15_acceptance >= 0`
- `bar_delta > 0`

Bear put trigger (falsified): symmetric.

V1A aggregate on held-out folds: mean_net_pct −0.354%, gap vs Control
A +0.16pp (narrow loss). V1B (same trigger + `iv_percentile` / `vrp`
gates) improved the gap to +1.02pp but still failed the bar.
V2 stress-test ablation confirmed that removing the `vwap_reclaim_state`
feature from the learned model improves it.

### Current: V2 learned-scorer baseline (PROMOTED)

Implementation: [v2/analysis/mechanical_baseline_v2_learned_scorer.py](../analysis/mechanical_baseline_v2_learned_scorer.py).
Plan: [mechanical_baseline_v2_plan.md](mechanical_baseline_v2_plan.md).
Stress test verdict: passed all three promotion gates (bootstrap
gap_vs_A 95% CI [+0.19%, +4.92%], null p = 0.010 for gap_vs_A,
ablation story intact). See [`../lab_notebook.md`](../lab_notebook.md)
for the full stress-test entry.

Pipeline (unchanged across V1A → V1B → V2 except for the bar-selection
mechanism):

1. **V1B premium gates.** Per fold from `train_days` only, 3×3 grid
   over `iv_percentile ≤ {0.6, 0.7, 0.8}` × `vrp ≤ {median, p75, 0
   if feasible}`. Every fold selected `iv ≤ 0.8, vrp ≤ p75`.
2. **Candidate universe.** Every V1B-admissible bar (gates pass +
   contract exists after the dual spread gate) in `[BAR_LO=15,
   BAR_HI=120]`, both sides (C and P).
3. **Learned scorer.** `sklearn.ensemble.RandomForestRegressor(
   n_estimators=200, max_depth=5, min_samples_leaf=20)` over the
   12-core shortlist features + side indicator. Trained on simulated
   trade outcomes (realized `net_pct`) at every `train_days`
   candidate.
4. **Per-day selection.** Score every candidate on each test day,
   take argmax. No threshold.
5. **Exits and controls.** Unchanged from V1B.

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
