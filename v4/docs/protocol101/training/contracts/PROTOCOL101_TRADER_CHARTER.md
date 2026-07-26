# The Trader Charter — Protocol101's Root Definition of "Profitable"

Written: 2026-07-19, from the owner's answers to the three root questions.
Status: SIGNED FINAL on 2026-07-25. This is the document every
gate, metric, and loop must trace back to. It is deliberately written in
plain language; the mapping table at the end is the bridge to machinery.
Per the loop-architecture principle: this judgment comes from outside the
machinery and cannot be computed by it. Changing it requires the owner —
never the optimizer.

## Who this trader is

A base-hits trader with home-run capacity, protected by a survival
guarantee — the Pickles profile. It buys SPXW 0DTE options — capped-loss,
uncapped-gain tickets — one contract at a time. Most trades end small and
harmless because the exit discipline scratches anything that stops
working; a meaningful minority run to large wins; losses are cut fast and
big losses are rare. The equity line should move up and to the right —
smooth-ish from the many small outcomes, punctuated by jumps from the big
ones. High win rate is the natural BYPRODUCT of that exit discipline,
never a number we optimize directly (optimizing win rate itself teaches a
system to scratch everything and bleed out in fees).

## The outcome distribution we aim for (the Pickles north star)

The owner's model trader, 17 years / 58,251 trades:

| Category | Share of trades | Avg return |
|---|---:|---:|
| Big wins | 18% | +40% |
| Small wins / scratches | 73% | +0.1% |
| Small losses | 8% | -0.5% |
| Big losses | 1.6% | -30% |

Nearly all profit lives in the big-win column; the 73% scratch column is
risk management wearing a win's clothing; the killer discipline is the
bottom row staying under 2%. Our version must respect one difference of
scale: at one contract on a ~$10k account, fees (~$3 round trip) mean a
literal +0.1% scratch is a small net LOSS. Our "scratch" is therefore
defined fee-aware: any exit between -5% and +5% is a scratch, and the
distribution target is directional, not literal — MANY scratches and
small wins, a REAL minority of big wins, big losses rare enough to count
on fingers. Every Stage-1/2 packet reports this four-bucket outcome
distribution so drift from the profile is always visible.

## The four commitments

**1. Droughts are flat, never deep.** (Owner: "a relatively flat account
is what I'd be aiming for in a drought... avoid large losses where I'm
clawing back just to break even.") When the market offers nothing, the
trader's job is to abstain, not to force trades. Sideways-for-months is
acceptable and expected; a deep hole that takes months to climb out of is
not. Depth of drawdown matters more than duration of flatness. The
machine expression of patience is trade-count dropping toward the low end
of the frequency band, not degraded trades.

**2. The floor is SPY; the dream is convex — reached the Pickles way.** (Owner: "at the very
minimum beat the S&P 500 that year... 3x to 10x... 10k -> 100k would have
me genuinely ecstatic.") The first-year pass/fail line for live trading
is: beat what the same dollars would have done sitting in SPY that same
year. Below that line, the honest conclusion is "index and walk away."
Above it, ambition is open-ended — 3x-10x is the aspiration band, never a
promise and never a gate. No hardcoded profit cap may exist anywhere in
the system's final form: fixed exit targets are Stage-1 measurement
scaffolding only, scheduled for replacement by learned exits (Stage-2)
whose explicit mandate is harvesting the rare huge winner.

**3. No single day may wound the decision-maker.** (Owner: "a 10% loss in
a single day is unacceptable... aim for a max of a 3-5% drawdown as the
worst single day... it really depends on the account value.")
Daily circuit breaker: when a session's realized loss reaches **5% of
the account value recorded at that session's start**, the trader is done for
the day — no revenge trades, no averaging down, ever. The base is frozen for
the session and scales as the account grows ($500 on $10k; $5,000 on $100k).
Structural support:
single-contract sizing and OTM premiums mean most individual losses are
naturally 1.5-5% of the starting account; the breaker guards the pile-up,
not the single trade. (Historical footnote: the May protocol farm
independently used a $500/day stop on $10k — this rule has precedent.)

**4. Survival outranks everything.** Inherited from the signed G4 v2: the
account never falls below half its starting capital under any accepted
strategy, and every dollar of drawdown must be purchased by at least a
dollar of realized profit. Ruin is the only unrecoverable outcome; no
opportunity justifies risking it.

## What this trader is NOT

- Not a win-rate optimizer: win rate is tracked and expected to be HIGH
  as exit skill matures (the Pickles byproduct), but the optimized
  objective is always fee-adjusted PnL — a system rewarded for win rate
  itself learns to scratch everything and pay fees for nothing.
- Not a scratch-mill: exits must clear fee gravity; churning breakeven
  trades at our size is slow bleeding, not safety.
- Not allergic to lumpy profits: while the line should trend up-and-right,
  the big-win column carrying most of the profit is the design, not a
  red flag (concentration metrics stay report-only).
- Not a scaler (yet): one contract per trade until the owner signs a
  sizing change; no martingale, no doubling, no exceptions.
- Not a style-drifter: if this style stops working, the answer is an
  owner-level decision, never silent drift into selling premium,
  scalping, or holding overnight.

## Mapping: every charter sentence to machinery

| Charter commitment | Existing machinery | New (to implement) |
|---|---|---|
| Droughts flat, never deep | G6 era guard (no systematically negative eras); G7 low bound 0.3/day permits near-abstention; G4a Calmar bounds hole depth vs profit | **Underwater-duration report**: longest time below high-water mark, report-only, so "flat vs clawback" is visible per candidate |
| Floor = SPY | — | **SPY benchmark line**: every live/paper year-to-date report shows strategy PnL vs same-capital SPY return; G3 remains the training-time analog (beat the best heuristic) |
| Dream = 3x-10x, uncapped | Stage-2 learned exits (mandated); menu includes time-based uncapped shapes | **Harvest ratio**: realized PnL / peak available PnL per trade, tracked from Stage-1 day one — Stage-2's report card |
| Pickles outcome profile | Path diagnostics already record MFE/MAE per trade | **Four-bucket outcome distribution** (big win / scratch / small loss / big loss, fee-aware thresholds) in every packet; Stage-2 exit objective explicitly rewards BOTH loss-truncation (the scratch engine) and tail capture (the big-win engine) |
| 5% daily circuit breaker | Serial simulator supports daily-loss stops (farm precedent `dailyloss500`) | **Set to 5% of session-starting equity** in Stage-1 sim config and all live/paper guards; percent-based, scales with account |
| Survival floor | G4 v2(b): equity never below $5,000/fold (training); guard layer live | Confirm live guard mirrors the 50%-of-starting-capital rule |
| No profit caps in final form | Stage-2 objective doc (to be written after Stage-1 evidence) | Charter language binds Stage-2's design: exit model must be rewarded for tail capture, not smoothness |
| One contract, no martingale | Simulator hard-coded single contract; affordability enforced | Unchanged until owner-signed sizing revision |
| Win rate is not an objective | Gates doc: "win rate is a diagnostic, never an objective" | Unchanged |

## Expectation honesty (so future disappointment is calibrated)

Beating SPY on $10k with single-contract 0DTE longs after ~$3/trade fees
is a genuinely hard floor — it requires real edge, not just discipline.
The 3x-10x band requires exceptional tail capture on top of real edge.
The charter therefore defines SUCCESS TIERS for year one of live trading:
- **Tier 0 (failure):** underperforms SPY → index and stop.
- **Tier 1 (pass):** beats SPY with all four commitments intact.
- **Tier 2 (strong):** 2-3x while commitments intact.
- **Tier 3 (ecstatic):** 3x+ — celebrate, then audit before believing.

Owner signature:  Owen Heidenreich  Date: 07/25/2026
