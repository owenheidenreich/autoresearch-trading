# The dollar cap holds, the breaker is not mis-sized, and the real throttle is ruin

**2026-08-16. Model-free simulation on already-owned evidence. No fit, no purchase, no vendor
contact, no read of the partially-acquired backfill.** Declaration
[`SCALE_SENSITIVITY_DECLARATION_V1.json`](../../work/lifecycle-training/SCALE_SENSITIVITY_DECLARATION_V1.json)
(self-hash `b6b36b28…`) was frozen and its 28-cell family hashed before any outcome was computed.

> **This study assumes a hypothetical edge.** Win rate is a swept parameter, not a measurement. The
> measured win rate of this population at random entry is **31.65%**, and every economic measurement
> on it is negative. **No number below is evidence that an edge exists.**

## What the owner asked, answered

### Q1 — The dollar cap holds. But the code that builds the action mask does not implement it.

**Simulated: the cap holds exactly.** Across all four account levels including $100,000, the largest
premium any path ever bought was **$1,990** — the largest eligible ticket in the population. It does
not grow with equity, because a dollar ceiling cannot.

**But the owner's fear is real, and it is live in the code.** Two components express the ceiling as a
**share of equity**:

| Component | Expression | At $10,000 | At $100,000 |
|---|---|---:|---:|
| [`audit_causal_day_coverage.py:33`](../../ops/audit_causal_day_coverage.py) — builds `entry_eligible`, the action mask | `SESSION_START_EQUITY_USD * TICKET_CEILING_SHARE` | $1,300 | **$13,000** |
| [`check_occupancy_risk.simulate`](../../ops/check_occupancy_risk.py) | `start * premium_ceiling >= premium` | $1,300 | **$13,000** |

Both currently evaluate against a frozen $10,000, so neither drifts **today**. The hazard is that the
2026-08-16 serial-account repair made equity compound for the first time; any future wiring of live
equity into either expression converts a static constant into a drifting one, and $13,000 buys deep
ITM. **The mechanism is a share-denominated ceiling, and the fix is to state it in dollars** —
`MAX_ENTRY_ASK_USD` must become a fixed constant equal to the signed $2,000, not a product of equity
and a share. Phase 2 must do this before it hardcodes masks.

**The dollar cap alone does not bar in-the-money contracts.** Measured over 1,880,427 two-sided
ladder rows: of the 77.55% of contracts eligible under a $2,000 cap, **37.3% are in the money**, and
the 99th percentile sits at **+18.4 ITM points**. The `moneyness_band` bar is therefore load-bearing
and must not be treated as redundant with the price cap. The current mask happens to be stricter than
the charter — `eligible_entry` admits only `money < 0`, i.e. OTM only.

**One limit stated rather than hidden:** the owned ladder is bounded to **±25 ITM points**, so this
study cannot observe whether a wider ceiling would admit contracts deeper than that. Zero cap-eligible
contracts exceed 25 ITM points because none exist in the population, not because the cap excludes
them.

### Q2 — The breaker is **not** mis-sized. The owner's premise does not hold.

The concern was that 20% of $10,000 is $2,000, which equals one maximum ticket, so the first full-size
loss would end the session. **That assumes max loss equals premium paid, which the signed amendment
replaced.** Under the −40% declared stop, a maximum ticket loses about $800, not $2,000.

Measured loss distribution under the stop, cap-eligible (n = 20,959):

| | Loss | Two of them | Trips the $2,000 breaker? |
|---|---:|---:|---|
| Median | $373 | $746 | no |
| 90th percentile | $683 | $1,366 | no |
| Worst observed | $1,293 | $2,586 | **only two worst-case losses in one session** |

Simulated breaker firing rate is **0.11% of sessions** at worst ($10,000, 30% win rate) and **0.00%**
everywhere else. Affordability blocked **0.00%** of offered trades at every account level.

**Conclusion: no defect, and therefore no amendment is drafted.** The 20% breaker admits two
maximum-size losses as intended and never becomes the strategy. The instruction to report a defect
"as such, not work around it" is honoured by reporting that the measurement does not find one.

**The observed throttle at $10,000 is real but is not the breaker.** Trades per session fall to
**0.447** at a 30% win rate — because 95.5% of paths are dead, not because they are stopped from
trading. Death censors trading; the breaker does not.

### Q3 — Required edge, per account

Threshold declared before running: **ruin ≤ 5%**, where ruin is equity breaching 50% of starting
capital within 252 sessions. Chosen because the charter's survival floor is already 50% and the
project's risk machinery already declares a 5% tolerance — reusing it avoids inventing a standard.

| Account | Break-even win rate | Win rate for ruin ≤ 5% | Ruin at 40% |
|---|---:|---:|---:|
| $10,000 | ~37% | **between 45% and 50%** (7.5% → 1.9%) | 24.0% [23.4, 24.6] |
| $25,000 | ~35% | **between 35% and 40%** (29.6% → 3.0%) | 3.0% [2.7, 3.2] |
| $50,000 | ~34% | **between 35% and 40%** (5.9% → 0.1%) | 0.1% [0.1, 0.1] |
| $100,000 | ~33% | **between 30% and 35%** (5.8% → 0.0%) | 0.0% [0.0, 0.0] |

**This is the number Phase 5 should be judged against.** At $10,000, breaking even needs ~37% but
*surviving* needs ~47% — a ten-point gap that exists purely because a small account cannot absorb its
own drawdowns. Brackets are reported rather than interpolated points because the win-rate grid was
frozen at 5-point steps before the run; the arrows show the ruin figures at each bracket edge.

### Q4 — Per-trade economics is not size-coupled. The apparent movement is survivorship.

Per-trade net moves with account size in the raw output — at a 30% win rate it runs −$116 at $10,000
against −$43 at $100,000. **That is not the trade changing.** Two checks establish the mechanism:

- Affordability blocked **0.00%** of offers at every level, and max premium bought was **$1,990**
  identically everywhere, so the trade available is the same trade.
- Where ruin vanishes, the numbers converge exactly: at a 50% win rate, per-trade net is **$193.22 /
  $193.58 / $193.34** at $25k / $50k / $100k — identical within noise — while $10,000 reads $187.04
  with 1.9% ruin.

The divergence is **censoring**: paths that die stop trading, so the surviving sample at small
accounts is drawn from luckier sequences. It is an artifact of measuring per-trade means inside a
compounding walk, not a size coupling in the economics. **Any Phase 5 packet reporting per-trade P&L
across account sizes must report it on surviving paths only and say so**, or repeat this artifact.

### Q5 — Where the constraints stop binding

The cap and the breaker essentially never bind at any level: affordability blocks 0.00% of offers
everywhere, and the breaker fires at most 0.11% of sessions. **The binding constraint is the survival
floor, and it stops binding between $25,000 and $50,000** at a plausible edge:

| Account | Ruin at 35% | Ruin at 40% |
|---|---:|---:|
| $10,000 | 63.6% | 24.0% |
| $25,000 | 29.6% | **3.0%** |
| $50,000 | **5.9%** | 0.1% |
| $100,000 | 0.0% | 0.0% |

**This gives the $25,000 amendment-expiry review a measured basis.** $25,000 is approximately the
equity at which ruin falls below the declared 5% threshold at a 40% win rate — the round number in
the signed amendment turns out to sit almost exactly on the measured transition, which is fortunate
rather than designed.

## What this does not say

It does not say the strategy works, that a 40% win rate is attainable, or that the measured 31.65%
random-entry rate can be improved on. It prices what *would* be required. Two further limits: outcomes
are resampled from a 251-session random-entry population and assume the trade distribution is
stationary as the account grows, which nothing here tests; and win/loss draws are independent across
trades, so any within-session or across-session clustering of outcomes is absent by construction and
would make small accounts worse, not better.

## Method

20,000 compounding paths × 252 sessions × 28 declared cells. Outcomes are measured `(ticket, net)`
**pairs** resampled from the quote-priced path population under the declared −40% stop — never a
parametric assumption, and never a quantile grid, which trims the right tail and biased an earlier
generation of these numbers pessimistic by 6–19%. Resampling pairs preserves the joint distribution of
ticket size and outcome. Ruin intervals are Wilson 95%; trades-per-session intervals are path
quantiles; per-trade intervals are normal-approximation on the path mean.

## Evidence

- Receipt: `v4/audit/autoresearch/scale_sensitivity_2026_08_16/receipt.json`
- Declaration: [`SCALE_SENSITIVITY_DECLARATION_V1.json`](../../work/lifecycle-training/SCALE_SENSITIVITY_DECLARATION_V1.json)
- Code: [`research/scale_sensitivity.py`](../scale_sensitivity.py),
  [`ops/run_scale_sensitivity.py`](../../ops/run_scale_sensitivity.py)
- Signed law: [`CHARTER_AMENDMENT_TICKET_AND_BREAKER_2026_08_16.md`](../../governance/CHARTER_AMENDMENT_TICKET_AND_BREAKER_2026_08_16.md)
  and [`ADDENDUM_STOP_LEVEL_AND_UNDERPOWER_2026_08_16.md`](../../governance/ADDENDUM_STOP_LEVEL_AND_UNDERPOWER_2026_08_16.md)
