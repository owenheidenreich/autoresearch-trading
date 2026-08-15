# How to approach the 0DTE options bot

**Job 24. Written 2026-08-13 on 905 usable 0DTE sessions spanning 2022-06 to 2026-07, acquired free.**

## The constraints that do not move

0DTE SPXW. Long calls and long puts only. One position at a time. Everything else — charter, hold length,
entry time, model class, data — is open.

## One sentence

The bar is not lowered by better exits or shorter holds, both of which are worth about 1.5 points and are
near-noise; it is lowered by **occupancy**, which cuts the accuracy a candidate must prove from **57.11% to
54.12%** against a break-even of 52.65% — so the plan is to trade a 15-minute serial clock rather than the
one-trade-per-session, 60-minute clock this project has assumed since the beginning.

## What was measured to get here

All of it on the 905-session corpus, at a $25 near-ATM round trip, computing no policy.

**Exit rules do not lower the bar.** Break-even is `L / (W + L)`, so an exit that cuts losses should help.
It does not, because it cuts winners in almost exactly the same proportion:

| Exit rule | W (correct) | L (wrong) | Break-even |
|---|---:|---:|---:|
| hold to horizon | 871 | 901 | 50.87% |
| stop −20% | 444 | 444 | 50.01% |
| stop −30% | 607 | 593 | **49.42%** |
| stop −40% | 733 | 722 | 49.62% |
| stop −50% | 817 | 813 | 49.88% |

Best case −1.45 points, and non-monotonic across levels, which means noise. The exit remains worth having:
it halves the account hit when wrong, 7.55% to 4.29%. But it is a **risk control, not an edge**, and the
plan does not budget for it as one.

**Hold length does not lower the bar either.** 52.62% at 15 minutes against 54.10% at 60 — about 1.5 points.

**Occupancy does.** One position at a time still permits `385 / hold` trades per session, and total trades
is what the measurement floor divides by:

| Hold | Trades/session | Total trades | Break-even | Provable at | Gap |
|---:|---:|---:|---:|---:|---:|
| 5 min | 77 | 69,685 | 52.73% | 53.57% | 0.84 |
| **15 min** | **25** | **22,625** | **52.65%** | **54.12%** | **1.48** |
| 30 min | 12 | 10,860 | 52.82% | 54.95% | 2.13 |
| 60 min — the assumed design | 6 | 5,430 | 54.10% | 57.11% | 3.01 |

Going from the 60-minute one-trade clock to a 15-minute serial clock is worth **3 accuracy points** on the
same data. That is larger than every other lever found in this project combined.

## Phase 0 — Resolve a contradiction before building on either number

**Two measurements of the same quantity disagree.** The pooled near-ATM 60-minute break-even is **54.24%**
by one route and **50.87%** by another, on the same corpus and the same band. The difference is how a
contract that stops trading before the exit minute is handled: one drops it, the other takes its last
traded price.

That is not a detail. Dropping contracts that went illiquid removes trades that most likely went to zero,
and which of the two is right decides whether the whole table above is stated a full three points too
optimistically. **Nothing else in this plan proceeds until it is settled**, and the settlement is a
measurement, not a choice: count how many contracts vanish before the exit, and what their last price was.

Every first number produced in this session has turned out to be the most optimistic one available. That
pattern is the reason this phase is first.

## Phase 1 — Extend the corpus to 2013, free

0DTE does not require *daily* expiries. Before 2022, SPXW expired Monday, Wednesday and Friday, and those
sessions carry same-day expiries — verified on 2019-06-05 and 2021-06-02. OPRA `ohlcv-1m` is $0.00 at every
era back to **2013-04-01**.

Roughly **1,400 further 0DTE sessions** at no cost, taking the corpus to about **2,300**. The gain is only
`sqrt`, so it is worth perhaps half a point of provable accuracy — but it is free, it widens the regime
coverage that Phase 2 depends on, and the machinery already exists.

## Phase 2 — Choose the cell, on stated criteria, before seeing any outcome

Three things trade off and the choice must be declared rather than discovered:

1. **Provability** favours the shortest hold: 5 minutes needs 53.57%.
2. **Charter risk limits** favour the longest: 77 trades a session at 13% of equity per ticket turns over
   the account many times a day, and must be checked against the 5% daily circuit breaker and the 50%
   survival floor before it can be declared. A cell that cannot satisfy them is not a candidate.
3. **Plausibility** favours neither: 53.57% at a five-minute horizon and 54.12% at fifteen are both above
   what this project has ever demonstrated, and the honest prior on both is poor.

**Recommendation: 15 minutes**, unless the risk check rules it out. It keeps most of the occupancy gain,
quarters the per-session turnover against 5 minutes, and sits at a horizon the existing feature and clock
machinery already supports.

## Phase 3 — Declare and freeze, then the known-answer campaign

Unchanged in discipline from the G1 attempts, and unchanged in order:

1. Freeze one hypothesis: entry rule family, exit rule, hold, moneyness band, entry-premium band. Content
   hashed before any replay.
2. Build the power receipt against the frozen index.
3. **Known-answer campaign first** — matched-surrogate false pass ≤5%, shared-term fixture ≤5%, recovery
   ≥80% at the declared threshold. This is what stopped both previous attempts, and if it stops this one
   the economics are not read.
4. Only then, one replay, one verdict.

The alpha ledger prices every attempt, including every constraint setting the outer loop occupies. The
knob registry refuses any setting that would touch a frozen or uncertified constant.

## What would make this fail, stated in advance

- **Phase 0 settles at 54.24% rather than 50.87%.** Every provable-accuracy figure rises about three
  points and the 15-minute cell needs 57%, which is back where the project already failed.
- **The risk check kills the occupancy gain.** If 25 trades a session at 13% per ticket breaches the daily
  breaker, the cell that provides the gain is not tradeable under the charter, and the charter's risk
  limits are not the part worth amending.
- **The recovery criterion fails again.** Two attempts have died there. A third is the pre-committed close.

## What is explicitly not in this plan

Buying quote data, a longer tenor, spreads, short premium, more than one position, or a different
instrument. All were considered and all are outside the constraints given.

## Stopping rule

The 2026-08-12 rule was overridden once, on 2026-08-13, and that override is
[recorded with the case against it](../../governance/STOPPING_RULE_OVERRIDE_2026_08_13.md). It says
plainly that a further failure closes the programme. This plan is that further attempt.
