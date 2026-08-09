# G1 verdict: UNDERPOWERED — the screen cannot see an edge worth trading

**One sentence: the gate works correctly, but on 247 owned sessions it can only detect an edge roughly
22 to 88 times the cost of trading, so it was stopped before any real profit-and-loss was computed.**

This is one of the three verdicts declared in advance. It was reached the way the declaration said it
must be — from the known-answer campaign and the recomputed detection floor, **before** the economic
outcome was read. No member's profit and loss has been computed at any point, and none should be.

## What was measured

The frozen 18-member family, on the frozen 247/243 session index, judged by the six declared pass
criteria. Everything below is synthetic: matched surrogates and fixtures with a known truth.

### The gate does not report edges that are not there

| Known-answer test | Result | Declared limit | |
|---|---:|---:|---|
| Matched surrogate false pass | **0.7%** (7/1000) | ≤ 5.0% | **pass** |
| — Wilson 95% upper | **1.29%** | ≤ 7.5% | **pass** |
| Shared-term fixture (row 183's shape) | **1.0%** (10/1000) | ≤ 5.0% | **pass** |

The shared-term result is the one worth pausing on. Row 183 recorded a screen whose headline was 78.5%
reproducible by surrogates because its feature and its target shared a price level. This family is
immune to that specific trap, and now measurably so: **1.0%**. The reason is structural — every G1 score
and every G1 target is a *difference*, never a level.

### But it cannot see an edge of plausible size

The declared criterion is that an effect of minimum-detectable size is recovered at least 80% of the
time. It is not. Sweeping the injected effect until 80% recovery is reached gives the gate's real
detection floor:

| Mechanism | Detects at | Occupancy | Per executed trade | Against the 0.358-point cost bar |
|---|---:|---:|---:|---:|
| M3 overnight gap | 8 pts/session | 99.2% | 8.06 pts | **22.5×** |
| M1 opening range | 16 pts/session | 56.0% | 28.57 pts | **79.8×** |
| JOINT | 16 pts/session | 50.7% | 31.56 pts | **88.2×** |

At $50 per ES point that is **$400 to $800 per session**, every session, for a year. An edge that large
would be visible without a research programme.

Recovery is smooth and monotonic, so this is a power limit rather than a broken gate — M3 recovers 17.5%
at 4 points, 55.8% at 6, 85.0% at 8, 96.7% at 10, and 100% at 16.

## Why this is worse than the measurement review predicted

The measurement review estimated 2.2–4.0 net points per session as detectable. The gate actually needs
**8–16**. The review was not wrong; it answered a different question. Its figure is the minimum
detectable effect of a *single z-test on the mean*. The real gate is a conjunction of six criteria —
an absolute bootstrap lower bound, a paired lower bound against a causal comparator, four-of-five folds
positive on **both**, and gross-per-trade clearing friction — judged with a session-block bootstrap and
corrected for eighteen members. Every one of those costs power, and they compound.

**The general lesson: an analytic MDE describes the test you wish you were running. Measure the one you
are actually running.**

## The one permitted null repair, and what it caught

The declaration allows a single repair if the campaign fails. It was needed and it was used.

The first campaign measured **11.7% familywise false pass** against a 5% limit. The cause was that the
family was judged one member at a time at the raw 95% level, then reported as "any of eighteen passed" —
eighteen attempts at significance quoted at the price of one. The repair applies a Bonferroni correction
inside the gate (per-member α = 0.05/18 = 0.00278), so the null campaign and any future real replay are
judged by identical arithmetic. False pass fell to 0.7%.

That repair is spent. There is no second one.

## Two fixture errors found and corrected before they could mislead

Both were caught by the campaign disagreeing with itself, which is the point of running it.

1. **The shared-term fixture was built with mean reversion.** A path pulled back toward a level is
   *genuine tradeable predictability*, not an artifact — and the gate passed it 100% of the time,
   correctly. The fixture now uses independent increments around a large wandering level, which is row
   183's actual shape. A test pins the lag-1 autocorrelation of its increments near zero.
2. **The injected effect was spread across the whole remaining session.** A 15-minute horizon therefore
   collected 15/355 of it — below friction — and recovery came out *non-monotonic* in the injected size,
   which is how the error surfaced. The drift is now per-minute and calibrated to a reference horizon.

## What this does and does not mean

- It means **no large edge is detectable here**, not that no edge exists. The screen was released under
  "large edge or stop," and this is the stop.
- It does **not** mean the mechanisms are disproven. They were never measured. Their real economics
  remain unread, which keeps them legitimately available to a future screen with more sessions.
- It does **not** license reading the outcome "just to look." The campaign is what makes a positive
  result meaningful; without it a positive number is the row-183 failure with extra steps.

## What would change the answer

Only more sessions. The detection floor scales as `1/sqrt(n)`, so halving it needs roughly four times
the data. Reaching the cost-scale edge the project actually cares about is not reachable by cleverness
on the owned year.

| Target | Sessions needed | Calendar |
|---|---:|---|
| Halve the floor to ~4 pts/session (M3) | ~988 | ~4 years |
| Reach ~1 pt/session | ~15,800 | ~63 years |

This is the same wall the measurement review hit, now measured on the real gate instead of estimated.

## Reopening conditions

A genuinely new attempt needs at least one of:

- **substantially more sessions** — the only lever that moves the floor;
- **a materially higher-occupancy design** — more independent decisions per session lowers the
  per-decision effect needed, which the current one-trade-per-session clock forbids;
- **a different instrument or horizon** with a better signal-to-friction ratio, declared and frozen
  before any outcome is inspected.

Re-running this family on this corpus is not one of them.

## Provenance

Family hash `157fe437998d968d41e7312a914dd65fa110b82c25600df0523c62a01699985a` (247 M1-eligible, 243
gap-eligible). 1,000 campaigns per ceiling criterion, 120 per recovery point. Code:
[`campaign.py`](../direction/campaign.py), [`gate.py`](../direction/gate.py),
[`fixtures.py`](../direction/fixtures.py), [`surrogate.py`](../direction/surrogate.py).
