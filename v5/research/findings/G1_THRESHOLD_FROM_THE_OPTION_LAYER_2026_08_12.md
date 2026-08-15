# What G1 has to prove for the option layer to survive it

**Dated 2026-08-12. Derived before any G1 economics exist, so it can be frozen as a pass criterion rather
than fitted to a result.**

**One sentence for the owner: G1's current bar is ES friction, which is the wrong bar — the right bar is
what the option layer needs to survive translation, and that is 6.23 gross ES points per trade at 60
minutes, a number the owned corpus cannot see but the expanded one can.**

## Why this exists

G1 measures ES direction. It exists only to justify buying SPXW options: do-not-retest ledger row 181
closed the 0DTE long-premium class *structurally* and named "demonstrated directional skill on the
underlying" as one of three ways to reopen it. So a G1 pass is not an end in itself — it is a key cut for a
specific lock, and it has to fit that lock.

Its current bar is **0.358 ES points**, which is ES round-trip friction. That bar answers "would this
trade be profitable in ES?" It does not answer "would the option built on it be profitable, and could we
tell?" Those are different questions with different, larger answers.

## The option layer's own measurement problem

The gate-chain audit measured 852 real 60-minute option trajectories. At the measured $3.08 fee, a correct
directional call returns **+$308.17** and a wrong one **−$425.33**. Break-even accuracy is therefore
**57.99%**, which the audit already recorded.

But being profitable and being *measurable* are different bars, exactly as they were for ES. The per-trade
payoff standard deviation is about **$362** — enormous relative to the edge — and there are only **251
owned option sessions**.

| | Accuracy required |
|---|---:|
| To be profitable | **57.99%** |
| To be detectable on 251 option sessions, single hypothesis | **65.73%** |
| Same, were G2 ever a multi-member search | 79.33% |

**The option layer must be 7.7 accuracy points better than break-even simply to be seen.** This is the same
trap job 15 found in ES, one level up, and it is why a G1 pass at the ES friction bar would mean nothing
downstream.

## Translating that into a G1 threshold

If the sign accuracy of the directional call is `p`, and the magnitude of the move is independent of
whether the call was right, then expected gross ES points per trade is `(2p − 1) × E|move|`. At the G1
09:35 slot with a 60-minute horizon the move standard deviation is 24.807 points, so
`E|move| = 24.807 × √(2/π) = 19.79` points.

| Directional accuracy | Option P&L per trade | Implied ES gross points |
|---:|---:|---:|
| 50.00% | −$58.58 | 0.00 |
| **57.99%** (option break-even) | $0.03 | **3.16** |
| 60.00% | +$14.77 | 3.96 |
| **65.73%** (option detectable) | **+$56.80** | **6.23** |
| 70.00% | +$88.12 | 7.92 |

> **The G1 threshold should be 6.23 gross ES points per trade at the 60-minute horizon**, not 0.358. That
> is 17.4x the ES friction bar, and it is set by what the option layer needs rather than by what ES costs.

## Correction, same day: the threshold must be stated in accuracy, not points

**The paragraph above is wrong in its units, and the expanded corpus is what exposed it.** The 24.807-point
move dispersion it rests on is the *owned year's*. Measured across the full ten years the pooled figure is
**15.971**, and the per-year figures range **7.08x**:

| Year | 60-min move SD | Year | 60-min move SD |
|---|---:|---|---:|
| 2017 | **3.88** | 2022 | 21.85 |
| 2018 | 8.24 | 2024 | 16.45 |
| 2020 | 16.45 | 2025 | 23.63 |
| 2021 | 14.12 | 2026 | **27.51** |

A fixed points threshold means a different thing in each of those years. The same 65.73% accuracy implies
**0.93** ES points in 2017 and **6.84** in 2026 — a sevenfold spread. Freezing "6.23 points" would set a
bar that is far too strict in calm years and far too lax in violent ones, and pooling across them would let
high-volatility years dominate the result.

> **Corrected threshold: 65.73% directional accuracy at the 60-minute horizon.** Accuracy is scale-free, it
> is the quantity the option layer actually requires, and it is invariant to the regime heterogeneity the
> ten-year corpus contains.

Detectability, recomputed as a proportion test against a 50% coin flip with the conservative conjunction
penalty:

| ES sessions | Smallest detectable accuracy | Required 65.73% |
|---:|---:|---|
| **247** (before the purchase) | 71.77% | **invisible** |
| **2,531** (after) | **56.80%** | **detectable, with margin** |

The conclusion is unchanged and now rests on a scale-free quantity: the owned corpus could not see what the
option layer needs, and the expanded one can. The points figures remain useful as a per-year sanity check,
never as the frozen bar.

**Consequence for Phase 2 beyond the units.** A 7x volatility range across the corpus makes the
fold-stability diagnostic essential rather than optional. An edge that lives only in the violent years is a
volatility artifact, not a strategy, and pooling would hide that. The diagnostic stays a report rather than
a pass criterion, as planned, but it must be read.

**Stated assumption, which Phase 2 must declare rather than inherit:** move magnitude is treated as
independent of whether the call was correct. A strategy that is right more often on *large* moves would
need less accuracy than this table says; one that is right more often on *small* moves would need more.
Nothing here measures which, and the direction of that error is unknown.

## Can G1 see it?

This is the part that decides whether the Phase 1 purchase was the right move. Using the invariant in
[`research/statistics.py`](../statistics.py) at one trade per session, single hypothesis, with the
conservative 2.75x conjunction penalty:

| ES sessions | Detection floor | Required 6.23 pts |
|---:|---:|---|
| **247** (owned before the purchase) | 10.81 pts/trade | **invisible** |
| **2,603** (after the ten-year purchase) | **3.33 pts/trade** | **detectable, with margin** |

**The purchase is what converts this chain from unmeasurable to measurable.** On the owned corpus, the
edge the option layer needs sits below G1's own detection floor — G1 could have passed or failed and
neither outcome would have told us anything about the option bot. After the purchase there is real margin
between what must be detected and what can be.

## What this does not establish

**That 65.73% accuracy is achievable.** It is not. This finding says only that the question becomes
answerable, which it was not before.

The bar is high and the project's own record argues against it: break-even alone is 57.99%, row 181
measured directional expectancy *below* break-even at minute cadence (−$13.00/trade gross before any
friction), and row 183's apparent directional skill was 78.5% reproduced by surrogates with no
predictability by construction. Nothing in the record demonstrates 60-minute directional accuracy above
chance that survived its controls.

So the honest position is the one the pre-committed stopping rule already anticipates: the expanded corpus
buys the ability to ask a question that was previously unaskable, the prior on the answer is poor, and a
measured "no" is worth far more than the "unknown" it replaces.

## What Phase 2 must do with this

1. Freeze **6.23 gross ES points per trade at 60 minutes** as the declared threshold, alongside the
   hypothesis, before any replay runs.
2. Declare the magnitude-independence assumption above explicitly, or replace it with a measured
   relationship if one can be derived without inspecting outcomes.
3. Recompute the exact figure once the expanded index is final — 6.23 depends on the 09:35-slot move
   dispersion, which will be re-derived on ~2,603 sessions rather than 247.
