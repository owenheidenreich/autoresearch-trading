# The morning asymmetry pays nothing — the overshoot join

**2026-08-23. Supersedes the table in
[`MORNING_NEAR_MONEY_ASYMMETRY_2026_08_23.md`](MORNING_NEAR_MONEY_ASYMMETRY_2026_08_23.md) and closes
its open question.** Owner-authorised on 2026-08-23 as a historical option-path join under STATUS row 48.

---

## 1. The answer

The asymmetry is real and reproduces **stronger** than published. It is worth **nothing**.

Across 36 cells (4 clocks x 3 strikes x 3 stated exit rules) over 1,011 clean sessions,
**no cell has a mean P&L whose 95% bootstrap interval clears zero.** Thirty-one of 36 lose outright.
The five with a positive point estimate are indistinguishable from zero and none survives its interval.

The best cell, 09:35 at-the-money held to the 20-minute horizon:

| | |
|---|---:|
| mean P&L per trade | **+$2.01** |
| 95% bootstrap CI | **[−$34.61, +$36.98]** |
| median trade | **−$61.54** |
| profitable trades | **44.5%** |
| ticket | $1,370 |

Pooled over every cell, each stated exit rule loses: hold-to-horizon **−$8.49**, target-at-R
**−$11.48**, target-2R **−$12.05** per trade.

The three late cells are confidently negative — 15:00 ATM is **−$29.43 [−$50.99, −$6.27]** — which is
the same result the parent finding reported as `J = 0.0`, now priced.

## 2. Why an 80% race is worth zero

The parent finding asked whether SPX travels `R` before `J`. It does: at 09:35 ATM, `R = 1.14` against
`J = 10.94`, and that race is won **79.8% (CI 77.2–82.2)** — better than the 72% published.

**Winning that race means crossing zero, not profiting.** What the reconstruction adds is the
distribution on the far side, and it is symmetric enough to consume the entire advantage:

- Wins are small and frequent; the median win under target-at-R is **+$31**.
- Losses are large and rare; a stop costs roughly **$500**, near the declared −40% of the ask.
- The flat state is a **full loss to theta**, not a neutral outcome, and runs 5–32%.

**The 12:1 ratio of `J` to `R` is not a payoff ratio.** It is the ratio of two thresholds whose
crossing probabilities are already in the price. Reading it as a reward-to-risk figure is the error;
the option market is quoting it correctly. This is `RIGHT IDEA, WRONG UNITS` in a new costume — a
favourable-looking ratio that is not denominated in dollars.

## 3. What was wrong with the published table

The parent table had **no code behind it**: both join commits (`74f0e3d1`, `f280507f`) touched only
`LOG.md`, and the stored race grid cannot have produced it, since its `favourable_threshold_points`
bottoms out at **2.0** while the reported `R` is near **1**. The numbers were also revised three times
under changing IV assumptions ($2,278 → $2,266 → $1,748).

[`economics_race_join.py`](../economics_race_join.py) reconstructs it reproducibly, taking entry cost
from the **real ask in the ladder** rather than a modelled price, so no spot or spread is assumed.
Every level comes out lower, in the same direction, across all six published cells:

| cell | published | reconstructed |
|---|---:|---:|
| 09:35 ATM IV | 10.0% | **9.59%** |
| ask | $1,748 | **$1,370** |
| `R` | 1.2 | **1.14** |
| `J` | 14.5 | **10.94** |
| win | 72% | **79.8%** |

The consistency of the direction points at the original's entry pricing, not at noise. **The published
table is superseded.** Its qualitative claims — early and near-the-money is favourable, the asymmetry
decays to a coin flip, the 15:00 25-point contract is dead on arrival — all survive.

## 4. Method, and the two bugs the preflight caught

Per session, at its own spot, own IV and own spread: compute `R` and `J` from the real ask by
repricing through the pinned pricer, race the session's actual path against **that session's** `R` and
`J`, then reprice the position at the exit the rule produces. Pooling `R` and racing against the
average would be a different and wrong question.

The known-answer preflight ran before any result was read, and failed twice:

1. **Three defect sessions were included** that the upstream receipt excludes for interior whole-book
   freezes. The population now matches upstream at exactly 1,011.
2. **A test fixture was internally incoherent** — an ask of 9.7 paired with an unrelated 10% sigma,
   so the pricer valued the contract at 12.88 and `R = 0` was the correct answer to a meaningless
   question. `self_iv` is solved *from* each quote; the pricer reproduces the real mid exactly.

Preflight now covers race ordering, `J = 0` as an instant loss, unreachable `R`, Wilson intervals,
`R`/`J` monotonicity in distance, that reaching `R` returns exactly $0.00, and that a stop lands
near −40% of the ask.

## 5. What this does and does not close

**Closes:** the long single-leg 0DTE route as an *unconditional* proposition. The census closed it once
on fee-only grounds; this closes it on measured expectancy with the asymmetry granted in full.

**Does not close:** conditional entry. Every number here is unconditional — it says the *average* trade
in these cells loses, not that no signal can select a better subset. **No alpha has been spent on a
signal.** The ledger stands at 6 against a 0.66059463 bar. Any conditional test must overcome a
baseline that is genuinely negative, which is a materially harder starting point than the parent
finding implied.

**Unresolved and unchanged:** era stability is still confounded with source; the corpus remains
`NOT-USABLE` for certification; calendar, not money, is the binding constraint at 93.3% of all
sessions that exist.

## 6. Nothing is adopted

No strategy, no rule, no fit, no purchase, no order, no reserved session. This is a measurement that
says stop.
