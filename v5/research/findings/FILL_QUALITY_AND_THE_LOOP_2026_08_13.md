# The loop is affordable. The instrument is not.

**2026-08-13**, answering "can we run an autoresearch loop that keeps testing ideas until it is profitable?"

[Fill-quality receipt](../../../v4/audit/autoresearch/fill_quality_2026_08_13/receipt.json) ·
[code](../../ops/measure_fill_quality.py) · [tests](../../tests/test_fill_quality.py) ·
[alpha ledger](../autoresearch/budget.py)

## Two answers, and the second is the one that matters

**Yes, the loop is affordable — far more than I expected.** The alpha ledger prices every experiment by
raising the bar it must clear, and that bar rises with `sqrt(log k)`, which is very nearly flat:

| Experiments run | Bar a screen must show | True accuracy needed |
|---:|---:|---:|
| 1 | 55.29% | 57.89% |
| 30 | 56.11% | 58.71% |
| 400 | 56.57% | 59.17% |
| 5,000 | 56.95% | **59.55%** |

Going from **one** experiment to **five thousand** costs **1.66 accuracy points**. The fear that searching
hard would burn the corpus is quantitatively wrong; multiplicity is cheap here.

**But the entry price is the problem.** Even experiment number one needs a **true directional accuracy of
57.89%** against a break-even of 53.72%. The best any measured rule has reached is 53.52%, and it lost
money. A loop of five thousand ideas searches a space whose admission fee nothing has ever come close to
paying.

## Why more ideas cannot help: the conclusion is a ratio

Everything measured this session reduces to one comparison — what the option is mispriced by, against what
it costs to trade. A signal search moves neither term. So the only lever left was execution, and it had
never been measured. Now it has, on **1,912,157 near-ATM contract-minutes across 251 sessions**:

| Entry hour | Mean mid | Mean spread | Spread as % of mid | Aggressive round trip |
|---|---:|---:|---:|---:|
| 09:00 | $1,740 | $22.01 | **1.44%** | $25.09 |
| 11:00 | $1,372 | $18.26 | 1.85% | $21.34 |
| 13:00 | $1,093 | $18.81 | 2.81% | $21.89 |
| 14:00 | $940 | $19.61 | 4.10% | $22.69 |
| **15:00** | $805 | $23.87 | **9.04%** | $26.95 |
| **all** | $1,206 | **$19.92** | 3.44% | **$23.00** |

Two things follow, and the second is fatal.

**The carried $25 constant was honest.** Measured, the aggressive round trip is **$23.00**. Every economic
conclusion in this project rested on a number nobody had checked; it checks out, slightly conservative.

**The spread is worst exactly where the edge is.** The variance premium is concentrated in the final hour —
2.74% of premium at 15:00 on a fifteen-minute hold. The spread at 15:00 is **9.04% of mid**. In the one
window where the option is most mispriced, the toll is **3.3 times** the prize. The two effects are not
independent: dealers widen precisely where the gamma risk they are warehousing is largest.

## The last hope, measured and closed

The one cell that ever cleared its corrected bound did so at **$3.08 a leg** — fees only, no spread — which
requires resting passive and being filled. So: would a resting order actually get filled, and at what price?

| Patience | Side | Fill rate | Spread saved | Adverse selection | Net |
|---|---|---:|---:|---:|---:|
| 1 min | buy | 47.5% | +$10.03 | **−$83.98** | **−$73.95** |
| 1 min | sell | 41.8% | +$9.82 | **−$85.56** | **−$75.75** |
| 5 min | buy | 76.4% | +$9.84 | −$86.24 | −$76.40 |
| 5 min | sell | 67.9% | +$9.62 | −$60.34 | −$50.72 |

**Resting at the midpoint saves about $10 of spread and costs about $80 in selection.** You are filled
when the market is leaving, and not filled when it is not. Waiting longer raises the fill rate — 41.8% to
67.9% for a seller — and does not repair the selection.

The $3.08 execution assumption is not merely difficult. It is **measured to be unavailable to anyone who
posts and waits**. Fee-only fills belong to whoever is quoting, not to whoever is resting.

## So what would a loop actually be searching for?

Given the above, a signal loop inside the current charter is searching for a rule with **57.89% true
directional accuracy** in an instrument where the cost of trading exceeds the mispricing by three to five
times. I do not think that search is worth the compute, and I would rather say so than run it and report
whichever of five thousand rules looked best.

**The loop machinery is built, correct and ready** — hash-chained ledger, per-experiment bar, an
`exhausted` flag and an `experiments_remaining()` count that answers "how long may this run?" before it
starts. It should be pointed at a question where the answer could change. On this evidence there are three,
and all three are charter amendments rather than searches:

1. **Sell instead of buy.** Every positive number measured all session is on the short side. This is the
   single largest change available and it needs one clause struck.
2. **Trade a structure whose cost is not the full spread.** A vertical spread pays a narrower net spread
   than two outright legs and caps the 529%-of-premium tail that makes short straddles unfundable. Barred.
3. **Quote rather than take.** The entire seller edge — 0.55% of premium — is smaller than the 1.66% the
   spread pays whoever is on the other side. This is a different business, not a different model.

## What is now measured rather than assumed

- The round trip: **$23.00**, from 1.9 million observations, against a carried $25.
- Its shape through the day: **1.44% of mid at the open, 9.04% at 15:00**.
- Passive fills: **41.8%–76.4%** achievable, at **−$60 to −$86** of adverse selection against **+$10** of
  spread saved.
- The loop's own economics: **1.66 accuracy points from experiment 1 to 5,000**.

## Honest limits

- The passive-fill test is a **proxy**. Minute bars cannot show an order resting inside a minute. It
  ignores queue position, which makes the fill rate optimistic, and it misses fills that reverse inside one
  minute, which makes it pessimistic.
- Adverse selection is measured at the end of the patience window rather than at the instant of fill.
- It is measured **per leg**. A short straddle's two legs are adversely selected in opposite directions, so
  the pair would suffer less than twice the single-leg figure. That effect is real, it is not measured
  here, and it is the one place this finding could be too harsh.
- The quote corpus covers 251 sessions from 2025-08 to 2026-07. Spreads in 2022 were not observed.
