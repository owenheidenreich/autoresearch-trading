# The definitive answer: the market charges half a percent, and we pay three

> **SCOPE CORRECTED 2026-08-13.** This work prices a **fixed-horizon** policy — buy at one minute, sell at a fixed minute later. That is sound for that policy and **does not transfer to a scalp with a dynamic exit**, which lives on the path rather than the terminal value. Measured afterwards: a near-ATM contract reaches a median **+28.2%** within thirty minutes and selling at the best minute close would earn **+$321/trade** against **-$32** for holding to a clock. See [the exit is the strategy](THE_EXIT_IS_THE_STRATEGY_2026_08_13.md).

**2026-08-13.** Five experiments, run in sequence, each one following from the last.
This closes the question the project has been asking since it began.

[Magnitude screen](../../../v4/audit/autoresearch/magnitude_screen_2026_08_13/receipt.json) ·
[variance premium](../../../v4/audit/autoresearch/variance_premium_2026_08_13/receipt.json) ·
[leg asymmetry](../../../v4/audit/autoresearch/leg_asymmetry_2026_08_13/receipt.json) ·
[direction screen](../../../v4/audit/autoresearch/intraday_direction_screen_2026_08_13/receipt.json) ·
code: [dataset](../../ops/build_magnitude_dataset.py), [magnitude](../../ops/screen_magnitude.py),
[premium](../../ops/measure_variance_premium.py), [asymmetry](../../ops/measure_leg_asymmetry.py)

## One paragraph

A near-ATM 0DTE option is priced above what the underlying actually delivers, by **0.55% of premium over
fifteen minutes** and **1.42% over an hour**. That is the seller's edge and it is statistically solid. The
cost of crossing the spread on both legs is **2.83% of premium**. The market's mispricing is real,
measurable, stable across every year of the corpus — and **about one fifth the size of the toll charged to
touch it**. Buying loses, selling loses, and the only thing that separates the profitable participants from
us is that they collect the spread rather than pay it.

## How the five experiments chain together

### 1. Direction is not predictable here — seven declared rules, all negative

Reported separately in [the direction screen finding](DIRECTION_SCREEN_AND_A_LOOKAHEAD_I_INTRODUCED_2026_08_13.md).
The useful part was not the null but the mechanism: `breakout_with` called direction correctly **53.52%**
of the time and still lost $22.70 a trade, because it was right precisely where the option was dearest.
That pointed at magnitude.

### 2. Magnitude *is* predictable — and it is priced to the penny

Eight declared rules, thresholds calibrated on prior sessions only, scored on the profit and loss of a real
position rather than on whether the range turned out large. Fifteen-minute hold, 895 sessions after
warm-up:

| Rule | Trades | Realised move | Straddle paid | Net/trade |
|---|---:|---:|---:|---:|
| baseline: trade everything | 18,692 | 4.67 pts | $1,707 | **−$29.3** |
| **high_recent_range** | 6,303 | **7.05 pts** | **$2,420** | **−$29.6** |
| low_recent_range | 6,805 | 2.99 pts | $1,178 | −$28.0 |

**The signal works.** Selecting on the prior thirty minutes' range raises the realised move by 51%. The
option price rises by 42%. The profit and loss moves by **thirty cents**. The same holds at sixty minutes:
move +47%, premium +41%, P&L identical to within a dollar.

Nothing in the family cleared zero, at either hold, at any Bonferroni-corrected bound. This is not a
failure to predict volatility. It is a demonstration that predicting it is worthless, because the price
already contains the prediction.

### 3. The variance premium, measured directly

Buying the straddle is a pure bet that the underlying travels further than the option charged, with no
model in between. Its gross profit and loss **is** the premium:

| Hold | Straddle premium | Gross P&L | As % of premium | 95% CI | Negative with confidence |
|---|---:|---:|---:|---:|---|
| 15 min | $1,766 | **−$9.7** | **−0.55%** | [−12.6, −6.4] | **yes** |
| 60 min | $1,937 | **−$27.5** | **−1.42%** | [−44.9, −10.3] | **yes** |

And it is concentrated in the last two hours, which is where the gamma is and where buyers pay up:

| Entry hour | 15-min gross as % of premium | 60-min |
|---|---:|---:|
| 10:00 | −0.34% | −0.89% |
| 12:00 | −0.25% | +0.86% |
| 14:00 | −1.06% | **−5.55%** |
| 15:00 | **−2.74%** | — |

### 4. What either side actually keeps

A straddle is two legs, so buyer and seller both pay two round trips.

| | 15-min | 60-min |
|---|---:|---:|
| Gross premium to the seller | +$9.7 (0.55%) | +$27.5 (1.42%) |
| Cost of two legs at $25 | −$50.00 (2.83%) | −$50.00 (2.58%) |
| **Buyer nets, aggressive** | **−$59.7** | **−$77.5** |
| **Seller nets, aggressive** | **−$40.3** | **−$22.5** |
| Buyer nets at fee-only $3.08 | −$15.9 | −$33.6 |
| **Seller nets at fee-only** | **+$3.5** | **+$21.3** |

**At retail execution both sides lose.** The spread is five times the mispricing at fifteen minutes and
1.8 times it at sixty. The seller only turns a profit at fee-only execution, which means resting passive
and accepting non-fills — which is to say, behaving like a market maker.

And the seller's tail is what the charter's survival floor exists for: at sixty minutes the 99th percentile
loss is **$2,234** against a mean gain of $27.50, and the worst single slot in 5,195 was **$22,878** —
**549% of the premium collected**. Roughly 830 average wins to pay for one bad hour.

### 5. No skew edge, and the premium is stable

If puts were dearer than their own realised downside, a long call would be a cheap synthetic long rather
than a volatility position, and the direction null would not apply to it. Measured:

| Hold | Call gross | Put gross | Call minus put | Distinguishable? |
|---|---:|---:|---:|---|
| 15 min | −0.30% | −0.79% | +$4.3, CI [−6.6, 14.4] | **no** |
| 60 min | −0.63% | −2.21% | +$15.2, CI [−29.4, 58.9] | **no** |

Puts lean dearer, as equity skew predicts, but not separably. There is no free synthetic here.

The fifteen-minute premium is present in **every year**, and shrinking:

| Year | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|
| Gross as % of premium | −0.77% | −0.57% | −0.54% | −0.46% | −0.44% |
| Negative with confidence | yes | yes | yes | yes | no |

A structural, decaying edge — consistent with a market that has become more competitive as 0DTE volume
grew. It is not a regime artifact, and it is not getting easier.

## The answer to "people trade 0DTE all the time"

They do, and this measurement says what separates them from this project:

- **They sell rather than buy.** Every number above says the seller has the edge. The charter permits only
  buying, and buying is the wrong side of a 0.55% premium.
- **They do not pay the spread.** The entire seller edge — 0.55% at fifteen minutes — sits inside a 2.83%
  round trip. Whoever collects that spread is the one being paid. A retail taker is on the wrong side of a
  toll five times larger than the prize.
- **They cap the tail.** A naked short straddle risks 549% of premium in an hour. Spreads exist for this,
  and the charter bars them too.

None of that is a criticism of the charter. It is a measurement of what the charter's three constraints —
long premium only, single contracts, aggressive execution — cost together. **They are the binding
constraint, not the account size, not the statistics, and not the absence of a model.**

## What would have to change for anything here to work

In descending order of how much they move the number, and every one is an owner decision:

1. **Sell premium instead of buying it.** Turns a −0.55% headwind into a +0.55% tailwind. Requires
   amending the charter's long-only clause and accepting a tail the survival floor currently forbids.
2. **Stop paying the spread.** The entire edge lives inside the bid-ask. Passive or midpoint execution is
   worth 2.5% of premium, which is five times any signal effect measured in this project. Nothing has ever
   been built to measure or optimise fill quality.
3. **Use defined-risk spreads.** Cuts both the cost and the tail. Barred by the charter.
4. **Concentrate in the last two hours**, where the premium is 2.74% at fifteen minutes rather than 0.25%
   at midday. This is the one change compatible with the current charter, and on its own it is not enough:
   a seller at 15:00 still needs execution better than $14 a leg.

## What is now definitively closed

**Buying near-ATM 0DTE premium on a serial clock, with or without a directional or magnitude signal, on
1,045 sessions.** The instrument is negative-expectancy before costs, the costs are five times the
mispricing, direction was not predictable by seven declared rules, and magnitude was predictable but fully
priced. No further rule on this label, this instrument and this side is worth testing.

## What is still genuinely open

- **Fill quality.** Everything rests on a $25 round trip carried from the quote corpus. Whether a real
  order policy pays $25, $14 or $6 has never been measured, and it is worth more than any signal here.
- **The short side and defined-risk structures**, both charter-barred and both where the measured edge
  actually is.
- **Event windows.** Every measurement here is a serial clock indifferent to the calendar. Whether pricing
  is less efficient around scheduled catalysts is untested, and it is the one place a buyer might still
  find something.

## Honest limits

- The underlying is a put/call parity estimate from traded option prices, not an SPX print.
- Prices are last trades, so they carry bid/ask bounce. That widens dispersion but does not bias the mean.
- The $25 and $3.08 costs are carried from the owned quote corpus into 2022–2025 sessions whose spreads
  were never observed. Since the central claim is a ratio of spread to mispricing, this is the most
  load-bearing carried number in the finding.
- The rule sets were chosen with earlier findings in view. Bonferroni charges the declared family, not the
  path that led to it.
- The hour and regime splits in experiments 3 and 5 are descriptive cuts, not declared rules, and the
  60-minute 12:00 cell being positive is exactly the kind of thing twelve buckets produce by chance.
