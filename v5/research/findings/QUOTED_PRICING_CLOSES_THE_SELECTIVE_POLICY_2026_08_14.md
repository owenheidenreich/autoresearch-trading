# Priced at the touch, the edge is zero

**2026-08-14.** The follow-up the previous finding asked for, run on owner authorization.
[quote-trained receipt](../../../v4/audit/autoresearch/selective_policy_quoted_2026_08_14/receipt.json) ·
[out-of-time receipt](../../../v4/audit/autoresearch/policy_out_of_time_2026_08_14/receipt.json) ·
code: [dataset](../../ops/build_quoted_dataset.py), [out-of-time scorer](../../ops/score_policy_out_of_time.py) ·
[tests](../../tests/test_quoted_dataset.py)

## What this changes

**The selective policy has no timing edge. Measured with the spread removed entirely it is worth
−$3.5 per trade, which is zero.** The +$20.9 it showed on last-trade bars was the tape, exactly as
suspected, and this settles it rather than arguing it. The class is closed on measurement, not on
reasoning.

## What was done

The previous finding showed that a model free to pick its own minute learns to buy prints that landed
on the bid, and that its profit equalled the price distortion it selected to within $1.50. The remedy
named there was to price the same policy where the spread is **charged** rather than inferred.

The owned quote corpus carries bid, ask and size every minute for **251 sessions**, 2025-08-01 to
2026-07-31. Rebuilt on it: **707,171 candidate trades**, entry charged at the **ask**, exit paid at
the **bid**, and only the measured $3.08 of fees added on top — charging a modelled round trip too
would bill the spread twice. Features are read from the **mid**, which is the honest split: a trader
observes the market at the midpoint and transacts at the touch. Greeks are recomputed from the mid
rather than taken from the vendor, because a vendor greek is a train/live divergence.

The feature set is byte-identical to the trade-corpus run, so the only thing that changed is the
price source. Volume was joined from the trade corpus for the same sessions; a contract that printed
nothing in a minute traded zero, which is the true value rather than a gap.

## Two runs, one answer

**Trained and scored on the quote corpus**, the policy is negative at every operating point and
**loses to its own matched control** — −$123.2/trade against −$37.2 at the most selective setting. Its
remaining hit-rate advantage over random selection (45% against 33%) is entirely instrument choice:
the composition-matched control reaches the same 45–48%.

That leaves one fair objection — 151 scored sessions is far less training data than the 745 the trade
corpus gave it, so perhaps the signal is real but unlearnable here. So the second run removes it.

**Trained on 794 trade-corpus sessions (2022-06-01 to 2025-07-31) and scored out-of-time on all 251
quote sessions (2025-08-01 to 2026-07-31).** The training window ends before the scoring window
begins, and the scoring window prices at the touch. Full training data, honest prices.

| Target | Mid-to-mid | Spread | Fees | Net |
|---:|---:|---:|---:|---:|
| 0.35 | −$8.2 | −$47.7 | −$3.08 | −$56.8 |
| 0.50 | −$2.5 | −$48.1 | −$3.08 | −$51.6 |
| 0.55 | −$0.6 | −$48.7 | −$3.08 | −$52.0 |
| 0.60 | −$0.0 | −$50.4 | −$3.08 | −$54.3 |
| 0.70 | +$3.6 | −$50.6 | −$3.08 | −$51.3 |

**The mid-to-mid column is the finding.** That is the policy's edge with the spread removed
completely — a world in which trading is free. It averages **−$3.5 per trade**. Not reduced by the
spread: absent before the spread is charged at all.

The rest is arithmetic. The spread on the contracts it chose is **$48.70, or 1.21% of premium**, and
that is what turns zero into −$52.

## Why the earlier result looked so strong

Everything the project knows how to check, it passed. Chronological validation. Causality tests that
mutate the future and assert no feature moves. Thresholds taken only from the training fold. A
shuffled-label null. A composition-matched control it beat by $68/trade. Four of four out-of-sample
years positive. One position at a time.

**All of that was true and none of it was relevant, because the contamination was in the price.** The
put/call-parity residual on last-trade bars has a standard deviation of $102.60 against a $19.79
round trip — the noise in the entry price is five times the toll — and a model with freedom to choose
its minute finds that before it finds anything about the market.

The general lesson is worth more than the result: **a control tests the mechanism it was aimed at.**
Validation splits test leakage across time. Null labels test whether the target carries information.
Matched controls test composition. None of them looks at whether the *price* is real, so a defect
there passes every one of them at once, and the more controls pass the more convincing the artifact
becomes.

## What is now closed, and what is not

**Closed:** selective entry models fitted or scored on `ohlcv-1m` last-trade prices where the model
may choose its own entry minute or contract. The measurement cannot support them at any effect size
smaller than the print noise, and that noise is five times the cost bar.

**Closed:** this policy, on the long side, at these horizons, on the owned quote corpus. Its edge
before costs is zero, so no improvement in execution rescues it. There is nothing to rescue.

**Not closed by this:** the exit. Every number here uses a fixed 30-minute clock and no exit model has
ever been fitted on a quote-priced stream. The exit was solved once, on prints, and reached gross ≈ $0
— which now has to be read as a measurement made in the same contaminated units as this one.

**Not closed by this:** the short side and defined-risk structures, where the measured premium accrues
rather than drains, both charter-barred and both owner decisions.

## Honest limits

- The quote corpus is 251 sessions against the trade corpus's 1,045, and covers one year rather than
  four. The out-of-time run addresses the training-data objection but the *scoring* sample is still
  one year of market conditions.
- Exit at the bid is the worst case for a taker and matches the charter's aggressive execution. A
  policy resting passively would pay less spread and suffer adverse selection instead, which was
  separately measured at −$74 to −$76 and is worse.
- The policy chose contracts averaging $3,983 of premium. Even at zero cost that is 40% of a $10,000
  account and outside the charter's ticket ceiling.
- Bid and ask are the top of book. Whether size was available at those prices for a real order is not
  established by this corpus.
