# The first signal test on the option corpus — and a look-ahead filter I built into everything before it

**2026-08-13.** Owner asked for the recommended tests to be run. Two results came back: a clean negative
from the first directional screen this project has ever run on option data, and a defect in my own
measurement chain that makes every number I reported earlier today too optimistic.

[Screen receipt](../../../v4/audit/autoresearch/intraday_direction_screen_2026_08_13/receipt.json) ·
[code](../../ops/screen_intraday_direction.py) · [tests](../../tests/test_intraday_screen.py) ·
[corrected occupancy](../../../v4/audit/autoresearch/hold_occupancy_2026_08_13/receipt.json)

## One sentence

Seven declared rules, two holds, 1,045 sessions: **every one loses money and none clears zero** — and the
reason is not that direction is unpredictable but that **long 0DTE premium is a bet on magnitude, not on
direction**, which is a different question from the one this project has been asking for months.

## The defect first, because it changes numbers I gave earlier today

Every module I wrote this session located the underlying by taking the strike where call and put prices are
closest. That quantises the underlying to the **five-point strike grid**. Each module then skipped any slot
where that quantised spot was unchanged between entry and exit.

**That is a look-ahead filter.** Nothing at the entry minute says how far the underlying will travel, so a
population defined by having travelled is not one the bot could have selected. Measured exactly, by
replicating the filter and scoring both sides of it (60-minute hold, near ATM, $25 round trip):

| Population | Slots | Share | Net per right call | Net per wrong call | Break-even | Net/trade |
|---|---:|---:|---:|---:|---:|---:|
| **Every slot** | 6,241 | 100% | $492 | $571 | **53.71%** | **−$39.4** |
| Kept by my modules | 5,077 | 81.3% | $628 | $639 | 50.44% | −$5.6 |
| **Silently discarded** | 1,164 | **18.7%** | −$99 | $275 | unwinnable | **−$186.9** |

The discarded slots are pure losers, because a slot where the underlying went nowhere is exactly where a
long option decays. Removing them moved the break-even **3.27 accuracy points** in the flattering
direction.

This is the same class of error as the one Phase 0 found and corrected this morning — a filter that
conditions on the outcome — and I introduced it while correcting that one. The estimator is now put/call
parity, `S = K + C − P`, averaged over strikes near the money, which is continuous; no slot is dropped for
a small move; and there is a regression test that a half-point move survives.

### What that does to what I told you earlier

| | I reported | Corrected |
|---|---:|---:|
| 60-min break-even, near ATM | 50.43% | **53.69%** |
| 60-min accuracy a screen could prove | 52.28% | **55.36%** |
| 15-min break-even | 51.01% | **55.77%** |
| Trades per session at 5 minutes | 36.1 | **75.2** |

The corrected occupancy table, on 1,045 sessions:

| Hold | Trades/session | Break-even | Provable at |
|---:|---:|---:|---:|
| 5 min | 75.2 | 58.82% | 59.30% |
| 10 min | 37.6 | 56.51% | 57.17% |
| 15 min | 24.7 | 55.77% | 56.59% |
| 30 min | 11.9 | 54.49% | 55.69% |
| **60 min** | 5.9 | **53.69%** | **55.36%** |

Three things follow, and I got all three wrong earlier today:

1. **The bar did not fall.** I said the 60-minute cell needed 52.28% rather than the 57.11% the project
   believed. It needs **55.36%**. The project's original understanding was close to right and my
   correction of it was the artifact.
2. **Occupancy is not neutral, it is harmful.** Trading every five minutes needs **59.30%** against
   **55.36%** for hourly holds — a 3.94-point penalty, monotone in occupancy. The fixed round trip is
   amortised over a smaller payoff and more decay is paid per unit of movement. The job-24 plan wanted to
   raise occupancy; the honest reading is the opposite.
3. **The ES screen is not the harder one.** That claim used the contaminated option payoffs. Corrected, the
   option corpus needs 53.69% at 60 minutes against an ES break-even of 51.40%, so the option layer is
   harder and the [training-precondition draft](../../governance/TRAINING_PRECONDITION_AMENDMENT_2026_08_13.md)
   loses its central argument. It should not be signed as written.

## The screen: seven declared rules, scored whole

Declared and hashed before running (`433136d73ef8…`): seven closed-form entry rules on causal features —
five, fifteen and thirty-minute returns and position in the session range — at two holds, family size 14,
Bonferroni z 2.690, session-block bootstrap. Nothing fitted, no threshold searched, every member reported.

60-minute hold, 1,045 sessions, $25 round trip:

| Rule | Trades | Stood down | Accuracy | Net/trade | Lower bound | Clears zero |
|---|---:|---:|---:|---:|---:|---|
| breakout_with | 1,734 | 66.6% | **53.52%** | −$22.7 | −$68.8 | no |
| confirmed_momentum | 3,191 | 38.6% | 51.61% | −$20.2 | −$59.8 | no |
| fast_momentum_with | 5,194 | 0.0% | 51.37% | −$24.3 | −$51.8 | no |
| momentum_with | 5,194 | 0.0% | 50.17% | −$38.6 | −$67.3 | no |
| momentum_against | 5,194 | 0.0% | 49.83% | −$38.9 | −$67.9 | no |
| fast_momentum_against | 5,194 | 0.0% | 48.63% | −$53.2 | −$81.3 | no |
| contrarian_stretch | 1,734 | 66.6% | 46.48% | −$74.0 | −$124.1 | no |

The 15-minute results are the same shape and are in the receipt. **Nothing clears zero.** The test harness
carries its own null and recovery controls: a rule with no information does not clear the bar, a planted
62% edge is found, and the mirror of a winner loses.

## The result that actually matters

**`breakout_with` calls direction correctly 53.52% of the time — comfortably above the 53.69% break-even
region — and still loses $22.70 a trade.** That is not a rounding problem. It falsifies the assumption
every accuracy bar in this project rests on, declared on 2026-08-12 as "the magnitude of the move is
independent of whether the call was right".

It is not independent, and it runs against us. Conditional payoffs by rule at 60 minutes:

| Rule | Accuracy | Paid when right | Lost when wrong | Break-even it actually faces |
|---|---:|---:|---:|---:|
| unconditional | 50.00% | $449 | $527 | 53.97% |
| breakout_with | 53.52% | **$406** | $516 | **55.98%** |
| confirmed_momentum | 51.61% | $466 | $539 | 53.62% |

A breakout rule is right more often **and paid less for it**, because the option is more expensive exactly
where the move looks likely. Its own break-even rises faster than its accuracy does.

## What long 0DTE premium actually is

Break-even by how far the underlying moved during the hold, 60 minutes, near ATM, $25 cost:

| Move size | Share of slots | Break-even | Net/trade |
|---|---:|---:|---:|
| 0–2 points | 16% | unwinnable | −$181 |
| 2–5 points | 23% | 95.91% | −$159 |
| 5–10 points | 27% | 64.86% | −$108 |
| **10–20 points** | 22% | **49.54%** | **+$6.5** |
| **20+ points** | 12% | **35.53%** | **+$453** |

The median hour moves **6.79 points** and loses money. **Only the top third of hours by movement is
profitable at all, and the top eighth carries essentially all of it.**

This is the answer to "people trade 0DTE all the time". Long premium is not a direction product; it is a
**magnitude** product with a direction filter attached. A signal worth having predicts *when the underlying
is about to move a lot* — and if it has that, it barely needs to call direction, because the 20+ point
bucket breaks even at 35.53%. Every screen this project has run, including today's, has asked the other
question.

## What this suggests testing next, and what it does not

**Genuinely new:** a magnitude or volatility signal — does anything causal at the entry minute predict that
the next hour's range will be large? That is a different label, a different family, and it has never been
tested here. Its payoff structure is far more forgiving than direction's.

**Not new, and not worth repeating:** more directional rules on this corpus, other holds on the same ladder,
other moneyness bands, or a learned ranker on the same direction label. The label is the problem, not the
rule set.

**Also worth saying plainly:** the charter forbids the two things that make the small-move majority
tradeable — selling premium and spreads. On this evidence that constraint, not the account and not the
statistics, is what makes the instrument hard. That is an owner decision and this finding does not make it.

## What is now known versus assumed

**Known:** the conditional payoff structure of near-ATM 0DTE across 1,045 sessions, uncontaminated; that
seven declared directional rules lose money on it; that payoff depends steeply on realised movement.

**Still untested:** any magnitude signal; any selective policy on a magnitude label; what a real order
policy costs; anything the charter bars.

## Honest limits

- The underlying is a parity estimate from traded option prices, not an SPX print. It is now continuous and
  agrees with the strike grid where both can be read, but it is still an estimate.
- Prices are last trades, so they carry bid/ask bounce.
- The $25 round trip is carried from the owned quote corpus.
- The seven rules are a small family chosen for being closed-form and symmetric. A negative on them is a
  negative on *them*, not on directional prediction in general.
- The screen read economics on the option corpus for the first time. That spends the corpus for this
  family; the ledger row records what may not be re-run.
