> **The owner was right about the gap.** Every screen before this one tested
> *buying*. Testing one side of a rule and calling the result "no edge" is an
> omission, not a conclusion. This finding fixes it: both sides of every declared
> rule, plus a fitted model free to buy, sell or stand aside.

# A model that can sell as well as buy — and what it found

> **SCOPE CORRECTED 2026-08-13.** This work prices a **fixed-horizon** policy — buy at one minute, sell at a fixed minute later. That is sound for that policy and **does not transfer to a scalp with a dynamic exit**, which lives on the path rather than the terminal value. Measured afterwards: a near-ATM contract reaches a median **+28.2%** within thirty minutes and selling at the best minute close would earn **+$321/trade** against **-$32** for holding to a clock. See [the exit is the strategy](THE_EXIT_IS_THE_STRATEGY_2026_08_13.md).

**2026-08-13.** [Model receipt](../../../v4/audit/autoresearch/two_sided_model_2026_08_13/receipt.json) ·
[code](../../ops/train_two_sided_model.py) · [tests](../../tests/test_two_sided_model.py)

## One paragraph

Scoring the short side changed the picture immediately: two declared rules turn
positive at the full aggressive round trip, and selling straddles in the last
ninety minutes keeps **+$23.50 a trade** where the long side lost. But it does
not survive scrutiny. Corrected for the thirty-two cells actually inspected, its
lower bound is **−$50.90**; it is **negative in two of five years**; its worst
single trade lost **529% of the premium collected**; and a gradient-boosted model
given twelve causal features and free choice of side produced an out-of-fold
prediction/actual correlation of **+0.03**, with the shuffled-label control
outscoring it at fifteen minutes. **The short side is where the edge is, and it
is still too small to survive the spread.**

## What scoring both sides revealed

Same declared rules, same slots, now scored long *and* short. Sixty-minute hold,
895 sessions after warm-up:

| Rule | Trades | Straddle gross | Long @ $25 | **Short @ $25** | Short @ $3.08 |
|---|---:|---:|---:|---:|---:|
| trade everything | 4,448 | −$28.1 | −$78.1 | −$21.9 | +$22.0 |
| **late_session** | 887 | **−$82.3** | −$132.3 | **+$32.3** | **+$76.1** |
| **cheap_vs_recent** | 1,575 | −$55.3 | −$105.3 | **+$5.3** | +$49.2 |
| range_edge | 1,494 | −$49.6 | −$99.6 | −$0.4 | +$43.4 |
| dear_vs_recent | 1,374 | +$4.3 | −$45.7 | −$54.3 | −$10.5 |

Two things worth noticing. `late_session` short is positive at the aggressive
cost, which no long-only scoring could have shown. And `dear_vs_recent` — buying
when the option looks *expensive* against recent movement — is the **best** long
cell, the opposite of the naive reading, because high recent realised movement
predicts *lower* future movement. The cheapness feature carries real information
with its sign inverted.

## Why the best cell still fails

The candidate is: sell the near-ATM straddle with ninety minutes or less to
close. Under proper scrutiny, correcting for the 8 rules × 2 sides × 2 holds =
**32 cells** actually inspected:

| Hold | Cost | Net/trade | Bonferroni 99.84% lower bound | Clears? |
|---|---:|---:|---:|---|
| 60 min | $25.00 | +$23.5 | **−$50.9** | no |
| 60 min | $14.00 | +$45.5 | −$28.9 | no |
| 60 min | $3.08 | +$67.4 | −$7.0 | no |
| 15 min | $25.00 | −$28.6 | −$40.3 | no |
| 15 min | $14.00 | −$6.6 | −$18.3 | no |
| **15 min** | **$3.08** | **+$15.2** | **+$3.5** | **yes** |

**Exactly one cell clears, and it requires fee-only execution** — four passive
fills, two legs in and two out, with no slippage. That is the market maker's
economics, not a taker's.

The sixty-minute version, which has the largest headline number, fires once per
session on a single slot (14:35), and:

- it is **negative in two of five years**: 2022 −$26, 2023 +$76, 2024 +$10,
  2025 +$68, 2026 −$68;
- its worst trade lost **$6,674 against a $1,326 mean premium — 529%**;
- 33% of its trades lose;
- 1,036 trades is one per session, so the sample is thin where it matters most.

## The fitted model

Twelve causal, scale-free features. Label: straddle profit and loss as a fraction
of its own premium. Gradient boosting, depth 3. **Chronological walk-forward** —
the first 300 sessions train the first model, each later fold of 100 sessions is
predicted by a model fitted only on strictly earlier sessions. The policy takes
the side of the predicted edge when that edge exceeds 1.5× the round trip it is
about to pay, and stands aside otherwise.

| Hold | Run | $/leg | Trades | Short | Net/trade | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| 15 min | model | 3.08 | 9,857 | 93% | −$0.3 | [−6.2, 5.2] |
| 15 min | **null: shuffled labels** | 3.08 | 11,062 | 94% | **+$4.5** | [−0.3, 9.4] |
| 15 min | model | 25.00 | 201 | 64% | −$50.9 | [−121.8, 22.0] |
| 60 min | model | 3.08 | 3,398 | 73% | +$5.8 | [−16.5, 28.7] |
| 60 min | null: shuffled labels | 3.08 | 3,409 | 70% | −$14.7 | [−44.1, 11.5] |
| 60 min | model | 25.00 | 1,465 | 75% | −$4.9 | [−41.9, 37.9] |

**Out-of-fold prediction/actual correlation: +0.0257 at fifteen minutes, +0.0318
at sixty.** Effectively zero.

**At fifteen minutes the shuffled-label null beats the real model.** That is the
clearest possible statement that the model learned nothing conditional. What it
did learn is to be short — 93% of trades — which simply harvests the
unconditional variance premium and is not a model at all.

Nothing clears zero, at any hold, at any cost. And the model **underperforms the
simple declared rule**: at sixty minutes and $25 it nets −$4.9 against
`late_session`'s +$23.5, because it spreads 1,465 trades across the session
instead of concentrating in the window where the premium actually lives.

The test harness carries its own controls, and they behave: a planted
relationship is recovered at correlation >0.5, shuffling destroys it, and the
first predicted fold cannot see a signal planted only in later sessions.

## What this changes, honestly

**It does not overturn the conclusion, and it materially strengthens the
evidence for it.** Before this, "no edge" rested on long-only tests, which was a
fair criticism. Now: both sides tested, a model free to choose either, proper
chronology, and two null controls. The answer is the same and the reasoning is
much harder to fault.

**It does move where the edge is.** Every positive number in this work is on the
short side. The charter permits only buying, and buying is measurably the wrong
side of a 0.55%-per-fifteen-minutes premium.

**It sharpens the binding constraint to one thing: fill quality.** The single
cell that clears does so at $3.08 a leg and fails at $14. The entire question is
now whether a real order policy pays $3, $14 or $25 — and that has never been
measured. It is worth more than every signal effect in this project combined.

## What I would test next, in order

1. **Fill quality on the owned quote corpus.** For 251 sessions we hold bid, ask
   and size every minute. What share of passive orders at the midpoint would
   have filled within thirty seconds, and at what adverse selection? This is
   measurable today, needs no purchase, and decides everything above.
2. **The short late-session cell as a declared hypothesis with its own
   known-answer campaign**, if and only if the charter is amended to permit
   selling and the tail is sized. It is the only candidate that has ever reached
   this point in this project.
3. **Defined-risk spreads**, which cut both the round trip and the 529% tail.
   Charter-barred and an owner decision.

## Honest limits

- The costs are carried from the owned quote corpus into 2022–2025 sessions
  whose spreads were never observed, and the whole conclusion is a ratio of
  spread to premium.
- The model is one architecture with one declared configuration. A different
  feature set could do better; nothing here proves no model can work, only that
  this one, given these features, found nothing beyond the unconditional premium.
- Short straddles are barred by the charter. Nothing here is a recommendation to
  trade them, and the tail is reported but not sized against the survival floor.
- The 32-cell Bonferroni charges the grid that was inspected, not the path of
  reasoning that produced the rules.
