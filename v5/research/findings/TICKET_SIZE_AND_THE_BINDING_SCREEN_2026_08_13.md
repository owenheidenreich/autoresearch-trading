# The charter is not what is stopping us. The account is — and the wrong screen is.

> **SUPERSEDED IN ITS NUMBERS 2026-08-13.** Every payoff below was measured on a slot population that silently dropped 18.7% of slots — the ones where the underlying barely moved, averaging **-$186.90 per trade**. That was a look-ahead filter I introduced, and it moved break-even 3.27 accuracy points in the flattering direction. Corrected figures and the mechanism: [direction screen and the look-ahead](DIRECTION_SCREEN_AND_A_LOOKAHEAD_I_INTRODUCED_2026_08_13.md). The *comparative* conclusions here survive where both arms were filtered identically; every absolute level does not.

**Measured 2026-08-13**, on 1,045 usable 0DTE sessions, to answer one question: would changing the charter
improve the chance of a profitable model?

[Ticket surface](../../../v4/audit/autoresearch/ticket_size_surface_2026_08_13/receipt.json) ·
[charter grid](../../../v4/audit/autoresearch/charter_settings_2026_08_13/receipt.json) ·
[screen requirement](../../../v4/audit/autoresearch/screen_requirement_2026_08_13/receipt.json) ·
code: [ticket](../../ops/measure_ticket_size_surface.py), [grid](../../ops/search_charter_settings.py),
[screen](../../ops/derive_screen_requirement.py)

## Two sentences

**Loosening the risk limits does not help.** A wider daily breaker buys occupancy and nothing else; what
decides whether the account survives is how large one ticket is against it, and every way of shrinking the
ticket raises the accuracy the model must reach, because the round trip goes from 1.6% of a $1,600 contract
to 27% of a $50 one.

**But one rule is costing us for no reason.** Model training is blocked until the ES direction screen
passes, and that screen is now measurably **harder** than a direct screen on the option corpus — 52.50% to
55.12% against 51.83% to 52.52% — so the block is holding the project to a test tougher than the one that
actually matters.

## Test 1 — what a cheaper ticket costs

At every slot and on each side, the contract whose entry premium is closest to a declared target, priced
under the settled exit-price convention and charged the round trip its own moneyness band was measured to
carry. 60-minute hold:

| Ticket | Share of a $10k account | Round trip as share of premium | Where it lands | Break-even | Provable at |
|---:|---:|---:|---|---:|---:|
| $50 | 0.5% | **26.6%** | deep OTM | 69.09% | 70.80% |
| $100 | 1.0% | 15.2% | OTM | 57.65% | 59.48% |
| $200 | 2.0% | 8.9% | OTM | 53.22% | 55.06% |
| $400 | 4.0% | 5.2% | near ATM | 51.38% | 53.23% |
| **$800** | 8.0% | 2.9% | near ATM | **50.69%** | **52.54%** |
| **$1,600** | 15.8% | 1.6% | near ATM | **50.59%** | **52.44%** |
| $3,200 | 30.2% | 1.9% | ITM | 52.52% | 54.36% |

**There is no cheap ticket that is also winnable.** The cost of trading is roughly fixed per contract, so
halving the premium nearly doubles the cost as a share of it, and the accuracy required climbs steeply:
$800 needs 52.54%, $200 needs 55.06%, $50 needs 70.80%. The charter's existing judgement that deep OTM is
unwinnable is confirmed and extended — it is not a cliff at the bottom of the ladder, it is a slope all the
way down.

The ladder also turns back up at the top: $3,200 lands in the money, where the dollar spread grows faster
than the payoff. **The optimum is $800–$1,600 of premium, near the money**, which is where the signed
charter amendment already put it.

## Test 2 — every combination of ticket size and daily breaker

Seven ticket sizes against six breaker levels from the signed 5% to 25%, five holds, at break-even
accuracy and at the accuracy a screen could prove. 420 declared cells, enumerated in advance.

**Widening the breaker does not fix anything.** It buys occupancy — that is arithmetic, since the breaker
is the thing cutting occupancy — but it leaves the survival floor where it was, because ruin is driven by
how much of the account rides on each trade, not by when the session stops. At a $10,000 account, 60-minute
hold, at the provable accuracy, the share of years that breach the 50% floor:

| Ticket | 5% breaker | 10% | 25% |
|---:|---:|---:|---:|
| $200 | 37.7% | 37.4% | 37.5% |
| $400 | 52.7% | 56.1% | 56.2% |
| $800 | 69.2% | 73.7% | 75.4% |

Wider is flat or slightly worse. **The breaker is not the binding clause and should not be amended.**

## The number that actually decides it: account size

Same cells, sweeping the account instead. Share of years breaching the survival floor, 60-minute hold, at
the provable accuracy:

| Ticket | Accuracy needed | $10,000 | $25,000 | $50,000 | $100,000 |
|---:|---:|---:|---:|---:|---:|
| $100 | 59.48% | 18.9% | 0.1% | 0.0% | 0.0% |
| $200 | 55.06% | 37.7% | 2.5% | 0.0% | 0.0% |
| $400 | 53.23% | 52.7% | 10.2% | 0.3% | 0.0% |
| $800 | 52.54% | 69.2% | 24.5% | 2.9% | 0.0% |
| $1,600 | 52.44% | 84.4% | 41.8% | 10.4% | 0.4% |

Read it diagonally and the trade-off is exact. **The bigger the account, the bigger the ticket it can carry;
the bigger the ticket, the lower the accuracy the model has to reach.** The lowest bar this instrument
offers is about **52.4%**, and collecting it requires roughly a **$100,000** account. At $10,000 the best
survivable choice is a $100–$200 ticket, which asks for **55.1% to 59.5%** — a bar this project has never
come close to.

**That is the whole answer to "would changing the charter help".** The charter's limits are not
mis-calibrated; they are correctly protecting an account that is too small for the instrument. The one
clause that *is* mis-calibrated is the 13% ticket ceiling, which permits a size that ruins 84% of
simulated years at $10,000.

## Test 3 — the screen the training block is calibrated to

Training is prohibited until the ES direction screen passes, at a bar of **65.73%**. That number was derived
on 2026-08-12 from **251 option sessions**, one trade each, charged the **fee-only $3.08** rather than the
$25 round trip. Both inputs have since moved: the option corpus is **909 sessions** with five to thirty-six
trades in each, and the payoff pair is measured on the settled convention at full cost.

Recomputing the requirement on both candidate corpora:

| Horizon | Option corpus break-even | Provable at | ES corpus break-even | Provable at | Harder on |
|---:|---:|---:|---:|---:|---|
| 5 min | 51.88% | 52.52% | 54.83% | 55.12% | ES |
| **10 min** | 51.01% | **51.83%** | 53.45% | 53.86% | ES |
| 15 min | 51.01% | 51.99% | 52.84% | 53.36% | ES |
| 30 min | 50.69% | 52.05% | 52.03% | 52.80% | ES |
| 60 min | 50.43% | 52.28% | 51.40% | 52.50% | ES |

**The ES screen is harder than the option screen at every horizon.** It was the easier one when the option
corpus held 251 sessions and ES held 2,435; buying 909 option sessions for $0.00 reversed that, and nothing
in the gate chain noticed. Keeping "the ES screen must pass first" now means holding the project to a test
0.2 to 2.6 accuracy points tougher than the one the option layer actually needs.

The legacy bar of 65.73% is not wrong for what it was computed on. It is simply about a corpus we no longer
have to use.

## What I recommend changing, and what I recommend leaving alone

**Leave alone:** the 5% daily breaker, the 50% survival floor, one contract per trade, long premium only,
the near-ATM band. The grid says none of them is the binding constraint and the breaker in particular
cannot be traded for anything.

**Change:** the 13% ticket ceiling, which is measurably too large; the absence of any minimum-account rule,
which is what actually protects the survival floor; and the training precondition, which points at the
harder of two screens. Drafts, unsigned:
[ticket and account](../../governance/CHARTER_AMENDMENT_TICKET_AND_ACCOUNT_2026_08_13.md),
[training precondition](../../governance/TRAINING_PRECONDITION_AMENDMENT_2026_08_13.md).

## A defect found and repaired during this work

The risk simulations resampled payoffs from a hundred-point quantile grid. A draw from such a grid can
never exceed the 99.5th percentile, so its mean is a trimmed mean — and option payoffs carry a large part
of their expected value in the right tail. Measured, this understated the mean winning trade by **6% to
19%** while leaving the bounded losing side accurate to within 1%: a systematic bias against the strategy
in every risk number produced before it was found.

It is replaced by equal-probability strata, whose resampled mean is exact by construction, with regression
tests. The Phase-2 risk finding has been corrected in place and the correction is recorded there rather
than quietly applied — the earlier claim that 96–100% of years breached the survival floor at $10,000 was
wrong; the true figure under that model is 0.0%.

## What this is not evidence of

That any edge exists. No policy is computed, no model fitted, no threshold searched. Every figure describes
the instrument and the arithmetic of trading it.

## Honest limits

- **The round trip is carried per band from the owned quote corpus** and applied to 2022–2025 sessions
  whose spreads were never observed. It is the single most load-bearing carried number here: the entire
  cheap-ticket result is a statement about cost as a share of premium.
- **The simulations resample trades independently within a session**, so a day that trends against every
  position is under-represented and real ruin is at least as high as reported.
- **The ES side of Test 3 uses move dispersion measured on 247 owned sessions**, scaled by the measured
  ten-year to owned-year ratio at 60 minutes. That scaling is an approximation applied to the other
  horizons, and it is the weakest input in that table.
- **420 cells were enumerated.** Choosing a setting from this surface spends multiplicity, and any setting
  carried into a screen must be declared before the screen runs and charged against the alpha ledger.
