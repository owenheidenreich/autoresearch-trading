# Charter amendment: the ticket ceiling, and the account size that makes the survival floor real

**Status: DRAFT. NOT SIGNED. Binds nothing.**
Proposed 2026-08-13. Amends [`CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md`](CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md),
signed earlier the same day, under the amendment procedure that document established.

## One sentence for the owner

The 13% ticket you signed this morning is too large — at a $10,000 account it breaches the survival floor
in **84% of simulated years** even when the strategy is working — and the fix is not to loosen the daily
breaker, which measurement shows buys nothing, but to cut the ticket and to write down the account size
below which this instrument cannot be traded at all.

## Why this amendment exists so soon after the last one

The signed amendment raised the ceiling to 13% on a sound argument: a −30% exit takes 4.29% of the account
rather than 13%, so the ticket is survivable. That argument was about **one trade**. It was never checked
against a **year** of them, and a year is what the survival floor is a rule about.

Nothing in the earlier evidence was wrong. This amendment adds the missing dimension.

## 1. The ticket ceiling falls from 13% to 4% of session-starting equity

**Was:** single contract, premium up to 13% of session-starting equity.
**Becomes:** single contract, premium up to **4%** of session-starting equity.

Measured over 20,000 simulated years per cell on 1,045 sessions of real payoffs, 60-minute hold, with the
strategy assumed to reach exactly the accuracy a screen on this corpus could prove
([receipt](../../v4/audit/autoresearch/charter_settings_2026_08_13/receipt.json)):

| Ticket as share of a $10,000 account | Years breaching the 50% survival floor |
|---:|---:|
| 15.8% — roughly what 13% permits | **84.4%** |
| 8.0% | 69.2% |
| **4.0%** | **52.7%** |
| 2.0% | 37.7% |
| 1.0% | 18.9% |

4% is not chosen because it is safe at $10,000 — it is not, and clause 3 says so. It is chosen because it
is the largest ticket at which the loss from a single wrong call, after the declared exit, stays inside the
daily breaker with room for a second trade. At 13% the first losing trade ends the session, which is why
the occupancy measurement found every hold length collapsing to about one trade a day.

## 2. The declared exit becomes mandatory, not advisory

**Was:** the −30% exit is described as the normal exit and is the basis of the sizing case.
**Becomes:** whenever the ticket exceeds **2%** of session-starting equity, the −30% exit is **required**.

The sizing evidence does not survive without it: the measured account hit when wrong is **4.29%** with the
exit and **7.55%** without, and only the first fits inside a 5% daily breaker. A ceiling justified by an
exit that the bot is not obliged to use is not a ceiling.

**Recorded honestly:** this project's attempt to re-measure the exit on the trade corpus produced a result
it could not defend — a break-even of 48.14%, which would make a coin flip profitable. Two candidate
artifacts were tested and ruled out and the mechanism remains unproven, so **nothing in this amendment
credits the exit with improving returns**. It is required here only for the loss-capping effect measured on
the owned quote corpus, which is the reliable part.

## 3. A minimum account size, which the charter has never had

**New clause.** The bot may not trade an account below the size at which its declared ticket keeps the
modelled one-year probability of breaching the survival floor at **zero**, measured on the same machinery
before the account is funded.

For the current near-ATM design, measured:

| Declared ticket | Accuracy the model must reach | Smallest account with zero modelled breaches |
|---:|---:|---:|
| $100 | 59.48% | $25,000 |
| $200 | 55.06% | $50,000 |
| $400 | 53.23% | $50,000 |
| $800 | 52.54% | $100,000 |
| $1,600 | 52.44% | $100,000 |

**This is the clause that matters most and the one the owner will like least.** The 50% survival floor has
been in the charter from the beginning as an absolute rule. Measurement now says that at $10,000 no
tradeable configuration honours it: the cheap tickets that would be survivable demand 55% to 59% accuracy,
which is far beyond anything this project has demonstrated, and the tickets whose accuracy bar is reachable
breach the floor in half to five-sixths of years.

Either the floor binds and the account must be larger, or the floor is a target rather than a rule and
should say so. **That is an owner decision and this draft does not make it.** The clause above assumes the
floor binds, because that is what the signed charter says.

## 4. Occupancy is stated as measured, and is not to be bought with risk

**New clause.** The trades-per-session a design assumes must be the measured figure, not `tradeable
minutes / hold`. Measured on 909 sessions: **4.9** at 60 minutes, 8.8 at 30, 16.0 at 15, 22.0 at 10, 36.1
at 5 — against theoretical maxima of 6, 12, 25, 38 and 77.

And the reason this clause exists: across that whole range the accuracy a screen could prove spans **0.69
points**, from 51.83% at ten minutes to 52.52% at five. Occupancy is nearly worthless here, so it must
never be used to justify a larger ticket, a wider breaker, or a shorter hold.

## What does not change

The **5% daily circuit breaker**, the **50% survival floor**, one contract per trade, long premium only,
no overnight holds, the near-ATM moneyness band, and Tier-0 SPY. The breaker was tested at 5, 8, 10, 15, 20
and 25 percent: widening it buys occupancy and leaves the survival floor unchanged to within noise, so
there is nothing to gain by trading it away.

## Review and expiry conditions

1. **Void if the measured account hit when wrong exceeds 6.0%** on any twenty-session window — carried
   unchanged from the signed amendment.
2. **Void if the exit fires on more than 85% of trades** — carried unchanged.
3. **The 4% ceiling and the minimum-account table expire when the round trip is re-measured** on a corpus
   that carries quotes for the traded era. Every figure here rests on per-band costs carried from
   2025–2026 evidence, and the cheap-ticket result is entirely a statement about cost as a share of
   premium.
4. **Re-derive at $100,000 of equity**, where a $1,600 ticket becomes 1.6% and the whole table shifts.

## What this amendment is not evidence of

That any edge exists. Every figure characterises the instrument and the arithmetic of trading it. No policy
was evaluated, no model fitted, no threshold searched.

## Alpha

Choosing settings from a 420-cell enumerated grid spends multiplicity. This draft declares the grid, its
size, and the criteria, all fixed before the run. Any setting carried into a screen must be declared before
that screen runs and charged against the alpha ledger.
