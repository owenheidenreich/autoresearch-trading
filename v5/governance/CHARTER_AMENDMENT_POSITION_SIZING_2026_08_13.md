# Charter amendment: position sizing and the contract universe

**Status: SIGNED AND IN FORCE — owner signed 2026-08-13.**

Amends [`PROTOCOL101_TRADER_CHARTER.md`](../../v4/docs/protocol101/training/contracts/PROTOCOL101_TRADER_CHARTER.md).
The charter itself lives in frozen v4 and is not edited; this document supersedes the clauses it names.

## One sentence for the owner

The bot may buy a contract worth up to about 13% of the account instead of about 3%, because the measured
loss after an early exit is **4.29% of the account** rather than the 13% the ticket price suggests — but
the amendment also bounds the contract universe on **both** sides, because the cheap end of the ladder is
not merely risky, it is unwinnable.

## What changes

### 1. The contract universe is declared by moneyness, not by price

**Was:** an implicit `$3.00–$8.00` price filter, never written down as a decision.
**Becomes:** an explicit `moneyness_band` declaration, registered as a searchable knob.

The price filter was a proxy that happened to select the worst tradeable part of the ladder. Measured
2026-08-13 over 232 sessions and 82,709 contract observations, at measured per-contract spreads
([receipt](../../v4/audit/autoresearch/option_payoff_by_moneyness_2026_08_13/receipt.json)):

| Band | Premium | Round trip | Break-even accuracy |
|---|---:|---:|---:|
| deep OTM | $35 | $9 | **impossible (181%)** |
| OTM — what `$3–8` selects | $333 | $12 | 54.50% |
| **near ATM** | $1,945 | $25 | **50.60%** |
| ITM | $6,778 | $84 | 52.59% |
| deep ITM | $19,865 | $369 | 60.17% |

**The band is bounded on both sides.** Deep OTM is barred outright: being right about direction earns +$4
while being wrong costs −$21, so no exit discipline and no directional skill can rescue it. Deep ITM is
barred by affordability and by its own widening spread. This is the "smart trades, not dumb expensive
ones" requirement made mechanical: expensive is not the goal, and cheap is not safety.

### 2. Position size rises to a measured ceiling

**Was:** single contract, implicitly ~3% of a $10k account through the price filter.
**Becomes:** single contract, **premium up to 13% of session-starting equity**.

The justification is not the ticket price but the loss actually taken after an early exit. Measured over
1,394 contracts averaging $1,280 premium, entered 09:35 and held to 10:35
([receipt](../../v4/audit/autoresearch/stop_effectiveness_2026_08_13/receipt.json)):

| | Mean account hit when the direction call was wrong |
|---|---:|
| No exit discipline | **7.55%** |
| Exit at −30% of premium | **4.29%** |

The declared stop is **not outrun**: the mean realised fill is −33.9% against a declared −30%, so 3.9
points of slippage, not a collapse. The worst observed fill was −70%, which is the tail this ceiling
accepts rather than denies.

### 3. What the exit discipline is for, stated honestly

The exit model is a **risk control, not a return engine**, and the charter should not imply otherwise.
Measured mean outcome was **+1.30%** with no stop and **+1.52%** at −30% — but **+0.99%** at −40% and
**+1.66%** at −50%. That ordering is non-monotonic, which means the differences are noise. The exit
earns its place by halving the loss when wrong, and by nothing else yet demonstrated.

Two consequences the charter must carry:

- **A −30% exit fires on 70.3% of trades at this premium.** It is the normal exit, not an emergency brake.
  Any exit model trained here is choosing when to *stay*, not when to flee.
- **It costs about one winner in five.** 19.9% of stopped paths would have recovered to profit. That is
  the price of the risk reduction and it must appear in every packet rather than being netted away.

## What does not change

- **One contract per trade. No martingale, no doubling, no exceptions.** Unchanged and not proposed for change.
- **Survival floor:** the account never falls below half its starting capital.
- **5% daily circuit breaker** on session-starting equity.
- **Long premium only.** No selling premium, no spreads, no overnight holds. Style drift remains an
  owner-level decision.
- **Win rate is a diagnostic, never an objective.**
- **Floor is SPY.** Tier 0 remains: underperform SPY and stop.

## The four-bucket distribution has to be re-derived

The charter's scratch band is **−5% to +5%**, set when its own text records friction as "4.68% of premium
— almost exactly one scratch-band width". At near-ATM premiums the measured round trip is **$25 on $1,945,
which is 1.3%**. A band calibrated to one friction width no longer describes one friction width.

**This amendment does not set a new band.** It records that the old one is no longer meaningful and that
the band must be re-derived in friction-multiples before any candidate is judged against it.

## Review condition, because this is a hypothesis and not a conclusion

The owner's stated intent is to let a model "take some risks at the beginning" while the account is small,
and for this to matter less as the account grows. That makes the amendment provisional by design:

1. **It expires at $25,000 account equity**, at which point 13% is $3,250 and the sizing question should be
   re-asked rather than inherited.
2. **It is void if the measured account hit when wrong exceeds 6.0%** on any twenty-session window of real
   or shadow trading — the halfway point between the 4.29% measured here and the 7.55% unprotected figure.
3. **It is void if the exit fires on more than 85% of trades**, which would mean the exit is not selecting
   at all.

## What this amendment is not evidence of

That any edge exists. Every figure here characterises the **instrument** and the **cost of a declared exit
rule**. None of it evaluates a policy, and none of it was produced by a search. The gate chain is unchanged
and G1 has not passed.

## Amendment procedure, established here

The charter has had no defined way to change, which is why a price filter became binding law without ever
being a decision. From this document onward:

1. An amendment is a dated file under `v5/governance/`, naming the clauses it supersedes.
2. It states its evidence, or states that it has none.
3. It carries a review or expiry condition. An amendment with no exit is refused, on the same principle
   that refuses a divergence axis with no settling condition.
4. **An amendment that widens what the bot may trade spends alpha** from the autoresearch ledger, because
   widening the search space is a hypothesis like any other.
5. The owner signs. Unsigned amendments are drafts and bind nothing.

## Signature

- Drafted **2026-08-13** by Claude Opus 5, on owner instruction.
- **Signed: repository owner, 2026-08-13.** Instruction given in conversation:
  *"lets change the charter"*, then *"proceed"* after the amendment was read back in full,
  including the measured evidence, the three review conditions, and the two clauses that
  decline to claim more than the data supports.

*Drafting this contacted nothing, traded nothing, and changed no runtime state.*
