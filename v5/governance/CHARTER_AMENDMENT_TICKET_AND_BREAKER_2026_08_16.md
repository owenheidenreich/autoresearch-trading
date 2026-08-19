# Charter amendment — per-trade risk cap, declared stop, and the daily breaker

**Status: SIGNED 2026-08-16 by the repository owner.** The instruction, given in conversation after
the draft was read back in full including §6's ruin table: *"please sign and authorize. we need to
allow the model to have this freedom to choose more contracts so it can have more room for its edge
discovery. yes its a bet it has an edge. but we will fine tune an entry and exit model so it doesnt
take massive whopping 50% losses."* Drafted 2026-08-16 by Claude Opus 5 on owner instruction, under the amendment procedure
established by
[`CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md`](CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md)
§"Amendment procedure". That document's signed bytes are not edited; the clauses below supersede it.

## 1. One sentence for the owner

The bot may risk up to **$2,000 of premium per trade** instead of $500, because the cheap tickets the
old cap forced it into pay **10.8% of premium in friction against 1.4%** at $1,000–2,000 — but the
daily breaker must rise to **20%** at the same time, because a 5% breaker and a $2,000 ticket cannot
both hold, and the measured survival evidence says a bigger ticket is only safe if the model actually
has an edge.

## 2. Clauses superseded

| Superseded | Source | Replaced by |
|---|---|---|
| "premium up to **13% of session-starting equity**" | 2026-08-13 amendment §2 | §3 below: a **$2,000 absolute dollar** cap |
| "**5% daily circuit breaker** on session-starting equity" | 2026-08-13 amendment, "What does not change" | §5 below: **20%** of session-starting equity |
| Max loss charged as **premium paid** | job-46 work-packet risk law (`$500` ticket-plus-fee rule) | §4 below: a **declared, enforced stop**, checked against realised losses |

Everything else in the 2026-08-13 amendment stands, including the `moneyness_band` declaration, one
contract per trade, no martingale, the 50% survival floor, long-premium-only, and the SPY floor.

## 3. Per-trade cap: $2,000, stated in dollars

**Becomes:** one contract, **entry premium plus fees at most $2,000**, as an absolute dollar figure.

It is deliberately **not** a percentage of equity. A percentage drifts upward as the account grows and
would eventually authorise the deep-ITM tickets the owner has explicitly refused; measured on the
owned chain, the whole-ladder tail reaches **$17,000 per contract**. A dollar cap cannot drift.

Two ceilings therefore bind together, and both must hold:

1. **$2,000 absolute premium ceiling**, unchanged by account growth; and
2. the existing **`moneyness_band` bar on deep ITM**, retained in full.

Raising the ceiling above $2,000 requires a further amendment, not a recalculation.

## 4. Max loss is a declared stop, not the premium paid

**Was:** the risk law charged every ticket its full premium, i.e. −100%.

**Becomes:** the risk law charges a **declared exit level**, which must be enforced in the simulator
and verified against realised outcomes afterwards.

The old assumption is measured to be wrong about the typical trade. Over 32,976 quote-priced trades on
a 60-minute frame:

| Drawdown reaches | Share of trades |
|---|---:|
| ≥ 99% of premium | **0.34%** |
| ≥ 95% | 2.27% |
| ≥ 90% | **4.32%** |
| ≥ 75% | 15.25% |
| ≥ 50% | 43.28% |

The mean realised loss on losing trades is **47.5% of premium**, not 100%. Charging every ticket the
catastrophic case overstates the typical loss roughly twofold, and that overstatement is what forced
the bot into the cheap tail.

**Three conditions bind this clause, and they are the reason it is safe to relax:**

1. **The stop must be enforced, not assumed.** Any stop used to justify a ticket size must be
   implemented in the simulator and fire in the replay. A risk law resting on an exit the simulator
   does not take is not a risk law.
2. **The realised worst loss must be reported and checked.** Every packet must state the worst loss
   actually taken. If it exceeds the declared maximum, the strategy fails the risk check regardless of
   its P&L — a policy that only looked safe because it got lucky must not pass.
3. **A stop slips, and the slippage is measured.** With a −50% stop declared, the worst realised loss
   at a $2,000 cap was **−$1,473**, not −$1,000: the fill comes at the next available bid, which can
   gap past the level. The 2026-08-13 amendment measured the same effect at −30% (mean fill −33.9%,
   worst −70%). The declared stop is a control, never a guarantee, and the ceiling must be sized to
   survive the slip.

## 5. The daily breaker rises to 20%, resolving the row-21 conflict

STATUS row 21 has carried an unresolved conflict since 2026-08-13: the 13%-of-equity ticket and the 5%
daily breaker cannot both hold, because one losing trade trips the breaker and ends the session. This
amendment rules on it explicitly.

**Becomes:** a **20% daily circuit breaker** on session-starting equity.

The arithmetic: at a $2,000 cap with a declared −50% stop, one maximum-size loss is about $1,000, so
two are about $2,000 — 20% of a $10,000 account. A breaker below that would end the session on the
first loss, which is not a circuit breaker but a one-trade-a-day rule wearing its name.

Measured breaker behaviour at a $2,000 cap (20,000 simulated years, 252 sessions, two trades a
session, resampling measured dollar outcomes):

| Daily breaker | Share of sessions tripping it |
|---|---:|
| 5% (current) | **10.06%** — fails the declared 5% tolerance |
| 10% | 3.44% |
| 15% | 1.18% |
| **20% (this amendment)** | **0.41%** |

At 20% the trade cap becomes the binding constraint in the normal case and the breaker binds on
slippage and on the rare deep loss. That is intentional: a breaker that fires on ordinary trading has
become the strategy rather than a control on it.

## 6. What the measurement actually says, including against this amendment

**The friction case for a bigger ticket is confirmed.** Median round trip as a share of premium, over
933,198 two-sided candidates in the tradeable OTM band:

| Ticket | Candidates | Median round trip |
|---|---:|---:|
| under $200 | 241,498 | **10.77%** |
| $200–500 | 248,638 | 3.63% |
| $500–1,000 | 261,511 | 2.29% |
| $1,000–2,000 | 161,912 | **1.37%** |
| over $2,000 | 19,639 | 1.03% |

**The cap opens the chain**, measured over the same population:

| Cap | Share of OTM-band candidates eligible | Share of the whole chain |
|---|---:|---:|
| $500 | 52.52% | 26.93% |
| $1,000 | 80.55% | 46.69% |
| $1,500 | 93.08% | 63.20% |
| **$2,000** | **97.90%** | 77.55% |

**Three figures in the amendment request did not reproduce and are not relied on here.** The request
cited 8.89% eligible at $500, 49.55% at $1,000, 79.75% at $1,500, 92.99% at $2,000, and a 43.63%
share in the $1,000–2,000 band. Measured on the owned ladder those are 52.52 / 80.55 / 93.08 / 97.90%
with a 17.40% band share (OTM band), or 26.93 / 46.69 / 63.20 / 77.55% (whole chain). The cited
figures may come from a population this draft could not identify. **The direction of the argument
survives on either population and the conclusion is unchanged**, but the specific percentages above
are the ones this amendment stands on. The friction claim traces cleanly to the STATUS §5 moneyness
table (deep-OTM $35 premium against a $9 round trip, 25.7%), which is a different population from the
band measured here and is not in conflict with it.

### The trade-off, stated against this amendment

A larger ticket lowers the friction bar and raises the loss per mistake. Measured on the same
20,000-simulated-year machinery, using **random-entry outcomes — that is, a policy with no edge**:

| Cap | Ruin (years breaching the 50% floor) | Median year-end equity |
|---|---:|---:|
| $500 | 0.0% | 0.78x |
| $1,000 | 0.0% | 0.67x |
| $1,500 | 1.9% | 0.57x |
| **$2,000** | **84.6%** | 0.48x |

**This is the honest warning and it must not be read away.** At $2,000, an edgeless policy destroys
the account in roughly five years out of six. Two things follow, and both belong in the record:

- **No cap makes a negative-expectancy strategy safe.** At $500 the same policy still bleeds to 0.78x
  a year; the cap changes the speed and the variance, not the sign. The protection in this amendment
  comes from §4's enforced stop and realised-loss check and from §5's coherent breaker — not from a
  small ticket.
- **The $2,000 cap is a bet that the model has an edge.** If it does not, the cap accelerates the
  loss. That is why §9's review condition is written against realised results rather than against a
  date.

A more conservative reading of this same table would set the cap at **$1,500**, which still opens
93.08% of the band and holds ruin under 2%. The owner asked for $2,000 and the request is recorded as
made; §9 is the mechanism that catches it if $2,000 proves to be the wrong call.

## 7. The account is serial and compounding — and was not

The charter declares one $10,000 account carried across sessions. **It was not implemented that way.**
Found while verifying this amendment: `v5/ops/causal_day_simulator.py` opened **every** session with
`cash = STARTING_EQUITY_USD` and measured the daily breaker against that same constant, so the account
silently re-seeded $10,000 daily and the breaker never moved with real equity. No multi-session walk
existed at all.

Repaired 2026-08-16, before any corpus is built on it:

- `simulate_session` takes `starting_equity_usd` and opens the session with it;
- all three daily-breaker sites now measure against session-starting equity;
- `simulate_serial_account` walks sessions chronologically, carries ending cash forward, refuses a
  replay that opens at anything other than the carried equity, stops at the survival floor, and
  carries realised P&L through a blocked terminal rather than restoring the account; and
- eight regression tests in `v5/tests/test_serial_account.py` fail if the account ever re-seeds again.

Every prior figure computed per session is unaffected — a single session opening at $10,000 is
correct. What was wrong was the absence of compounding across sessions, which no result to date had
claimed.

## 8. What does not change

- One contract per trade. No martingale, no doubling.
- The `moneyness_band` declaration and its bar on deep ITM.
- The 50% survival floor and the SPY floor.
- Long premium only; no selling, spreads, or overnight holds.
- Account starts at **$10,000**, serial and compounding across sessions.
- The **$75 data ceiling** and every other rail of `DEVELOPMENT_CHARTER_2026_08.md` §4.
- Win rate remains a diagnostic, never an objective.

## 9. Review condition

An amendment without an exit is refused, so this one carries three:

1. **It expires at $25,000 account equity.** At that point $2,000 is 8% of equity and the sizing
   question should be re-asked with real results rather than resampled ones.
2. **It expires if the realised worst loss exceeds the declared maximum** in any evaluated packet.
   That is a failed risk check under §4.2 and returns the cap to $1,000 pending a fresh amendment.
3. **It expires if the first full evaluation shows no edge.** The survival table in §6 makes the
   $2,000 cap conditional on an edge existing; an edgeless result withdraws its justification, and the
   cap returns to $1,000 rather than being retested at size.

## 10. Alpha charge and signature

Per procedure item 4, an amendment that widens what the bot may trade **spends alpha**. This one
widens the action space from 52.52% to 97.90% of the tradeable band and must be charged as one
declared hypothesis against the job-46 experiment ledger before any fit runs under it.

Evidence: [`ticket_risk_law_2026_08_16/receipt.json`](../../v4/audit/autoresearch/ticket_risk_law_2026_08_16/receipt.json),
produced by [`ops/measure_ticket_risk_law.py`](../ops/measure_ticket_risk_law.py); loss profile and
survival tables computed from the owned quote corpus with
[`ops/check_occupancy_risk.py`](../ops/check_occupancy_risk.py)'s existing machinery, parameterised
rather than modified.

- Drafted **2026-08-16** by Claude Opus 5, on owner instruction. No fit, mask, or corpus build had
  occurred at signature time.
- **Signed: repository owner, 2026-08-16.** The owner accepted §6's warning explicitly rather than
  by omission — "yes its a bet it has an edge" — and stated the intended mitigation: the entry and
  exit models are to be tuned so the policy does not take the large losses the cap now permits.
  **That intent is binding through §4:** the declared stop must be enforced in the simulator and the
  realised worst loss reported and checked, so "we will tune it not to" is verified after the fact
  rather than assumed. The stop level itself is set per fit declaration and is not fixed here; the
  −50% used in §4 and §6 is the measurement's illustration, not the declared policy stop.
- **Review conditions in §9 remain live and were not waived by this signature.**
