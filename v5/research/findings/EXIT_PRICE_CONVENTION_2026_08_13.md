# What a 0DTE contract is worth when it stops trading — and why the answer was backwards

> **SUPERSEDED IN ITS NUMBERS 2026-08-13.** Every payoff below was measured on a slot population that silently dropped 18.7% of slots — the ones where the underlying barely moved, averaging **-$186.90 per trade**. That was a look-ahead filter I introduced, and it moved break-even 3.27 accuracy points in the flattering direction. Corrected figures and the mechanism: [direction screen and the look-ahead](DIRECTION_SCREEN_AND_A_LOOKAHEAD_I_INTRODUCED_2026_08_13.md). The *comparative* conclusions here survive where both arms were filtered identically; every absolute level does not.

**Phase 0 of job 24, measured 2026-08-13.**
[Receipt](../../../v4/audit/autoresearch/exit_price_convention_2026_08_13/receipt.json) ·
[code](../../ops/resolve_exit_price_convention.py) · [tests](../../tests/test_exit_price_convention.py)

## One sentence

The contradiction is settled in favour of the **optimistic** number — the pooled near-ATM 60-minute
break-even is **50.94%, not 54.24%** — and the reasoning that made the pessimistic number look plausible
was exactly wrong: the contracts that stop printing are not dying, they are **winning**.

## What was in dispute

Two measurements of the same quantity on the same corpus and the same moneyness band disagreed by 3.37
accuracy points. The only difference was what happens to a contract that has no trade print at the exit
minute:

| Route | Handling | Break-even |
|---|---|---:|
| [payoff by era](../../../v4/audit/autoresearch/option_payoff_by_era_2026_08_13/receipt.json) | inner-join entry to exit, so the contract is **dropped** | 54.24% |
| the exit-rule study | keep it at its **last print** | 50.87% |

The plan that raised this
([job 24](../../work/option-approach/PLAN.md)) said dropping them "removes trades that likely went to
zero", which would make the *low* number the flattering one. Everything in this project's recent history
supported that suspicion: every first number produced had turned out to be the most optimistic available.

**That reasoning was wrong, and the measurement says so in four independent ways.**

## What was measured

One code path now produces both disputed numbers, so the difference cannot be a second implementation
detail hiding behind the first. 909 sessions, 19,502 near-ATM entries at 09:35 held to 10:35, $25
round trip carried from the owned quote corpus.

| Convention | Entries | Net when right | Net when wrong | Break-even |
|---|---:|---:|---:|---:|
| drop the vanished contracts | 18,178 | +$726 | −$859 | **54.21%** |
| value them at their last print | 19,502 | +$826 | −$857 | **50.94%** |
| mark them to zero | 19,502 | +$302 | −$880 | 74.46% |
| mark them to exercise value | 19,502 | +$775 | −$864 | 52.71% |

Both disputed figures reproduce to within 0.03 and 0.07 points. The small residual is a half-open bucket
edge: the era study's band was `(−25, +25]` and this one is `[−25, +25]`, so this measurement carries one
extra strike level. Nothing else differs.

## Why dropping them is wrong — the causal objection

**A contract that stops printing cannot be identified at the entry minute.** Nothing at 09:35 says which
contracts will still be trading at 10:35, so a population defined by that fact is not one the bot could
have selected. Dropping them is a look-ahead filter regardless of which direction it biases, and it is
therefore inadmissible before its number is even read.

That argument alone settles the choice between the two. What follows settles the *magnitude*, and it
matters because it points the opposite way from what the plan assumed.

## Why dropping them is wrong — the contracts are winners

Across all 909 sessions the vanishing contracts are, on average:

- **91.9% on the correct side**, against 47.0% for the contracts that keep printing.
- Last seen **+55.4% above their entry price**, median +47.1%.
- Entered at **$2,659** of premium against $1,573 for the rest, and at a **wider** mean absolute moneyness
  (17.1 against 13.2) — that is, already in the money on entry.
- **None** of them stopped printing at the entry minute. Every one traded at least once more.

The mechanism is ordinary and, once stated, obvious: a near-ATM 0DTE contract that goes sharply into the
money becomes expensive and its spread widens, so it prints in fewer minutes. The cheap losers keep
printing constantly. **Illiquidity at this horizon is a symptom of winning, not of dying.**

This is not a recent-liquidity artifact. It holds in every year of the corpus, and the gap between the two
conventions is 2.6 to 5.1 points in every year:

| Year | Sessions | Entries | Vanished | Of those, on the right side | Last print vs entry | Drop | Last print |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 134 | 2,769 | 12.8% | 90.7% | +52% | 59.11% | 54.00% |
| 2023 | 203 | 4,308 | 7.1% | 89.5% | +44% | 55.26% | 52.14% |
| 2024 | 217 | 4,643 | 7.0% | 89.8% | +48% | 52.03% | 49.04% |
| 2025 | 215 | 4,708 | 5.2% | 96.7% | +68% | 54.47% | 50.88% |
| 2026 | 140 | 3,074 | 3.1% | 99.0% | +98% | 52.17% | 49.56% |

## The settlement: the price that actually existed

The profile above says the vanishing contracts are winners. It does not prove that a *stale* price is a
fair stand-in for the price at the exit minute. That needs the one thing a trade-only corpus cannot supply,
and the owned quote corpus has it: a bid and an ask for **every** contract at **every** minute, whether or
not it traded.

The two corpora overlap on 230 sessions. On that overlap, every convention was recomputed on one identical
population, and then one more — identical to the last-print convention in every respect **except** that the
vanishing contracts are valued at the quote that really existed:

| | Break-even |
|---|---:|
| drop the vanished contracts | 52.06% |
| value them at their last print | **49.6244%** |
| **value them at their real quote, everything else unchanged** | **49.6246%** |
| every contract at its quote mid | 49.98% |
| every contract at the quote bid | 50.57% |

**Correcting the stale prices to the real ones moves the break-even by 0.0002 accuracy points.** On the
161 vanished contracts the quote corpus could price:

- assumed value from the last print, **$5,229.12**; actual quoted value, **$5,229.01**. Mean error
  **−$0.11** on a $5,229 position, or 0.002%.
- **Not one was worthless.** Zero of 161 quoted below $0.05.
- Their real value averaged **+86.6% above entry**, confirming the profile on prices rather than on prints.

The last-print convention is not merely the causal one. It is **numerically almost exact**.

## What this changes

1. **The plan's stated failure condition did not occur.** Job 24 pre-committed that "Phase 0 settling at
   54.24%" would push every provable-accuracy figure up three points and put the 15-minute cell back where
   the project has already failed twice. It settled at 50.94%, so that branch is closed and the plan
   proceeds on its own terms.
2. **Every break-even in the plan's occupancy table was computed under the wrong convention** and is two to
   five points too pessimistic. Rebuilding it lowered every bar by 1 to 5 points **and destroyed the
   plan's central claim**, because the drop convention taxed long holds about twice as hard as short ones
   and that differential tax was the entire apparent occupancy gain. See
   [occupancy and charter risk](OCCUPANCY_AND_CHARTER_RISK_2026_08_13.md).
3. **The near-ATM band's advantage over the `$3–8` price filter is larger than recorded**, because the
   50.60% figure in the signed [charter amendment](../../governance/CHARTER_AMENDMENT_POSITION_SIZING_2026_08_13.md)
   came from the quote corpus, where this convention question does not arise. The quote-based and
   trade-based routes now agree to within half a point, which is a cross-corpus confirmation the project
   did not previously have.

## What this is not evidence of

That any edge exists. Every figure here characterises the **instrument** — what a correct or incorrect
direction call is worth in near-ATM 0DTE premium. No policy is computed, no rule is fitted, no threshold is
searched, and the gate chain is unchanged.

## Honest limits

- **The quote settlement reaches only 230 of 909 sessions**, all from 2025-08 onward, and only 161 vanished
  contracts within them. Vanishing is four times more common in 2022 (12.8%) than in 2026 (3.1%), so the
  years where the convention matters most are the years the settlement cannot reach directly. The
  year-by-year profile is what carries the conclusion back to them, and it is a profile of prints rather
  than of quotes.
- **The $25 round trip is carried, not measured here.** It was measured on the owned quote corpus at
  near-ATM and is applied to 2022–2025 sessions whose spreads were not observed.
- **Prices are trade prints, not mids**, so they carry bid/ask bounce that a mid does not. The settlement
  bounds this: the all-quote-mid break-even is 49.98% against the trade-print 49.62% on the same
  population, a difference of 0.36 points.
- **Entry and exit are bar labels.** A bar labelled 09:35 covers 09:35:00–09:36:00, so a fill is assumed
  somewhere inside that minute. This is the project's existing fill law and is unchanged.

## Reopening condition

This is settled unless a corpus arrives that carries **quotes** for the pre-2025 years, which would let the
settlement be measured directly where it is currently carried by the year-by-year profile. Re-running the
same measurement on the same evidence is not new.
