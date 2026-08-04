# Wave 3 — Delta-Banded Universe + Confirmatory 10:30 Test — PRE-REGISTRATION (2026-08-04)

**Frozen before execution.** Nothing below may be edited after the run starts; amendments are dated
appendices that leave the original intact. Codex execution goal is §7.

---

## 1. What changed, and why this is not another sweep

**The owner identified a confound in the candidate universe.** The entry universe was filtered to a
**$3–8 premium band**. Premium is not an instrument property — it mixes moneyness, time-to-expiry, and
volatility. Holding *price* constant across the day therefore silently changes *what is bought*:

| Hour (ET) | avg premium | distance from ATM | \|delta\| |
|---|---|---|---|
| 10am | $609 | **10.9 pts** | **0.34** |
| 1pm | $557 | 7.1 pts | 0.39 |
| 3pm | $537 | **4.6 pts** | **0.48** |

At ~constant price the filter buys something 11 points OTM at delta 0.34 in the morning and 4.6 points OTM
at delta 0.48 in the afternoon. **Those are different trades wearing the same price tag**, and the model
could never learn "buy this moneyness at this hour" because a price filter that cannot see the clock had
already decided for it.

**Disambiguation performed before freezing this document.** Because the confound makes time-of-day a proxy
for moneyness, the previously observed 10:30–10:59 ET effect had to be checked against delta. Gross % of
premium, by half-hour × delta band:

| | <.25 | .25–.35 | .35–.45 | .45–.55 | >.55 |
|---|---|---|---|---|---|
| **10:30** | **+8.05** | **+1.12** | **+1.24** | **+0.77** | **+10.38** |
| all other half-hours | mostly negative | negative | negative | negative | mixed |

**10:30 is positive in all five delta bands**, so it is a time effect, not a strike-selection artifact.
Delta alone carries nothing (−2.45 / −2.25 / −2.26 / −2.10 across the main bands).

## 2. Honest status of the 10:30 hypothesis — read before interpreting any result

The 10:30 window came from an **11-way half-hour scan**. This wave tests it as **one pre-specified
window**, and:

- **No adjacent windows.** No 10:00, no 11:00, no re-scanning. Testing neighbours re-opens the multiplicity
  the pre-specification is meant to close.
- **The 55-cell table above is not a menu.** The *row* being uniformly positive is the robust observation.
  Individual cells (+10.38%) are noise-prone and may not be selected.
- **This is a quasi-replication, not a confirmation.** The sessions are the same 166 OOF sessions that
  generated the hypothesis; only the *contract universe* changes. Session-level regime is shared. The
  protected holdout is SPENT and may not be opened.

**Therefore the highest claim available from this wave is "worth a forward live-paper test", never
"confirmed".**

## 3. The change under test

**Stop selecting on price.** Rebuild the candidate universe on **delta bands**, with premium demoted from
filter to feature:

- Selection axis: `|delta|` bands, so moneyness is explicit and held constant by construction.
- **Premium becomes an observed feature**, not a filter.
- **Add the greeks and the clock the model never had**: delta, gamma, theta, vega, minutes-to-expiry.
  Every feature must be causal at the **t−60s** clock and live-twin available on Databento OPRA.

Stacked friction reductions, both already measured:
- **Passive entry**, one-tick-penetration model: **+$12.57/trade** (the honest figure, not the optimistic
  offer-touch +$22.76).
- ~~Owner's real fee $0.65/side rather than the frozen law's $1.50: +$1.70/round trip.~~ **WITHDRAWN.**
  **Corrected 2026-08-04 by measurement.** `$0.65/side` is the **IBKR fixed commission line item only**, not the all-in cost. Per `PROTOCOL101_STAGE1_OBJECTIVE_AND_GATES_PROPOSAL.md` (2026-07-19) the all-in is IBKR $0.65 + CBOE SPXW proprietary ~$0.70/side + regulatory ~$0.05-0.10/side. A guarded paper round trip on 2026-08-04 measured **$1.54/side = $3.08 round trip** (avgCost 81.54028 on a 0.80 fill; RealizedPnL -3.08 on a price-flat round trip). **The frozen `FILL_LAW` at $1.50/side is very nearly correct and slightly UNDERcharges.** Fees are not a source of conservatism; the $10.00 tick-through is the only real one. There is **no fee saving**. The only friction reduction available is passive entry.

Passive entry alone takes round-trip friction from **$26.48 to $13.91** ($26.48 − $12.57). That figure
is unaffected by the withdrawn fee claim, because it was measured on the entry price rather than on fees.
The frozen `FILL_LAW` is **not** modified; the passive model is a separate, explicitly labelled
counterfactual. Downstream breakevens in §4 were computed at ~$14 and therefore still stand.

## 4. The charter risk ceiling — this bounds the experiment

The hard commitments retained under Charter Amendment 1 cap how far premium can usefully go. On a $10,000
account the 5% daily breaker is $500, and **56.3% of 60-minute holds lose more than 25%** (measured):

| Premium | routine 25% loss | vs $500 breaker | share of account |
|---|---|---|---|
| $1,000 | $250 | within | 10% |
| **$1,500** | **$375** | **within** | 15% |
| $2,000 | $500 | **TRIPS** | 20% |
| $2,500 | $625 | **TRIPS** | 25% |

**Premium is capped at $2,000 for this wave**, and $1,500 is the comfortable working ceiling. This matters:
the arithmetic that made higher premium attractive (+$1.90/trade at $1,500, +$12.50 at $2,500) runs into
the charter's own risk limits right where it starts to pay. The viable window is roughly **$1,500–2,000** —
real, but tight, and it must not be widened to make a result work.

## 5. Frozen hypotheses — budget 4, and the declared count IS the maxT family

| ID | Hypothesis | Horizon |
|---|---|---|
| W3-H1 | Delta-banded universe, **10:30–10:59 ET only**, passive entry, premium ≤ $2,000 | 60m |
| W3-H2 | Same as H1 | 25m (the original reference hold) |
| W3-H3 | Delta-banded universe, **all day** (isolates whether 10:30 is doing the work) | 60m |
| W3-H4 | Original $3–8 price band, 10:30 only (isolates whether the **wider universe** is doing the work) | 60m |

H3 and H4 are the attribution arms. Without them a positive H1 cannot be attributed to either the window or
the universe change.

**Comparators:** not trading ($0.00); all-day passive on the same universe; the frozen-law $3–8 band at
10:30.

**Negative controls, every hypothesis:** sign-reversed, session-shuffled, constant. **Any accepted control
makes the wave `INVALID` and the results are discarded.**

## 6. THE GATE

`TIER_A` requires ALL of:

1. Net PnL > best comparator **and** > $0;
2. Positive paired delta in ≥4 of 5 outer folds;
3. One-sided 95% session-bootstrap LCB > 0;
4. Family-adjusted maxT p ≤ 0.05 across all 4 declared members;
5. ≥100 filled serial trades across ≥30 sessions and all five folds;
6. All four rejection tests pass (`v4/research/pathd_model_gate.py`);
7. All negative controls fail;
8. No single session or weekday supplies >50% of positive delta;
9. Causal-clock, mutate-future, identity and reproducibility checks pass.

The four charter diagnostics are **reported, not gated** (Charter Amendment 1).

Anything else is `NO_SIGNAL`. **Do not widen the premium cap, add windows, or extend the budget to reach a
pass.**

**Claude's prior, recorded in advance:** `NO_SIGNAL` is more likely than not. The gross drift needed is
+1.06% and the friction is ~2.5% of a $565 premium; the case rests on premium scaling that the charter caps
at $2,000. Recording this so a surprise must be argued for.

---

## 7. Codex execution goal

1. **Prior-art check first** (`prior_art_check` in `v4/research/pathd_research_loop.py`) for every
   mechanism. A do-not-retest row or a rejecting protocol verdict is a STOP.
2. **Rebuild the candidate universe on delta bands** from raw OPRA on the SSD. Verify the confound is gone:
   at a fixed delta band, mean |delta| must be stable across hours (it currently drifts 0.34 → 0.48).
3. **Materialize features** including premium, delta, gamma, theta, vega, minutes-to-expiry — all causal at
   t−60s. Run `label_balance` **before** any fit.
4. **Run the four hypotheses** through `run_wave` with budget 4, registry on the SSD.
5. **Score** with `acceptance_tier` plus all four rejection tests and all four charter diagnostics.
6. **Report** per §6, append any falsification to the do-not-retest ledger, update the roadmap, and end
   with `STOP_FOR_CLAUDE_VERIFICATION`.

**Environment:** `AR_TRADING_DATA_ROOT` / `AR_TRADING_SCRATCH_ROOT` = `/Volumes/AR_TRADING_DATA`,
`AR_TRADING_ARTIFACT_ROOT` = `/Volumes/AR_TRADING_DATA/artifacts`. Prefix long runs with `caffeinate -dimsu`;
`storage-preflight` before and after each long stage.

**Hard stops:** protected 36-session firewall SPENT — never reopen, `holdout_open_count` stays 0. Do not
modify `FILL_LAW`, the causal t−60s clock, the label law, or the OOF firewall. No paid data, no broker,
no paper submit, no promotion or default change, no runtime/launchd edits. No reward-hacking; `NO_SIGNAL`
is a successful outcome.

*Signed: Claude Opus 5 — 2026-08-04 — FROZEN before execution.*
