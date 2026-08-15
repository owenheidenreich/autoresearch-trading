# Rejected strategic packet — MES is outside the owner-approved universe

**Date:** 2026-08-14  
**Status:** **REJECTED BY OWNER — DO NOT EXECUTE**

On 2026-08-14 Owen Heidenreich instructed: **“not approved. SPXW and SPX only.”** This document is
preserved as historical analysis. It does not satisfy the breakthrough criterion, authorize a charter
change, authorize data, or define an active research route. The governing rejection is
`v5/governance/MES_STRATEGIC_PIVOT_REJECTION_2026_08_14.md`.

## Decision

Stop paying the option layer to express a 10–30-point view. The next research game should use one Micro
E-mini S&P 500 future (MES), with one shared clock-aware sequential model choosing LONG, SHORT or WAIT and
managing only positions it actually opens.

This is not “futures because options were hard.” It follows the measured failure mechanism:

- the compact SPXW policy lost **-$15.93/trade at midpoint**, before spread;
- long 0DTE also pays variance premium, theta and the option spread;
- cheap OTM SPXW tickets make fixed costs a prohibitive share of premium;
- short premium contains a small premium but exposes an unacceptable tail, while the tested four-leg
  defined-risk translation lost **-$60.57/session** at touch; and
- MES preserves the underlying 10–30-point move without premium, Greeks, expiry, strike selection or a
  four-leg toll.

The conclusion is superiority of the *trading game*, not evidence that direction is predictable.

## Executable scale

CME specifies MES at **$5 per S&P point** and a **0.25-point/$1.25 tick**. The project measured the
elevated-volatility ES spread at 1.0734 ticks. Translating that tick count to MES and adding the current
published IBKR $0.25 commission plus $0.35 exchange fee per side gives **0.508 points / $2.54** per round
trip. The research packet rounds this against the strategy to **0.55 points / $2.75**. Sources checked
2026-08-14: [CME contract](https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.margins.html),
[IBKR commission](https://www.interactivebrokers.com/en/pricing/commissions-futures.php), and
[IBKR CME fees](https://www.interactivebrokers.com/en/accounts/fees/CME.php).

That 0.55-point figure is a design assumption, not a MES fill measurement. Exact MES BBO must replace the
ES proxy before fitting or claiming economics.

With one MES and a five-point stop, the frozen translation is:

| Target | Net win | Net stop loss | Reward/risk | Target-hit break-even |
|---:|---:|---:|---:|---:|
| 10 points | $47.25 | -$27.75 | 1.70 | 37.0% |
| 20 points | $97.25 | -$27.75 | 3.50 | 22.2% |
| 30 points | $147.25 | -$27.75 | 5.31 | 15.9% |

A stop is not a guaranteed maximum loss. Any future test still dies if its historical worst trade exceeds
$500, and it may take at most two trades per session, one at a time, with no overnight position.

## Evidence requirement

The current 254-session ES corpus is too small for this decision. The project's measured simple
one-hypothesis requirement at a 60-minute horizon is **3,807 sessions for a 1-point net edge, 952 for a
2-point edge and 423 for a 3-point edge**. Controls, family correction and the 4/5 chronology rule can
only make those requirements stricter.

MES launched on 2019-05-06, so roughly 1,800–1,830 pre-cutoff sessions exist. Acquire all regular-session
MES OHLCV-1m and BBO-1s from launch through 2026-07-31, reserve the final 252 sessions before any policy
construction, and cap the first model at 50 parameters. One shared model gets causal clock features;
10:00 and 13:30 remain diagnostics rather than trade triggers.

The project’s actual vendor estimates scale to **about $150.52** for 1,830 sessions: $2.5922/311 sessions
for OHLCV-1m plus $1.4783/20 for BBO-1s. Use a **$200 hard cap** because the exact MES request has not been
quoted. A read-only exact cost preflight must abort if the cap is exceeded.

## Exact owner-controlled change

Replace the long-SPXW-only research charter with a MES research charter and authorize up to **$200** for
exact `MES.FUT` OHLCV-1m plus BBO-1s history from launch through 2026-07-31. This would authorize research
data only—no broker contact, paper/live orders, promotion or real-money action.

If that change is not authorized, the honest project state is not “try another options model.” It is:
the owned long-0DTE game is closed and the best different game lacks the independent executable history
required to test it.

## Frozen future test

- one shared time-aware model, at most 50 built parameters;
- actions LONG / SHORT / WAIT while flat and HOLD / EXIT while open;
- morning/afternoon position ownership follows the opening state across the boundary;
- first eligible BBO after each causal decision, exact fees, one- and two-tick adverse-fill stresses;
- outcome-blind time/direction-matched control and identical shuffled-label null;
- corrected lower bounds above zero absolutely and against both controls;
- positive absolute and paired results in at least four of five chronological folds;
- per-feature timestamp and future-mutation audits; and
- $10,000 account, one MES, five-point initial stop, two trades/day maximum, -$500 worst-trade kill.

Machine evidence:
`v4/audit/autoresearch/mes_strategic_pivot_2026_08_14/receipt.json`.
