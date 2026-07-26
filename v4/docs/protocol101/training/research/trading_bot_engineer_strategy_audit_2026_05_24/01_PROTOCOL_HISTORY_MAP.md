# Protocol History Map

This is a family-level map, not a line-by-line review of every protocol number. The project has many experiments, but most fall into a few recurring strategy families.

## Family Taxonomy

| Family | Representative Protocols | What It Tried | Outcome | Lesson |
|---|---|---|---|---|
| A+ timing/value entry | 001-024, 039 | Encode post-open/late-afternoon A+ 0DTE entry timing, teacher margin, value scoring, basic robustness | Found positive edge but with fragile stress behavior and limited scope | Trader timing priors mattered from the beginning |
| Contract-quality gates | 022-038 | Filter puts/calls by overpay risk, spread, theta, breakeven, gamma, ladder quality | Often over-filtered convex winners | "Looks expensive" is not enough to reject an option in a convex 0DTE setup |
| Sequential lifecycle | 029-081 | Improve exits/holding, recover after adverse moves, avoid clipping winners | Later versions improved materially, but lifecycle remained hard and context-dependent | Exit logic is strategy, not just a risk clamp |
| Protocol101 lineage | 092, 097, 101 | Learn event-level candidate selection with causal short history and wait action | Protocol101 cleared the strict serial gate and became paper default | The most successful default combined trader priors, causal features, and serial replay |
| Readiness/falsification | 102-130 | Stress Protocol101 with paper-account, timing, high-res, no-order/live readiness, capital realism | Supported paper-default status but flagged timing/live/fill blockers | Profitability in replay is not execution proof |
| Account/serial realism | 161-163 | Compare independent opportunities against strict one-slot account replay | Independent opportunity counts overstated tradable performance | The single open-position slot is economically central |
| Full-action surface | 164-194, 210-215 | Rank the wider ATM +/- $50 SPXW surface rather than only Protocol101 candidates | Some strong challengers beat Protocol101 in exposed replays, but timing/live parity blocked promotion | Full surface can find opportunity, but it is very timing-sensitive and sparse |
| Premium/capital efficiency | 221-248 | Favor lower premium or blended dollar/return utility to capture more convexity | Higher gross PnL, lower quality, worse drawdown and churn | More opportunity is not automatically a better bot |
| Policy routers | 254-265 | Route between Protocol101-like and challenger proposal streams; test baseline-anchored continuation | Strong research evidence, but not deployable and lifecycle extensions were split-mixed | The best clue was not routing itself, but baseline-relative opportunity pricing |
| Unified serial policy | 270-276 | Build wait/enter/hold/exit labels and integrated entry+lifecycle replay | Protocol276 failed older protected blocks badly | A coherent game contract is necessary but not sufficient |
| Learned defer | post-276 | Learn cost of blocking Protocol101 and allow challenger only after charging slot opportunity cost | Positive diagnostic replay, but blocked by calibration, live parity, fills, and exposed validation | Baseline-relative conservative improvement is the right shape, but not yet promotable |

## Important Winners That Are Still Not Paper Defaults

Several challengers beat Protocol101 on exposed diagnostic splits. This is crucial: Protocol101 is not the highest-PnL historical artifact. It is the strongest operational default.

| Protocol | Result Versus Protocol101 | Why It Did Not Replace Protocol101 |
|---|---|---|
| Protocol194 | Beat Protocol101 on March, Q1, Q3, Q4, and recent diagnostics | One-minute timing stress turned every split negative; live timing parity not proven |
| Protocol215 | Beat Protocol101 on every listed block after history feature repair | Still needed attribution, chart inspection, live/training parity, and promotion gates |
| Protocol240 | Premium-leaning blend beat Protocol101 with large PnL deltas | Lower win rate, lower PF, worse drawdown/churn, and no paper-default readiness |
| Protocol260 | Router beat Protocol101 across listed splits | It did not beat the best challenger stream consistently and remained research-only |
| Protocol265 | Beat Protocol101 on March/Q1/recent paper-default comparison | Baseline-anchored extension did not beat its own Protocol261 base in March/Q1; lacked deployable parity |
| Learned-defer challenger | Positive preregistered diagnostic deltas across slippage levels | Calibration, live parity, fills, formal overfit controls, and untouched holdout remain blocked |

## Important Failures

| Protocol Line | Failure | What It Taught |
|---|---|---|
| Protocol092/097 | Positive but failed strict serial gates in some blocks | Event selection needed better causal history |
| Protocol163 | Recent strict one-account model was profitable but did not beat recent Protocol101 | Serial realism reduces apparent opportunity |
| Protocol200 | Lifecycle continuation overheld and blocked later profitable entries | Holding longer can destroy slot value |
| Protocol209 | Unified entry+lifecycle failed March/Q1 and lifecycle baseline | Joint modeling did not automatically solve serial economics |
| Protocol276 | Integrated Protocol271 entry + Protocol275 lifecycle failed Q3/Q4/Q1/March | Entry selected many negative-advantage trades; lifecycle overheld on the wrong state distribution |
| Rowwise full-surface attempts | Learned mostly abstention or unstable entry scores | The positive-action surface is sparse and candidate choice is listwise/serial, not rowwise |

## Repeated Project Lessons

1. **Timing dominates.** Many apparent edges decay sharply with one-minute entry delay, and sometimes with much smaller delay.
2. **The one-slot account is a real scarce resource.** A trade must beat waiting and must not block better future trades.
3. **Trader priors are not optional.** Time-of-day, premium/moneyness, side, quote freshness, and candidate quality strongly shape outcomes.
4. **More gross PnL often came with worse trade quality.** The best challenger-style stream traded more, won less often, and drew down more.
5. **Lifecycle is not a universal "hold longer" rule.** Some exits clip winners; some holds destroy PnL and block the next trade.
6. **Validation splits are research-exposed.** Q3/Q4/Q1/March/recent should be treated as diagnostics, not sacred promotion evidence.

