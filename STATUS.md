# Project Status — the one page

**Last updated: 2026-08-05.** This is the only status document. If another document disagrees with this
one, this one wins and the other one is stale. Written in plain English on purpose.

---

## 1. What we are building

An automated day-trading bot that buys SPX 0DTE call and put options — a machine-learning model that
decides which way the market is about to go, buys the matching option, and sells it the same day.

## 2. Where we actually are

**The plumbing works. We do not yet have a reason to trade.**

Everything mechanical has been built and proven: we can pull live market data, turn it into features,
make a decision, qualify a real option contract, and place a guarded paper order with the broker. What we
have never demonstrated is a way to predict the market that makes money after costs.

Five separate research campaigns have each returned "no edge." That is not a broken machine — that is the
machine honestly reporting that the things we tried do not work.

## 3. The single most important finding

On 2026-08-04 the project proved something that changes the order of the work:

> **Buying SPX 0DTE options and holding them minute-to-minute loses about $13 per trade *before you pay
> any commission or spread*.** It lost money in all 5 test folds. This is the option losing value as time
> passes — it is a property of the instrument, not of the prediction. **No model, no neural network, no
> feature set, and no exit rule fixes it.**

This is recorded as row 181 of the do-not-retest ledger.

**Read carefully, because this does not kill the goal.** The same ledger row lists what would legitimately
re-open it, and one condition is **"demonstrated directional skill on the underlying"** — meaning: show
you can predict which way SPX itself moves. Row 180 confirms the option version is closed *because*
direction has not yet cleared that bar.

So the honest position is:

> **We cannot yet predict which way SPX is going. Until we can, wrapping that non-prediction in 0DTE
> options just adds decay and friction on top of a coin flip. Once we can, the 0DTE call/put bot becomes
> buildable again.**

That reduces the whole project to **one question**:

> ### Can we predict SPX direction over 15–60 minutes well enough to clear measured trading costs?

Everything else is downstream of that answer.

## 4. The cost bar every idea must clear

Costs are measured, not assumed:

| Thing | Measured value |
|---|---|
| Round-trip friction, ES futures | **$17.92 = 0.358 ES points** |
| ES spread | 1.04 ticks (1.07 in the top volatility quartile) |
| Round-trip friction, SPX options | **$3.08** ($1.54/side) |
| Friction as a share of a $565 average option premium | **4.68%** |

A candidate that does not beat 0.358 points per trade gross is not a strategy, however good its
statistics look.

## 5. What is genuinely proven (do not redo these)

- **Live execution works.** Guarded paper round trip on account `DU***40`, 2026-08-04. The contract we
  qualify live (`SPXW  260804C07775000`) matches the training-data format exactly, so the
  data-to-broker bridge is closed.
- **Safety guards work.** They refuse a real-money account, refuse without the enable flag, refuse without
  each acknowledgement. Verified by direct exercise, not just unit tests.
- **Live market data works.** Databento OPRA definitions + market capture ran end to end. Real arrival
  timing observed: CBBO-1m p99 584.6 ms.
- **Validation machinery works.** Replay, negative controls, bootstrap confidence bounds, multiple-testing
  correction, and a prior-art check that blocks re-running closed experiments. All tested against
  known-bad inputs where they must fail — and they do.
- **Costs are measured** (§4), replacing three earlier wrong estimates.

## 6. What is closed and must not be retried

From the do-not-retest ledger — the most valuable document in this repo:

| Closed | Why |
|---|---|
| 0DTE long premium at minute cadence, **any** model or features | −$13.00/trade before costs, 5/5 folds |
| `omar` (SPX position within session range) as a tradable signal | −0.324 points/trade on raw data |
| Options-chain risk-reversal direction signal | Failed 5/5 fold gate (only 3/5 passed) |
| The 18-feature entry contract | Zero ranking power — deciles flat and non-monotonic |
| The exit-repair family (6 arms) | All lost to their comparator |

## 7. The gate chain — everything between here and a working bot

Each gate must pass before the next is attempted.

| # | Gate | Plain meaning | Status |
|---|---|---|---|
| **G1** | **Direction** | Predict SPX/ES direction over 15–60 min, beating 0.358 pts/trade gross, under one-position-at-a-time accounting | 🔴 **OPEN — this is the whole project right now** |
| **G2** | Option wrapper re-opens | Only if G1 passes. Row 181's stated re-entry condition | 🔴 blocked by G1 |
| **G3** | Feature certification | Prove each feature is actually available live at decision time, so we never train on data we would not have had | 🟡 in flight — 8/73 admitted; capture 08-06/07 |
| **G4** | Train | Fit entry + exit models on certified features only | ⬜ needs G1 + G3 |
| **G5** | Validation replay | Beat the comparator, positive confidence bound, 4/5 folds, negative controls clean — **7 conditions, all required** | ⬜ needs G4 |
| **G6** | Runtime parity | Live decisions match offline decisions **100%**. This is the gate that caught the `signed18` leak at 18.44% | ⬜ built, ready to run |
| **G7** | Live shadow | Model runs on live data, decides, submits nothing | ⬜ not built |
| **G8** | Guarded paper | Real paper orders over a pre-registered window. **This replaces the spent holdout** | ⬜ not built |
| **G9** | Real money | Separate owner decision and safety packet | ⬜ out of scope |

**G3 runs in parallel and is free. G1 is what actually blocks everything.**

## 8. Why G6 and G8 exist (the expensive lesson)

A model called `signed18` posted a confirmed **+$540/session** edge. It was then found to have used an SPX
price bar stamped at time `t` whose closing value is not knowable until `t+60s`. Under the correct clock
only ~10% of its decisions reproduced. Proving it wrong consumed the protected holdout — the reserved
data set kept for final confirmation.

**Consequence: there is no clean historical confirmation left.** The only remaining out-of-sample test is
fresh live paper trading, which costs calendar time and cannot be bought with compute. That is why G8 is
not optional.

## 9. What is running right now

| What | When | Purpose |
|---|---|---|
| Track-A live OPRA capture | 2026-08-06 and 08-07, two windows each (06:28 and 09:10 PT) | Measure when market data actually arrives, so features can be certified for G3 |
| Everything else | — | Nothing runs unattended |

2026-08-05's capture is **excluded from the sample** — its open window was lost to a file-sync fault and
its midday window only confirms the repair. Owner review: **Saturday 2026-08-08**.

## 10. Where things live

| Path | What |
|---|---|
| `STATUS.md` | This page. The only status document. |
| `CLAUDE.md` / `AGENTS.md` | Ground rules for agents. Identical content. |
| `v4/docs/protocol101/training/history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md` | **The do-not-retest ledger.** Every closed idea and why. Read before proposing anything. |
| `v4/docs/protocol101/training/contracts/` | Signed decisions and build order |
| `v4/docs/protocol101/training/research/` | Individual studies — evidence, not plans |
| `v4/audit/autoresearch/` | Raw evidence: receipts, captures, results |

*Maintained by: Claude. Update this page whenever a gate moves — do not create a second status document.*
