# Reality-Check Brief for Fable — Path-D, 2026-08-04

**What we want:** an outside read on whether this program is solving the right problem, and whether we are
missing something obvious. Be blunt. `STOP` and `NO_EDGE` are respectable answers; so is "you are building
the wrong thing."

**Context you should trust but verify:** every number below is reproducible from artifacts on the SSD at
`/Volumes/AR_TRADING_DATA` or from committed receipts. Paths are given throughout.

---

## 0. The finding that prompted this brief

While assembling this document I enumerated the entry model's actual feature contract from
`/Volumes/AR_TRADING_DATA/artifacts/entry_v2/oof_scores.parquet`. It has 25 columns. It contains **none**
of the following:

| | present? |
|---|---|
| time of day | **ABSENT** |
| time to expiry | **ABSENT** |
| bid-ask spread | **ABSENT** |
| bid/ask sizes, imbalance | **ABSENT** |
| theta | **ABSENT** |
| vega | **ABSENT** |
| IV level | **ABSENT** |
| realized volatility | **ABSENT** |

What it *does* contain: contract identity (right, strike, atm_strike, strike_offset, is_call), raw
bid/ask/mid, **eleven** SPX context variants (vwap gap ×3, session range, momentum 5m/15m ×4, OMAR, three
alignment flags), three near-ATM composites, and delta + gamma.

**This is a 0DTE options model that cannot see the clock, the decay rate, or its own transaction cost.**

It matters because the single positive signal anywhere in our data is a **time-of-day effect** — the
10:30–10:59 ET window shows +$5.99 mean gross, positive in all five delta bands (§4). The model is
structurally incapable of representing the one variable the data says matters. We concluded "the feature
contract has zero ranking power" and treated that as a finding about the market. It may simply be a finding
about the feature list.

**Question 1 for Fable:** is that the whole story, or a convenient explanation? Deciles are flat
(top −$15.37, middle −$10.79, bottom −$16.32) — is that consistent with "missing the key variable," or does
it indicate something worse?

---

## 1. The pipeline as built

```
TRAIN     Databento OPRA (SPXW 0DTE) + ThetaData (SPX/VIX)   12 months, 251 sessions
   │      corpus: /Volumes/AR_TRADING_DATA/vendor/pathd_2025-08-01_2026-07-31
   ▼
DECIDE    Databento Live OPRA          (same vendor as training, by design)
   ▼
EXECUTE   IBKR via IB Gateway          (execution only; supplies no alpha)
```

The vendor split is deliberate: Path-D exists to escape the cross-vendor parity treadmill that consumed
June–July, when corrected historical replay produced 38 entry signals and live produced **zero**.

**Key source files**
| Purpose | Path |
|---|---|
| Entry dataset + fill law | `v4/research/pathd_phase1_entry.py` (`FILL_LAW` at line 46) |
| Exit model | `v4/research/phase1_exit_model.py` |
| Four-box replay / acceptance | `v4/research/pathd_phase1_replay.py` (gate at 311–330) |
| Executable gate + charter diagnostics | `v4/research/pathd_model_gate.py` |
| Autoresearch loop | `v4/research/pathd_research_loop.py` |
| Feature admission law (enforcement) | `v4/research/pathd_feature_admission_ledger.py` |
| Live feature catalog (16 families) | `v4/research/autoresearch_v2/entry_live_feature_catalog.py` |
| Live OPRA capture | `v4/scripts/capture_databento_live_opra_training_twin.py` |
| IBKR paper guard / executor | `v4/live/ibkr_paper_guard.py`, `v4/live/ibkr_paper_executor.py` |

**Governing documents**
- Build order: `v4/docs/protocol101/training/contracts/PATHD_BUILD_ORDER_2026_08_04.md`
- Trader Charter (signed, + Amendment 1): `.../contracts/PROTOCOL101_TRADER_CHARTER.md`
- Program status: `.../contracts/PATHD_PROGRAM_STATUS_WORKSHEET_2026_08_04.md`
- Prior-campaign ledger + do-not-retest: `.../training/history/PROTOCOL101_PRIOR_CAMPAIGN_DISTILLATION.md`
- Protocol 002–160 decoder: `.../training/history/PROTOCOL101_PROTOCOL_FARM_LINEAGE_2026_07_19.md`

---

## 2. What is proven

- **Execution plane.** Guarded paper round trip 2026-08-04 on `DU***40`. A live-qualified contract came
  back as `SPXW  260804C07775000` — the `localSymbol` matches the OSI format in the training corpus
  exactly, closing the contract-identity question June's crisis was about. Evidence:
  `v4/audit/autoresearch/pathd_phase0b_trackc_paper_transitions_2026_08_04/`
- **Guard layer.** Refuses live (`U`-prefix) accounts, missing env flag, missing acknowledgements.
- **Databento live path.** 60 s smoke capture end to end. Measured arrival clock: **cbbo-1m p99 584.6 ms**,
  cbbo-1s 504.8 ms.
- **Fee truth-up.** **$1.54/side, $3.08 round trip**, measured. The frozen `FILL_LAW` $3.00 is nearly exact.
- **Research machinery.** Gate, loop, prior-art blocking, maxT, negative controls — all validated against
  known-bad artifacts where they must fail.

## 3. What is not

- **8 of 73 features certified**, all calendar/geometry. **Zero market observables admitted.** Ledger:
  `v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04/feature_admission_ledger.json`
- 65 blocked behind one receipt: `cbbo1m_native`'s multi-session arrival distribution (capture armed
  2026-08-05..07).
- **No entry model exists.** It cannot be trained — enforcement refuses non-admitted features.
- The exit model is degenerate: exits at the first step in **95.1%** of trajectories.

---

## 4. The four negatives (none fixes another)

| # | Finding | Number |
|---|---|---|
| 1 | 0DTE long premium is negative-EV **before any cost** | −$13.00/trade gross, negative 5/5 folds |
| 2 | Feature contract has zero ranking power | deciles top −$15.37, **middle −$10.79**, bottom −$16.32 |
| 3 | The April-validated exit repair failed | 6/6 arms lost, maxT p 0.967–0.994 |
| 4 | Outcome shape incompatible with the Charter | 23–69% big-loss vs a 2% limit; 3–15% scratch vs 73% target |

**Survived every attack:** horizon dominates instrument choice (116%→57% required win rate, 1 min → 60 min).
**The one lead:** 10:30–10:59 ET, +$5.99 gross, positive in all five delta bands, net still −$20.08.

---

## 5. What the last 10 commits taught us

Ordered newest first. Every one is a *correction*, which is itself a signal.

| Commit | Lesson |
|---|---|
| `61ced22b` | Status worksheet: capability proven, edge absent |
| `f2158438` | I clobbered two ledger-referenced receipts; the hash chain caught it. Referenced receipts are immutable |
| `5639a3b9` | Track C parity PASSES but stays BARRED — no measured arrival clock. Refusing to admit on an unmeasured clock |
| `e5ff11e6` | Fee correction across 4 docs |
| `36715544` | **$0.65/side was wrong.** Measured $1.54. It is the IBKR commission line item, not all-in. Also caught a SELL-limit-below-bid bug that left a position open |
| `1f9ae17d` | Smoke test caught a wrapper bug that would have silently failed the first capture |
| `d9ed0c0c` | Verifying interfaces found 3 defects in my own staged wrapper (invented module, wrong CLI, 900 s vs a 300 s cap) |
| `28d54efc` | Capture window was midday-only; latency peaks at the open. Sample *placement* was wrong |
| `078da115`, `41af2604` | **`implied_spot` was admitted with a 4.898 ms local compute p99 while its inputs arrive at ~585 ms — a 119× clock error.** My Phase-0 spec omitted its quote parent |

**The pattern:** nine of ten are me catching my own errors. The machinery works. But we produced **18 commits
today and trained zero models.**

**Question 2 for Fable:** the project's own ledger records this exact failure mode — *"the governance
apparatus massively outran the research question; the lean fit answered it in an hour."* Are we repeating
it? Is the certification programme (Build Order, admission ledger, enforcement) correct discipline, or
elaborate avoidance of a cheap experiment?

---

## 6. Hypotheses we want attacked

**H-A — The feature list, not the market, explains the null.** §0. The model lacks time, TTE, spread,
theta, vega, IV. Test: does adding a clock alone move the deciles?

**H-B — We are trading the wrong side.** Gross expectancy is −$13/trade *before costs*. That is theta. The
owner wants the long-only bot working before considering spreads or premium selling. Is that ordering
defensible, or is it insisting on the losing side of a structural trade?

**H-C — The Charter's economics make this unwinnable regardless of signal.** One contract, $10k account,
$565 average premium ⇒ friction is **4.68% of premium**. Larger premium, larger account, or defined-risk
structures all change that ratio. Charter Amendment 1 already demoted the outcome profile to report-only.
Should the sizing/account constraints go too?

**H-D — The data is too narrow.** The corpus is **0DTE only** — one expiry per session file, verified. No
1DTE, no weeklies, no other underlying. Theory says longer tenor has far less theta per unit time and a
better friction ratio. We have never measured it because we never bought the data.

**H-E — We should stop and clean up.** ~1,000 audit directories, six frozen governance generations, a
retracted +$540/session "edge," and four negatives. Is the right move a hard reset onto the smallest honest
experiment?

---

## 7. Specific questions

1. Given §0, is "zero ranking power" a market finding or a feature-list artifact?
2. Is certify-before-train (`PATHD_BUILD_ORDER_2026_08_04.md`) right, or is it this project's
   documented failure mode wearing new clothes?
3. Is there a cheap decisive experiment we are not running?
4. Is the 10:30 ET effect worth pursuing, or is it 1-of-11 multiplicity we are rationalising?
5. What would you *stop* doing?
6. Anything plainly obvious we have missed?

**Constraints that are real:** protected holdout is SPENT (forward confirmation = fresh live paper only);
causal clock is t−60s; paid data needs owner authorization; `FILL_LAW`, the clock, the label law and the
OOF firewall are frozen.

*Prepared by Claude Opus 5 — 2026-08-04.*
