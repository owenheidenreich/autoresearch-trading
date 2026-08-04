# Master Plan — Mechanism Screen to Trained Model (2026-08-05 → 2026-08-10)

**Status: PROPOSED — awaiting owner sign-off. Freezing §3 is what starts the clock.**

Owner chose "mechanism screen, then train" on 2026-08-04 after asking for a path from here to a trained
model by Monday 2026-08-10.

This **reopens** the programme stood down earlier the same day
([`PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md`](../contracts/PATHD_PROGRAMME_STAND_DOWN_RECORD_2026_08_04.md)).
That record's restart conditions govern this plan and are honoured in §1.

---

## 1. The rule that governs the week

**Raw economics first. Nothing else gets built until a mechanism clears the bar on raw owned data.**

No null, no surrogate, no model, no governance document, no receipt schema — until a candidate shows a
mean dollar edge per trade that clears measured friction on raw data. This is the single lesson that cost
the most: the number that closed `omar` took one pass over owned data and no machinery, and it sat
computable for a day while three studies argued about an information coefficient.

**The bar: 0.358 ES points per trade** ($17.9176 measured round-trip friction ÷ $50/point).
**The kill rule: a mechanism must clear 1.5× the bar gross — 0.537 points — to survive to training.**
The 1.5× is headroom for estimation error and backtest-to-live decay, and it is fixed now, before any
number is seen.

**Why "mechanism" and not "feature":** every one of the five negatives came from asking *is there edge in
this dataset?* That is a search, and with the confirmation firewall spent a search returns noise or
nothing. Each candidate below carries a stated economic reason it should work — a reason that survives
being said out loud — and dies on its own economics, not on a p-value.

## 2. Timeline

| Day | Date | Work | Output |
|---|---|---|---|
| 1–2 | Wed 08-05, Thu 08-06 | Raw-economics screen, all mechanisms | One ranked table |
| 3–5 | Fri 08-07 → Sun 08-09 | Train **only** what cleared | Trained model + four-box + gate |
| 6 | **Mon 08-10** | Verdict and write-up | Go / no-go on paper trading |

If nothing clears on Day 2, **that is the Monday deliverable** and no training happens. Two days spent
instead of a campaign is the plan working, not failing.

## 3. The mechanisms — FREEZE THIS LIST BEFORE RUNNING

Each is tested on owned data, causal at decision time, scored in dollars per trade against the 0.537-point
survival bar. **The declared list is the family; nothing may be added mid-week.**

| # | Mechanism | Economic reason it should exist | Status |
|---|---|---|---|
| **M1** | **Opening range, 09:30–10:00 ET** | The opening auction clears accumulated overnight order imbalance. The first 30 minutes carry the day's highest volume and its widest information asymmetry, and liquidity providers demand compensation to absorb that flow. | **NEVER TESTED** — every screen and campaign started at 10:00/10:01 |
| **M2** | **Closing hour, 15:20–16:00 ET** | MOC imbalance publication at 15:50 and index-tracking flow create a **calendar-locked, publicly announced** order imbalance. This is a scheduled mechanism, not a pattern. | **NEVER TESTED** — every campaign ended at 15:00/15:20 |
| **M3** | **Overnight gap** | Overnight information is repriced at the open. Gaps either fill (liquidity overshoot) or continue (genuine repricing); conditioning on gap size in volatility units separates the two. | Never tested — all work was intraday |
| **M4** | **VIX regime conditioning** | The volatility regime conditions both the size of moves and whether the tape trends or reverts. | **VIX HAS NEVER ENTERED ANY MODEL** — declared in the contract, hardcoded `float("nan")` in the kernel, absent from `feature_matrix`'s required inputs. 338 owned sessions, zero used |
| **M5** | **Calendar effects** — day-of-week, opex, month-end, Fed days | Systematic flows are calendar-locked: index rebalances, option expiry pinning, and scheduled announcements move known amounts of capital on known dates. | `day_of_week` is 1 of only 8 ADMITTED features and has never been tested economically |
| **M6** | **SPX–ES basis** | Basis is a mechanical arbitrage relationship with a known fair value; deviations must revert on a bounded timescale or arbitrage is free. | Owned both series. **Distinct from Protocol 028**, which rejected stitched ES *VWAP as a feature* — basis is a different object |

**Excluded by the do-not-retest ledger and not to be revisited:** 0DTE long premium in any form; the nine
side-free SPX context features; `omar` as a tradable directional signal; stitched ES VWAP as a feature.

## 4. What the screen actually computes

For each mechanism, on raw owned data, with **no surrogate and no null**:

- Mean dollars per trade, and the trade count, at 15/30/60-minute horizons.
- The same split by fold, so instability is visible immediately.
- Nothing else. No IC, no p-value, no maxT on this pass.

A mechanism that cannot clear 0.537 points gross on the raw pooled number dies in minutes and is recorded
as dead. Significance testing is a **second** pass, applied only to survivors — and any null used there
must pass a known-answer gate before its verdict is believed.

## 5. Days 3–5: train what cleared

Only surviving mechanisms are trained. Existing machinery, unmodified:

- Causal **t−60s** clock; feature-availability-clock parity verified, not just future-outcome guards.
- Shallow ranker inside a fixed game (April's calibration: RF 200 trees, depth 5, leaf 20 → PF 1.291).
  Anything materially above April's honest best is a bug until proven otherwise.
- `pathd_phase1_replay.py` four-box, session-bootstrap LCB, all three negative controls.
- `pathd_model_gate.py` — four rejection tests plus the four Charter diagnostics.
- Session-blocked maxT over the declared family.
- `prior_art_check` before every hypothesis.

## 6. Stop rules

- **Nothing clears 1.5× on Day 2** → report and stop. No training.
- **A negative control clears the gate** → the wave is `INVALID`, results discarded.
- **No mechanism may be added after §3 is frozen.** Adding one mid-week is the search error that produced
  the first five negatives.
- **No widening** of a window, threshold, or horizon to make a result work.
- **No paid data, no broker, no promotion, no runtime or launchd mutation** without fresh owner
  authorization.
- `FILL_LAW`, the causal clock, the label law and the OOF firewall stay frozen.

## 7. What Monday can and cannot be

**Can:** a trained, causally-validated model with an honest verdict, and a clear go/no-go on whether it is
worth putting into forward paper.

**Cannot:** proven edge, or "ready to trade real money." **The protected holdout is SPENT** — opened once
on 2026-08-02 for `signed18`, which was then invalidated. There is no confirmation firewall left, so the
only confirmation available is forward live-paper observation, which is calendar time and not compute. A
clean result on Sunday still needs weeks of paper before it is tradeable with real money.

Anyone who promises otherwise by Monday is selling a backtest.

## 8. First command

```bash
PYTHONPATH=. python -m v4.research.pathd_mechanism_economics_screen --list
```

(prints the frozen mechanism list and the bar, runs nothing — the screen itself is written Day 1.)

*Drafted: Claude Opus 5 — 2026-08-04. Not in force until the owner signs and §3 is frozen.*
