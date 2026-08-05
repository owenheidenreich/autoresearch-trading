# Master Plan — Mechanism Screen to Trained Model (2026-08-05 → 2026-08-10)

> **SUPERSEDED 2026-08-05 — see [`STATUS.md`](../../../../../STATUS.md) for current status.**
> Never signed, and its §3 carries a DO-NOT-FREEZE banner because five of six mechanisms died on verification. Retained as the audit trail for that verification. The surviving direction question is now gate G1 in STATUS.md.

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

## 3. The mechanisms — ⚠ DO NOT FREEZE. FIVE OF SIX DIE ON VERIFICATION.

> **CORRECTION 2026-08-04, after adversarial review by Codex and a second Opus reviewer.** I verified the
> contested claims myself. **Three premises in the table below are false, and the table must not be frozen
> as written.**
>
> **The root defect: the instrument is undeclared.** The 0.358-point bar is **ES futures** friction
> ($4.50 commissions + 1.0734 ticks × $12.50, $50/point). But §5's exit language — convex tail, p99 $3,422
> vs $86, "the entire reason to hold a long option" — is **option** language, and ledger row 180 closes the
> option route for any exit policy. This is unavoidably an **ES** plan, and §5 describes the wrong
> instrument. Most of the disagreement between the two reviews traces to this one ambiguity.
>
> | Mechanism | Verified status |
> |---|---|
> | **M3 overnight gap** | **DEAD — untestable on owned data.** The ES corpus is **RTH-only**: 390 rows, 09:30→15:59 ET, and **zero** pre-09:30 bars across 25 sampled sessions. There are no overnight bars to compute a gap from. *(This also kills two of the four members in Codex's proposed K=4 family, which assumed gap data exists.)* |
> | **M4 VIX** | **DEAD — my premise was false.** VIX **has** entered models: `vix_level`, `vix_change_5m`, `vix_change_15m`, `vix_minus_spx_rv15_pct_points` are in `v4/research/lean_autoresearch/harness.py`, and that S4/S5 campaign returned **`NULL_NO_NEW_ENTRY_EDGE`**. I verified only the canonical Stage-1 path and generalised to "any model". Separately, `vix_change` is a `QUARANTINED_ALPHA_TOKEN` enforced by `assert_model_alpha_firewall`, so M4 could not run without editing a contract §6 declares frozen. |
> | **M1 opening range** | **Survives, but reframed and not novel.** Minutes 31–350 is a **feature-warmup constraint, not an arbitrary exclusion**: `history_minutes=30` and `momentum_15m` needs 15 prior bars, so the kernel *structurally cannot* produce features before 10:01. M1 therefore needs a **new warmup law**, not a flipped filter. And "NEVER TESTED" overstates: V1A/V1B opening-structure reversion failed with controls. Different instrument, so not closed — but not virgin ground either. |
> | **M2 closing/MOC** | **DROP from this wave.** Owned OHLCV contains **no signed MOC imbalance feed**; a clock window alone supplies no direction. Prior late-session trigger work went 0/8. |
> | **M5 calendar** | **SPLIT AND DEFER.** Day-of-week, opex, month-end and FOMC are at least four distinct families, and none currently defines a signed trade. |
> | **M6 SPX–ES basis** | **BLOCKED pending contract work.** Needs carry, dividends, rates, roll handling and synchronised causal clocks. `ES.c.0` **is** the stitched object Protocol 028 rejected, with 4 in-corpus rolls. |
>
> **Also corrected:** the owned-data counts in the handoff were inflated. SPX 1m is **251 official** (not
> 505 — 254 are proxy files the kernel rejects) and VIX 1m is **251 official** (not 338). Of those 251, **36
> are the spent holdout.**
>
> **Net: one mechanism survives, and only with new warmup work.** §4's per-trade bar is also wrong — see
> the correction below.

## 3b. The original list, retained as the audit trail — SUPERSEDED BY §3

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

## 3c. The survival bar is wrong — CORRECTED 2026-08-04

Both reviewers independently reached the conclusion I flagged as a suspected hole, and they are right.

**A per-trade bar is not a survival rule.** One account holds one position. Over a 390-minute ES session
the non-overlapping capacity is **26 / 13 / 6** trades at 15 / 30 / 60 minutes, and the existing six-block
structure already caps a session at 6 entries. A mechanism firing 132 times per session cannot take them
all, so a per-trade average overstates what is harvestable — and 0.537 points per trade at one trade per
session leaves **0.179 points = $8.95** net for the whole session.

**The primary screen statistic must be net points per calendar session under a frozen one-account serial
allocator** — causal selection, frozen tie-breaking, one occupied position, no overlap, flat by close —
reported with a session-bootstrap LCB and 4-of-5 fold sign-stability. The 0.537-point figure survives only
as a **necessary microeconomic diagnostic**, never as sufficient.

**Two further gaps, both real:**

- **No capital-productivity hurdle is declared.** A minimum dollars-per-session or return-on-margin bar
  must be frozen *before* results are seen; it cannot be inferred from the winning candidate. Currently
  `UNKNOWN`.
- **No comparator.** Against an absolute bar, ES drift alone can carry a long-biased mechanism over the
  line. Flat must be the primary comparator, with sign-reversal, session-shuffle and constant-action as
  invalidation controls.

**And the hurdle should be said out loud:** clearing 1.5× friction at a 15-minute horizon on ES demands
roughly **54.5% directional accuracy**. By this project's own calibration rule, a pass at that level is a
bug until proven otherwise — not a celebration.

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
