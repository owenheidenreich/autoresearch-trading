# Methodology Overhaul — Final Summary — 2026-04-22

## TL;DR

**New champion: V0 (model's chosen direction) + time-stop.** Aggregate
PF **1.132** across 780 OOS days over 13 disjoint 60-day windows.

The previous champion — "V1 (always_put) + A3 L3 @ 0.19, OOS PF 2.847
on 20 days" — is retired. That claim was built on a single
chop-bearish window. In the broader 13-window rolling re-evaluation,
V1 loses to V0 in **12 of 13 windows** on a strict comparison and in
**10 of 13 windows** by a wider `> 0.10 PF` margin; aggregate PF is 0.888.

The switch from V1 to V0 is the single most important methodological
correction in this project's history.

---

## Workstream overview

| Phase | Goal | Outcome |
|---|---|---|
| R1 | Build rolling-window harness (60d non-overlap, 180d min train) | 13 windows, 780 OOS days |
| R2 | Audit L2 features + engineer 9 intraday-developing additions | 6 Cat A + 3 Cat B, 0 NaN |
| R3 | Retrain Layer-2 per window with 51-feature input | 13 models, 1.0 min CPU total |
| R4 | Per-window V0 vs V1 evaluation | V0 wins 12/13 strict, 10/13 by `> 0.10 PF` |
| R5 | Conditional V0/V1 rule using entry-bar intraday features | TIE (oracle ceiling only +0.17 above blanket V0) |
| R6 | Lock new champion, honest comparison | V0 + time-stop, PF 1.132 |

Total effort: ~1 day CPU-only, single laptop.

---

## New champion config

| Component | Value |
|---|---|
| Layer-2 entry | Retrained per rolling window (51-feature input) |
| Layer-2 direction | **V0 (model's chosen direction, call or put)** |
| Layer-3 exit | **Time-stop at session end** (learned L3 deferred — see below) |
| Sizing | 1 contract per trade |
| Guardrails (Layer 0) | premium cap $1200, daily brake $1500, ≤2 full-premium losers |

---

## Cross-window V0 performance

| Metric | Value |
|---|---:|
| Aggregate PF | **1.132** |
| Aggregate DD % | 95.9% (cold-start equity-curve artifact) |
| Aggregate mean $ | +$62 / trade |
| Total trades | 432 |
| Per-window PF mean | 1.188 |
| Per-window PF median | 1.033 |
| Per-window PF std | 0.546 |
| Per-window PF range | [0.586, 2.274] |
| Windows with PF ≥ 1.0 | 7/13 |
| Windows with PF ≥ 1.2 | 4/13 |
| Windows with PF ≥ 1.5 | 3/13 (windows 5, 9, 11) |
| Strictly beats V1 in | **12/13 windows** |
| Beats V1 by ≥ 0.10 PF in | **10/13 windows** |

Per-window breakdown: see [v3/reference/rolling_directional_eval_2026_04_22.md](rolling_directional_eval_2026_04_22.md).

---

## Why V0 beats V1

V1 (always_put) was the previous champion because it won a single
20-day OOS window (2026-03-05 to 2026-04-01) where the market was
chop-bearish and put-biased selections paid off.

Across 13 disjoint 60-day regimes:
- **5 of 13 windows have PF ≥ 1.2 under V0** (windows 5, 6, 9, 11, partly 8)
- **Only 1 window has V1 beat V0** (window 7, 2024 Q4)
- Bullish / trending-up regimes (5, 9, 11) get crushed under V1
  (V1 PF 1.177 vs V0 PF 2.274 in window 9)
- Chop-bearish regimes (where V1 shines) are a minority of the
  historical record

V1 is essentially a regime-specific fix that was mistakenly generalized.

---

## What the old "PF 2.847" claim really meant

- Built on 20 days (2026-03-05 to 2026-04-01)
- Used augmented L3 trained on days that included some rolling OOS
  days — i.e., had leakage by design, because the 5-fold structure
  overlapped with future periods
- PF 2.847 was an artifact of this specific window + L3 config
- The window happened to be a chop-bearish regime where both V1
  (always_put) AND A3 L3 (sparse exits) helped simultaneously
- Neither intervention generalizes cleanly when tested on 13 other
  60-day windows

The true story: **on 780 OOS days spanning 2023-01 to 2026-02, a
trader using V1+L3@0.19 would have earned aggregate PF 0.888 (losing
money), while a trader using V0+time-stop would have earned PF 1.132
(modest profit)**.

---

## Conditional V0/V1 rule: tested and failed

R5 trained a walk-forward classifier on 51 intraday-developing features
at the chosen entry bar to detect V1-favorable days (rare: 4.9% of
day-trades).

Result:
- Classifier AUC 0.72-1.00 on most windows (signal IS present)
- Precision at threshold 0.5 is 0 on most windows (classifier ranks
  well but doesn't fire confidently)
- Conditional rule PF 1.092 vs blanket V0 PF 1.132 (−0.040)
- Oracle upper bound (perfect foresight) PF 1.302, so best-case
  conditional beats blanket V0 by only +0.17

The conditional approach doesn't reliably beat blanket V0 at this
sample size and class imbalance (21 positive examples across 432 trades).

---

## Layer-3 (learned exits): deferred

The Phase 2A "A3 = augmented L3 minus mfe_norm" config lifted the
20-day V1 OOS PF from 2.169 to 2.847 (+0.678). If L3 generalizes
with similar lift cross-window, composed V0+L3 could reach ~1.6-1.8.

But we cannot apply the existing L3 model honestly here, because its
training pool (chosen + teacher trades from the 5-fold structure's
training days) overlaps with rolling OOS windows. Proper L3
re-evaluation requires **walk-forward L3 retraining per rolling
window** — 2-3 hours of additional pipeline development + CPU time.

**Flagged as next workstream.** Until then, the honest champion is
V0 + time-stop, PF 1.132.

---

## What this means for deployment

If deploying capital:
- Use V0 (model's chosen direction), not V1
- Expect aggregate PF ~1.1-1.2 with wide per-window variance (0.59 to 2.27)
- Be prepared for losing windows (6/13 historical) — fold-level
  stop-losses or capital floors are the right safety net
- Mean $/trade is +$62, so a year of ~250 trades = ~+$15,500 gross
  on a 1-contract-per-trade basis at $25k
- Per-window PF std 0.546 means the strategy is genuinely regime-
  sensitive; paper trading over multiple 60-day windows before
  committing capital is prudent

If NOT deploying capital yet:
- Complete the L3 walk-forward retraining workstream (likely +0.3-0.7 PF)
- Collect more forward OOS data to expand from 13 to ~15-20 windows
- Revisit conditional V0/V1 with larger training data

---

## Comparison table: old vs new

| Claim | Old | New |
|---|---|---|
| Champion directional rule | V1 (always_put) | **V0 (model direction)** |
| Exit rule | Augmented L3 @ 0.19 (minus mfe_norm) | Time-stop (L3 deferred) |
| OOS PF | 2.847 | 1.132 |
| OOS sample | 20 days | 780 days (13 × 60d) |
| Methodological basis | Single chop-bearish window | 13 disjoint non-overlapping windows |
| Confidence | Low (small N, no cross-window validation) | Higher (but per-window PF still variable) |
| Deployment reliability | Unknown | Historically profitable with known variance |

The old claim was optically better but methodologically fragile. The
new claim is modest but built on far more evidence.

---

## Critical caveat

The new champion's **aggregate PF 1.132 is modest**. At this PF and
mean $62/trade, the strategy is marginally profitable and highly
sensitive to:
- Regime shifts (per-window PF spans 0.59 to 2.27)
- Execution costs (not stress-tested in this pass)
- Single-day large losses (max DD per window reaches 74.6%)

Before deploying capital:
1. Complete the walk-forward L3 retraining (could lift PF to ~1.5-1.8)
2. Add Tier-1 fractional Kelly sizing (may amplify edge if deployed
   with good risk discipline)
3. Stress test slippage (was 0 in this pass; realistic $10-25/RT
   could materially degrade PF)
4. Set up Layer-0 guardrails + daily circuit breakers in the live
   environment

The methodology overhaul has REDIRECTED the strategy toward V0 but
has NOT yet produced a deploy-ready PF. More work is needed before
capital commitment.

---

## Files

**New in this workstream:**
- `v3/harness/rolling_windows.py` (R1)
- `v3/analysis/intraday_feature_audit.py` (R2)
- `v3/analysis/rolling_l2_retrain.py` (R3)
- `v3/analysis/rolling_directional_eval.py` (R4)
- `v3/analysis/conditional_directional_rule.py` (R5)
- `v3/analysis/methodology_overhaul_summary.py` (R6)

**Reference docs:**
- `v3/reference/rolling_directional_eval_2026_04_22.md` (R4 per-window detail)
- `v3/reference/conditional_directional_rule_2026_04_22.md` (R5 conditional test)
- `v3/reference/methodology_overhaul_summary_2026_04_22.md` (this doc)

**Artifacts:**
- `v3/artifacts/rolling_l2/` (13 per-window L2 models + manifest)
- `v3/artifacts/rolling_directional_eval/` (per-window + aggregate variant eval)
- `v3/artifacts/conditional_directional_rule/` (conditional classifier)
- `v3/artifacts/methodology_overhaul_summary/summary.json`

---

## Retired claims

1. ~~"V1 (always_put) is the right default"~~ — V0 wins 12/13 strict windows, 10/13 by `> 0.10 PF`
2. ~~"OOS PF 2.847 is reproducible"~~ — window-specific artifact
3. ~~"V1+L3@0.19 is a deployable champion"~~ — uses leaked L3 + wrong direction

## Active claims (as of 2026-04-22)

1. **V0 is the right default directional rule** (across 13 regimes)
2. **Aggregate cross-window V0 PF is 1.132** — the baseline for
   any future improvement to beat
3. **L3 walk-forward re-evaluation is the single highest-EV next
   workstream** — could materially lift the aggregate

---

## Verification

- [x] All 13 rolling windows verified disjoint (R1)
- [x] 9 new features spec'd with 0 NaN across 986 days (R2)
- [x] 13 L2 models retrained with augmented features (R3)
- [x] Per-window V0 vs V1 table with beat_V1 counts (R4)
- [x] Walk-forward conditional classifier tested; tie (R5)
- [x] Oracle upper bound computed (1.302); establishes ceiling
- [x] R6 summary + summary.json
- [ ] Commit
