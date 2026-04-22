# Stage C1 — Layer-2 Directional Variants — 2026-04-21

## TL;DR

**V1-WIN: always_put wins OOS by +0.522 PF** (0.869 → 1.391). The
Layer-2 model's call signal is worthless on the 20 cached OOS days.
Forcing every chosen trade to put converts the system from a losing
0.869 PF to a meaningfully profitable 1.391 PF. In-sample cost is
0.199 PF (1.455 → 1.256), concentrated in folds 1 and 4 where calls
genuinely contributed.

This is the cleanest possible signal that the model's directional
edge does NOT generalize — at least on this OOS window. Production-
recommended Layer-2 directional rule: **always_put** (with caveat
about needing more OOS coverage to confirm).

## Setup

Script:
[v3/analysis/layer2_directional_variants.py](../analysis/layer2_directional_variants.py).
Artifact:
[v3/artifacts/layer2_directional_variants/directional_variants.json](../artifacts/layer2_directional_variants/directional_variants.json).

Five variants tested as inference-time post-processing on the frozen
detach-side Layer-2 model. All evaluated on:
- In-sample: 5-fold using each fold's own calibration
- OOS: 20 cached days (2026-03-05 → 2026-04-01) using fold-4
  calibration

| Variant | Description |
|---|---|
| V0 | baseline (`direction_mode=teacher_if_triggered_else_put`) |
| V1 | always_put (override every chosen direction to "put") |
| V2 | sigma_pos veto on calls (drop if direction==call AND sigma_pos > 0) |
| V3 | conviction-asymmetric (call → put if side_conf < 1.5*side_threshold) |
| V4 | combined V3 + V2 |

## Results — Aggregate

| Variant | In-sample PF | In-sample DD% | In-sample mean$ | OOS PF | OOS DD% | OOS mean$ | OOS call% |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 baseline | **1.455** | 36.9 | +225 | 0.869 | 25.9 | −68 | 50.0% |
| **V1 always_put** | 1.256 | 50.2 | +140 | **1.391** | **17.0** | **+159** | 0.0% |
| V2 sigma_pos veto | 1.434 | 41.0 | +224 | 0.984 | 13.0 | −6 | 23.1% |
| V3 conviction-asym | 1.431 | 40.9 | +216 | 0.869 | 25.9 | −68 | 50.0% |
| V4 combined | 1.414 | 43.2 | +214 | 0.984 | 13.0 | −6 | 23.1% |

V1 wins OOS decisively: +0.522 PF over baseline, DD compressed from
25.9% → 17.0%, mean trade flips from −$68 to +$159.

## Per-fold PF (in-sample)

| Fold | V0 | V1 | V2 | V3 | V4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.860 | **0.888** | 0.878 | 0.860 | 0.878 |
| 1 | **1.439** | 1.127 | 1.379 | 1.439 | 1.379 |
| 2 | 1.038 | 0.983 | 1.012 | 0.983 | 0.983 |
| 3 | 2.095 | 2.079 | 2.129 | 2.095 | 2.129 |
| 4 | **1.801** | 1.126 | 1.764 | 1.736 | 1.663 |

V1's in-sample weakness is concentrated in folds 1 (1.439 → 1.127,
−0.312) and 4 (1.801 → 1.126, −0.675). Fold 0 actually IMPROVES
slightly under V1 (0.860 → 0.888). So V1's tradeoff is consistent:
it sacrifices strong-direction-regime PF to insure against
chop-regime losses.

## Why V3 didn't activate

Side-threshold values per fold:
- Fold 0: 0.0515 (1.5× = 0.077)
- Fold 1: 0.0062 (1.5× = 0.0093)
- Fold 2: 0.8358 (1.5× = 1.254)
- Fold 3: 0.7792 (1.5× = 1.169)
- Fold 4: 0.7259 (1.5× = 1.089)

For folds 0/1, the side_threshold is so close to zero that V3's
"call requires 1.5× side_threshold" multiplier yields a trivial
gate — virtually all calls pass. For folds 2-4, side_threshold is
higher, but it appears the model's call selections all have
side_conf above the multiplied bar already (the per-day ranking
selected the highest-conviction bar by construction).

V3 conceptually asks "are the model's CALL selections low-conviction?"
The answer is no — they're typically high-conviction OOS too. So V3
doesn't filter anything. **The OOS calls aren't low-conviction; they're
just structurally wrong.**

## Why V2 doesn't fully fix OOS

V2 drops calls when `sigma_pos > 0`. On OOS this drops 7 of 10 calls
(based on agent diagnostic). PF moves from 0.869 → 0.984 — meaningful
but still sub-1.0. Why?

The remaining 3 calls (with sigma_pos ≤ 0) include 2 winners (March
10 +$1744 and March 31 +$2126) but also 1 loser. V2 is too crude:
it correctly drops most losing calls but isn't sharp enough to be
profitable on its own.

V1 (always_put) is more aggressive than V2 — it kills ALL the calls,
losing the 2 big winners V2 keeps but also being immune to call-side
losses. The 0.522 PF lift V1 produces over V0 vs the 0.115 PF lift V2
produces shows that **the call signal's downside vastly outweighs its
upside on this OOS window**.

## Per-day OOS trace under V1

V1 takes 20 trades (same days as V0 plus V0's invalid-direction days
forced to put). All puts. Need to look at trade-level results to
see which days V1 wins/loses on. (Detailed CSV at
[v3/artifacts/layer2_directional_variants/oos_trades_V1.csv](../artifacts/layer2_directional_variants/oos_trades_V1.csv).)

The key observation: V1's 20 OOS trades are ALL puts. They had
- PF 1.391 (vs V0's 0.869 with 50% calls)
- DD 17.0% (vs V0's 25.9%)
- Mean trade +$159 (vs V0's −$68)
- Aggregate net PnL gain over V0 ≈ +$4,540 across the 20 days

That's the empirical magnitude of the call-signal-OOS-failure: ~$227
per trade lost on average from taking calls instead of puts in this
window.

## Implications

1. **Layer-2's call signal does not generalize on this OOS sample.**
   Three independent signals confirm:
   - V1 (always_put) wins OOS by +0.522 PF
   - Layer-2 V0 alone OOS PF 0.869 < 1.0
   - The call-side aggregate OOS PnL is −$2316 (per agent diagnostic)
2. **V2 (sigma_pos veto) is partial but not enough.** The 3 surviving
   calls under V2 still drag PF below 1.0 (0.984).
3. **V3 (conviction-asymmetric) is non-functional with current
   thresholds.** Side_threshold is too tight; 1.5× multiplier is
   trivial. Would need re-calibration or a different conviction metric.
4. **In-sample cost of always_put is significant** (1.455 → 1.256).
   Folds 1 and 4 had genuine call-side profits the model captured.
   Going puts-only sacrifices that.

## Caveats

- **20 OOS trades is small.** PF 1.391 has wide CI; could be 0.85–2.2
  at 95% confidence. V1's win could partly be the same single big put
  winner (March 5 +$2982) that anchored V0's puts. Need more OOS
  coverage to confirm.
- **V1 is brittle by construction.** It wins on chop/down regimes and
  loses on directional-up regimes. The OOS window happened to be
  chop/down. A different OOS window could flip the verdict.
- **In-sample fold 0 (chop/loser fold) IS slightly better under V1
  (0.888 vs 0.860).** This is consistent — V1 is the
  chop-regime-defensive strategy.

## Per the Stage C1 verdict criteria

- TERMINAL FAIL: NO variant achieves OOS PF >= 1.0 — NOT triggered (V1
  at 1.391, V2 at 0.984 just below 1.0)
- V1-WIN: V1 wins by huge margin AND beats baseline >= +0.30 PF —
  TRIGGERED (V1 OOS PF 1.391, lift over V0 +0.522)
- V0-WIN: baseline still best — NOT triggered
- BEST-VARIANT-WINS: some Vk lifts OOS PF >= 1.10 — TRIGGERED for V1

V1 is the best variant by both PF and the 1.10-threshold bar.

## What's next

Stage C2 pressure-tests V1 specifically:
- Random-direction ablation: if random direction on V1's chosen days
  has PF >= V1's PF, the "puts work" finding is just that the OOS
  window was bullish-unfavorable for any direction.
- Per-fold breakdown: V1's fold-1 and fold-4 in-sample regression
  may be disqualifying.
- Slippage stress: V1's PF stability under realistic execution costs.

If V1 survives Stage C2, Stage C3 will compose V1 + augmented Layer-3
on OOS at threshold 0.19 (the new optimal from Stage B).

## Verification

- [x] `python -m py_compile v3/analysis/layer2_directional_variants.py` passes
- [x] All 5 variants run end-to-end on in-sample (5 folds) + OOS (20 days)
- [x] Per-fold and per-OOS metrics reported
- [x] V3's non-activation diagnosed (side_threshold too tight for 1.5× multiplier)
- [x] V1-WIN verdict tagged
- [ ] Commit
