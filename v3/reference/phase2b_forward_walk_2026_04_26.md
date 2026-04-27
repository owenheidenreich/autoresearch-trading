---
date: 2026-04-26
parent: phase1_filter_validation_2026_04_26.md
spec_id: phase_c_gpt55_v0_2026_04_26
status: FW GATE FAILS on oracle PF + per-seed crashes; veto NOT deployed; ships disabled-by-default; Phase 3 runs to explain why offline lift didn't generalise
---

# Phase 2b: post-L2 veto layer + 8-metric forward-walk gate

## Setup

`v3/layer2/post_filters.py` exposes `apply_avoid_filters_mask` and
`report_favor_diagnostics`. `v3/analysis/forward_walk.py` was extended
with `--apply-veto` (default off) — when set, `apply_avoid_filters_mask`
is applied to `pred_df` BEFORE `_select_daily_trades`, which gives
rescan-after-veto semantics (matches runtime, per Phase 2a).

Comparison setup:
- Champion: `spx_combined_3seed_001`, 5 seeds (42-46), window 12 model
- Dataset: `layer2_action_surface_dataset.pkl` (91148 rows, through 2026-04-24)
- Oracle: `simulated_l3_oracle_spx_live_0945_1130_seedN_balanced_fresh.npz`
  (the rebuilt oracle that covers the FW window; `_balanced` does NOT)
- Forward-walk window: 2026-02-25 → 2026-04-24 (42 days, 3819 candidate rows per seed)

## Headline result

| Metric | Baseline | Veto | Δ | Gate result |
|---|---|---|---|---|
| Mean PF (with oracle) | **1.700** | 1.647 | -0.054 | **Gate 1 FAIL** (target ≥ 1.700) |
| Mean PF (no oracle, dataset label) | 1.140 | 1.226 | +0.087 | informational |
| Mean PF (time_stop, naive hold) | 1.236 | 1.328 | +0.092 | informational |
| Mean DD (with oracle) | 8.38% | 8.81% | +0.43 pp | Gate 4 PASS (≤ +20%) |
| Max DD (with oracle) | 14.49% | 13.75% | -0.74 pp | Gate 4 PASS |
| Trade count (5-seed sum) | 53 | 50 | -5.7% | Gate 3 PASS (≥ 70%) |

Veto application detail: dropped 596–911 of 3819 candidate bars per seed
(15.6%–23.9%) before the daily-best pick. Seed 42 saw 0 days replaced
(its baseline winners didn't match avoid rules); other seeds had 1–9
trades shifted to alternate bars on rescan.

## Per-seed PF deltas (oracle metric — Gate 8)

| Seed | Baseline | Veto | Δ | Verdict |
|---|---|---|---|---|
| 42 | 0.505 | 0.505 | 0.000 | neutral |
| 43 | 0.079 | 0.082 | +0.004 | neutral |
| 44 | **3.672** | 3.395 | **-0.277** | **CRASH (Δ < -0.10)** |
| 45 | 0.358 | 0.176 | **-0.182** | **CRASH** |
| 46 | 3.889 | 4.076 | +0.188 | improvement |

**Two seeds (44, 45) crash the per-seed -0.10 floor — Gate 8 FAILS.**

## Per-seed PF on non-oracle metrics

The story is opposite on metrics that don't depend on the trained
L3 oracle (full label coverage, no model bias):

| Seed | base time_stop | veto time_stop | Δ |
|---|---|---|---|
| 42 | 1.785 | 1.785 | 0.000 |
| 43 | 0.319 | 0.339 | +0.020 |
| 44 | 1.301 | 1.292 | -0.009 |
| 45 | 0.162 | 0.411 | **+0.249** |
| 46 | 2.615 | 2.813 | +0.198 |

| Seed | base no_oracle | veto no_oracle | Δ |
|---|---|---|---|
| 42 | 1.616 | 1.616 | 0.000 |
| 43 | 0.294 | 0.312 | +0.018 |
| 44 | 1.212 | 1.212 | 0.000 |
| 45 | 0.147 | 0.379 | **+0.232** |
| 46 | 2.429 | 2.612 | +0.183 |

On time_stop and no_oracle, seed 44 is flat and seed 45 is significantly
UP — the seeds that "crash" on the oracle metric. The crash exists only
in the oracle-PnL evaluation.

## Why the oracle metric crashes while others lift

The L3 oracle (built by `v3/layer2/build_simulated_l3_oracle.py`) was
trained per seed on the bars/actions the L2 baseline model was likely
to pick. When the veto removes the L2-baseline daily-winner bar, rescan
picks a different bar with a different `chosen_action_id`. The L3
oracle's prediction at that alternate (row, action) pair was trained
on a smaller / less representative subset of trajectories than the
top-1 picks were.

Mechanically:
- The dataset's `time_stop_pnl` and `hybrid_live_no_oracle` labels are
  computed deterministically from market data — they have no
  model-dependent training distribution.
- The L3 oracle's `l3_exit_pnl` is computed by running a learned
  classifier — its calibration is best on the bars the classifier saw
  most frequently during training.

So the FW oracle PF drop is partly an oracle-distribution-mismatch
artifact, not pure veto deterioration. The non-oracle metrics suggest
the veto is selecting bars with comparable or slightly better realised
forward-PnL; the oracle simply gives less-bullish predictions for those
bars.

This is a **Phase 3 finding**: the L3 oracle's training procedure
couples the model and the oracle in a way that makes any post-L2
filtering layer look worse on the oracle metric than it actually is.
Phase 3 documents this as one of the architectural pathologies and
recommends decoupling the oracle's training from the L2's top-1
selection.

## Decision-tree verdict

Per the Phase 2b decision tree:
- Gate 1 (PF ≥ 1.700) — FAIL on oracle metric
- Gate 8 (per-seed Δ ≥ -0.10) — FAIL on oracle metric (seeds 44, 45)
- Gates 3, 4, 6, 7 (trade count, DD, concentration, per-cell floor) — pass or near-pass
- Gates 2, 5 (hl/day, worst-5) — improvement on non-oracle metrics, neutral on oracle

Verdict: **Mixed near-failure**. Per the plan's flowchart this routes to
"revert veto code; Phase 3 explains why."

## What ships

- `v3/layer2/post_filters.py` and `post_filters_v0.py` are kept in the
  tree as research infrastructure.
- `v3/analysis/forward_walk.py` retains the `--apply-veto` flag (default
  OFF). It is opt-in; default behaviour is unchanged.
- `_select_daily_trades` in `v3/layer2/train_unified_policy.py` is
  **not modified**. The veto is applied externally in the FW script.
- No production deployment of the veto layer.

## What Phase 3 must explain

1. Why does L2 emit anti-calibrated `decision_margin` for puts, especially
   in cells where puts win the regime? (calibration objective vs ranking
   objective conflict)
2. Why does the L3 oracle's prediction quality depend on the L2 model's
   top-1 selection? (oracle/policy training coupling)
3. Why are calls with `orc_triggered=True` still entered with margin ~0.60?
   (gate decoupling from feature)
4. Whether the L2 retraining direction should be:
   - architectural redesign (shared-state bottleneck)
   - config flip (turn on side_balance_weight, w_side_contrastive)
   - target re-encoding (action-surface dataset rebuild)
   - L3 oracle decoupling (so the oracle's training distribution
     covers the candidate space, not just L2's top-1 picks)

## Cost

- Dev: ~1.5 hours (post_filters.py + forward_walk.py integration + debugging)
- Compute: ~5 minutes per FW run × 2 runs = ~10 minutes
- Total Phase 2b: ~2 hours, $0 GPU

## Files

- `v3/layer2/post_filters.py` (research-only veto module)
- `v3/analysis/forward_walk.py` (added `--apply-veto` flag, default off)
- `v3/artifacts/forward_walk/spx_combined_3seed_001_baseline_fresh.json`
- `v3/artifacts/forward_walk/spx_combined_3seed_001_with_veto_fresh.json`
- `v3/reference/phase2b_forward_walk_2026_04_26.md` (this doc)
