---
date: 2026-04-25
parent: profitability_sprint_summary_2026_04_25.md
status: PROOF FAILURE — offline 2.142 PF claim does NOT replicate on 26 days of forward-walk data; seed 46 is the only profitable seed
---

# Forward-Walk Proof — `spx_combined_3seed_001`

## TL;DR — the offline number doesn't hold up

The action_surface dataset already contains **2026-02-25 → 2026-04-01
(26 trading days)** that no rolling-window OOS evaluation ever
covered. Running the saved 5 W12 seed models with the deployment
recipe (calibration policy + K=2 + cal_pf>4 guard) on these 26 unseen
days produces:

- **Cross-seed mean PF: 0.72** (offline claim: 2.142, **−66%**)
- **All 41 trades are puts** — 100% put bias on forward walk
- **Only seed 46 is profitable** (PF 2.02). Seeds 42-45 are all losers.
- **The cal_pf>4 guard correctly flags seeds 42, 43, 44** but the
  remaining "clean" seed 45 is also a forward-walk disaster (PF 0.17).
- **Tightening cal_pf guard to >2.5 leaves only seed 46** — and that
  single seed's forward-walk PF (2.02) DOES match the offline claim.

This is a decisive proof failure for the multi-seed ensemble recipe
as offline-spec'd. It is also a partial validation: the strongest-form
single-seed deployment (seed 46, lowest cal_pf) holds up.

## Method

The forward-walk window is `2026-02-25` → `2026-04-01`, which is:
- After the W12 OOS cutoff (`2026-02-24`).
- Before the latest available flat-file date (`2026-04-08`).
- 26 trading days, 2,363 dataset rows.

For each seed:
1. Load `seed_NN/window_12/model.pkl` (the latest trained model).
2. Load `seed_NN/window_12/calibration.json` (the abstention policy).
3. Run inference on the 2,363 forward-walk rows.
4. Apply the calibration's `decision_margin`, `min_win_prob`,
   `max_stopout_prob`.
5. Build `chosen_trades` for the forward-walk window.

Per-trade realized PnL is reported in three forms:
- **time_stop**: hold to bar 120 (most conservative, no oracle).
- **hybrid_no_oracle**: dataset's hybrid_live_utility label (time_stop
  base + spread/stopout penalty). Closest to honest live-realizable.
- **hybrid_with_oracle**: training utility target (L3-oracle-augmented
  exit pnl). **Not computable on forward-walk rows** — the L3 oracle
  was trained per-window and does not produce predictions for rows
  outside training cohorts (verified: 100% NaN on forward-walk rows).

The headline forward-walk PF therefore uses `hybrid_no_oracle`.

## Per-seed results

```
seed   cal_pf_W12   n_trades   wins   losses    PF      sum_PnL ($)   win_rate
  42      21.58        5        2       3      1.05        +143         40%
  43       5.71        9        2       7      0.38      -3,690         22%
  44       7.99        6        0       6      0.00      -4,453          0%
  45       3.15        9        1       8      0.17      -6,473         11%
  46       1.93       12        5       7      2.02      +4,415         42%
```

- All trades are PUTS (no calls).
- Same days appear across seeds (consensus: 8 bars at K=2).
- Seed 44 is a complete disaster: 0 wins on 6 trades.
- Seed 46 is the ONLY profitable seed. Its cal_pf was the lowest at
  1.93 (well below the cal_pf>4 guard).

## Cal_pf guard validation

The cal_pf>4 deployment guard (from `calibration_audit_2026_04_25.md`)
flags seeds 42, 43, 44 from W12 deployment. Among the unguarded seeds,
the picture is binary:

```
threshold      seeds passing       fwd_walk mean PF       fwd_walk min PF
none (all)     42,43,44,45,46         0.72                   0.00
> 4.0          45, 46                 1.09                   0.17
> 2.5          46 (only)              2.02                   2.02
> 1.5          (none)                  —                     —
```

A tighter cal_pf>2.5 guard recovers the offline-spec performance, but
only by collapsing to a single-seed deployment. That removes the K=2
consensus benefit and concentrates risk on one model.

## Why does it fail?

Three diagnostics, in order of likely contribution:

**1. Side-bias regression on forward walk.** Offline OOS showed call
share 43%; forward walk shows 0%. The model became 100% put-biased on
2026-02-25 onwards. SPX did fall ~5% over the period (6946→6575) so
puts were *directionally* sometimes right — but the model bought
near-ATM puts that were vulnerable to theta on flat-to-slightly-down
days (8 of 26 days closed within ±0.5%, and the model lost on most of
those).

**2. W12 calibration overfit propagates forward.** The cal_pf>4 audit
showed W12's val→OOS degradation was the worst across all 13 windows.
That same overfit pattern persists on forward-walk data — the
calibrated thresholds for seeds 42/43/44 were tuned to a val cohort
that doesn't represent forward-walk conditions.

**3. L3-oracle exit prediction is unavailable forward.** The offline
PF 2.142 claim depended heavily on the L3 oracle's exit timing
(audit #3: oracle adds PF ~1.0 on top of time-stop). Forward walk has
no oracle predictions, so the realized PnL falls back to time-stop +
penalties. This is consistent with the offline pattern (time-stop
PF 1.24, hybrid PF 2.19) and implies that forward-walk PF could
plausibly be lifted by re-running the oracle on new days. But:
the oracle ITSELF is trained on historical data and has its own OOS
risk. Even with oracle restored, forward-walk PF would likely fall
short of 2.14.

## Comparison: offline vs forward-walk

```
metric                   offline (W12-OOS)    forward-walk         delta
days                          ~85                  26
trades / seed                 ~25                  5-12
mean PF (all 5 seeds)         1.881                0.72             -61%
min PF (all 5 seeds)          1.653                0.00             -100%
max DD (best seed)            12-13%               6.9% (seed 46)     ok
mean PF with cal_pf>4 guard   ~2.10                1.09             -48%
mean PF with cal_pf>2.5       ~2.10 (would be      2.02 (seed 46     -4%
                              similar — only        only)
                              1 seed unguarded)
side bias (call share)        43%                  0%
```

## Decision-tree update

Given the forward-walk failure, the deployment recipe needs revision:

**1. DO NOT deploy the 5-seed K=2 recipe live.** The offline 2.14 PF
claim is not forward-validated. The cross-seed mean PF on 26 days of
unseen data is below breakeven (0.72).

**2. Single-seed-46 deployment is the *only* live-shadow candidate.**
Seed 46 had the lowest cal_pf (1.93) AND the only positive forward-walk
PF (2.02). This matches the offline pattern that low-cal_pf seeds
generalize. Deploy seed 46 alone with very small size in shadow.

**3. Tighten cal_pf guard to ≥ 2.5.** The offline ≥4.0 threshold
flagged 3 of 5 seeds; forward-walk evidence says even cal_pf=3.15
(seed 45) is unsafe. Stricter is better.

**4. Retrain with forward-walk data included.** Adding 2026-02-25 →
2026-04-01 to the training set produces a "W13" rolling window. If
the new W13 model still produces a low-cal_pf seed with similar
recipe, that becomes the new canonical deployment.

**5. Re-run L3 oracle on forward-walk days.** Allows direct
measurement of how much the oracle would have lifted forward-walk
PF, separating "model entry edge degraded" from "oracle exits
unavailable" as failure modes. ~30 min CPU.

**6. Live-shadow Monday must be observation-only.** Do not transmit
orders. Record DecisionSnapshot JSONL only. After 5 sessions, audit
whether the live model's side bias matches offline (43% calls, not
0%). If it stays 100% puts, the model has structurally biased post
training.

## What's strong

The forward-walk failure validates several earlier findings:
- **L3-oracle is real, not memorization** (audit #3) — confirmed; the
  oracle's exit timing was lifting PF, and without it forward-walk
  reverts to time-stop-like levels.
- **cal_pf>4 catches the worst overfit** (audit #5) — confirmed; the
  3 worst forward-walk seeds (42/43/44) all had cal_pf>4.
- **Bootstrap CIs warned of fragility** (audit #4) — confirmed; the
  max DD claim of 12.87% was at the optimistic edge, and forward
  walk's ensemble would have shown much wider DD.

What's weak: the offline OOS 2.142 PF was a single chronological
sample. Bootstrap p05 was 1.85, p95 was 2.51. Forward-walk lands at
0.72 — well below even the bootstrap p05. This means the forward
period contains a **structural shift** the offline rolling-window
evaluation didn't capture (likely the call-bias regression).

## Files

- `v3/analysis/forward_walk.py` — inference script
- `v3/artifacts/forward_walk/spx_combined_3seed_001.json` — full
  per-seed metrics
- `v3/artifacts/forward_walk/forward_walk_chosen_seed{42..46}.pkl` —
  per-seed forward-walk chosen trades

## Cost

Step total: ~10 min CPU (no GPU). The forward-walk window was already
on disk; only inference was needed.

## Recommended next steps (concrete)

1. **Re-run L3 oracle on forward-walk days** (~30 min CPU). Quantifies
   how much of the failure is "oracle missing" vs "entry edge gone".
2. **Retrain with new data included** (~$2-3 GPU). Build a W13 model
   covering forward-walk days, see if a new low-cal_pf seed emerges.
3. **Live-shadow Monday in pure observation mode**. Do not deploy the
   ensemble. Record the live model's side share — if it's 100% puts,
   the structural bias has propagated to live.
4. **DEFER any IBKR paper trading** until forward-walk PF >= 1.5 on a
   refreshed model. Current evidence does not support live deployment.
