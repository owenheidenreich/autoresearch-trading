# W1 Surface Branch — Sequence + Contract Surface — 2026-04-22

## Verdict

The first W1 branch is live and working.

It directly addresses the audit's main criticism of the old Layer-2:
instead of seeing only a flattened point-in-time bar vector, the new
branch sees:

- the current scalar bar-state features
- the last `20` bars of sequential context
- the top `8` passing calls and top `8` passing puts as explicit tokens

On the existing 5-fold Layer-2 harness, a short-budget run (`6` epochs,
shared encoder, model-direction only) produced:

- `287` trades
- `PF 1.571`
- `DD 15.3%`
- mean `+$235/trade`
- `0.957` trades/day

Artifact:
- [v3/artifacts/layer2_surface_shared/replay_report.json](../artifacts/layer2_surface_shared/replay_report.json)

## Why this matters

The stable CPU flat-feature reference in the same 5-fold world was:

- `287` trades
- `PF 1.122`
- `DD 56.2%`

So even this first structured-input branch materially improves the old
flat Layer-2 path.

This does **not** prove the final architecture yet. It only proves that
the audit's criticism was directionally right: giving the model a richer
view of recent price action and executable contracts helps.

## Surface bundle

New dataset artifact:
- [v3/artifacts/layer2_surface_dataset.pkl](../artifacts/layer2_surface_dataset.pkl)

Shapes:

- rows: `89,692`
- sequence tensor: `89,692 × 20 × 15`
- call contract tensor: `89,692 × 8 × 8`
- put contract tensor: `89,692 × 8 × 8`

Sequence features:

- `underlying_close`
- `vwap_dist`
- `vwap_slope`
- `volume_ratio`
- `first15_range_pct`
- `bars_since_break_above_first15`
- `bars_since_break_below_first15`
- `sigma_pos`
- `omar_retest_dist_norm`
- `omar_range_pct`
- `last10_range_over_omar`
- `inside_first15`
- `late_window_40_120_flag`
- `omar_mid_pos_units`
- `last10_break_state`

Per-contract token features:

- `strike_offset_pct`
- `premium`
- `delta`
- `abs_delta`
- `spread_fraction`
- `gamma_dollar`
- `theta_to_premium`
- `selection_score`

## Caveat

This is still a **bridge branch**, not the final W1 verdict.

- It uses the existing 5-fold Layer-2 harness, not the newer 13-window
  rolling methodology.
- It keeps the current decomposed entry + side heads.
- It still uses threshold calibration and `per_day_choice`.

So the honest interpretation is:

1. richer inputs help
2. the audit's architectural diagnosis is supported
3. the next real step is to move this branch into the rolling-window
   methodology, not to declare victory

## Files

- [v3/layer2/export_surface_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/export_surface_dataset.py)
- [v3/layer2/surface_dataset.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/surface_dataset.py)
- [v3/layer2/surface_neural.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/surface_neural.py)
- [v3/layer2/train_surface_model.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/train_surface_model.py)
