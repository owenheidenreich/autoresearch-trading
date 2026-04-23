# Simulated-L3 Oracle Per Candidate Contract — Design — 2026-04-22

## Why this is the only remaining defensible outer-loop target

Three failed composed-utility targets have now produced the same
seed-dependent collapse pattern:

| target | falsification | common failure |
|---|---|---|
| side-contrastive weight sweep | cycle 7 | seed 43 W05 overtrade |
| `best_exit_pnl` blend (α=0.10, 0.30) | cycles 8-9 | late-session peaks L3 can't capture |
| fixed-horizon (60 bars) target | cycles 10-12 | bimodal L3 hold ignored |

Across all three, seed 42 benefited and seeds 43/44 collapsed. Each
proxy was misaligned with what the champion Layer-3 actually
realizes at deployment.

The only target that by construction *cannot* be misaligned with
deployment is the oracle that **literally simulates what champion
Layer-3 would do on each candidate contract**. That is this cycle's
design.

## Goal

For every `(day, bar, candidate_contract_token)` in the action-
surface dataset where `tradeable_mask == 1`, produce an
`l3_exit_pnl` label: the PnL the champion `CPU w=0.00 + L3 robust
0.90` policy would have realized if it had entered that contract
at that bar.

Flat action (`action_id=0`) → `l3_exit_pnl = 0.0`.

## Champion specification the oracle must match

Champion is fully defined by two artifacts:

1. **Per-window Layer-3 classifier**: `HistGradientBoostingClassifier`
   trained on the seed-42 champion run's `chosen_trades.pkl` for
   windows `0..W-1`, evaluated at window `W`. 79 features =
   `v2_features + trade_state (7)`. Label is
   `int(current_pnl >= suffix_max)`. Fold-0 uses time-stop
   fallback (no classifier available).

2. **Per-window robust-90 threshold**: Chosen by
   `calibrate_threshold.py --policy prior_window_robust
   --robust-slack 0.90`. Lowest threshold within `0.90 × best_PF`
   on prior windows' OOS rows.

Recorded per-window for seed 42 in
`v3/artifacts/layer3_unified_cpu_w000_seed42/rolling_layer3_calibrated_robust_90.json`.

## Algorithm (per candidate simulation)

```
inputs: day, entry_bar, contract_strike, contract_right, window_idx
output: l3_exit_pnl, l3_exit_bar, l3_exit_trigger

1. load sidecar(day); get contract path for (strike, right)
2. compute entry_mid, entry_spread_fraction, entry_ask
3. for t in (entry_bar+1 .. session_end_bar):
     compute current_pnl_t using spread-adjusted exit_bid
     compute trade_state_t = [bars_since_entry, bars_to_end,
                              current_pnl_norm, mfe_norm,
                              mae_norm, mfe_bar_age, is_call]
     build feature row_t = concat(v2_state_features_t, trade_state_t)
4. if window_idx == 0 (no prior-window data):
     exit at session_end_bar (time-stop)
     l3_exit_pnl = PnL at session_end
     l3_exit_trigger = "fold0_time_stop"
5. else:
     model, threshold = champion_models[window_idx]
     probs = model.predict_proba(feature_rows)[:, 1]
     for i, p_i in enumerate(probs):
       if p_i >= threshold:
         exit at bar i
         l3_exit_pnl = current_pnl_i
         l3_exit_trigger = "model"
         break
     else:
       exit at session_end (time-stop fallback)
       l3_exit_trigger = "time_stop_fallback"
```

This is identical to `replay_trade_set` in
`v3/layer3/common.py`, run once per (bar, contract) tuple.

## What the existing `train_models_by_window` trains on

Champion L3 models are trained on chosen trades, NOT on arbitrary
candidates. Specifically: `seed-42 unified entries' chosen_trades.pkl`.

### Option A — champion-match (chosen)

Train L3 per window on champion seed-42 chosen trades only. Apply
to all candidate contracts at labeling time. This **is** what the
addendum asked for: "PnL the current champion Layer-3 policy would
have realized."

**Distribution shift**: at labeling time, we apply a model trained
on chosen (higher-quality) entries to arbitrary candidates. The
trade_state trajectories of "random contracts" will skew more
negative in current_pnl / mae than the training distribution, which
the classifier may handle unreliably.

**Known risk, accepted for this first cycle**: the addendum is
explicit that we compute "what the champion L3 would have realized,"
so we respect that. If results are promising we revisit with
candidate-trained L3 in a future loop.

### Option B — candidate-trained L3 (rejected for this cycle)

Train L3 on all candidate contracts. Matches label-time
distribution, but is no longer the champion L3. Out of scope.

## Artifact schema

Single `.npz` file under `v3/artifacts/`:

```
v3/artifacts/simulated_l3_oracle_seed42.npz
  row_order            (n_rows,)      int32  — row index in action-surface dataset
  l3_exit_pnl          (n_rows, 25)   float32 — PnL (0 for flat, NaN for non-tradeable)
  l3_exit_bar          (n_rows, 25)   int32   — absolute exit bar (or -1)
  l3_exit_trigger      (n_rows, 25)   int8    — 0=flat, 1=model, 2=time_stop_fallback,
                                                3=fold0_time_stop, -1=non_tradeable
  meta_json            str            — seed, champion config, build stats
```

Aligned to the existing action-surface dataset rows: the i-th row
in `l3_exit_pnl` corresponds to the i-th row in
`v3/artifacts/layer2_action_surface_dataset.pkl`.

Single canonical oracle, seed-42-based. If ensemble needed later,
produce `simulated_l3_oracle_seed43.npz` etc. separately.

## Infrastructure scope

New files:
- `v3/layer2/build_simulated_l3_oracle.py` (build script, ~200 lines)

Modifications:
- `v3/layer2/train_unified_policy.py`: add
  `--utility-target simulated_l3 --simulated-l3-oracle <npz>` flag;
  `_slice_inputs` reads `l3_exit_pnl` when target=simulated_l3

No schema changes to the action-surface dataset. The oracle is a
sidecar artifact.

## Compute estimate

- Tradeable cells in dataset: ~1.0M
  (89,692 rows × avg 11 tradeable candidates per row)
- Per simulation: ~300 bars of feature construction + classifier
  inference (`HistGradientBoostingClassifier`, 200 trees, depth 4)
- Per simulation time (benchmark needed): expected 5-10ms
- Total build time: ~1-3 hours on CPU

Checkpoint/resume: process days one at a time; persist partial
results. Crash-safe.

## Validation before running 3-seed retrain

Before re-training L2 policies against the oracle, sanity-check:

1. **Oracle PnL on champion seed-42 chosen trades must match
   the champion's realized composed PnL.** For each
   (day, bar, chosen_action_id) in `layer3_unified_cpu_w000_seed42/
   layer3_trades_calibrated_robust_90.csv`, look up
   `l3_exit_pnl[row, chosen_action_id]` in the oracle. The two must
   match within floating-point noise. Any material gap is a bug.

2. **Oracle distribution**: mean/median/std, winners rate, vs
   time_stop_pnl distribution. Expect:
   - lower std than time_stop_pnl (early exits cap losses)
   - more winners than time_stop_pnl (L3 takes profits)
   - higher mean than time_stop_pnl (positive L3 alpha)

Only if both checks pass do we proceed to retrain + compose.

## Stop conditions (per addendum)

- If oracle validation (1) fails → bug; debug before shipping oracle
- If oracle is trustworthy but first retrain regresses the PF floor → stop
- If infrastructure requires broader rewrites than above → stop

## Scope for THIS loop

Minimum three concrete cycles:

| cycle | step | commit |
|---|---|---|
| 13 (now) | design note | this note |
| 14 | build script + oracle file + validation-1 | plumbing |
| 15 | trainer wiring + 3-seed retrain + L3 compose + compare | experiment |

If cycle 14 reveals the build takes multi-hour scale or a schema
issue appears, stop and commit partial progress.
