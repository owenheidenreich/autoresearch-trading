# Mechanical Baseline V2 — Learned Scorer, Implementation Plan

**Thesis source:** [strategy_card_opening_reversion.md](strategy_card_opening_reversion.md)
**Upstream plans:**
- [mechanical_baseline_plan_opening_reversion.md](mechanical_baseline_plan_opening_reversion.md) (V1A/V1B)
- V1A result logged in [../lab_notebook.md](../lab_notebook.md) (2026-04-19)
- V1B result logged in same
**Feature source:** [feature_shortlist_opening_reversion.md](feature_shortlist_opening_reversion.md)

V1A and V1B falsified the hand-coded reclaim/reject trigger but did not
falsify the broader opening-reversion thesis: V1B widened the strategy-vs-
Control-A gap to +1.02pp (from V1A's +0.16pp), but the aggregate mean is
still essentially zero. V2 tests whether a learned scorer over the 12-core
features can separate from Control A where the hand-coded trigger could not.

## Context

The V1B trigger evaluates five conjunctive conditions per bar
(overextension + reclaim + first15_acceptance alignment + bar_delta
alignment + session window). A conjunction of continuous conditions
thresholded at hard cut-offs is a crude ranker: it produces a binary
{trigger, no-trigger} classification instead of a continuous preference
across the same feature space. V2 asks whether replacing that binary with
a learned regression score improves bar selection.

## Hypothesis

A learned scorer over the 12-core opening-reversion features can rank bars
within V1B-admissible days better than the hand-coded reclaim/reject
trigger, and better than same-day random-bar control.

## What is frozen from V1B (do not change)

- Opening-structure reversion thesis.
- V1B premium gates — `iv_percentile ≤ iv_max` and `vrp ≤ vrp_max` —
  selected per fold from the same 3×3 grid on the same train-only data
  using the same `MIN_N_TRAIN = 30` rule. Every fold selected
  `iv_max=0.8, vrp_max=train_p75` in V1B; V2 reuses that selection
  mechanism, so the configs are expected to be identical, but V2 is not
  required to re-pick by hand.
- Contract selection: delta band `[0.45, 0.55]`, dual spread gate
  (context `option_spread_pct ≤ 0.20` + per-contract `spread_fraction ≤
  0.20`), tiebreaker = `|delta − 0.50|` nearest → lowest `spread_fraction`
  → lowest `|moneyness_pct|`.
- Exit engine: VWAP re-cross > first-15 boundary touch > 30-min or 11:30
  time stop > EOD safety.
- One trade per day.
- 5-fold walk-forward; train on `fold.train_days`; test on
  `fold.test_days`; no test peeking.
- Two controls: A (same-day random-bar, strategy-side), B (same-bar
  random-day, strategy-side), both with V1B gates applied.

## What changes

- **Bar selection.** The hand-coded V1A/V1B reclaim/reject trigger is
  replaced by a learned regression scorer. At each V1B-admissible bar
  (gates pass, contract exists), the scorer predicts expected `net_pct`
  for that (bar, side) pair. The day's trade is the highest-scoring
  (bar, side) pair. No threshold: if any admissible pair exists in the
  day, V2 trades it.
- **Side determination.** Also by the scorer. For each admissible bar,
  both sides (C, P) are scored independently (same model, side indicator
  as input). The argmax across (bar, side) in the day becomes the trade.

## Interpreting "V1B-admissible"

"V1B-admissible" is read at the **bar level**: a bar is admissible if the
V1B gates (iv_percentile and vrp thresholds for that fold) pass AND a
side-specific contract exists at that bar passing the dual spread gate.
This is the natural candidate universe for a bar-ranker: the trigger is
fully replaced.

This is broader than "V1B-fired bars" (which would also require the
hand-coded reclaim/reject pattern). The broader reading is the fairer
test of the hypothesis, because the claim being tested is that the
trigger CONSTRUCT is limiting, not just which-bar-within-trigger is.

## Learned scorer specification

### Inputs (13 dimensions)

Twelve core features from the
[feature_shortlist](feature_shortlist_opening_reversion.md), plus a side
indicator. Values are read from `data["X"]` — the rolling-z-score
normalized tensor in `v2/data.pt` — which is standard practice for ML
consumption and matches how a downstream ML stage would naturally be fed.

| # | feature | idx in X | notes |
|--:|---|--:|---|
| 1 | `vwap_dist` | 6 | normalized |
| 2 | `vwap_reclaim_state` | 37 | bounded {-1, 0, 1}, not renormalized |
| 3 | `vwap_slope` | 28 | normalized |
| 4 | `opening_gap_pct` | 32 | normalized |
| 5 | `session_open_dist` | 33 | normalized |
| 6 | `first15_close_position` | 35 | bounded [0, 1], not renormalized |
| 7 | `first15_acceptance` | 36 | bounded [-1, 1], not renormalized |
| 8 | `bar_delta` | 22 | bounded [-1, 1], not renormalized |
| 9 | `volume_climax_signal` | 44 | bounded [0, 3], not renormalized |
| 10 | `atm_gamma` | 49 | normalized |
| 11 | `atm_theta_per_bar` | 50 | normalized |
| 12 | `option_spread_pct` | 53 | normalized |
| 13 | `side` | — | `+1` for C, `−1` for P |

Indices are looked up by name from `data["feature_names"]` at runtime
(robust to any reshuffling).

### Output

Regression target: `net_pct` — simulated trade net return at that (bar,
side) pair using the frozen V1B pipeline (contract select → exit engine →
adaptive spread cost).

### Model class

`sklearn.ensemble.RandomForestRegressor`, fixed hyperparameters:

- `n_estimators = 200`
- `max_depth = 5`
- `min_samples_leaf = 20`
- `random_state = <fold_seed>` (see below)
- `n_jobs = -1`

Rationale: linear is too limited for the conjunctive conditional
structure the hand-coded trigger represents (the trigger is literally a
conjunction of inequalities). RF captures those interactions naturally
while staying interpretable via feature importance. Hyperparameters are
fixed (not tuned) to avoid another source of train-time selection bias;
the user-locked discipline is "choose gate thresholds from train only",
not "sweep model hyperparameters on train". One predeclared model, one
fit per fold, no variation.

### Training data construction (per fold)

For each `day` in `fold.train_days`:

1. Load sidecar, compute V1B gate values at each bar (iv_percentile, vrp
   from the unnormalized `X_sim`).
2. For each bar `t ∈ [BAR_LO, BAR_HI]`:
   - Check V1B gates; skip if fail.
   - For each `side ∈ {C, P}`:
     - Run contract selection; skip side if no contract.
     - Run exit engine; skip side if exit is unpriced.
     - Compute `net_pct` via the same `compute_pnl` call used elsewhere.
     - Record `(X[gi, core_idx], side_indicator, net_pct)`.
3. Stack all records into `(n_train_samples, 13)` feature matrix +
   `(n_train_samples,)` target vector.

Expected scale per fold: ~800 train days × ~30-80 gate-passing bars/day ×
up to 2 sides × contract-available-fraction ≈ 30,000-60,000 training
samples. Tractable on CPU.

### Training

Fit `RandomForestRegressor` on the fold-specific training matrix. Store
the trained estimator and log feature importances for the fold report.

### Inference (per test day)

1. Load sidecar, compute V1B gate values at each bar.
2. For each bar in `[BAR_LO, BAR_HI]`:
   - Check V1B gates; skip if fail.
   - For each `side ∈ {C, P}`:
     - Run contract selection; skip side if no contract.
     - Enqueue the candidate `(bar, side, features_row)`.
3. If no candidates: no trade that day.
4. Otherwise: score each candidate via the trained model. Take argmax on
   predicted `net_pct`. Run the exit engine on that (bar, side). Record
   the realized trade.

No threshold is applied. If any candidate exists, V2 trades. This is
deliberate: the first pass tests whether ranking helps at all. If V2
passes, a follow-up can examine whether thresholding the top-score
improves it further.

## Controls (both apply V1B gates)

- **Control A — same-day, random-bar, same-side-as-V2**. Draw one random
  bar uniformly in `[BAR_LO, BAR_HI]`, use the V2-chosen side for that
  day, run V1B gates and contract selection. If any check fails, skip
  with the reason.
- **Control B — same-bar (V2's chosen bar), random-day, same-side-as-V2**.
  Randomly pick a different test-day from the same fold, use V2's chosen
  bar and side on that day.

Controls preserve V1B's apples-to-apples logic: the only thing they do
not have is the scorer's bar choice. Any edge V2 shows over them is
attributable to the scorer.

## Falsification criteria

Same structure as V1A/V1B:

- `N < 20` → `inconclusive`
- `mean_net_pct ≤ 0` → failure contributor
- `target_hit_frac ≤ stop_hit_frac` → failure contributor
- `control_A_mean_net_pct ≥ strategy_mean − 0.5 × stderr` → primary
  failure contributor (this is the decisive comparator per the user
  framing)
- `control_B_mean_net_pct ≥ strategy_mean − 0.5 × stderr` → secondary
  failure contributor

Clear fail (`mean_net_pct ≤ −0.01`) → accept falsified; do not patch.
Near-miss (`mean > 0` and one control margin breached) → flag for
follow-up, but no V3 plan is pre-committed (per user: "no V1C-style
trigger tweaking").

## Fold 2 postmortem (out of V2 scope)

Fold 2 (summer 2025) dragged both V1A (−7%) and V1B (−6.7%) aggregates.
It is explicitly **not** the V2 branch. V2 is evaluated on the aggregate
and on per-fold detail; if V2 also struggles on Fold 2, that's evidence
for a regime limit, documented after the V2 run is complete.

## Locked parameters

| parameter | locked value | source |
|---|---|---|
| Session window | `[15, 120]` | strategy card |
| Time stop | 30 bars or bar 120 (11:30 ET) | strategy card |
| Delta band | `[0.45, 0.55]` | strategy card |
| Spread gate (context) | `option_spread_pct ≤ 0.20` | strategy card |
| Spread gate (contract) | `spread_fraction ≤ 0.20` | V1B plan |
| Gate selection grid | `iv_max ∈ {0.6, 0.7, 0.8} × vrp_max ∈ {median, p75, zero}` | V1B user lock |
| `MIN_N_TRAIN` (gate selection) | 30 | V1B |
| Model class | RandomForestRegressor | V2 (this plan) |
| Model hyperparameters | `n_estimators=200, max_depth=5, min_samples_leaf=20` | V2 (this plan) |
| Trades per day | 1 | V1A/B |
| Folds | 5 canonical, test_days only | V1A/B |
| Random seed (controls + RF) | `fold_idx * 10007 + 1` | V1A/B |

## Critical files

**Read (unchanged):**
- [v2/analysis/mechanical_baseline_opening_reversion.py](../analysis/mechanical_baseline_opening_reversion.py) — V1A helpers
- [v2/analysis/mechanical_baseline_v1b_opening_reversion.py](../analysis/mechanical_baseline_v1b_opening_reversion.py) — gate selection + TriggerEval collection pattern
- [v2/core/chain_data.py](../core/chain_data.py) — `load_sidecar_cached`, `sidecar_path`
- [v2/core/walkforward.py](../core/walkforward.py) — `generate_folds`

**Write:**
- `v2/analysis/mechanical_baseline_v2_learned_scorer.py` (new, ~600 lines)
- `v2/artifacts/mechanical_baseline_opening_reversion_v2/` (results)

## Output contract

```
v2/artifacts/mechanical_baseline_opening_reversion_v2/
├── summary.json                  # aggregate metrics, verdict, per-fold rows
├── trades.csv                    # all trades (strategy + controls)
├── trades_fold{0..4}.csv
├── skips.csv
├── controls.json
└── report_fold{N}.json           # selected_gate_config + feature_importance
                                  # + grid_diagnostics + strategy_trades
```

Trade records include all V1B fields plus `predicted_net_pct` (the
scorer's output at the chosen bar) and `score_rank_in_day` (rank of the
chosen score among that day's candidates).

## Implementation checklist

1. New module `v2/analysis/mechanical_baseline_v2_learned_scorer.py`
   importing shared helpers (`detect_trigger` NOT needed — replaced;
   `select_contract`, `run_exit`, `compute_pnl`, `context_spread_at_bar`,
   `first15_levels`, `load_data`, `day_ranges`, `feature_index_map`,
   `build_folds`, `summarize_trades`, `falsification_verdict`,
   `write_csv`, `write_json`, `TRADE_FIELDS`, `SKIP_FIELDS`,
   `BaselineTrade`, `SkipRecord`, `BAR_LO`, `BAR_HI`) from V1A, and
   `pick_best_config`, `check_gates`, `V1BGateConfig`, `GATE_IV_CANDIDATES`,
   `MIN_N_TRAIN`, `evaluate_triggers_on_day` from V1B (for gate
   selection).
2. `CORE_FEATURE_NAMES` constant = list of 12 names exactly as in the
   shortlist.
3. `collect_training_samples(fold_spec, sidecar_dir, X, X_sim, spot,
   day_to_range, idx_map, gate_config)` — walks train days, enumerates
   gate-admissible (bar, side) pairs, simulates each, returns
   `(features_matrix, labels_vector, metadata_list)`.
4. `train_scorer(features, labels, seed)` — fits RF, returns model.
5. `score_day_candidates(sc, X_day, X_sim_day, spot_day, idx_map,
   core_feature_indices, gate_config, model)` — enumerates test-day
   candidates, scores them, picks argmax.
6. `run_v2_strategy_day(...)` — wraps score_day_candidates + trade
   construction.
7. `run_v2_control_A(...)` / `run_v2_control_B(...)` — same as V1B but
   use V2's chosen side (which is per-trade, not per-fold).
8. `run_v2_fold(...)` — orchestrates gate selection, training set
   collection, training, test-day scoring, controls.
9. CLI `main()` with `--fold`, `--out-dir`, `--controls/--no-controls`.
10. py_compile; fold-0 smoke; full 5-fold run; log in lab notebook;
    commit.

## What V2 does not do

- No hyperparameter search (RF hparams are frozen in this plan).
- No threshold selection (always trade top candidate if one exists).
- No V1C / V3 plan is pre-committed.
- No reopening of continuation or vol-expansion theses.
- No changes to the trigger construct, contract-selection rule, exit
  engine, or gate grid.
- No feature engineering beyond the 12 shortlist cores.
- No cross-fold model sharing; each fold trains from scratch.
- No Fold 2 special-casing in V2 evaluation.
