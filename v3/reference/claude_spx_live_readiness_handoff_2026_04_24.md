# Claude Handoff: SPX Live-Readiness Repair

Date: 2026-04-24
Branch: `codex/v3-orc-xsp-10k`

## Context

Codex implemented the SPX champion stack repair scaffolding, but intentionally did not run the full expensive pipeline:

- full SPX dataset/artifact rebuild
- 3-seed simulated-L3 oracle rebuild
- 3-seed promotion retrain
- routed Layer-3 composition

The code path passed focused unit tests and a tiny smoke training run. The smoke run only validates schema/runtime wiring; it does not say anything about model quality or promotion.

The main thing to keep in mind: the project is being steered away from "best historical simulated-L3 backtest" and toward a live-executable SPX policy. Distribution and intent mismatch are the blocker, not another scalar tuning knob.

## Files To Review First

Core data contract and action surface:

- `v2/pipeline/build_v2_dataset.py`
  - Review `_build_iv_history_seeds` and the `iv_history_seed` worker plumbing. This is the fix for `iv_percentile` being same-day-only.
- `v3/layer2/action_surface_dataset.py`
  - Review `validate_action_surface_bundle`, risk-band selection, new contract metadata, next-bar fill labels, and `hybrid_live_utility`.
- `v3/layer2/export_action_surface_dataset.py`
  - Review the new `--contract-selection-mode` option. Default is now `risk_band`.

Unified policy:

- `v3/layer2/unified_policy.py`
  - Review the multi-head model outputs and the new losses: utility, dollar, return multiple, win probability, clean entry, stopout, risk-band ranking, cross-band ranking.
- `v3/layer2/train_unified_policy.py`
  - Review `hybrid_live` target construction, schema validation, chosen-contract diagnostics, and `_calibrate_abstention_policy`.
  - Important: this script currently takes one `--simulated-l3-oracle` path per run. For per-seed oracles, run each seed separately or patch the script to accept a seed-to-oracle map.

Layer 3 and oracle:

- `v3/layer2/build_simulated_l3_oracle.py`
  - Review the next-bar fill change in `_simulate_candidate`.
- `v3/layer3/common.py`
  - Review `_build_trade_data`; it now treats `bar_index` as decision bar and fills at `entry_fill_bar` / `bar_index + 1`.
- `v3/layer3/train_rolling.py`
  - Review `--l3-training-source routed_experts`, which trains chosen-distribution and candidate-surface experts and routes by prior-window evidence.

Live shadow:

- `v3/live_shadow/schema.py`
- `v3/live_shadow/resolver.py`
- `v3/live_shadow/feature_parity.py`
- `v3/live_shadow/snapshot.py`

Tests:

- `v3/tests/test_live_readiness_repairs.py`
  - Covers schema failure, risk-band determinism, hybrid utility safety, next-bar fill labels, and feature parity reporting.

Smoke artifact from Codex:

- `v3/artifacts/live_readiness_smoke_time_stop/manifest.json`
  - Do not use this as performance evidence.

## What Passed

Run these again first:

```bash
.venv/bin/python -m compileall \
  v2/pipeline/build_v2_dataset.py \
  v3/layer2/action_surface_dataset.py \
  v3/layer2/export_action_surface_dataset.py \
  v3/layer2/train_unified_policy.py \
  v3/layer2/unified_policy.py \
  v3/layer2/build_simulated_l3_oracle.py \
  v3/layer3/common.py \
  v3/layer3/train_rolling.py \
  v3/live_shadow \
  v3/tests/test_live_readiness_repairs.py

.venv/bin/python -m unittest v3.tests.test_live_readiness_repairs
```

Codex also ran this smoke successfully:

```bash
.venv/bin/python -m v3.layer2.train_unified_policy \
  --tier smoke \
  --device cpu \
  --max-epochs 1 \
  --patience 1 \
  --batch-size 4096 \
  --run-dir v3/artifacts/live_readiness_smoke_time_stop
```

Again: the smoke has one OOS trade and is not a model result.

## Next Work

### 1. Rebuild the SPX data contract

If the current `v2/data.pt` predates the IV-history patch, rebuild it so `iv_percentile` has prior-day rolling context:

```bash
.venv/bin/python -m v2.pipeline.build_v2_dataset \
  --output v2/data.pt \
  --sidecar-dir v2/data_sidecars
```

Then export the repaired default action surface:

```bash
.venv/bin/python -m v3.layer2.export_action_surface_dataset \
  --output v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \
  --contract-selection-mode risk_band \
  --execution-start-bar 15 \
  --execution-end-bar 120
```

Optional widened-window variants for diagnostics only, not promotion by default:

```bash
.venv/bin/python -m v3.layer2.export_action_surface_dataset \
  --output v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1230.pkl \
  --contract-selection-mode risk_band \
  --execution-start-bar 15 \
  --execution-end-bar 180

.venv/bin/python -m v3.layer2.export_action_surface_dataset \
  --output v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1500.pkl \
  --contract-selection-mode risk_band \
  --execution-start-bar 15 \
  --execution-end-bar 330
```

### 2. Rebuild simulated-L3 oracles per seed

Use the current promoted champion paths unless you discover a newer promotion manifest. Likely current references:

- `v3/artifacts/layer2_unified_policy_simL3_objfix_seed42/seed_42/chosen_trades.pkl`
- `v3/artifacts/layer2_unified_policy_simL3_objfix_seed43/seed_43/chosen_trades.pkl`
- `v3/artifacts/layer2_unified_policy_simL3_objfix_seed44/seed_44/chosen_trades.pkl`
- `v3/artifacts/layer3_unified_cpu_simL3_objfix_seed42/rolling_layer3_calibrated_robust_90.json`
- `v3/artifacts/layer3_unified_cpu_simL3_objfix_seed43/rolling_layer3_calibrated_robust_90.json`
- `v3/artifacts/layer3_unified_cpu_simL3_objfix_seed44/rolling_layer3_calibrated_robust_90.json`

Command template:

```bash
for seed in 42 43 44; do
  .venv/bin/python -m v3.layer2.build_simulated_l3_oracle \
    --dataset v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \
    --champion-chosen-trades v3/artifacts/layer2_unified_policy_simL3_objfix_seed${seed}/seed_${seed}/chosen_trades.pkl \
    --champion-calibration v3/artifacts/layer3_unified_cpu_simL3_objfix_seed${seed}/rolling_layer3_calibrated_robust_90.json \
    --output v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed${seed}.npz \
    --seed ${seed}
done
```

Before launching the full run, do a small `--max-days` smoke and confirm:

- dataset fingerprint matches the rebuilt action surface
- `l3_exit_pnl` shape is `(rows, 25)`
- non-flat actions use next-bar fill semantics
- selected champion trades still validate exactly or explain any expected drift from next-bar repair

### 3. Train the live-utility stack

Run seeds separately so each seed gets its own oracle:

```bash
for seed in 42 43 44; do
  .venv/bin/python -m v3.layer2.train_unified_policy \
    --tier promotion \
    --device cuda \
    --seed ${seed} \
    --dataset v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \
    --utility-target hybrid_live \
    --simulated-l3-oracle v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed${seed}.npz \
    --run-dir v3/artifacts/layer2_unified_policy_spx_live_hybrid_seed${seed}
done
```

If CUDA is unavailable, do not silently run a full promotion on CPU unless you intentionally accept the time cost. Use smoke/dev first.

### 4. Compose with routed two-expert Layer 3

After each seed trains, compose with routed exits:

```bash
for seed in 42 43 44; do
  .venv/bin/python -m v3.layer3.train_rolling \
    --entry-source unified \
    --chosen-trades v3/artifacts/layer2_unified_policy_spx_live_hybrid_seed${seed}/seed_${seed}/chosen_trades.pkl \
    --candidate-dataset v3/artifacts/layer2_action_surface_dataset_spx_live_0945_1130.pkl \
    --l3-training-source routed_experts \
    --seed ${seed} \
    --out-dir v3/artifacts/layer3_unified_spx_live_hybrid_routed_seed${seed}
done
```

Check whether routed experts actually route by window/regime in the report. If it collapses to one expert everywhere, treat that as a diagnostic, not a failure by itself.

### 5. Promotion gate

Current champion remains the benchmark until a new stack clears the floor-safe gate. Stop or quarantine the run if:

- min PF falls below `1.786`
- aggregate PF falls below `1.976`
- mean DD exceeds `12%` without a clear PF lift
- trade count becomes unstable
- W5/W6/W8/W12 regressions explain the mean lift

Do not promote on mean PF alone.

### 6. Required diagnostics

Produce an offline report comparing old vs new:

- chosen side
- risk band
- moneyness bucket
- delta bucket
- premium
- spread fraction
- return on premium
- time bucket
- W5/W6/W8/W12 targeted regressions for seeds `42/43/44`
- old nearest-to-spot token distribution vs new risk-band token distribution
- default `09:45-11:30` vs diagnostic windows `09:45-12:30` and `09:45-15:00`

If widened windows help, do not make them default until they pass the same promotion floor. The point is to discover whether later-day selection fixes ITM/OTM behavior, not to widen because it raises the mean.

### 7. Shadow path comes after offline parity

The new `v3/live_shadow` package is only scaffolding. It does not connect to IBKR and does not place orders.

Next live-shadow tasks:

- build historical-bar to live-style feature parity replay
- build IBKR SPX/SPXW option-chain resolver around `reqSecDefOptParams`
- subscribe to underlying and option market data for quote/Greek snapshots
- write `DecisionSnapshot` JSONL for every completed-bar decision
- add deterministic replay from captured snapshots

No IBKR paper orders until five full shadow sessions pass:

- no missing critical features
- no unresolved selected contracts
- no stale quote decisions
- quote/Greek availability audited
- replay/shadow intent parity within tolerance

## Important Cautions

- SPX/SPXW only. Do not reintroduce XSP assumptions.
- OTM is not a hard filter. Risk-band representation should let OTM emerge when the utility supports it.
- The hybrid target must remain live-executable: decision observes completed bar `t`, fill labels use next executable bar/quote.
- Do not treat the exit layer as ground truth. The two-expert router is meant to reduce distribution mismatch, not hide it.
- The current repo has other dirty/untracked files. Do not revert unrelated changes while continuing this work.
