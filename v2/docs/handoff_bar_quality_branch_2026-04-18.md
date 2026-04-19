# Bar-Quality Two-Stage Branch (2026-04-18)

This branch implements a stricter two-stage supervision path:

- `OPP_LABEL=bar_quality`
- `GATE_TARGET_MODE=bar_quality`
- ranker trains only on bars that clear a bar-quality threshold
- optional rank weighting by bar quality via `RANK_WEIGHT_BY_BAR_QUALITY=1`

## Code Changes

- [v2/train.py](/Users/gduby/Documents/autoresearch-trading/v2/train.py:1)
  - added continuous `_compute_bar_opportunity_quality(...)`
  - added `OPP_LABEL=bar_quality`
  - added `GATE_TARGET_MODE=bar_quality`
  - added `BAR_QUALITY_*` env knobs
  - selection loss now optionally weights passed bars by bar quality
  - fixed a real bug: `LINEAR_SCORE_HEADS=1` now correctly makes **both** score heads linear
- [v2/analysis/gate_label_audit.py](/Users/gduby/Documents/autoresearch-trading/v2/analysis/gate_label_audit.py:1)
  - now supports `--mode bar_quality`
  - reports `avg_bar_quality`
- [v2/core/policy.py](/Users/gduby/Documents/autoresearch-trading/v2/core/policy.py:1), [v2/replay.py](/Users/gduby/Documents/autoresearch-trading/v2/replay.py:1), [v2/ops/run_experiment_wf.py](/Users/gduby/Documents/autoresearch-trading/v2/ops/run_experiment_wf.py:1)
  - added `gate_threshold_floor` / `POLICY_GATE_MIN_THRESHOLD`
  - quantile calibration can now be capped so it does not force the gate below a chosen floor

## Local Validation

- `python3 -m unittest v2.tests.harness_integrity.test_gate_path_cleanup` passed
- `python3 -m py_compile ...` on touched files passed
- `python3 -m v2.analysis.gate_label_audit --mode bar_quality --screen-mode mini`
  - fold 0: `10.28%`
  - fold 2: `9.48%`
  - fold 4: `10.54%`
  - overall: `10.10%`
- tiny real-data CPU smoke completed cleanly

This is the first target in this project that is both:

- genuinely abstention-first
- still trainable on the current sidecars without a dataset rebuild

## Partial GPU Readout

`exp_next_c1_screen_mini` did **not** finish cleanly because the lease died mid-run.

What we did learn before the provider dropped:

- Fold 0 trained successfully under the new target.
- Validation replay selected `epoch 1` with `PF=0.000`, `DD=0.0%`, `score=0.0000`.
- Held-out fold 0 replay, with quantile calibration forcing `10%` pass rate, produced:
  - `PF=0.559`
  - `DD=87.7%`
  - `Trades=288`
  - `TPD=4.80`
  - gate failure

## Important Inference Mistake Found

The partial GPU run exposed a real policy mismatch:

- the new gate is trained as a quality margin around `0`
- but replay quantile mode was allowed to set a **negative** threshold
- on fold 0 that forced the gate open (`threshold=-0.3155`) even though raw validation gate pass was effectively zero

That means the inference policy was partially overriding the abstention behavior the model had just learned.

This is now fixed by:

- `DecisionPolicy.gate_threshold_floor`
- replay-time threshold capping
- env knob: `POLICY_GATE_MIN_THRESHOLD`

For the bar-quality branch, the right first rerun is to set:

```bash
POLICY_GATE_MIN_THRESHOLD=0.0
```

so quantile calibration can never push the gate below the learned no-trade boundary.

## Next Rerun Command

```bash
TRAIN_ENV="TRAIN_FEATURE_SET=full79 LINEAR_SCORE_HEADS=0 GATE_ARCH=decoupled_mlp OPP_LABEL=bar_quality GATE_TARGET_MODE=bar_quality BAR_QUALITY_PASS_THRESHOLD=0.83 BAR_QUALITY_MIN_PNL=0.20 BAR_QUALITY_TARGET_PNL=0.45 BAR_QUALITY_MFE_TARGET=0.08 BAR_QUALITY_MAX_MAE=-0.10 BAR_QUALITY_BREAKEVEN_MAX=8.0 BAR_QUALITY_LOSS_SCALE=5.0 RANK_WEIGHT_BY_BAR_QUALITY=1 SEL_TARGET_MODE=soft_pnl SOFT_TEMP=0.40 POLICY_GATE_THRESHOLD_MODE=quantile POLICY_GATE_TARGET_PASS_RATE=0.10 POLICY_GATE_MIN_THRESHOLD=0.0 CKPT_SELECTION_MODE=val_replay" ./v2/ops/deploy.sh run_screen_mini exp_next_c1_floor0
```

If that rerun still returns near-zero trades on validation and no useful held-out PF, the next conclusion is that the model is not just miscalibrated at inference; it is still failing to discover enough high-quality bars even under the stronger target.
