# PREREGISTERED_UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_SPEC_V1

What is this: preregistration for exactly one next unified neural policy run
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Live orders: no

## Fixed Training Recipe

- Training script: `v4/scripts/run_unified_conservative_neural_policy.py`
- Output: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1`
- Seed: `1`
- Epochs: `5`
- Batch size: `4096`
- Hidden dim: `128`
- Flat train rows: `450000`
- Flat eval rows: `120000`
- Holding train rows: `450000`
- Holding eval rows: `150000`
- Flat train positive fraction: `0.35`
- Flat positive class weight: `16`
- Flat positive regression weight: `8`
- Flat tail class weight: `1`
- Train splits: `q3_2025`, `q4_2025`
- Validation split: `q1_2026`
- Diagnostic split: `recent_2026`

## Frozen Learned Defer Overlay

- Replay script: `v4/scripts/run_unified_slot_opportunity_learned_defer_overlay_replay.py`
- Output: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_replay_v1`
- Slot estimator: `v4/audit/autoresearch/unified_slot_opportunity_cost_estimator/model_artifacts/slot_opportunity_cost_estimator.joblib`
- Min net advantage margin: `0`
- Blocked-cost uncertainty weight: `0.25`
- Max blocked Protocol101 entries: `3`
- Slippage grid: `0`, `0.10`, `0.25`

## Pass/Fail Criteria

Pass for this diagnostic foundation stage requires:

1. Strict one-account serial replay completes without overlap or unaffordable challenger entries.
2. Q1 2026 and Q3 2025 same-scope deltas versus Protocol101 are nonnegative under all three slippage levels.
3. Challenger entries are nonzero, so the result is not merely all-defer Protocol101.

This run cannot promote a model or change the paper default. Protocol101 challenge claims remain blocked until calibrated fills, untouched holdout data, live no-order parity, and formal validation controls pass.
