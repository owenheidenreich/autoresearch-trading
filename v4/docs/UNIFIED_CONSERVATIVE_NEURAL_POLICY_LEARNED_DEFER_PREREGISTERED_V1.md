# CHALLENGER_UNIFIED_CONSERVATIVE_NEURAL_POLICY_V1

Training spec: `PREREGISTERED_UNIFIED_CONSERVATIVE_NEURAL_TRAINING_V1`
What is this: preregistered model training / conservative unified neural policy
Does it change the paper-trading default: no
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training: yes
Decision: `conservative_neural_policy_trained_replay_and_challenge_still_blocked`
Challenge allowed: `False`

## Scope

- Oracle decision: `unified_serial_dp_oracle_ready_for_baseline_aligned_training_scope`
- Train splits: `['q3_2025', 'q4_2025']`
- Validation split: `q1_2026`
- Diagnostic split: `recent_2026`
- Flat rows loaded: `1919518`
- Holding rows loaded: `2856795`
- Flat train positive rate: `0.0725`
- Holding train positive rate: `0.9098`

## Metrics

| head | split | rows | MAE | corr | allowed rows | allowed true adv mean | allowed true positive rate |
|---|---|---:|---:|---:|---:|---:|---:|
| flat_entry | train | 450000 | 478.44 | 0.363 | 1152 | -409.57 | 0.536 |
| flat_entry | validation | 120000 | 647.08 | 0.205 | 798 | -1080.31 | 0.209 |
| flat_entry | diagnostic_recent | 120000 | 516.11 | 0.135 | 184 | -615.19 | 0.272 |
| holding_lifecycle | train | 450000 | 624.46 | 0.589 | 336247 | 1122.57 | 0.939 |
| holding_lifecycle | validation | 150000 | 983.63 | 0.422 | 109508 | 1537.55 | 0.931 |
| holding_lifecycle | diagnostic_recent | 150000 | 767.82 | 0.246 | 87058 | 1141.23 | 0.936 |

## Challenge Blockers

- strict serial replay not yet run for this model
- calibrated stochastic fill model unavailable
- untouched holdout data pending
- live no-order full-action parity pending
- formal validation controls pending

## Next Required Evidence

1. Run strict one-account serial replay with the trained flat-entry and holding-lifecycle heads.
2. Keep Protocol101 as paper default until replay, fill, untouched holdout, live parity, and validation controls pass.

## Artifacts

- manifest: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts/manifest.json`
- flat_entry_model: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts/flat_entry_model.pt`
- holding_lifecycle_model: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts/holding_lifecycle_model.pt`
- flat_scaler: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts/flat_scaler.json`
- holding_scaler: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/model_artifacts/holding_scaler.json`

## Outputs

- summary: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/summary.json`
- report: `v4/audit/autoresearch/unified_conservative_neural_policy_learned_defer_preregistered_v1/report.md`
- doc: `v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_LEARNED_DEFER_PREREGISTERED_V1.md`
