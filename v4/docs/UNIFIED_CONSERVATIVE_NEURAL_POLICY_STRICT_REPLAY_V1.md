# REPLAY_UNIFIED_CONSERVATIVE_NEURAL_POLICY_STRICT_SERIAL_V1

What is this: strict one-account serial replay of the trained conservative neural policy with Protocol101 defer fallback
Does it change the paper-trading default: no
Trained policy: `CHALLENGER_UNIFIED_CONSERVATIVE_NEURAL_POLICY_V1`
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training in this runner: no
Decision: `strict_replay_complete_model_deferred_to_protocol101_flat_gate_too_conservative`
Challenge allowed: `False`

## Interpretation

The trained flat-entry head never cleared the conservative override gate in strict replay. The artifact is therefore an abstention/pass-through policy, not evidence of Protocol101 improvement.

## Scope

- Seed: `1`
- Included sessions: `191`
- Flat rows loaded: `1919518`
- Holding rows loaded: `2856795`
- Baseline event rows: `68247`

## Stress Results

| slippage | split | PnL | same-scope Protocol101 | delta | trades | challenger | defer | PF |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | q1_2026 | 80450.00 | 80450.00 | 0.00 | 217 | 0 | 217 | 3.70 |
| 0.00 | q3_2025 | 60420.00 | 60420.00 | 0.00 | 238 | 0 | 238 | 5.18 |
| 0.00 | q4_2025 | 91230.00 | 91230.00 | 0.00 | 252 | 0 | 252 | 7.75 |
| 0.00 | recent_2026 | 3640.00 | 3640.00 | 0.00 | 72 | 0 | 72 | 1.16 |
| 0.00 | total | 235740.00 | 235740.00 | 0.00 | 779 | 0 | 779 |  |
| 0.10 | q1_2026 | 76110.00 | 76110.00 | 0.00 | 217 | 0 | 217 | 3.45 |
| 0.10 | q3_2025 | 55660.00 | 55660.00 | 0.00 | 238 | 0 | 238 | 4.66 |
| 0.10 | q4_2025 | 86190.00 | 86190.00 | 0.00 | 252 | 0 | 252 | 6.89 |
| 0.10 | recent_2026 | 2200.00 | 2200.00 | 0.00 | 72 | 0 | 72 | 1.09 |
| 0.10 | total | 220160.00 | 220160.00 | 0.00 | 779 | 0 | 779 |  |
| 0.25 | q1_2026 | 69600.00 | 69600.00 | 0.00 | 217 | 0 | 217 | 3.11 |
| 0.25 | q3_2025 | 48520.00 | 48520.00 | 0.00 | 238 | 0 | 238 | 3.94 |
| 0.25 | q4_2025 | 78630.00 | 78630.00 | 0.00 | 252 | 0 | 252 | 5.78 |
| 0.25 | recent_2026 | 40.00 | 40.00 | 0.00 | 72 | 0 | 72 | 1.00 |
| 0.25 | total | 196790.00 | 196790.00 | 0.00 | 779 | 0 | 779 |  |

## Remaining Challenge Blockers

- calibrated stochastic fill model unavailable
- untouched holdout data pending
- live no-order full-action parity pending
- formal validation controls pending

## Next Required Evidence

1. Diagnose why the flat-entry head is over-conservative before retraining: target imbalance, margin thresholds, and positive-class weighting.
2. Keep Protocol101 as the paper default.
3. Collect paper/no-order fill evidence before stochastic fill calibration.
4. Reserve and acquire untouched holdout data before any promotion claim.
5. Build live no-order full-action parity for this exact policy contract.
6. Add formal validation controls before a better-than-Protocol101 claim.

## Outputs

- summary: `v4/audit/autoresearch/unified_conservative_neural_policy_strict_replay_v1/summary.json`
- report: `v4/audit/autoresearch/unified_conservative_neural_policy_strict_replay_v1/report.md`
- doc: `v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_STRICT_REPLAY_V1.md`
