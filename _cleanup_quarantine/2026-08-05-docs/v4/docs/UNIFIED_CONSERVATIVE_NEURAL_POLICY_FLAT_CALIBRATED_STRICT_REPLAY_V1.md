# REPLAY_UNIFIED_CONSERVATIVE_NEURAL_POLICY_STRICT_SERIAL_V1

What is this: strict one-account serial replay of the trained conservative neural policy with Protocol101 defer fallback
Does it change the paper-trading default: no
Trained policy: `CHALLENGER_UNIFIED_CONSERVATIVE_NEURAL_POLICY_V1`
Paper default baseline: `PAPER_DEFAULT_PROTOCOL101`
Paid data downloaded: no
Broker endpoint called: no
Model training in this runner: no
Decision: `strict_replay_complete_protocol101_challenge_still_blocked`
Challenge allowed: `False`

## Interpretation

The trained model produced challenger overrides in strict replay, but promotion remains blocked until fill calibration, untouched holdout data, live no-order parity, and formal validation controls are complete.

## Scope

- Seed: `1`
- Included sessions: `191`
- Flat rows loaded: `1919518`
- Holding rows loaded: `2856795`
- Baseline event rows: `68247`

## Stress Results

| slippage | split | PnL | same-scope Protocol101 | delta | trades | challenger | defer | PF |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | q1_2026 | 69680.00 | 80450.00 | -10770.00 | 166 | 45 | 121 | 2.91 |
| 0.00 | q3_2025 | 59680.00 | 60420.00 | -740.00 | 235 | 5 | 230 | 5.00 |
| 0.00 | q4_2025 | 131050.00 | 91230.00 | 39820.00 | 213 | 25 | 188 | 9.39 |
| 0.00 | recent_2026 | 6820.00 | 3640.00 | 3180.00 | 65 | 3 | 62 | 1.30 |
| 0.00 | total | 267230.00 | 235740.00 | 31490.00 | 679 | 78 | 601 |  |
| 0.10 | q1_2026 | 66360.00 | 76110.00 | -9750.00 | 166 | 45 | 121 | 2.75 |
| 0.10 | q3_2025 | 54980.00 | 55660.00 | -680.00 | 235 | 5 | 230 | 4.50 |
| 0.10 | q4_2025 | 126790.00 | 86190.00 | 40600.00 | 213 | 25 | 188 | 8.57 |
| 0.10 | recent_2026 | 5520.00 | 2200.00 | 3320.00 | 65 | 3 | 62 | 1.24 |
| 0.10 | total | 253650.00 | 220160.00 | 33490.00 | 679 | 78 | 601 |  |
| 0.25 | q1_2026 | 61380.00 | 69600.00 | -8220.00 | 166 | 45 | 121 | 2.53 |
| 0.25 | q3_2025 | 47930.00 | 48520.00 | -590.00 | 235 | 5 | 230 | 3.80 |
| 0.25 | q4_2025 | 120400.00 | 78630.00 | 41770.00 | 213 | 25 | 188 | 7.48 |
| 0.25 | recent_2026 | 3570.00 | 40.00 | 3530.00 | 65 | 3 | 62 | 1.15 |
| 0.25 | total | 233280.00 | 196790.00 | 36490.00 | 679 | 78 | 601 |  |

## Remaining Challenge Blockers

- calibrated stochastic fill model unavailable
- untouched holdout data pending
- live no-order full-action parity pending
- formal validation controls pending

## Next Required Evidence

1. Attribute every challenger override versus Protocol101 by side, premium, time bucket, moneyness, and lifecycle exit reason.
2. Keep Protocol101 as the paper default.
3. Collect paper/no-order fill evidence before stochastic fill calibration.
4. Reserve and acquire untouched holdout data before any promotion claim.
5. Build live no-order full-action parity for this exact policy contract.
6. Add formal validation controls before a better-than-Protocol101 claim.

## Outputs

- summary: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_strict_replay_v1/summary.json`
- report: `v4/audit/autoresearch/unified_conservative_neural_policy_flat_calibrated_strict_replay_v1/report.md`
- doc: `v4/docs/UNIFIED_CONSERVATIVE_NEURAL_POLICY_FLAT_CALIBRATED_STRICT_REPLAY_V1.md`
