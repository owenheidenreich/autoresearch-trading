# FOUNDATION_UNIFIED_SLOT_OPPORTUNITY_DEFER_OVERLAY_V1

What is this: foundation / slot opportunity-cost defer overlay contract and oracle diagnostic
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `slot_opportunity_defer_overlay_oracle_target_repairs_q1_q3_ready_for_learned_estimator`

## Guardrail

The oracle overlay uses realized blocked Protocol101 PnL and is diagnostic only; it is not live-deployable.

## Oracle Overlay Diagnostic

| slippage_per_side | split | challenger_entries | oracle_kept_entries | original_delta_vs_protocol101 | oracle_overlay_delta_vs_protocol101 | blocked_protocol101_pnl_released |
|---|---|---|---|---|---|---|
| 0.00 | q1_2026 | 45 | 14 | -12880.00 | 32310.00 | 30550.00 |
| 0.00 | q3_2025 | 5 | 1 | -1760.00 | 1390.00 | 50.00 |
| 0.00 | q4_2025 | 25 | 9 | 38950.00 | 59400.00 | 20470.00 |
| 0.00 | recent_2026 | 3 | 1 | 2550.00 | 4950.00 | 580.00 |
| 0.10 | q1_2026 | 45 | 14 | -12040.00 | 32710.00 | 29350.00 |
| 0.10 | q3_2025 | 5 | 1 | -1740.00 | 1410.00 | -70.00 |
| 0.10 | q4_2025 | 25 | 9 | 39570.00 | 59640.00 | 19630.00 |
| 0.10 | recent_2026 | 3 | 1 | 2650.00 | 5030.00 | 480.00 |
| 0.25 | q1_2026 | 45 | 14 | -10780.00 | 33310.00 | 27550.00 |
| 0.25 | q3_2025 | 5 | 1 | -1710.00 | 1440.00 | -250.00 |
| 0.25 | q4_2025 | 25 | 9 | 40500.00 | 60000.00 | 18370.00 |
| 0.25 | recent_2026 | 3 | 1 | 2800.00 | 5150.00 | 330.00 |

## Causal Overlay Contract

- Decision rule: allow challenger only if predicted challenger advantage - estimated blocked Protocol101 opportunity cost - uncertainty >= fixed margin
- Promotion gate: nonnegative q1_2026 and q3_2025 same-scope deltas under $0.00/$0.10/$0.25 stress before broader claims
- Forbidden inputs:
  - future Protocol101 entries
  - realized blocked Protocol101 PnL
  - realized challenger exit or duration
  - future bid/ask path

## Diagnosis

- q1_2026 @ 0.00: oracle overlay keeps 14/45 overrides and changes delta from -$12,880 to $32,310.
- q3_2025 @ 0.00: oracle overlay keeps 1/5 overrides and changes delta from -$1,760 to $1,390.
- q4_2025 @ 0.00: oracle overlay keeps 9/25 overrides and changes delta from $38,950 to $59,400.
- recent_2026 @ 0.00: oracle overlay keeps 1/3 overrides and changes delta from $2,550 to $4,950.
- q1_2026 @ 0.10: oracle overlay keeps 14/45 overrides and changes delta from -$12,040 to $32,710.
- q3_2025 @ 0.10: oracle overlay keeps 1/5 overrides and changes delta from -$1,740 to $1,410.
- q4_2025 @ 0.10: oracle overlay keeps 9/25 overrides and changes delta from $39,570 to $59,640.
- recent_2026 @ 0.10: oracle overlay keeps 1/3 overrides and changes delta from $2,650 to $5,030.
- q1_2026 @ 0.25: oracle overlay keeps 14/45 overrides and changes delta from -$10,780 to $33,310.
- q3_2025 @ 0.25: oracle overlay keeps 1/5 overrides and changes delta from -$1,710 to $1,440.
- q4_2025 @ 0.25: oracle overlay keeps 9/25 overrides and changes delta from $40,500 to $60,000.
- recent_2026 @ 0.25: oracle overlay keeps 1/3 overrides and changes delta from $2,800 to $5,150.

## Next Required Evidence

1. Materialize causal blocked-Protocol101 opportunity-cost labels for candidate override events.
2. Train or calibrate a small opportunity-cost estimator using only causal inputs.
3. Rerun strict replay with the learned overlay; require Q1/Q3 nonnegative deltas under all deterministic stress levels.
4. Keep Protocol101 as paper default; oracle overlay diagnostics are not live-deployable evidence.

## Outputs

- summary: `v4/audit/autoresearch/unified_slot_opportunity_defer_overlay_foundation/summary.json`
- report: `v4/audit/autoresearch/unified_slot_opportunity_defer_overlay_foundation/report.md`
- oracle_overlay_summary: `v4/audit/autoresearch/unified_slot_opportunity_defer_overlay_foundation/oracle_overlay_summary.csv`
- doc: `v4/docs/UNIFIED_SLOT_OPPORTUNITY_DEFER_OVERLAY_FOUNDATION.md`
