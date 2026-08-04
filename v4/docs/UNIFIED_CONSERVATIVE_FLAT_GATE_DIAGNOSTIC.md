# DIAGNOSTIC_UNIFIED_CONSERVATIVE_FLAT_GATE_V1

What is this: diagnostic / flat-entry conservative gate abstention analysis
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `flat_entry_gate_overconservative_zero_model_overrides`

## Diagnosis

- Flat A_enter positives are rare: 2.9349% of 1,919,518 full-surface candidate rows.
- Gate component pass counts: advantage=0, positive_probability=138, tail=1,752,701, all=0.
- The advantage regression head is the binding bottleneck; target scaling or loss balance must be fixed before retraining/replay.

## Gate Summary

| split | rows | positive rate | adv pass | pos-prob pass | tail pass | all pass | pred adv p99 | pos prob max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q1_2026 | 551413 | 0.0286 | 0 | 119 | 469416 | 0 | -313.97 | 0.7488 |
| q3_2025 | 529645 | 0.0317 | 0 | 3 | 513635 | 0 | -260.18 | 0.6653 |
| q4_2025 | 534830 | 0.0296 | 0 | 8 | 483106 | 0 | -271.30 | 0.8332 |
| recent_2026 | 303630 | 0.0261 | 0 | 8 | 286544 | 0 | -370.15 | 0.7610 |
| total | 1919518 | 0.0293 | 0 | 138 | 1752701 | 0 | -283.52 | 0.8332 |

## Next Required Evidence

1. Before retraining, pre-register a class-imbalance/calibration fix for the flat-entry positive head.
2. Report gate-component pass counts on train/validation/recent before running strict replay.
3. Keep Protocol101 as the full fallback; do not loosen thresholds ad hoc to create trades.

## Outputs

- summary: `v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic/summary.json`
- report: `v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic/report.md`
- split_gate_summary: `v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic/split_gate_summary.csv`
- doc: `v4/docs/UNIFIED_CONSERVATIVE_FLAT_GATE_DIAGNOSTIC.md`
