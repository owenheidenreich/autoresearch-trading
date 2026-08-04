# DIAGNOSTIC_UNIFIED_CONSERVATIVE_FLAT_GATE_V1

What is this: diagnostic / flat-entry conservative gate abstention analysis
Does it change the paper-trading default: no
Paid data downloaded: no
Broker endpoint called: no
Model training: no
Decision: `flat_entry_gate_produces_model_overrides_needs_replay_attribution`

## Diagnosis

- Flat A_enter positives are rare: 2.9349% of 1,919,518 full-surface candidate rows.
- Gate component pass counts: advantage=11,518, positive_probability=713,885, tail=1,788,912, all=11,169.

## Gate Summary

| split | rows | positive rate | adv pass | pos-prob pass | tail pass | all pass | pred adv p99 | pos prob max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| q1_2026 | 551413 | 0.0286 | 7100 | 213698 | 490305 | 6855 | 275.65 | 0.9809 |
| q3_2025 | 529645 | 0.0317 | 823 | 202083 | 512543 | 822 | 122.50 | 0.9763 |
| q4_2025 | 534830 | 0.0296 | 2670 | 193170 | 496973 | 2614 | 194.46 | 0.9775 |
| recent_2026 | 303630 | 0.0261 | 925 | 104934 | 289091 | 878 | 141.15 | 0.9763 |
| total | 1919518 | 0.0293 | 11518 | 713885 | 1788912 | 11169 | 200.80 | 0.9809 |

## Next Required Evidence

1. Run strict replay attribution for the generated overrides.
2. Bucket overrides by side, time, moneyness, premium, and label advantage before any challenge claim.

## Outputs

- summary: `v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic_flat_calibrated_v1/summary.json`
- report: `v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic_flat_calibrated_v1/report.md`
- split_gate_summary: `v4/audit/autoresearch/unified_conservative_flat_gate_diagnostic_flat_calibrated_v1/split_gate_summary.csv`
- doc: `v4/docs/UNIFIED_CONSERVATIVE_FLAT_GATE_DIAGNOSTIC_FLAT_CALIBRATED_V1.md`
