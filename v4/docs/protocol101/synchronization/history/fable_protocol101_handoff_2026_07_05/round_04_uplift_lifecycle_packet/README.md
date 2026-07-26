# Round 04 Uplift / Lifecycle Packet

This folder is the packet to send back to Fable after implementing the combined uplift, sampler, candidate-matched null, and lifecycle attribution diagnostics.

## Send To Fable

Upload or paste:

1. `00_message_to_fable.md`
2. `02_uplift_lifecycle_report.md`
3. `03_uplift_lifecycle_summary.json`
4. `04_strategy_uplift_rows.csv`
5. `05_lifecycle_attribution_rows.csv`
6. `06_candidate_null_m_rows.csv`
7. `07_stratum_uplift_rows.csv`
8. `08_uplift_lifecycle_script.py`
9. `09_gate_null_variant_report_refreshed.md`
10. `10_gate_null_variant_summary_refreshed.json`
11. `11_gate_null_baseline_script_refreshed.py`

`01_fable_round3_response.md` is included for continuity.

## Main Finding

The combined packet supports these provisional conclusions:

- first-eligible sampling is evidence-backed harmful on March evidence;
- slot schedule is less harmful but not paper-ready;
- attempt130/attempt131 do not beat candidate-matched Null-M;
- lifecycle policy changes do not cleanly rescue the issue;
- the next critical path is likely backfill plus fold-aware evaluation before any resumed training.

## Guardrails

These experiments were offline diagnostics only:

- no model training;
- no threshold tuning;
- no broker calls;
- no paper-submit;
- no paid downloads;
- no default or promotion changes;
- no recorder days used for selection.

