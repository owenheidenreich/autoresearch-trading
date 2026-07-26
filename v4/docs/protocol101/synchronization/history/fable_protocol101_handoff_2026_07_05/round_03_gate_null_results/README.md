# Round 03 Gate/Null Results Packet

This folder is the packet to send back to Fable after implementing the Round 2 gate-only and random-in-gate null baseline experiments.

## Send To Fable

Upload or paste:

1. `00_message_to_fable.md`
2. `02_gate_null_base_report.md`
3. `03_gate_null_base_summary.json`
4. `04_gate_null_variant_report.md`
5. `05_gate_null_variant_summary.json`
6. `06_gate_null_baseline_script.py`

`01_fable_round2_response.md` is included for continuity but does not need to be re-uploaded if Fable already has it.

## Main Finding

Gate-only failed under strict serial replay and did not beat the random-in-gate null.

The near-VWAP gate was validation-positive but diagnostic-negative, which supports the concern that this branch is regime/selection fragile.

## Guardrails

These experiments were offline diagnostics only:

- no model training;
- no threshold tuning;
- no broker calls;
- no paper-submit;
- no paid downloads;
- no default or promotion changes;
- no recorder days used for selection.

