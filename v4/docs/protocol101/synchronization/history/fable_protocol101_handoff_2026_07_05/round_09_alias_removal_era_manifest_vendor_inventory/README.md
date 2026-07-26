# Round 9 Package

This folder contains the next response package for Fable.

Primary file to paste/upload first:

```text
00_message_to_fable.md
```

Recommended upload set:

```text
00_message_to_fable.md
02_protocol101_serial_simulator_v2_alias_removed.py
03_selected_candidate_replay_gate_v2_alias_removed.py
04_build_protocol101_session_era_manifest.py
05_run_protocol101_vendor_overlap_inventory.py
06_test_protocol101_serial_simulator_alias_removed.py
07_test_selected_candidate_replay_gate_alias_removed.py
08_test_protocol101_session_era_manifest.py
09_test_protocol101_vendor_overlap_inventory.py
10_PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2_alias_removed.md
11_session_era_manifest_report.md
14_vendor_overlap_inventory_report.md
17_test_and_artifact_output.txt
```

What this packet covers:

- Retires `simulator_semantics_hash` before first fold artifacts.
- Normalizes emitted `synthetic_exit_time` to UTC.
- Adds standalone fail-closed session era manifest generation.
- Runs data-plane-only vendor overlap inventory.
- Confirms local owned data does not include ThetaData option quote files in the scanned roots.

No broker connectivity, paper-submit, paid data downloads, model training, threshold tuning, promotion/default changes, or real-money paths were touched.
