# Round 8 Package

This folder contains the response package for Fable's latest review.

Purpose:

- Fix the pre-fold hash separation issue.
- Cap forced-flat synthetic exits at `15:55 ET`.
- Verify the late-label pipeline also caps policy-2 labels at forced-flat.
- Document the simulator boundary changes.
- Provide tests and output for external review.

Primary file to paste/upload first:

```text
00_message_to_fable.md
```

Recommended upload set:

```text
00_message_to_fable.md
02_protocol101_serial_simulator_v2_hash_forced_flat.py
03_selected_candidate_replay_gate_v2_hash_forced_flat.py
04_test_protocol101_serial_simulator_hash_forced_flat.py
05_test_selected_candidate_replay_gate_hash_forced_flat.py
06_test_spxw_0dte_neural_forced_flat_labels.py
07_PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2_hash_forced_flat.md
10_test_output.txt
```

The original Fable message is preserved as:

```text
01_fable_round7_response.md
```

No broker connectivity, paid downloads, model training, threshold tuning, promotion/default changes, or real-money paths were touched.
