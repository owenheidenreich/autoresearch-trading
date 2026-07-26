# Round 7 — Simulator Pins And Backfill Request

This folder contains the response packet after Fable approved the v2 simulator direction but requested three contract pins before fold artifacts.

## Key Changes

- Added simulator metadata/hashes.
- Added simulator-level `15:30 ET` entry cutoff.
- Added cooldown/max-hold mismatch fail-closed behavior when max hold is provided.
- Added archive note clarifying March diagnostics versus candidate strict replays.
- Added paid backfill approval-request draft.

## Files

```text
00_message_to_fable.md
01_fable_round6_response.md
02_protocol101_serial_simulator_v2_pinned.py
03_selected_candidate_replay_gate_v2_pinned.py
04_test_protocol101_serial_simulator_pinned.py
05_test_selected_candidate_replay_gate_pinned.py
06_PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2_pinned.md
07_PROTOCOL101_LIVE_FEATURE_CONTRACT_V1_pinned.md
08_PROTOCOL101_BACKFILL_APPROVAL_REQUEST_DRAFT_2026_07_05.md
09_test_output.txt
SHA256SUMS
```

## Verification

```text
50 targeted tests passed.
No broker, paid-data, paper-submit, training, threshold, promotion, or default-change actions occurred.
```

