# Round 6 — Simulator Contract Boundary

This folder contains the response packet after Fable verified Round 5 and flagged the daily-loss basis ambiguity.

## Key Result

The old selected-candidate replay gate used stressed PnL for daily-loss/cash state. That is now treated as archived v1 behavior only.

Forward simulator:

```text
protocol101_serial_simulator_v2
```

Forward semantics:

```text
daily_loss_basis = raw_realized_net_pnl
cash_basis = raw_realized_net_pnl
stress_application = metrics_only
```

## Files

```text
00_message_to_fable.md
01_fable_round5_response.md
02_protocol101_serial_simulator_v2.py
03_selected_candidate_replay_gate_v2.py
04_test_protocol101_serial_simulator.py
05_test_selected_candidate_replay_gate.py
06_PROTOCOL101_CANONICAL_SERIAL_SIMULATOR_V2.md
07_PROTOCOL101_LIVE_FEATURE_CONTRACT_V1.md
08_test_output.txt
SHA256SUMS
```

## Verification

```text
44 targeted tests passed.
No broker, paid-data, paper-submit, training, threshold, promotion, or default-change actions occurred.
```

