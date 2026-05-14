# Protocol 145: Tuesday Cold-Start Rehearsal

No paid data was downloaded. No market-data endpoint was called. No order endpoint was called.

- Decision: `pass_cold_start_api_ready`
- Ports before: `[]`
- Ports after: `[4002]`
- Start returncode: `0`

## Interpretation

IB Gateway opened and the paper API became reachable. Tuesday can move to account probe and live-data parity.

## Start Output

```text
IB Gateway API listener is up on port 4002 via IBC
```

## Next Gate

Run Protocol141 account probe, then Protocol119/124 live-data parity, then paper executor dry-run.
