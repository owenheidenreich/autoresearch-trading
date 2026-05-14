# Protocol 145: Tuesday Cold-Start Rehearsal

No paid data was downloaded. No market-data endpoint was called. No order endpoint was called.

- Decision: `expected_blocker_gateway_login_required_or_api_port_closed`
- Ports before: `[]`
- Ports after: `[]`
- Start returncode: `1`

## Interpretation

This is the expected failure when IB Gateway launches but credentials/2FA have not completed. The automation can open Gateway, but cannot bypass an unsaved login challenge.

## Start Output

```text
IB Gateway started, but no API port from [4000,4002,7497,7496,4001] was listening after 35s.
If Gateway is waiting for credentials/2FA, finish the login once and rerun the preflight.
```

## Next Gate

Finish IB Gateway paper login manually once and ensure API settings remain enabled. Then rerun this rehearsal; the next passing state should expose port 4002 or 4000.
