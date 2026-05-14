# Protocol 145: Tuesday Cold-Start Rehearsal

No paid data was downloaded. No market-data endpoint was called. No order endpoint was called.

- Decision: `expected_blocker_gateway_login_required_or_api_port_closed`
- Ports before: `[]`
- Ports after: `[]`
- Start returncode: `2`

## Interpretation

This is the expected failure when IBC cannot complete credentials/2FA or no API listener opens. IBC can enter the stored username/password, but it cannot bypass IBKR Mobile/2FA approval.

## Start Output

```text
IBC runtime config was not created. Run v4/ops/ibkr/store_ibkr_paper_credentials.sh locally.
{
  "detail": "Run v4/ops/ibkr/store_ibkr_paper_credentials.sh locally; do not paste credentials into chat.",
  "out": "/Users/gduby/.autoresearch-trading/ibc/runtime/ibc-paper.ini",
  "password_service": "autoresearch-trading-ibkr-paper-password",
  "status": "missing_credentials",
  "username_service": "autoresearch-trading-ibkr-paper-username"
}
```

## Next Gate

Run v4/ops/ibkr/store_ibkr_paper_credentials.sh locally, approve any IBKR Mobile/2FA challenge during startup, and rerun this rehearsal. The next passing state should expose port 4002 or 4000.
