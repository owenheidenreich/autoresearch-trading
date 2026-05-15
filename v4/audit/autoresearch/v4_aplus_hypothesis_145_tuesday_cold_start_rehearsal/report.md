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
{
  "account_count": 1,
  "account_id_redacted": "DU***40",
  "attempts_tail": [
    {
      "attempts": [
        {
          "port": 4002,
          "status": "socket_closed"
        },
        {
          "port": 4000,
          "status": "socket_closed"
        },
        {
          "port": 7497,
          "status": "socket_closed"
        },
        {
          "port": 7496,
          "status": "socket_closed"
        },
        {
          "port": 4001,
          "status": "socket_closed"
        }
      ],
      "connected": false
    },
    {
      "attempts": [
        {
          "port": 4002,
          "status": "socket_closed"
        },
        {
          "port": 4000,
          "status": "socket_closed"
        },
        {
          "port": 7497,
          "status": "socket_closed"
        },
        {
          "port": 7496,
          "status": "socket_closed"
        },
        {
          "port": 4001,
          "status": "socket_closed"
        }
      ],
      "connected": false
    },
    {
      "attempts": [
        {
          "port": 4002,
          "status": "socket_closed"
        },
        {
          "port": 4000,
          "status": "socket_closed"
        },
        {
          "port": 7497,
          "status": "socket_closed"
        },
        {
          "port": 7496,
          "status": "socket_closed"
        },
        {
          "port": 4001,
          "status": "socket_closed"
        }
      ],
      "connected": false
    },
    {
      "account_count": 1,
      "attempts": [
        {
          "port": 4002,
          "status": "connected"
        }
      ],
      "connected": true,
      "host": "127.0.0.1",
      "port": 4002,
      "primary_account_id_redacted": "DU***40"
    },
    {
      "account_count": 1,
      "attempts": [
        {
          "port": 4002,
          "status": "connected"
        }
      ],
      "connected": true,
      "host": "127.0.0.1",
      "port": 4002,
      "primary_account_id_redacted": "DU***40"
    }
  ],
  "broker_order_endpoint_called": false,
  "host": "127.0.0.1",
  "port": 4002,
  "stable_seconds": 10.0,
  "status": "pass"
}
```

## Next Gate

Run Protocol141 account probe, then Protocol119/124 live-data parity, then paper executor dry-run.
