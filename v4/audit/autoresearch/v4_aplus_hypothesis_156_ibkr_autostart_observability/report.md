# Protocol 156: IBKR Autostart Observability

No paid data was downloaded. No broker order endpoint was called. No orders were placed.

- Generated: `2026-05-19T08:11:35.769533-07:00`
- Session date: `2026-05-19`
- Decision: `pass_live_market_data_entitlements`
- Log directory: `/Users/gduby/Library/Logs/autoresearch-trading`
- Next action: Run Protocol101 no-order/paper session and inspect the live JSONL plus this status report after the session.

## What It Means

- The latest no-order entitlement probe confirms live SPX, live VIX, and live SPXW option NBBO are available.
- Detected signals: api_port_open_detected=1, holding_status_detected=1, ibkr_api_connected_detected=1, ibkr_competing_live_session=1, ibkr_keepalive_socket_disconnect=1, ibkr_market_data_not_subscribed=1, launchd_permission_denied=2, launchd_python_runtime_failed=2, missing_live_market_data_entitlements=1, pass_status_detected=1, pythonpath_missing_for_session_runner=1, socket_disconnect=1

## LaunchAgents

| label | loaded | state | runs | last exit | stdout | stderr |
| --- | ---: | --- | ---: | ---: | --- | --- |
| com.autoresearch.ibgateway.paper | `True` | `active` | `5` | `2` | `/Users/gduby/Library/Logs/autoresearch-trading/ibgateway-paper.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/ibgateway-paper.err.log` |
| com.autoresearch.protocol101.paper-preflight | `True` | `active` | `5` | `1` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-preflight.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-preflight.err.log` |
| com.autoresearch.protocol101.paper-session | `True` | `active` | `5` | `1` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.err.log` |

## Current IBKR API Ports

| port | open | error |
| ---: | ---: | --- |
| 4002 | `True` | `None` |
| 4000 | `False` | `ConnectionRefusedError` |
| 7497 | `False` | `ConnectionRefusedError` |
| 7496 | `False` | `ConnectionRefusedError` |
| 4001 | `False` | `ConnectionRefusedError` |

## Signal Counts

| signal | count |
| --- | ---: |
| api_port_open_detected | 1 |
| holding_status_detected | 1 |
| ibkr_api_connected_detected | 1 |
| ibkr_competing_live_session | 1 |
| ibkr_keepalive_socket_disconnect | 1 |
| ibkr_market_data_not_subscribed | 1 |
| launchd_permission_denied | 2 |
| launchd_python_runtime_failed | 2 |
| missing_live_market_data_entitlements | 1 |
| pass_status_detected | 1 |
| pythonpath_missing_for_session_runner | 1 |
| socket_disconnect | 1 |

## Latest Entitlement Probe

- Path: `v4/audit/ibkr_live_data_entitlements/summary.json`
- Exists: `True`
- Checked at: `2026-05-19T11:09:54.979918-04:00`
- Decision: `pass`
- Blocked reason: `None`
- IBKR connected: `True`
- IBKR port: `4002`
- Broker order endpoint called: `False`

```json
{
  "spx": {
    "live_price_available": true,
    "market_data_type": "live",
    "price": 7352.45
  },
  "spxw_options": {
    "contracts_qualified": 6,
    "contracts_requested": 6,
    "delayed_nbbo_rows": 0,
    "live_nbbo_rows": 6,
    "market_data_type_counts": {
      "live": 6
    }
  },
  "vix": {
    "live_price_available": true,
    "market_data_type": "live",
    "price": 18.12
  }
}
```

## Runtime Wrappers

| wrapper | exists | exports PYTHONPATH | prefers project venv | modified |
| --- | ---: | ---: | ---: | --- |
| run_protocol101_paper_session.sh | `True` | `True` | `True` | `2026-05-19T08:00:41.861522-07:00` |
| run_protocol101_paper_preflight.sh | `True` | `True` | `True` | `2026-05-19T08:00:41.859102-07:00` |
| run_ibkr_autostart_status.sh | `True` | `True` | `True` | `2026-05-19T08:00:41.863702-07:00` |

## Log Files

### gateway_stdout

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/ibgateway-paper.out.log`
- Exists: `True`
- Size bytes: `0`
- Modified: `2026-05-14T17:18:02.375757-07:00`
- Signals: `[]`

### gateway_stderr

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/ibgateway-paper.err.log`
- Exists: `True`
- Size bytes: `21071`
- Modified: `2026-05-19T08:09:01.462516-07:00`
- Signals: `['holding_status_detected', 'ibkr_api_connected_detected', 'ibkr_keepalive_socket_disconnect', 'pass_status_detected', 'socket_disconnect']`
- Recent JSON events:
  - `{"blocked_reason": "ibkr_keepalive_failed", "error": "Socket disconnect", "status": "blocked"}`

```text
  ],
  "broker_order_endpoint_called": false,
  "host": "127.0.0.1",
  "port": 4002,
  "stable_seconds": 10.0,
  "status": "pass"
}
{
  "account_count": 1,
  "account_id_redacted": "DU***40",
  "broker_order_endpoint_called": false,
  "client_id": 147,
  "hold_seconds": 28800.0,
  "host": "127.0.0.1",
  "port": 4002,
  "status": "holding"
}
{"status": "blocked", "blocked_reason": "ibkr_keepalive_failed", "error": "Socket disconnect"}
Peer closed connection.
/Users/gduby/.autoresearch-trading/launchd/start_ib_gateway_paper_ibc.sh: line 87: unexpected EOF while looking for matching `"'
```

### preflight_stdout

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-preflight.out.log`
- Exists: `True`
- Size bytes: `3110`
- Modified: `2026-05-18T07:07:50.872746-07:00`
- Signals: `['api_port_open_detected', 'missing_live_market_data_entitlements']`
- Recent JSON events:
  - `{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": false, "market_data_type": "live", "price": null}, "spxw_options": {"contracts_qualified": 0, "contracts_requested": 0, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {}}, "vix": {"live_price_available": false, "market_data_type": "live", "price": null}}, "ibkr_port": 4002}`
  - `{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": false, "market_data_type": "live", "price": null}, "spxw_options": {"contracts_qualified": 0, "contracts_requested": 0, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {}}, "vix": {"live_price_available": false, "market_data_type": "live", "price": null}}, "ibkr_port": 4002}`
  - `{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": false, "market_data_type": "live", "price": null}, "spxw_options": {"contracts_qualified": 0, "contracts_requested": 0, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {}}, "vix": {"live_price_available": false, "market_data_type": "live", "price": null}}, "ibkr_port": 4002}`
  - `{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": false, "market_data_type": "live", "price": null}, "spxw_options": {"contracts_qualified": 0, "contracts_requested": 0, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {}}, "vix": {"live_price_available": false, "market_data_type": "live", "price": null}}, "ibkr_port": 4002}`
  - `{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": false, "market_data_type": "live", "price": null}, "spxw_options": {"contracts_qualified": 0, "contracts_requested": 0, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {}}, "vix": {"live_price_available": false, "market_data_type": "live", "price": null}}, "ibkr_port": 4002}`

```text
    4001
  ],
  "host": "127.0.0.1",
  "port": 4002,
  "status": "port_open"
}
v4/audit/ibkr_live_data_entitlements/report.md
{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": false, "market_data_type": "live", "price": null}, "spxw_options": {"contracts_qualified": 0, "contracts_requested": 0, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {}}, "vix": {"live_price_available": false, "market_data_type": "live", "price": null}}, "ibkr_port": 4002}
{
  "candidate_ports": [
    4002,
    4000,
    7497,
    7496,
    4001
  ],
  "host": "127.0.0.1",
  "port": 4002,
  "status": "port_open"
}
```

### preflight_stderr

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-preflight.err.log`
- Exists: `True`
- Size bytes: `5390`
- Modified: `2026-05-19T06:29:02.154079-07:00`
- Signals: `['ibkr_competing_live_session', 'ibkr_market_data_not_subscribed', 'launchd_permission_denied', 'launchd_python_runtime_failed']`

```text
  sys.executable = '/Library/Developer/CommandLineTools/usr/bin/python3'
  sys.prefix = '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9'
  sys.exec_prefix = '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9'
  sys.path = [
    '/Users/gduby/Documents/autoresearch-trading',
    '',
    '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python39.zip',
    '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9',
    '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/lib-dynload',
  ]
Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encoding
Python runtime state: core initialized
Traceback (most recent call last):
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 982, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 925, in _find_spec
  File "<frozen importlib._bootstrap_external>", line 1414, in find_spec
  File "<frozen importlib._bootstrap_external>", line 1383, in _get_spec
  File "<frozen importlib._bootstrap_external>", line 1347, in _path_importer_cache
PermissionError: [Errno 1] Operation not permitted
```

### session_stdout

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.out.log`
- Exists: `True`
- Size bytes: `0`
- Modified: `2026-05-15T06:30:05.424299-07:00`
- Signals: `[]`

### session_stderr

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.err.log`
- Exists: `True`
- Size bytes: `2732`
- Modified: `2026-05-19T06:30:02.145351-07:00`
- Signals: `['launchd_permission_denied', 'launchd_python_runtime_failed', 'pythonpath_missing_for_session_runner']`

```text
  sys.executable = '/Library/Developer/CommandLineTools/usr/bin/python3'
  sys.prefix = '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9'
  sys.exec_prefix = '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9'
  sys.path = [
    '/Users/gduby/Documents/autoresearch-trading',
    '',
    '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python39.zip',
    '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9',
    '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/lib/python3.9/lib-dynload',
  ]
Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encoding
Python runtime state: core initialized
Traceback (most recent call last):
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 982, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 925, in _find_spec
  File "<frozen importlib._bootstrap_external>", line 1414, in find_spec
  File "<frozen importlib._bootstrap_external>", line 1383, in _get_spec
  File "<frozen importlib._bootstrap_external>", line 1347, in _path_importer_cache
PermissionError: [Errno 1] Operation not permitted
```

### Recent IBC Logs
- `/Users/gduby/Library/Logs/autoresearch-trading/ibc/ibc-gateway.err.log` size=58 modified=`2026-05-19T06:28:04.292274-07:00` signals=`[]`
- `/Users/gduby/Library/Logs/autoresearch-trading/ibc/ibc-gateway.out.log` size=16970 modified=`2026-05-19T08:09:00.916822-07:00` signals=`[]`

## Live Paper JSONL Logs

- Date directory exists: `False`
- Root: `v4/logs/paper_trading`
