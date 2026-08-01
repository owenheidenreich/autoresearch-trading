# Protocol 156: IBKR Autostart Observability

No paid data was downloaded. No broker order endpoint was called. No orders were placed.

- Generated: `2026-05-23T09:51:26.003884-07:00`
- Session date: `2026-05-23`
- Decision: `observe_previous_live_market_data_entitlements_no_current_port`
- Log directory: `/Users/gduby/Library/Logs/autoresearch-trading`
- Next action: LaunchAgents are loaded, but Gateway is not currently reachable. Let the Tuesday startup run, then check this report plus the premium-blend live surface autotest output.

## What It Means

- A previous no-order entitlement probe passed, but no IBKR API port is reachable right now. This is acceptable off-hours if Gateway is closed; Tuesday still needs the scheduled startup to open Gateway.
- Detected signals: api_port_open_detected=1, holding_status_detected=1, ibkr_api_connected_detected=1, ibkr_competing_live_session=1, ibkr_keepalive_socket_disconnect=1, ibkr_market_data_not_subscribed=1, launchd_permission_denied=1, launchd_python_runtime_failed=1, missing_live_market_data_entitlements=1, pass_status_detected=1, socket_disconnect=1

## LaunchAgents

| label | loaded | state | runs | last exit | stdout | stderr |
| --- | ---: | --- | ---: | ---: | --- | --- |
| com.autoresearch.ibgateway.paper | `True` | `not running` | `0` | `None` | `/Users/gduby/Library/Logs/autoresearch-trading/ibgateway-paper.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/ibgateway-paper.err.log` |
| com.autoresearch.protocol101.paper-preflight | `True` | `not running` | `0` | `None` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-preflight.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-preflight.err.log` |
| com.autoresearch.protocol101.paper-session | `True` | `not running` | `0` | `None` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.err.log` |
| com.autoresearch.protocol101.daily-monitor | `True` | `not running` | `0` | `None` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-daily-monitor.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-daily-monitor.err.log` |
| com.autoresearch.premiumblend.no-order-surface-check | `True` | `not running` | `0` | `None` | `/Users/gduby/Library/Logs/autoresearch-trading/premiumblend-live-surface-autotest.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/premiumblend-live-surface-autotest.err.log` |

## Current IBKR API Ports

| port | open | error |
| ---: | ---: | --- |
| 4002 | `False` | `ConnectionRefusedError` |
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
| launchd_permission_denied | 1 |
| launchd_python_runtime_failed | 1 |
| missing_live_market_data_entitlements | 1 |
| pass_status_detected | 1 |
| socket_disconnect | 1 |

## Latest Entitlement Probe

- Path: `v4/audit/ibkr_live_data_entitlements/summary.json`
- Exists: `True`
- Checked at: `2026-05-21T10:37:50.474802-04:00`
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
    "price": 7407.51
  },
  "spxw_options": {
    "contracts_qualified": 10,
    "contracts_requested": 10,
    "delayed_nbbo_rows": 0,
    "live_nbbo_rows": 10,
    "market_data_type_counts": {
      "live": 10
    }
  },
  "vix": {
    "live_price_available": true,
    "market_data_type": "live",
    "price": 17.58
  }
}
```

## Runtime Wrappers

| wrapper | exists | exports PYTHONPATH | prefers project venv | modified |
| --- | ---: | ---: | ---: | --- |
| run_protocol101_paper_session.sh | `True` | `True` | `True` | `2026-05-23T09:51:25.904794-07:00` |
| run_protocol101_paper_preflight.sh | `True` | `True` | `True` | `2026-05-23T09:51:25.903326-07:00` |
| run_ibkr_autostart_status.sh | `True` | `True` | `True` | `2026-05-23T09:51:25.910796-07:00` |
| run_premium_blend_live_surface_autotest.sh | `True` | `True` | `True` | `2026-05-23T09:51:25.907635-07:00` |

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
- Size bytes: `31644`
- Modified: `2026-05-21T11:54:11.835559-07:00`
- Signals: `['holding_status_detected', 'ibkr_api_connected_detected', 'ibkr_keepalive_socket_disconnect', 'pass_status_detected', 'socket_disconnect']`
- Recent JSON events:
  - `{"blocked_reason": "ibkr_keepalive_failed", "error": "Socket disconnect", "status": "blocked"}`

```text
      "primary_account_id_redacted": "DU***40"
    }
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
```

### preflight_stdout

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-preflight.out.log`
- Exists: `True`
- Size bytes: `3934`
- Modified: `2026-05-21T06:29:17.262378-07:00`
- Signals: `['api_port_open_detected', 'missing_live_market_data_entitlements']`
- Recent JSON events:
  - `{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": false, "market_data_type": "live", "price": null}, "spxw_options": {"contracts_qualified": 0, "contracts_requested": 0, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {}}, "vix": {"live_price_available": false, "market_data_type": "live", "price": null}}, "ibkr_port": 4002}`
  - `{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": false, "market_data_type": "live", "price": null}, "spxw_options": {"contracts_qualified": 0, "contracts_requested": 0, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {}}, "vix": {"live_price_available": false, "market_data_type": "live", "price": null}}, "ibkr_port": 4002}`
  - `{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": false, "market_data_type": "live", "price": null}, "spxw_options": {"contracts_qualified": 0, "contracts_requested": 0, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {}}, "vix": {"live_price_available": false, "market_data_type": "live", "price": null}}, "ibkr_port": 4002}`
  - `{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": true, "market_data_type": "live", "price": 7432.97}, "spxw_options": {"contracts_qualified": 6, "contracts_requested": 6, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {"live": 6}}, "vix": {"live_price_available": true, "market_data_type": "live", "price": 17.72}}, "ibkr_port": 4002}`

```text
    4001
  ],
  "host": "127.0.0.1",
  "port": 4002,
  "status": "port_open"
}
v4/audit/ibkr_live_data_entitlements/report.md
{"blocked_reason": "missing_live_market_data_entitlements", "decision": "blocked", "feed_status": {"spx": {"live_price_available": true, "market_data_type": "live", "price": 7432.97}, "spxw_options": {"contracts_qualified": 6, "contracts_requested": 6, "delayed_nbbo_rows": 0, "live_nbbo_rows": 0, "market_data_type_counts": {"live": 6}}, "vix": {"live_price_available": true, "market_data_type": "live", "price": 17.72}}, "ibkr_port": 4002}
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
- Size bytes: `6055`
- Modified: `2026-05-20T06:29:20.881562-07:00`
- Signals: `['ibkr_competing_live_session', 'ibkr_market_data_not_subscribed', 'launchd_permission_denied', 'launchd_python_runtime_failed']`

```text
Traceback (most recent call last):
  File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
  File "<frozen importlib._bootstrap>", line 982, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 925, in _find_spec
  File "<frozen importlib._bootstrap_external>", line 1414, in find_spec
  File "<frozen importlib._bootstrap_external>", line 1383, in _get_spec
  File "<frozen importlib._bootstrap_external>", line 1347, in _path_importer_cache
PermissionError: [Errno 1] Operation not permitted
positions request timed out
open orders request timed out
completed orders request timed out
account updates for DUP440540 request timed out
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 4000)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 7497)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 7496)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 4001)")
Make sure API port on TWS/IBG is open
```

### session_stdout

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.out.log`
- Exists: `True`
- Size bytes: `617`
- Modified: `2026-05-21T12:11:02.501623-07:00`
- Signals: `[]`

```text
{
  "decision": "completed_no_order_analysis",
  "report": "v4/audit/autoresearch/v4_aplus_hypothesis_147_protocol101_morning_session/2026-05-19/protocol101_no-order-shadow_2026-05-19/report.md",
  "trade_log": "v4/logs/paper_trading/2026-05-19/protocol101_no-order-shadow_2026-05-19.jsonl"
}
{
  "decision": "pass_persistent_no_entry_intents_to_paper_submit",
  "report": "v4/audit/autoresearch/v4_aplus_hypothesis_160_protocol101_persistent_paper_trader/2026-05-21/protocol101_persistent-paper_2026-05-21/report.md",
  "trade_log": "v4/logs/paper_trading/2026-05-21/protocol101_persistent-paper_2026-05-21.jsonl"
}
```

### session_stderr

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.err.log`
- Exists: `True`
- Size bytes: `144197`
- Modified: `2026-05-21T12:10:56.391293-07:00`
- Signals: `[]`

```text
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 4002)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 4000)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 7497)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 7496)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 4001)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 4002)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 4000)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 7497)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 7496)")
Make sure API port on TWS/IBG is open
API connection failed: ConnectionRefusedError(61, "Connect call failed ('127.0.0.1', 4001)")
Make sure API port on TWS/IBG is open
```

### monitor_stdout

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-daily-monitor.out.log`
- Exists: `True`
- Size bytes: `1408917`
- Modified: `2026-05-21T13:04:35.208129-07:00`
- Signals: `[]`

```text
  "decision": "pass_live_entry_monitor_ready",
  "html": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/protocol101_persistent-paper_2026-05-21/daily_monitor.html",
  "latest_html": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/latest_daily_monitor.html",
  "report": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/protocol101_persistent-paper_2026-05-21/report.md",
  "summary": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/protocol101_persistent-paper_2026-05-21/summary.json"
}
{
  "decision": "pass_live_entry_monitor_ready",
  "html": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/protocol101_persistent-paper_2026-05-21/daily_monitor.html",
  "latest_html": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/latest_daily_monitor.html",
  "report": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/protocol101_persistent-paper_2026-05-21/report.md",
  "summary": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/protocol101_persistent-paper_2026-05-21/summary.json"
}
{
  "decision": "pass_live_entry_monitor_ready",
  "html": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/protocol101_persistent-paper_2026-05-21/daily_monitor.html",
  "latest_html": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/latest_daily_monitor.html",
  "report": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/protocol101_persistent-paper_2026-05-21/report.md",
  "summary": "v4/audit/autoresearch/v4_aplus_hypothesis_157_protocol101_daily_ops_monitor/2026-05-21/protocol101_persistent-paper_2026-05-21/summary.json"
}
```

### monitor_stderr

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-daily-monitor.err.log`
- Exists: `True`
- Size bytes: `14090`
- Modified: `2026-05-21T07:35:50.355390-07:00`
- Signals: `[]`

```text
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
no rows found in v4/logs/paper_trading/2026-05-21/protocol101_no-order-shadow_2026-05-21.jsonl
```

### premium_blend_autotest_stdout

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/premiumblend-live-surface-autotest.out.log`
- Exists: `False`
- Size bytes: `0`
- Modified: `None`
- Signals: `[]`

### premium_blend_autotest_stderr

- Path: `/Users/gduby/Library/Logs/autoresearch-trading/premiumblend-live-surface-autotest.err.log`
- Exists: `False`
- Size bytes: `0`
- Modified: `None`
- Signals: `[]`

### Recent IBC Logs
- `/Users/gduby/Library/Logs/autoresearch-trading/ibc/ibc-gateway.err.log` size=58 modified=`2026-05-21T06:28:05.287484-07:00` signals=`[]`
- `/Users/gduby/Library/Logs/autoresearch-trading/ibc/ibc-gateway.out.log` size=23808 modified=`2026-05-21T11:54:11.731122-07:00` signals=`[]`

## Live Paper JSONL Logs

- Date directory exists: `False`
- Root: `v4/logs/paper_trading`
