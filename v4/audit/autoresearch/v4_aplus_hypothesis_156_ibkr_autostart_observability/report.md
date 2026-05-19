# Protocol 156: IBKR Autostart Observability

No paid data was downloaded. No broker order endpoint was called. No orders were placed.

- Generated: `2026-05-18T18:00:34.165839-07:00`
- Session date: `2026-05-19`
- Decision: `blocked_live_market_data_entitlements`
- Log directory: `/Users/gduby/Library/Logs/autoresearch-trading`
- Next action: Confirm the required IBKR paper market-data subscriptions/session state, then run the no-order shadow path again during market hours.

## What It Means

- Gateway/API startup reached the market-data probe, but IBKR refused at least one live data request. This is a data entitlement/session issue, not a model issue.
- Detected signals: api_port_open_detected=1, holding_status_detected=1, ibkr_api_connected_detected=1, ibkr_competing_live_session=1, ibkr_keepalive_socket_disconnect=1, ibkr_market_data_not_subscribed=1, missing_live_market_data_entitlements=1, pass_status_detected=1, pythonpath_missing_for_session_runner=1, socket_disconnect=1

## LaunchAgents

| label | loaded | state | runs | last exit | stdout | stderr |
| --- | ---: | --- | ---: | ---: | --- | --- |
| com.autoresearch.ibgateway.paper | `True` | `active` | `4` | `1` | `/Users/gduby/Library/Logs/autoresearch-trading/ibgateway-paper.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/ibgateway-paper.err.log` |
| com.autoresearch.protocol101.paper-preflight | `True` | `active` | `4` | `1` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-preflight.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-preflight.err.log` |
| com.autoresearch.protocol101.paper-session | `True` | `active` | `4` | `1` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.out.log` | `/Users/gduby/Library/Logs/autoresearch-trading/protocol101-paper-session.err.log` |

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
| missing_live_market_data_entitlements | 1 |
| pass_status_detected | 1 |
| pythonpath_missing_for_session_runner | 1 |
| socket_disconnect | 1 |

## Runtime Wrappers

| wrapper | exists | exports PYTHONPATH | modified |
| --- | ---: | ---: | --- |
| run_protocol101_paper_session.sh | `True` | `True` | `2026-05-18T17:59:34.618453-07:00` |
| run_protocol101_paper_preflight.sh | `True` | `True` | `2026-05-18T17:59:34.622199-07:00` |
| run_ibkr_autostart_status.sh | `True` | `True` | `2026-05-18T17:59:34.625765-07:00` |

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
- Size bytes: `20918`
- Modified: `2026-05-18T08:25:23.709114-07:00`
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
- Size bytes: `3454`
- Modified: `2026-05-18T07:07:39.671432-07:00`
- Signals: `['ibkr_competing_live_session', 'ibkr_market_data_not_subscribed']`

```text
Error 354, reqId 4: Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.SPX S&P 500 Stock Index/TOP/ALL, contract: Index(conId=416904, symbol='SPX', exchange='CBOE', currency='USD', localSymbol='SPX')
Error 354, reqId 6: Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.VIX CBOE Volatility Index/TOP/ALL, contract: Index(conId=13455763, symbol='VIX', exchange='CBOE', currency='USD', localSymbol='VIX')
Error 354, reqId 4: Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.SPX S&P 500 Stock Index/TOP/ALL, contract: Index(conId=416904, symbol='SPX', exchange='CBOE', currency='USD', localSymbol='SPX')
Error 354, reqId 6: Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.VIX CBOE Volatility Index/TOP/ALL, contract: Index(conId=13455763, symbol='VIX', exchange='CBOE', currency='USD', localSymbol='VIX')
Error 354, reqId 4: Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.SPX S&P 500 Stock Index/TOP/ALL, contract: Index(conId=416904, symbol='SPX', exchange='CBOE', currency='USD', localSymbol='SPX')
Error 354, reqId 6: Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.VIX CBOE Volatility Index/TOP/ALL, contract: Index(conId=13455763, symbol='VIX', exchange='CBOE', currency='USD', localSymbol='VIX')
Error 354, reqId 4: Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.SPX S&P 500 Stock Index/TOP/ALL, contract: Index(conId=416904, symbol='SPX', exchange='CBOE', currency='USD', localSymbol='SPX')
Error 354, reqId 6: Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.VIX CBOE Volatility Index/TOP/ALL, contract: Index(conId=13455763, symbol='VIX', exchange='CBOE', currency='USD', localSymbol='VIX')
Error 1102, reqId -1: Connectivity between IBKR and Trader Workstation has been restored - data maintained. All data farms are connected: usfarm; ushmds; secdefil.
Error 10197, reqId 4: No market data during competing live session, contract: Index(conId=416904, symbol='SPX', exchange='CBOE', currency='USD', localSymbol='SPX')
Error 10197, reqId 6: No market data during competing live session, contract: Index(conId=13455763, symbol='VIX', exchange='CBOE', currency='USD', localSymbol='VIX')
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
- Size bytes: `796`
- Modified: `2026-05-18T06:45:01.431239-07:00`
- Signals: `['pythonpath_missing_for_session_runner']`

```text
/Library/Developer/CommandLineTools/usr/bin/python3: Error while finding module specification for 'v4.scripts.run_protocol147_protocol101_morning_session' (ModuleNotFoundError: No module named 'v4')
/Library/Developer/CommandLineTools/usr/bin/python3: Error while finding module specification for 'v4.scripts.run_protocol147_protocol101_morning_session' (ModuleNotFoundError: No module named 'v4')
/Library/Developer/CommandLineTools/usr/bin/python3: Error while finding module specification for 'v4.scripts.run_protocol147_protocol101_morning_session' (ModuleNotFoundError: No module named 'v4')
/Library/Developer/CommandLineTools/usr/bin/python3: Error while finding module specification for 'v4.scripts.run_protocol147_protocol101_morning_session' (ModuleNotFoundError: No module named 'v4')
```

### Recent IBC Logs
- `/Users/gduby/Library/Logs/autoresearch-trading/ibc/ibc-gateway.err.log` size=58 modified=`2026-05-18T06:45:01.532039-07:00` signals=`[]`
- `/Users/gduby/Library/Logs/autoresearch-trading/ibc/ibc-gateway.out.log` size=18030 modified=`2026-05-18T08:25:23.693200-07:00` signals=`[]`

## Live Paper JSONL Logs

- Date directory exists: `False`
- Root: `v4/logs/paper_trading`
