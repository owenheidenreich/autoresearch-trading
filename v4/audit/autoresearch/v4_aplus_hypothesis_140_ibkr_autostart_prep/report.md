# Protocol 140: IBKR Paper Autostart Prep

No paid data was downloaded. No broker order endpoint was called. No orders were placed.

- Decision: `ready_to_install_ib_gateway_paper_autostart`
- IB Gateway app: `/Users/gduby/Applications/IB Gateway 10.45/IB Gateway 10.45.app`
- Configured API port: `4000`
- Candidate API ports: `[4000, 4002, 7497, 7496, 4001]`

## Checks

| check | passed | detail |
| --- | ---: | --- |
| ib_gateway_app_exists | `True` | `/Users/gduby/Applications/IB Gateway 10.45/IB Gateway 10.45.app` |
| jts_ini_exists | `True` | `/Users/gduby/Jts/jts.ini` |
| paper_mode_configured | `True` | `tradingMode=p` |
| api_only_configured | `True` | `ApiOnly=true` |
| api_port_configured | `True` | `LocalServerPort=4000` |
| trusted_localhost | `True` | `TrustedIPs=127.0.0.1` |

## Assets

- Gateway LaunchAgent: `v4/ops/launchd/com.autoresearch.ibgateway.paper.plist`
- Preflight LaunchAgent: `v4/ops/launchd/com.autoresearch.protocol101.paper-preflight.plist`
- Install script: `v4/ops/launchd/install_ibkr_paper_autostart.sh`
- Uninstall script: `v4/ops/launchd/uninstall_ibkr_paper_autostart.sh`

## Autostart Assumption

launchd can open IB Gateway in paper mode, but IBKR may still require saved credentials/2FA. If Gateway pauses at login, finish that login once; the preflight will keep reporting port-not-open until the API listener is available.

## Next Gate

Install the LaunchAgents when desired, then run the IBKR preflight during market hours. Paper-order submission remains blocked until the paper-order guard and live Protocol101 shadow parity pass.
