# Protocol 119: Protocol101 Live Readiness

No paid data was downloaded. This script did not call IBKR and did not place orders.

- Decision: `blocked_missing_live_market_data_entitlements`
- Protocol 101 artifact loadable: `True`
- Protocol 101 feature count: `55`
- Protocol 118 rehearsal decision: `pass_historical_no_order_shadow_rehearsal_live_capture_next`
- IBKR entitlement decision: `blocked`
- IBKR blocked reason: `missing_live_market_data_entitlements`
- Feature dependency status: `pass`

## Next Requirements

- IB Gateway is reachable, but live SPX/VIX and OPRA/SPXW market data are not available to this API session. Check market-data subscriptions and API market-data acknowledgement.
- Rerun the no-order entitlement check and require live SPX, VIX, and OPRA/SPXW NBBO market data before Protocol 101 no-order live capture.

## Feature Dependency

- Live-derivable features: `48`
- Surface edge generator loadable: `True`
- Protocol 101 entry router edge wired: `True`
- Upstream edge features: `['edge', 'hist_prev_call_minus_put_edge', 'hist_prev_max_edge', 'hist_prev_mean_edge', 'hist_roll3_call_minus_put_edge', 'hist_roll3_max_edge', 'hist_roll3_mean_edge']`
- Unknown features: `[]`

Protocol 121 proved the frozen Protocol 051/A+ surface edge scorer feeds the frozen Protocol 101 event-policy feature path without zero-filling edge features. The remaining readiness blocker is live market-data availability, not local edge-feature wiring.

## IBKR Attempts

```json
[
  {
    "host": "127.0.0.1",
    "port": 4002,
    "status": "connected"
  }
]
```
