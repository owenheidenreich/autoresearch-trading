# Protocol 124: Protocol101 Live-Data Parity Checkpoint

No paid data was downloaded. No broker order endpoint was called. No orders were placed.

- Decision: `blocked_live_subscriptions_delayed_plumbing_passed`
- Protocol119 decision: `blocked_missing_live_market_data_entitlements`
- IBKR blocked reason: `missing_live_market_data_entitlements`
- Delayed capture decision: `blocked`
- Delayed capture blocker: `delayed_market_data_not_promotion_grade`

## Checks

| check | value |
| --- | ---: |
| `protocol101_entry_router_wired` | `True` |
| `protocol119_feature_status_pass` | `True` |
| `ibkr_connected` | `True` |
| `spx_live_price` | `False` |
| `vix_live_price` | `False` |
| `spxw_live_nbbo_rows` | `0` |
| `delayed_plumbing_rows` | `24` |
| `delayed_plumbing_parity_pass` | `True` |
| `delayed_market_data_observed` | `True` |

## Required Before Protocol101 Live Shadow

- Enable live Cboe index market data for SPX and VIX in the IBKR API session.
- Enable live OPRA top-of-book data so SPXW option NBBO is available to the API session.

Delayed rows prove that the local IBKR pipe, SPXW chain qualification, quote parsing, and no-order safety checks can run. They do not prove promotion-grade live-data parity.
