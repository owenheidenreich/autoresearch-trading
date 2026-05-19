# IBKR SPXW Live Market-Data Entitlement Check

No orders were created or submitted.

- Decision: `blocked`
- Blocked reason: `missing_live_market_data_entitlements`
- IBKR connected: `True`
- IBKR port: `4002`
- Regular market hours: `True`

## Feed Status

```json
{
  "spx": {
    "live_price_available": false,
    "market_data_type": "live",
    "price": null
  },
  "spxw_options": {
    "contracts_qualified": 0,
    "contracts_requested": 0,
    "delayed_nbbo_rows": 0,
    "live_nbbo_rows": 0,
    "market_data_type_counts": {}
  },
  "vix": {
    "live_price_available": false,
    "market_data_type": "live",
    "price": null
  }
}
```

## Required Actions

- `Market Data API Acknowledgement`: IBKR can reject API market data until this Client Portal acknowledgement is enabled. Source: https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/
- `Cboe Streaming Market Indexes`: Needed for live SPX/VIX index L1 context. Source: https://www.interactivebrokers.com/en/pricing/market-data-pricing.php
- `OPRA Top of Book (L1)(US Option Exchanges)`: Needed for live SPXW option NBBO. Source: https://www.interactivebrokers.com/en/pricing/market-data-pricing.php
- `Underlying/index plus derivative data`: IBKR documents that options Greeks need both underlying and derivative subscriptions. Source: https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/

## Subscription Errors

```json
[
  {
    "contract": "Index(conId=416904, symbol='SPX', exchange='CBOE', currency='USD', localSymbol='SPX')",
    "error_code": 354,
    "error_string": "Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.SPX S&P 500 Stock Index/TOP/ALL",
    "req_id": 4
  },
  {
    "contract": "Index(conId=13455763, symbol='VIX', exchange='CBOE', currency='USD', localSymbol='VIX')",
    "error_code": 354,
    "error_string": "Requested market data is not subscribed. Check API status by selecting the Account menu then under Management choose Market Data Subscription Manager and/or availability of delayed data.Delayed market data is available.VIX CBOE Volatility Index/TOP/ALL",
    "req_id": 6
  }
]
```
