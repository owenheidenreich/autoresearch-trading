# IBKR SPXW Live Market-Data Entitlement Check

No orders were created or submitted.

- Decision: `pass`
- Blocked reason: `None`
- IBKR connected: `True`
- IBKR port: `4002`
- Regular market hours: `True`

## Feed Status

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

## Required Actions

- `Market Data API Acknowledgement`: IBKR can reject API market data until this Client Portal acknowledgement is enabled. Source: https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/
- `Cboe Streaming Market Indexes`: Needed for live SPX/VIX index L1 context. Source: https://www.interactivebrokers.com/en/pricing/market-data-pricing.php
- `OPRA Top of Book (L1)(US Option Exchanges)`: Needed for live SPXW option NBBO. Source: https://www.interactivebrokers.com/en/pricing/market-data-pricing.php
- `Underlying/index plus derivative data`: IBKR documents that options Greeks need both underlying and derivative subscriptions. Source: https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/
