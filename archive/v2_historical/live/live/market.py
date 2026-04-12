"""Market data streaming from IBKR.

Provides real-time 1-minute bars, VIX, and option chain data to the
decision engine. Aggregates 5-second IBKR bars into 1-minute bars and
computes features using v2/core/features.py.

v1 origin: training/live/features.py (FiveSecondMinuteAggregator,
LiveFeatureEngine) + training/live/service.py (IBKRMarketStream) +
training/live/context.py (LiveContextBundle, context refresh)
"""
# TODO: MarketDataStream class (IBKR 5-sec -> 1-min aggregation)
# TODO: OptionChainSnapshot class (real-time chain for candidate generation)
# TODO: LiveFeatureEngine (rolling feature computation via core.features)
# TODO: Context bootstrap (download recent history for normalization buffer)
