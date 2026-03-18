# Future Features — Live Trading Readiness

## Features Still To Add

### GEX / Dealer Positioning
Whether dealers are long or short gamma determines if moves get dampened or amplified.
True GEX requires expensive data ($200/mo Options Depth). We can compute a **GEX
approximation** from publicly available open interest data if Polygon's option chain
snapshot API is available on our tier. Planned as Phase 5C:
- 4 features: `gex_level`, `gex_sign`, `gex_flip_dist`, `max_gamma_strike_dist`
- **Risk**: Polygon API tier may not include `list_snapshot_options_chain`. Test first.
- **Limitation**: Approximation only — uses start-of-day OI, assumes all options are
  dealer-short. Better than nothing for regime detection, not for precise levels.

### Market Internals (TICK, Breadth)
NYSE TICK measures uptick/downtick imbalance across all stocks. Breadth (advance/decline
ratio) shows whether moves have participation or are narrow. Both are leading indicators.
Planned as Phase 5D (deferred):
- 3 features: `tick_level`, `tick_extreme`, `tick_momentum`
- **Blocked on**: IBKR TICK-NYSE availability (needs testing)
- Can revisit with alternative sources (ThinkorSwim, TradeStation) if IBKR doesn't have it

### Mamba/SSM Architecture (Deferred)
At LOOKBACK=24 (2 hours of 5-min bars), Transformer is fine. Mamba's O(N) scaling advantage
only matters at 500+ sequence length. Not worth the complexity unless autoresearch experiments
show LOOKBACK=48/96 scoring better or the model plateaus on architecture.

## Future plans for auto research-trader

Instead of the bot having full authority to place trades, it flags setups with high confidence (the same function as if it would take the trade itself, how we measure its performance)

But we Implement an expert style layer over the top. An LLM that is fine tuned / trained on all of the necessary literature about options and trading (the 0dte library, pickles personality text, pickles strategies, an internet research check (for market news). All rapidly occurring in superhuman time. This LLM instantly analyzes the confidence data (the trigger for the ML algo) and then it determines if the trade is worth entering or not. This LLM could be ran on an h100. The time from ML algo flagging, and LLM obersving and double checking before giving the go ahead- or stopping, should be about 1 second. (As fast as we possibly can)

Then we test 50 trading days chosen at random with this expert layer over the top. 
