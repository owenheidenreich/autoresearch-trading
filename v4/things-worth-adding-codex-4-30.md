**Working Hypothesis**

The v4 data is meaningfully better than v3 because labels now use SPXW CBBO ask-entry / bid-exit truth, but the current v4 model is still too small to prove edge. The Q4 failure looks less like “bad data again” and more like a representation/generalization failure: ATM-only call/put actions, simplified VWAP/OMAR features, weak side discrimination, and too much selection pressure on a small Jan-Feb window.

The next protocol should test whether a richer but still controlled model can generalize before buying more data.

**Things Worth Adding**

1. Stable walk-forward identity from [v2/core/walkforward.py](/Users/gduby/Documents/autoresearch-trading/v2/core/walkforward.py:1): deterministic window IDs, window-seeded training, stable fold ordinals, no promotion bypass.

2. Scope-separated reporting from [v2/core/cv_report.py](/Users/gduby/Documents/autoresearch-trading/v2/core/cv_report.py:1): pooled economics, per-window metrics, and stability must be separate. Pooled PF alone is dangerous.

3. Rolling-window design from [v3/harness/rolling_windows.py](/Users/gduby/Documents/autoresearch-trading/v3/harness/rolling_windows.py:1): expanding train, recent validation, disjoint future OOS. Use this once v4 has more clean months.

4. Frozen audit discipline: March 2026 and Q4 2025 stay audit-only. The Q4 report already rejected both current champions, so no threshold/loss/window tuning against Q4.

5. Trial manifest per run: hypothesis ID, data fingerprint, feature version, label policy, split dates, seeds, loss config, allowed knobs, and all rejected trials.

6. Multiple-testing protection: track number of trials and treat “best of many” as suspicious. White’s Reality Check, Hansen SPA, and Bailey/Lopez de Prado backtest-overfit work all point at this exact problem.

7. Prior-window calibration: thresholds/risk controls for window W can only use windows before W, borrowing the v3 L3 calibration lesson.

8. Bootstrap confidence intervals: PF, PnL, drawdown, positive-day rate, and top-day concentration should report uncertainty, not just point estimates.

9. Matched random baselines: same trade count, same time window, random side, ATM always-call/put, VWAP/OMAR prior rule, and random exit/time-stop baselines.

10. Slippage stress grid: current labels use bid/ask, good. Still add extra fee/slippage stress to prove the signal is not a spread artifact.

11. Concentration gates: no champion if one day or one volatility regime explains most PnL.

12. Feature-provenance audit: every feature must declare source, causal timestamp, whether it is label-derived, and whether it came from v2/v3 prior mining. This directly guards against the v3 leakage failure in [v3/reference/oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md](/Users/gduby/Documents/autoresearch-trading/v3/reference/oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md:1).

13. True VWAP sigma position: v2’s SPY-derived VWAP scaled into SPX space is much more precise than v4’s current fallback VWAP. See [v2/core/market_structure.py](/Users/gduby/Documents/autoresearch-trading/v2/core/market_structure.py:85).

14. True OMAR fields: OMAR high, low, mid, range from the first 1-minute bar. v4 currently reduces OMAR to a session-normalized move, which loses the actual trader level.

15. First-15 structure: `first15_acceptance`, `first15_close_position`, `first15_range_pct`, and opening gap. These were load-bearing in v2.

16. Last-10 structure: last-10 high/low/range and break state. v3 found last10 range over OMAR mattered in failure cells.

17. Sigma-position × IV-percentile cells: v3 Phase C showed side preference flips by VWAP/IV cell, and L2 picked the wrong side in specific cells.

18. Time-of-day × cell interactions: not hard gates, but model-visible state. Morning after first 30 minutes and late afternoon can remain features, not fixed destiny.

19. ATR-15 as statistical scale, OMAR as trader location. v3 found OMAR is useful as a level/confluence reference, but ATR-15 is better as a volatility scaler.

20. Option-quality features: spread fraction, bid/ask size, quote age, open interest, actual OHLCV option volume where available, and volume-missing flags.

21. Greek pressure features: delta, gamma, theta, gamma/theta ratio, theta per minute, premium decay burden. Near-ATM 0DTE behavior is heavily gamma-sensitive.

22. Move from ATM-only to action surface. Current [v4/model/action_pilot.py](/Users/gduby/Documents/autoresearch-trading/v4/model/action_pilot.py:1) chooses only no-trade / nearest ATM call / nearest ATM put. The model should choose flat plus call/put tokens across the $5 ladder.

23. Reuse the v3 unified policy idea, not the exact code: scalar encoder, sequence encoder, contract/action tokens, explicit flat action, multi-head outputs. See [v3/layer2/unified_policy.py](/Users/gduby/Documents/autoresearch-trading/v3/layer2/unified_policy.py:85).

24. Side-contrastive training: force same-minute call-vs-put comparisons so the model cannot hide behind generic “trade/no-trade” scoring.

25. Side-balance/cell-balance weights: v3 showed side bias can dominate and still look okay in aggregate.

26. Clean-entry and stopout-risk heads: useful as auxiliary heads, not as promotion metrics by themselves.

27. Keep entry and exit research separate. First prove entry/side/strike selection; later add in-trade exits.

28. Later exit features from [v3/layer3/h3a_features.py](/Users/gduby/Documents/autoresearch-trading/v3/layer3/h3a_features.py:1): realized vol, PnL velocity, and MFE decay are causal and worth porting once entry is stable.

29. Regime-stratified diagnostics: trend × vol, VWAP sigma × IV, time bucket, spread bucket, premium bucket. The model should learn across environments, but the report must show where it breaks.

30. Broad-purchase signal: a candidate must beat current Huber baseline on March and frozen Q4 without new knobs, with positive multi-seed behavior, controlled drawdown, non-concentrated PnL, and no obvious feature leakage.



Sources used: Cboe SPXW specs and PM settlement ([Cboe](https://www.cboe.com/tradable_products/sp_500/spx_weekly_options/specifications/)), Cboe SPXW expiration trading hours ([Cboe](https://www.cboe.com/tradable-products/sp-500/spx-options/spx-specifications)), Cboe 0DTE/gamma discussion ([Cboe](https://www.cboe.com/insights/posts/volatility-insights-evaluating-the-market-impact-of-spx-0-dte-options)), backtest overfitting ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2507040)), White Reality Check ([Econometric Society](https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping)), Hansen SPA ([Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1198/073500105000000063)), and purged/embargo CV notes ([mlfinlab docs mirror](https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html)).