# v2 Goal

## Mission

Same as v1: build a model that profitably trades SPX 0DTE long options.

v2 is a structural reset. The domain (SPX 0DTE), data sources (IBKR, Polygon),
and general approach (supervised learning + autoresearch loop) are unchanged.
What changes is the contract between training, replay, and live.

---

## What v2 Fixes

**v1 problem: fractured contract.**
Training emitted prediction arrays. Replay reconstructed trades from those
predictions using hardcoded rules. Live used a different set of rules via
DecisionIntent. The model never saw a real trade object during training.

**v2 solution: TradeIntent.**
One frozen dataclass flows through the entire system. The model learns to
emit it. Replay scores it. Live executes it. No translation layers.

**v1 problem: proxy labels.**
4 of 5 label heads used proxies (MFE/MAE entry gate, ATR risk targets,
SPX-based exit signal). Only direction labels used real stopped option P&L.

**v2 solution: trade-outcome labels.**
The oracle labeler searches the candidate universe at each bar and finds the
best executable trade under the evaluator's rules. Labels are derived from
what actually would have worked, not from proxies for what might have worked.

**v1 problem: prediction-based scoring.**
Score formula: `direction_accuracy * (1 + max(0, rank_correlation))`.
A model could score well by predicting direction without ever making a
profitable trade.

**v2 solution: P&L-based scoring.**
Promotion score is dominated by replay profit factor. Direction accuracy
is diagnostic only. If the model can't make money in replay, it doesn't
get promoted.

---

## Success Gates

v2 is complete when all of the following hold simultaneously:

1. **Replay evaluates exact trades, not abstract predictions.**
   The simulator receives TradeIntents and produces P&L. No intermediate
   prediction-to-trade translation.

2. **Strike selection is dynamic from real chain candidates.**
   The model selects from available contracts (ATM +/- 30 in 5-point steps),
   not a fixed 6-class enum.

3. **Stop loss and take profit are part of the learned trade object.**
   The TradeIntent carries explicit stop_price and take_profit_price.
   These are not bolted on by hardcoded rules after the model speaks.

4. **Training beats simple baselines on held-out days.**
   The model must beat random selection, ATM-always, and a simple rules
   baseline before any GPU time is spent. See baselines.md.

5. **The promoted model runs all day in IBKR paper autonomously.**
   Full RTH session (9:30-16:00 ET) without crashes, stale data, or
   orphaned orders. Same requirement as v1, preserved.

6. **Training, replay, and live all use the same trade contract.**
   TradeIntent is the interface. No DecisionIntent, no InferenceResult,
   no separate live-only trading rules.

---

## What Changes

| Area | v1 | v2 |
|------|----|----|
| Trade contract | DecisionIntent (live only) | TradeIntent (everywhere) |
| Action space | 6-class enum (call/put x ATM/OTM5/OTM10) | Dynamic candidate scoring |
| Labels | Proxy (MFE/MAE, ATR, SPX returns) | Trade outcomes via oracle |
| Scoring | direction_accuracy * (1 + rank_corr) | Replay profit factor primary |
| Strike selection | Hardcoded offset from model class | Model scores real candidates |
| Risk params | Partially learned, partially hardcoded | Fully part of TradeIntent |

---

## What Is Preserved

- **Domain knowledge:** 0DTE SPX options, theta decay, gamma profiles, VIX regimes,
  time-of-day patterns. All documented in docs/domain/.
- **Feature engineering:** 39 features covering price action, volume, volatility,
  options Greeks, market structure. Preserved as-is initially.
- **Data access:** IBKR historical, Polygon SPY volume, incremental caching.
- **IBKR execution:** Bracket orders, OCO logic, position tracking, audit trail.
- **Autoresearch loop:** Mutate train.py, train, score, keep/revert.
- **Governance:** Session limits, stop rules, anomaly detection, score fingerprinting.

---

## What Is Explicitly NOT Preserved as a Contract

- **Model architecture.** The current 5-head transformer is v1's design. v2 may
  use a different architecture. The model is the thing being researched; locking
  its internal shape defeats the purpose. Only the input contract (39 features)
  and output contract (TradeIntent) are fixed.
- **Loss formulation.** The specific weighted combination of Huber, BCE, and
  cross-entropy losses is a v1 artifact. v2 loss design follows from the new
  label scheme.
- **Current score formula.** The v1 formula (direction_accuracy * rank_corr)
  is replaced by the evaluator contract in evaluator.md.

---

## Non-Goals for Phase 0

Phase 0 creates the skeleton and writes the specs. It does not:
- Implement any v2 behavior
- Change any v1 code
- Retrain any model
- Touch data.pt or model weights
- Modify the autoresearch loop
