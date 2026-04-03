# v18 Handoff Document

**Date:** 2026-04-02
**Status:** ALL PHASES COMPLETE. Ready for data rebuild + first training run.

## Architecture: 5-Head TradingModel

```
Shared Transformer Backbone (causal, Pre-LN)
  d_model=64, heads=4, depth=3, ff_mult=3

  -> Market Head:    (batch, 3)  SPX returns at 15/30/60 bars
  -> Entry Gate:     (batch, 1)  sigmoid: should we enter?
  -> Risk Head:      (batch, 3)  [stop_distance, target_distance, conviction]
  -> Exit Head:      (batch, 1)  sigmoid: should we exit?
  -> Direction Head: (batch, 6)  6-class softmax: call/put x ATM/OTM5/OTM10
```

## Labels: Path-Quality (MFE/MAE)

All labels computed from forward price path, not hindsight P&L:
- **MFE/MAE**: max favorable/adverse excursion over 30 bars using SPX high/low
- **Entry gate**: sigmoid((MFE - |MAE|) / ATR - 1.0), zeroed during lunch/power hour
- **Risk**: stop_distance (MAE/ATR), target_distance (MFE/ATR), conviction (MFE/(MFE+|MAE|))
- **Exit**: 1.0 when near-term (5-bar) risk exceeds reward
- **Direction**: 6-class argmax across stopped P&L (call/put x ATM/OTM5/OTM10)
- **bar_weight**: morning 1.0, midday 0.5, lunch core 0.1, afternoon 0.8, power hour 0.0

## Loss Function

5-term loss with time-of-day weighting:
- Market prediction: Huber (delta=0.01)
- Entry gate: BCE, weighted by bar_weight (GATE_W=1.0)
- Risk parameters: Huber (delta=0.5) (RISK_W=0.5)
- Exit signal: BCE, weighted by bar_weight (EXIT_W=1.0)
- Direction: cross-entropy, weighted by bar_weight (DIR_W=1.0)

## What Was Done

### Phase 1: Feature swap + data quality fixes (COMPLETE, prior session)
- 5 feature swaps (gamma_pressure, vix_roc, iv_percentile, option_spread_width, prev_close_dist)
- 5-min aggregation, cross-day contamination removal, walk-forward normalization
- data.pt: 387,990 bars, 39 features, 999 days

### Phase 2: Path-quality labels in prepare.py (COMPLETE)
- compute_v18_labels() function added
- make_v18_dataloader() yields (x, y_pred, y_v18, bar_weight)
- All stored in data.pt under v18_* keys

### Phase 3: 5-head TradingModel in train.py (COMPLETE)
- TradingModel replaces PredictionModel
- v18_loss() with 5-term weighted loss
- Warm start gated on model_version='v18' (forces fresh start from v17)
- Score unchanged: dir_accuracy * (1 + max(0, rank_corr))

### Phase 4: Wire into replay.py (COMPLETE)
- load_model detects v18 via config['model_version']
- Model inference handles both v17 and v18 (gate_head detection)
- Entry uses direction_logits for 6-class strike selection
- Stop uses risk_head stop_distance (ATR-normalized)

### Phase 5: Wire into decision.py (COMPLETE)
- InferenceResult extended with risk_stop_distance, risk_target_distance, risk_conviction
- v18/v17 dual-path in infer(), build_entry_intent(), build_risk_update_intent()

### Verification (COMPLETE)
- All 20 tests pass
- best_train.py, monitor.py, program.md synced
- trading_rules.py unchanged (safety envelope)

## Next Steps

1. **Rebuild data.pt**: `python3 prepare.py --skip-download --use-spx`
   This will compute v18 labels and store them in data.pt.

2. **First training run**: `python3 train.py`
   Fresh start (v17 checkpoint will be skipped automatically).

3. **Evaluate**: Run replay on validation days to compare v18 vs v17.

4. **Tune loss weights**: GATE_W, RISK_W, EXIT_W, DIR_W via inner_loop.py.

## Key Files

| File | Status |
|------|--------|
| training/prepare.py | COMPLETE: v18 labels + dataloader |
| training/train.py | COMPLETE: 5-head TradingModel |
| training/best_train.py | SYNCED with train.py |
| training/replay.py | COMPLETE: v18/v17 dual-path |
| training/live/decision.py | COMPLETE: v18/v17 dual-path |
| training/trading_rules.py | UNCHANGED (safety envelope) |
| training/program.md | UPDATED: v18 loss weights |
| tools/monitor.py | UPDATED: v18 param abbrevs |
| tests/test_live_decision.py | UPDATED: v17/v18 tests |

## Important Design Decisions

- Model makes complete trading decisions (entry, exit, strike, risk) -- not just direction
- Path-quality labels avoid hindsight trap (MFE/MAE from price, not from knowing the outcome)
- Safety envelope (trading_rules.py) still applies: model output filtered through domain rules
- 6-class direction (call/put x ATM/OTM5/OTM10) -- deeper OTM excluded (no data coverage)
- v17 backward compatibility preserved in all consumer files
