# Lab Notebook

## System
SPX 0DTE | v17 prediction model | 38 features | 3 heads (return + confidence + action)

**v17 (2026-03-31, CURRENT):** Pure SPX prediction model. No P&L in training. No hindsight.
- Return head: predicts SPX % change at 15/30/60 bar horizons
- Confidence head: predicts realized volatility
- Action head: account-aware (trade_prob, position_size, exit_signal)
- Loss: Huber(returns) + MSE(confidence) + MSE(action targets)
- Score: direction_accuracy * (1 + rank_correlation)
- Fresh start. No warm start from v16 (completely different architecture + labels).

**Previous versions (v1-v16, archived):** All trained on hindsight P&L labels. The model learned "which option trade WOULD HAVE made money" rather than predicting market movement. 37 failed patterns documented. See archive/ for history.

## What Fails (do NOT retry)
| Pattern | Attempts | Result |
|---------|----------|--------|
| Hindsight P&L labels (training on future outcomes) | v1-v16 | Model memorizes patterns, doesn't generalize. Monte Carlo: HINDSIGHT_DEPENDENT |
| New nn.Module subclasses (brand new fresh-param modules) | 0/16+ | Not enough training time in 5-min budget |
| Score config gaming (modifying evaluation metric) | 0/7 | Inflates score without improving model |
| Multiple simultaneous changes | 0/many | Can't attribute improvement to any one change |
| Fixating on a single metric | 0/25+ | Tunnel vision, no improvement |

## Best Runs

| Run | Score | Dir Acc | Return MAE | Rank Corr | Notes |
|-----|-------|---------|------------|-----------|-------|
| (v17 not yet trained) | - | - | - | - | Fresh start |

## Next Priorities

Beat the current best score.
