# ART² Program (v18)

Single source of truth. If anything conflicts with this file, this file wins.

## Mission

Build a model that predicts SPX price movement. The model does NOT know about options, P&L, or trading. It predicts where SPX goes. Separate code converts predictions into 0DTE option trades.

## The Loop

One loop. Karpathy's autoresearch design.

1. Read this file + train.py + lab_notebook.md
2. Propose ONE small change to train.py (or run baseline)
3. Run experiment: `python3 tools/inner_loop.py experiment --mutation /tmp/mutation.py --summary "hypothesis"`
4. Check score. If better: KEPT. If not: REVERTED.
5. Repeat.

Expected cadence: ~7 min per experiment. ~8 per hour on GPU.

When stuck (3+ consecutive reverts): stop experimenting. Read the replay backtest data. Form a hypothesis about why. Then try again with a structural change.

When the human asks "what should we do next": analyze existing data first, present findings, propose options. Research before GPU spend.

## Model Contract

Three-head PredictionModel:
- **Return head**: (batch, 3) -- predicted SPX % change at 15/30/60 bar horizons
- **Confidence head**: (batch, 1) -- predicted volatility
- **Action head**: (batch, 3) -- [trade_prob, position_size, exit_signal] (sigmoid, account-aware)

Account state: 5 dims [growth, consec_losses/5, daily_pnl/balance, win_rate, drawdown]
Backbone: Transformer d=64, 4 heads, depth=3, Pre-LN, causal
Input: (batch, lookback, 39 features)

## Data Contract

`data.pt` contains:
- `features`: (N, 39) normalized market features
- `pred_return_15/30/60`: forward SPX % change
- `pred_volatility_30`: realized volatility
- `pred_action_target`: graded signal (0.0 / 0.5 / 1.0)
- `valid_mask`, `train_end_idx`, `val_start_idx`, `val_end_idx`

No options P&L. No hindsight.

## Score

```
score = direction_accuracy * (1 + max(0, rank_correlation))
```

One number. This is the ONLY input to the keep/revert decision.

## Loss

```python
total = huber(pred_returns, actual_returns)
     + CONF_W * mse(pred_conf, actual_vol)
     + ACTION_W * mse(action, action_targets)
```

## What You MAY Change in train.py

- Hyperparameters (LR, dropout, batch size, weight decay, loss weights)
- Loss function form (Huber to MSE, add terms, change delta)
- Regularization
- LR schedule
- Batch construction
- Prediction horizons
- New loss terms with env var weights

## What You MUST NOT Change

- forward() signature (inputs: x, account_state; outputs: pred_returns, pred_conf, action)
- Score formula
- D_MODEL, DEPTH, N_HEADS (architecture lock)
- Add options P&L to labels
- Add hindsight to features
- Skip validation

## Hyperparameters

| Variable | Default | Range |
|----------|---------|-------|
| TRAIN_LR | 2.5e-4 | 1e-5 to 5e-3 |
| TRAIN_DROPOUT | 0.30 | 0.05 to 0.40 |
| TRAIN_BATCH_SIZE | 1024 | 32 to 2048 |
| TRAIN_WEIGHT_DECAY | 0.08 | 0.0 to 0.3 |
| TRAIN_GATE_W | 1.0 | 0.0 to 10.0 |
| TRAIN_RISK_W | 0.5 | 0.0 to 5.0 |
| TRAIN_EXIT_W | 1.0 | 0.0 to 10.0 |
| TRAIN_DIR_W | 1.0 | 0.0 to 10.0 |

## Fresh Start vs Warm Start

**Warm start** when: hyperparameter-only changes, non-training bug fixes, eval-only changes.
**Fresh start** when: feature count changed, architecture changed, loss semantics changed, data.pt rebuilt, multiple structural changes at once.

Fresh start procedure:
```bash
rm training/best_model.pt
echo "-5.0" > training/.best_score
rm -f training/.inner_loop_state.json
cp training/train.py training/best_train.py
```

If uncertain, fresh start. Poisoned warm start costs more than retraining.

## Key Files

| File | Role | Modified by |
|------|------|------------|
| training/train.py | Model + training loop | Agent (mutations) |
| training/program.md | This file. Instructions. | Human only |
| training/prepare.py | Data pipeline, features | Fixed (rebuild with --skip-download) |
| training/replay.py | Backtest simulation | Fixed |
| training/best_model.pt | Current best checkpoint | Promoted by inner_loop.py |
| training/best_train.py | Code that produced best model | Promoted by inner_loop.py |
| training/.best_score | Current best score | Updated on KEEP |
| training/lab_notebook.md | What worked, what failed | Agent logs findings |

## Commands

| Task | Command |
|------|---------|
| Run experiment | `python3 tools/inner_loop.py experiment --summary "hypothesis"` |
| Run with mutation | `python3 tools/inner_loop.py experiment --mutation /tmp/mutation.py --summary "what and why"` |
| Replay backtest | `python3 training/replay.py --backtest --model training/best_model.pt` |
| Rebuild data | `python3 training/prepare.py --skip-download --use-spx` |
| Version check | `python3 -m pytest tests/test_version_consistency.py -v` |
| Deploy GPU | `./infra/deploy.sh boot && ./infra/deploy.sh start` |
| Paper trade | `python3 tools/paper_live.py --paper-auto --port 4002 --client-id 80` |
| Status | `cat training/.best_score && python3 tools/inner_loop.py status` |
| Fresh start | `rm training/best_model.pt; echo -5.0 > training/.best_score; rm -f training/.inner_loop_state.json; cp training/train.py training/best_train.py` |

## Domain Knowledge

**Time-of-day:** Morning (9:35-10:30) = strongest trends. Lunch (10:30-13:30) = avoid. Power hour (15:30+) = extreme gamma, avoid for long options.

**VIX regimes:** <15 tight ranges, 15-20 normal, 20-30 wide stops needed, >30 crisis.

**Theta decay:** 1/sqrt(T). Long options bleed fastest in final 2 hours. Hold duration matters.

**Key principle:** Always take profits off the table. First test of target = exit.

Reference: `docs/domain/pickles-trading-knowledge.md`, `docs/domain/0dte-domain-knowledge.md`

## Invariants

- best_model.pt, best_train.py, .best_score are always in sync
- Version consistency gate must pass before training
- One change per experiment
- Research before GPU spend
