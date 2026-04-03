# ART² Program (v18)

Single source of truth. If anything conflicts with this file, this file wins.

## Mission

Build a model that profitably trades SPX 0DTE long options: when to enter, which strike, how much risk, when to exit. See `training/principles.md` for the full definition of profitability goals, integrity rules, and the migration path from current prediction-based scoring to P&L-based scoring.

## The Autoresearch Loop

This project follows Karpathy's autoresearch design: the AI is an autonomous researcher. It modifies train.py, runs experiments, evaluates results, keeps or discards, and repeats. The human's role is writing this program.md file -- programming the research organization, not doing the research.

### Experiment Loop

1. Read this file + train.py + lab_notebook.md + principles.md
2. Propose ONE small change to train.py (or run baseline). Write hypothesis BEFORE GPU spend.
3. git commit the change
4. Run experiment: `python3 tools/inner_loop.py experiment --summary "hypothesis"`
5. Check score. If better: KEPT (branch advances). If not: REVERTED (git reset).
6. Log result in lab_notebook.md
7. Check stop rules (see `principles.md` Section 6). If no stop rule fires, go to step 1.

Session limits: max 50 experiments or 6 hours. Expected cadence: ~7 min per experiment.

### Decision Rules

- **KEEP** when score improves (higher is better)
- **DISCARD** when score is equal or worse -- revert to previous commit
- **CRASH** -- if it's a typo or easy fix, fix and re-run. If fundamentally broken, log it, discard, move on
- **Timeout** -- if experiment exceeds 10 minutes, kill and treat as crash

### When Stuck (3+ consecutive reverts)

Stop experimenting. Read the replay backtest data. Form a hypothesis about WHY. Then try again with a structural change. Do not keep hammering small hyperparameter tweaks when the issue is structural.

### When to Stop

The loop runs until a stop rule fires (see `principles.md` Section 6) or the human interrupts. Stop rules include: goal achieved, convergence plateau (8 experiments without >= 2% improvement), session budget exhausted (50 experiments or 6 hours), diminishing returns, stuck loop, or crash storm. When stopped: log findings to lab_notebook.md, summarize what worked, propose next research directions, and wait for human review before the next session.

## Model Contract

Five-head TradingModel (v18):
- **Market head**: (batch, 3) -- predicted SPX % change at 15/30/60 bar horizons
- **Entry gate**: (batch, 1) -- sigmoid: should we enter a trade?
- **Risk head**: (batch, 3) -- [stop_distance, target_distance, conviction]
- **Exit head**: (batch, 1) -- sigmoid: should we exit?
- **Direction head**: (batch, 6) -- 6-class softmax: call/put x ATM/OTM5/OTM10

Note: forward() returns 7 values (5 above + gate_logit + exit_logit for autocast-safe BCE).

Backbone: Transformer d=64, 4 heads, depth=3, Pre-LN, causal
Input: (batch, lookback, 39 features)

## Data Contract

`data.pt` contains:
- `features`: (N, 39) normalized market features
- v17 labels: `pred_return_15/30/60`, `pred_volatility_30`, `pred_action_target`
- v18 labels: `v18_mfe`, `v18_mae`, `v18_entry_gate`, `v18_risk_stop_distance`, `v18_risk_target_distance`, `v18_risk_conviction`, `v18_exit_label`, `v18_direction_label`, `v18_bar_weight`
- `valid_mask`, `train_end_idx`, `val_start_idx`, `val_end_idx`

No options P&L in training loss. No hindsight.

## Score

```
score = direction_accuracy * (1 + max(0, rank_correlation))
```

One number. This is the ONLY input to the keep/revert decision.

**Note:** This is the current score formula. The target is to incorporate replay P&L metrics (profit factor, risk-adjusted return) per `principles.md` Section 7, Phase 2.

## Loss

```python
total = huber(market_pred, ret_target, delta=0.01)
     + GATE_W  * bce_with_logits(gate_logit, gate_target)  # weighted by bar_weight
     + RISK_W  * huber(risk_params, risk_target, delta=0.5)
     + EXIT_W  * bce_with_logits(exit_logit, exit_target)  # weighted by bar_weight
     + DIR_W   * cross_entropy(dir_logits, dir_target)      # weighted by bar_weight
```

## What You MAY Change in train.py

- Hyperparameters (LR, dropout, batch size, weight decay, all loss weights)
- Loss function form (Huber delta, add terms, change weighting)
- Regularization
- LR schedule
- Batch construction
- Head architecture (layers, width, activation)
- New loss terms with env var weights

## What You MUST NOT Change

- forward() output contract (7-tuple: market_pred, entry_gate, risk_params, exit_signal, dir_logits, gate_logit, exit_logit)
- Score formula
- D_MODEL, DEPTH, N_HEADS (architecture lock on backbone)
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
| training/prepare.py | Data pipeline, features, labels | Fixed (rebuild with --skip-download) |
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
- Stop rules govern session boundaries (see `principles.md` Section 6)
