# V3 Lab Notebook

## v3_exp_001 — Baseline PPO (2026-04-11)

**Config:** 10 updates, 8 rollout days, lr=3e-4, entropy=0.01, gamma=0.99

**Results:** score=-0.50, return=-57.4%, dd=100.2%, sortino=-1.37, trades=147, win_rate=8.2%

**Training trajectory:**
- update 2:  score=-0.50, return=-0.57, dd=1.00, trades=147
- update 4:  score=-0.50, return=-0.60, dd=0.96, trades=175
- update 6:  score=-0.50, return=-0.75, dd=1.01, trades=305
- update 8:  score=-0.50, return=-0.90, dd=1.01, trades=337
- update 10: score=-0.50, return=-0.88, dd=1.01, trades=153

**Diagnosis:**
- Score flat at -0.50 across all checkpoints — no learning signal reached the policy
- Turnover was massive (18,256 contracts) — model sizes too aggressively
- 8% win rate — essentially random entries
- Exposure at 32% — model is trading frequently but with no edge
- The "best" checkpoint was update 2 (least negative return), meaning later updates made things worse
- Baselines: ATM-Fixed scored -0.50 with +0.0% return and 45% win rate (much better than learned policy)

**Key issues to address:**
1. 10 updates is far too few for a model this complex (dual transformer + pointer network)
2. Reward scale: equity changes dominate, penalties are negligible
3. No reward normalization or advantage normalization visible in the loop
4. Learning rate may need warmup or lower value for transformer encoders

**Next:** exp_002 — increase updates to 50 with early stopping

## v3_exp_002 — More Training + Early Stopping (2026-04-11)

**Config:** 50 updates (early-stopped at 25), 8 rollout days, lr=3e-4, entropy=0.01, patience=4, eval_interval=5

**Training trajectory:**
- update 5:  score=-0.50, return=-93.1%, dd=100.6%, trades=47
- update 10: score=-0.50, return=-13.1%, dd=100.7%, trades=1216
- update 15: score=-0.50, return=-0.5%, dd=38.4%, trades=103
- update 20: score=-0.50, return=-0.08%, dd=1.5%, trades=16
- update 25: score=-0.50, return=0.0%, dd=0.5%, trades=14 (early stop)

**Diagnosis:**
- Clear learning trajectory: model went from catastrophic losses to capital preservation
- But it converged to a near-NoTrade policy (14 trades in 60 days)
- The drawdown penalty (0.10) and terminal drawdown penalty (0.10) trained the model to avoid risk entirely
- Score stayed at -0.50 throughout because the scoring function bottoms out there
- Bug: best-checkpoint selection uses `score > best_score` — when all scores tie, it keeps the first (worst) eval. The saved checkpoint is from update 5 (return=-93.1%), not update 25 (return=0.0%)

**Key issues to address:**
1. Fix best-checkpoint tiebreaker: when scores tie, prefer lower drawdown or higher return
2. Reduce drawdown_penalty to let the model explore trades without being punished into passivity
3. The model needs a positive incentive for trading — currently "do nothing" is the optimal policy

**Next:** exp_003 — fix checkpoint selection, reduce drawdown penalties

## v3_exp_003 — Reduced Drawdown Penalties (2026-04-11)

**Config:** 50 updates (early-stopped at 25), patience=4, drawdown_penalty=0.02, terminal_drawdown_penalty=0.02

**Training trajectory:**
- update 5:  score=-0.50, return=-45.0%, dd=81.7%, trades=43
- update 10: score=-1.00, return=0.0%, dd=0.6%, trades=1
- update 15: score=-1.00, return=0.0%, dd=0.0%, trades=0
- update 20-25: full NoTrade, early-stopped

**Diagnosis:**
- Converged to NoTrade even faster than exp_002 (by update 15 vs update 25)
- Reducing drawdown penalties didn't help — the dense reward itself punishes exploration
- Every random trade loses money to commissions+slippage, so the gradient says "stop trading"
- The model needs a positive incentive to keep exploring trades during training

**Next:** exp_004 — add trade_exploration_bonus=0.005 (reward for opening trades), keep low dd penalties
