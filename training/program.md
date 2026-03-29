# Autoresearch Trading Contract (v2 Phase)

This file is the strict operating contract for autonomous loop experiments.
If anything else conflicts with this file, this file wins.

## Mission
Build a model that makes money trading SPX 0DTE options on IBKR paper trading. The score is a proxy — focus on improving actual TRADING BEHAVIOR (profit factor, win rate, regime consistency, drawdown) rather than optimizing the score metric itself. Use the trade-level diagnostics (best/worst trades, time-of-day splits, VIX regime breakdowns) to diagnose specific weaknesses and propose targeted fixes.

## Model Contract (Required)
- Four-head architecture (v14, restored from v10):
  - Gate head: 2 logits — `[NO_TRADE, TRADE]` (entry/exit signal)
  - Direction head: 14 logits — `[CALL_ATM, CALL_OTM5..OTM30, PUT_ATM, PUT_OTM5..OTM30]` (strike selection, only active when gate=TRADE)
  - Value head: 1 scalar — expected remaining P&L (disabled, VALUE_W=0.0)
  - Risk head: `[stop_pct, size_frac, conviction]` (3 outputs, account-aware risk management)
- Position state: 7 dims (in_trade, bars_held, unrealized_pnl, account_health, loss_streak, best_pnl, bars_since_high)
- Account state: 4 dims (growth_ratio, log_size, daily_pnl_frac, win_rate_20) — risk head only
- 16 actions (gate + direction combined):
  - Action 0: `DO_NOTHING` (while flat = stay flat, while holding = EXIT)
  - Actions 1-7: `BUY_CALL_ATM`, `BUY_CALL_OTM5`, ..., `BUY_CALL_OTM30`
  - Actions 8-14: `BUY_PUT_ATM`, `BUY_PUT_OTM5`, ..., `BUY_PUT_OTM30`

**Exit mechanics**: There is NO hardcoded profit target. Gate predicting NO_TRADE while holding triggers exit (after min hold bars). The **risk head** provides learned stop-loss distance, position sizing, and conviction. Exit priority: stop_loss > model_exit (gate=NO_TRADE, bars≥2) > max_hold > end_of_day.

## Data Contract (Required)
`data.pt` must include the target fields and option price arrays required by training and replay:
- Stopped P&L (used by EV loss — counterfactual P&L for each action):
  - `call_stopped_pnl`, `put_stopped_pnl` (med-level, at DYNAMIC_STOP_BASE=0.35)
  - `otm{5,10,15,20,25,30}_call_stopped_pnl`, `otm{5,10,15,20,25,30}_put_stopped_pnl`
- Multi-level stopped P&L (for ATM strikes, selected by IV+VIX per bar):
  - `call_stopped_pnl_tight`, `put_stopped_pnl_tight` (stop=0.20)
  - `call_stopped_pnl_wide`, `put_stopped_pnl_wide` (stop=0.50)

**CRITICAL**: Multi-component `sniper_loss` (v14) uses gate labels (`pnl_ok.long()` — pure hindsight P&L), direction soft targets (P&L-weighted strike selection), and PnL alignment (trade_prob × dir_probs × stopped_pnl). ATM calls/puts use IV+VIX-based stop selection (tight/med/wide).
- Prices:
  - `atm_call_prices`, `atm_put_prices`
  - `otm{5,10,15,20,25,30}_call_prices`, `otm{5,10,15,20,25,30}_put_prices`
- Metadata:
  - `day_boundaries` — tensor of indices where trading days start (for sequential batching)

Missing required fields must fail fast. No silent fallback behavior.

## Architecture Lock
Do not change tensor-shape-defining architecture values during this phase:
- `D_MODEL`
- `DEPTH`
- `N_HEADS`

Warm start loads existing weights when shapes are compatible. Changing these values requires a fresh start.

## Safety + Runtime Constraints
- No `torch.compile`.
- No `DataParallel` / `DistributedDataParallel`.
- No `torch.jit.trace` / `torch.jit.script`.
- Keep memory/runtime within the configured budget. BATCH_SIZE ≤ 1024.
- Keep changes focused: one coherent hypothesis per experiment.

## Score Formula

The composite score determines experiment quality:

```
score = ProfitFactor * TradeSharpe * freq_mult * penalties * bonuses
```

**freq_mult** (configurable trade frequency band):
- `tpd < 0.5` → hard floor: score = -10
- `tpd < freq_lo` → linear ramp from -5 to raw_score
- `freq_lo ≤ tpd ≤ freq_hi` → `min(1, tpd/freq_center)`
- `tpd > freq_hi` → quadratic decay: `max(0.1, (freq_hi/tpd)²)`

**Configurable penalty thresholds** (applied only when score > 0):
- Consecutive loss penalty: -5% per consecutive loss beyond threshold (floor 0.5×)
- Short-hold penalty: penalizes above threshold of trades held ≤1 bar (floor 0.7×)
- Stop-loss rate penalty: penalizes above threshold of stop-outs (floor 0.5×)

**Configurable bonuses** (all default to 0.0 = neutral):
- Win rate bonus: rewards consistent winners above 40% WR
- R:R bonus: rewards strategies where avg_win > avg_loss
- Drawdown penalty: penalizes deep cumulative drawdowns
- Hold time bonus: rewards average hold times in the 5-60 bar sweet spot

**Negative scores are possible and informative.** When PF < 1.0, TradeSharpe is typically negative, yielding a negative score. A score of -0.5 is clearly closer to profitability than -10.0, giving the agent useful gradient signal.

### Score Formula is LOCKED — Do Not Modify
The `_score_config` dictionary in train.py is **read-only**. You MUST NOT change its values, add new SCORE_* env var readers, or modify how the score is computed.

**Why:** The score formula defines what "good trading" means. Changing it inflates scores without improving the model — the model trains on loss functions (gate_loss, dir_loss, pnl_alignment), NOT on the score. Modifying score_config only changes the post-training evaluation, making scores incomparable across experiments. This was exploited in a prior run where SCORE_DRAWDOWN_PENALTY was reduced from 0.5→0.05, inflating the score from 1.77→10.29 while PF and TPD barely changed.

**What to do instead:** Improve the model's actual TRADING BEHAVIOR by modifying loss hyperparameters (ENTROPY_COEFF, DO_NOTHING_BONUS), model architecture biases, or training dynamics. Improvements should be visible in raw metrics: profit factor, win rate, trades per day, drawdown.

**Human-authorized update (2026-03-28):** `win_rate_bonus` 0→0.5, `rr_bonus` 0.3→0.1. Authorized by human operator based on Pickles' strategic directive: optimize for win rate with margin of error, not raw backtest P&L. Monte Carlo showed v14 is HINDSIGHT_DEPENDENT (30% WR, removing top 5% winners kills profitability). These values prior to this change should NOT be compared to values after.

### Loss Function Policy
The core loss is multi-component `sniper_loss` (v14, proven in v10 score 16.73):
- Gate: cross-entropy on binary trade/no-trade labels + time-of-day weighting
- Direction: KL-divergence against P&L-weighted soft targets
- PnL alignment: trade_prob × (dir_probs × stopped_pnl).sum()
- Confidence: penalize high confidence on losers
- DIRECTION_ENTROPY_BONUS = 0.20 (hardcoded, prevents direction collapse)
Current v14 defaults: GATE_W=0.95, DIR_W=1.5, PNL_W=1.5, EXIT_W=0.15, CONF_W=0.05, VALUE_W=0.0, RISK_W=0.2.

**Win-rate-first levers (v14.1):**
- `REG_GATE_MARGIN` (default 0.0): Minimum profit % to label a bar as TRADE. Filters marginal hindsight winners.
- `REG_PNL_CLIP` (default 0.0): Cap P&L values in alignment loss at ±X. Prevents fat-tail chasing.
- `REG_WIN_RATE` (default 0.0): Soft penalty when batch win rate < 45% target.

### What You MUST NOT Do
- Modify the `forward()` method signature of TradingModel
- Replace gate+direction decomposition with unified action head (v12 failure)
- Add `_env_float()` declarations outside of the allowed prefixes below

### New `_env_float` Rules
You MAY add up to 3 new `_env_float()` declarations per experiment, but ONLY with these prefixes:
- `SCHED_*` — loss weight scheduling parameters
- `WEIGHT_*` — sample weighting parameters
- `WARM_*` — warm start controls
- `REG_*` — regularization parameters

All new `_env_float` declarations MUST default to 0.0 (no-op when absent).

### What You MAY Do
- Tune existing `_env_float` values (ENTROPY_COEFF, DO_NOTHING_BONUS, DROPOUT, WEIGHT_DECAY, LR, FEATURE_NOISE_STD, etc.)
- Implement loss weight SCHEDULING (e.g., ramp entropy coeff over epochs)
- Add sample weighting within EV loss (harder examples, time-of-day weights, VIX regime weights)
- Change learning rate schedules (warmup, cosine, OneCycle, cyclical)
- Modify bias initialization for action head
- Adjust feature noise patterns (targeted noise on specific feature groups)
- Change batch construction strategy (DAY_SEQ_RATIO, hard example mining)
- Add gradient accumulation steps
- Freeze early transformer layers during warm start
- Add early stopping based on validation metrics (must save best checkpoint)

`worst_chunk_pf` and `chunk_details` are reported for analysis — the agent may use these to evaluate temporal consistency.

### Dynamic Stop Environment Variables
These control the per-trade stop-loss computed from gate confidence + market conditions:

| Variable | Default | Range | Effect |
|----------|---------|-------|--------|
| DYNAMIC_STOP_BASE | 0.35 | 0.20-0.50 | Base stop percentage before adjustments |

The dynamic stop formula: `stop = BASE * confidence_factor * iv_factor * vix_factor`, clamped to [15%, 60%].
- **confidence_factor**: `1.0 - (gate_confidence - 0.5) * 0.8` — tighter stops for high-confidence trades
- **iv_factor**: `1.0 + max(0, iv_feature) * 0.15` — wider stops in high-IV environments
- **vix_factor**: `1.0 + max(0, vix_feature) * 0.10` — wider stops in elevated-VIX regimes

Training uses stopped P&L labels (tight/med/wide selected by IV+VIX). The dynamic stop is the safety net; the gate head must learn to exit BEFORE hitting it.

## Position State Contract
The model receives a 7-dimensional position state tensor at inference time:
- `[0]` in_trade: 1.0 if in a trade, 0.0 if flat
- `[1]` bars_held: bars held / BARS_PER_DAY, clamped to [0, 1]
- `[2]` unrealized_pnl: tanh(unrealized_pnl * 5.0), clamped to [-1, 1]
- `[3]` account_health: account_balance / starting_capital (1.0 = full, 0.0 = wiped)
- `[4]` loss_streak: consecutive_losses / consec_loss_threshold, clamped to [0, 1]
- `[5]` best_pnl_since_entry: best unrealized P&L seen during current trade (0.0 if flat)
- `[6]` bars_since_pnl_high: bars since best_pnl was set / BARS_PER_DAY (0.0 if flat)

During training:
- **Random batches** (8% of training via `1 - DAY_SEQ_RATIO`) receive **flat position state**: not_holding, account_health=1.0. This matches the starting state in evaluate_trades().
- **Day-sequential batches** (92%) track real position state through each simulated trading day.
- PositionStateGenerator was **removed** — it injected random noise uncorrelated with market data, causing train/eval mismatch.

During evaluation/replay, dims 3-6 use real tracked account/trade state.

## Account-Aware Simulation
- Starting capital: $10,000 (STARTING_CAPITAL constant)
- Contract multiplier: $100 (SPX_MULTIPLIER constant)
- Affordability check: entries blocked if contract cost > account balance
- **Scaled position sizing**: `n_contracts = max(1, floor(balance * POSITION_RISK_TARGET / contract_cost))` where POSITION_RISK_TARGET = 0.05 (5%). Always whole contracts. Scales with account growth — at $10k with a $500 contract: 1 contract; at $30k: 3 contracts.
- Inline equity tracking: account balance updated at each trade exit (dollar_pnl *= n_contracts)
- Risk fraction penalty: penalizes models that consistently risk >30% of account per trade

## Output Metrics Contract (Required Keys)
The training script output must include these parseable metric keys:
- `score:`
- `profit_factor:`
- `trades_per_day:`
- `trade_sharpe:`
- `stop_loss_rate:`
- `worst_chunk_pf:`
- `rr_ratio:`
- `avg_hold_bars:`
- `model_exit_rate:`
- `hit_ruin:`
- `min_equity_frac:`
- `avg_risk_fraction:`
- `trades_blocked_by_balance:`

## Domain Knowledge (0DTE SPX Options)

### Theta Decay (Critical for 0DTE)
- Theta is NOT linear — follows 1/sqrt(T). Approximately doubles when remaining time quarters.
- At open (6.5h left): ~$2/hr. At 2pm (2h left): ~$5/hr. At 3:30pm (30min): ~$30+/hr.
- Long options bleed fastest in final 2 hours. Model should bias toward earlier exits for longs held past noon.

### Gamma
- ATM gamma: ~0.02-0.04 morning, spikes to 0.08-0.15 by 3pm (5-8x morning levels).
- High gamma = small SPX moves create large option P&L swings.
- Negative GEX days: momentum amplified (trend-following edge). Positive GEX: mean-reversion edge.

### Time-of-Day Patterns
- 9:30-9:35: Avoid (chaotic opening rotation). NO_TRADE_BEFORE_BAR=30 blocks first 30 minutes.
- 9:35-10:30: Strongest trends, best directional entries.
- 10:00-10:30: "Magic time" — session direction often decided here.
- 11:30-13:30: Lunch chop — low volume, erratic, avoid or go small.
- 13:30-15:30: Charm flows dominate. SPX gravitates toward high-OI strikes / max pain.
- 15:30-16:00: Power hour — extreme gamma, avoid unless high conviction.
- Top-of-hour: Systematic tendency for reversals at :00 marks.

### VIX Regimes
- VIX < 15: Tight ranges, slow moves. Favor premium selling, tight stops.
- VIX 15-20: Normal. Balanced strategies.
- VIX 20-30: Wide moves, rich premiums. Need wider stops, smaller size.
- VIX > 30: Crisis. Trade tiny or sit out.
- Regime TRANSITIONS (VIX 14→22) more dangerous than steady high VIX.

### Strike Selection
- ATM: Highest gamma, most responsive, most whipsaw risk.
- OTM 0.5-1%: Better Sharpe ratio empirically, lower gamma risk.
- OTM trades in backtest: -601% cumulative. Model should strongly prefer ATM unless high directional conviction.

### Professional Exit Patterns
- Exits matter more than entries. Best traders manage exits more actively.
- Time-based: If thesis hasn't played out by 2pm, take what's left.
- Trailing: If P&L drops >50% from peak, exit (momentum lost).
- Regime awareness: Tighter exits during power hour, wider during morning trends.

### Regime Risk and Temporal Robustness
- 0DTE dynamics change across VIX regimes. A model trained on elevated-VIX data (large moves, big OTM payoffs) may degrade when VIX normalizes (smaller moves, theta-dominated).
- The 70/30 temporal train/val split means the model trains on earlier data and validates on later data. If market character changes, validation performance degrades from early to late dates.
- `worst_chunk_pf` reports the minimum PF across 5 date chunks. Per-chunk details (PF, WR, date range) are printed in training output. Robust models should have consistent per-chunk PF.
- The score formula evaluates validation trades with equal weight. A model with PF=7 on chunk 1 and PF=0.5 on chunk 5 has a regime problem even if aggregate score is high.

## Reliability Policy
A candidate can be kept only if:
- Score improves over current best, and
- No contract violation, and
- No critical anomaly flags.

Near-tie improvements require stability confirmation from guard-band metrics.

## Autonomy Rules
- Modify `train.py` only.
- Preserve parseable output format.
- Avoid broad rewrites unless strongly justified by experiment evidence.
- Prefer small, testable deltas that can be reverted cleanly.
