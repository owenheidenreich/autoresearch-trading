# Autoresearch Trading Contract (v2 Phase)

This file is the strict operating contract for autonomous loop experiments.
If anything else conflicts with this file, this file wins.

## Mission
Build a model that makes money trading SPX 0DTE options on IBKR paper trading. The score is a proxy — focus on improving actual TRADING BEHAVIOR (profit factor, win rate, regime consistency, drawdown) rather than optimizing the score metric itself. Use the trade-level diagnostics (best/worst trades, time-of-day splits, VIX regime breakdowns) to diagnose specific weaknesses and propose targeted fixes.

## Model Contract (Required)
- Three-head architecture (v5):
  - Gate head: `[NO_TRADE, TRADE]` (2 logits)
  - Direction head: `[CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]` (6 logits)
  - Value head: scalar prediction of remaining P&L (MSE regression)
- 8 effective actions:
  - `DO_NOTHING`
  - `BUY_CALL_ATM`, `BUY_CALL_OTM5`, `BUY_CALL_OTM10`
  - `BUY_PUT_ATM`, `BUY_PUT_OTM5`, `BUY_PUT_OTM10`
  - `EXIT` (gate=NO_TRADE while holding a position)

**Exit mechanics**: There is NO hardcoded profit target. The model's gate head is the PRIMARY exit mechanism — it must learn when to take profits and cut losses. The **value head** provides a secondary exit signal: when predicted remaining P&L drops below `VALUE_EXIT_THRESHOLD`, it triggers a value exit (requires ≥2 bars held). A **dynamic stop-loss** adapts per-trade based on gate confidence + market conditions (IV, VIX). Exit priority: stop_loss > model_exit > value_exit > max_hold > end_of_day.

## Data Contract (Required)
`data.pt` must include the target fields and option price arrays required by training and replay:
- Targets (unstopped P&L — entry to EOD, no stops; kept for reference only):
  - `call_pnl`, `put_pnl`
  - `exit_call_label`, `exit_put_label` (hindsight-optimal exit timing signals)
  - `otm5_call_pnl`, `otm5_put_pnl`
  - `otm10_call_pnl`, `otm10_put_pnl`
- Targets (stopped P&L — with DYNAMIC_STOP_BASE applied; **used by sniper_loss**):
  - `call_stopped_pnl`, `put_stopped_pnl` (med-level, at DYNAMIC_STOP_BASE=0.35)
  - `otm5_call_stopped_pnl`, `otm5_put_stopped_pnl`
  - `otm10_call_stopped_pnl`, `otm10_put_stopped_pnl`
- Multi-level stopped P&L (for ATM strikes, selected by IV+VIX per bar):
  - `call_stopped_pnl_tight`, `put_stopped_pnl_tight` (stop=0.20)
  - `call_stopped_pnl_wide`, `put_stopped_pnl_wide` (stop=0.50)

**CRITICAL**: `sniper_loss` selects tight/med/wide stopped P&L per bar based on market conditions (IV, VIX) to align training with the dynamic stop-loss used in replay/live. Do NOT add raw P&L as an additional loss signal — this creates conflicting gradients and was tried 9 times without success.
- Prices:
  - `atm_call_prices`, `atm_put_prices`
  - `otm5_call_prices`, `otm5_put_prices`
  - `otm10_call_prices`, `otm10_put_prices`
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

**What to do instead:** Improve the model's actual TRADING BEHAVIOR by modifying loss weights (GATE_W, DIR_W, PNL_W, EXIT_W), model architecture biases, or training dynamics. Improvements should be visible in raw metrics: profit factor, win rate, trades per day, drawdown.

### Loss Function Policy
The loss function STRUCTURE is locked — `total = gate + dir + pnl` must not change. Exit behavior is integrated into gate targets (exit-labeled bars override gate targets to NO_TRADE), not a separate loss term.

### What You MUST NOT Do
- Modify the `forward()` method signature of TradingModel
- Add new loss terms to the `total = gate + dir + pnl` summation
- Add `_env_float()` declarations outside of the allowed prefixes below

### New `_env_float` Rules
You MAY add up to 3 new `_env_float()` declarations per experiment, but ONLY with these prefixes:
- `SCHED_*` — loss weight scheduling parameters (e.g., SCHED_GATE_RAMP_EPOCHS)
- `WEIGHT_*` — sample weighting parameters (e.g., WEIGHT_HARD_EXAMPLE_RATIO)
- `WARM_*` — warm start controls (e.g., WARM_FREEZE_EPOCHS, WARM_LR_MULT)
- `REG_*` — regularization parameters (e.g., REG_L1_LAMBDA, REG_GRAD_PENALTY)

All new `_env_float` declarations MUST default to 0.0 (no-op when absent). This ensures baseline behavior is preserved if the experiment is reverted.

### Regularization Exception
You MAY add ONE regularization penalty term to the training loop (not to `total_loss` directly, but as a separate `optimizer` gradient source or weight penalty), controlled by a `REG_*` _env_float defaulting to 0.0. Examples:
- L1 sparsity on gate logits
- Gradient penalty for temporal smoothness
- Weight decay scheduling (varying across layers)

This does NOT change the loss structure (`total = gate + dir + pnl`). The regularization is applied separately.

### What You MAY Do
- Tune existing `_env_float` values (DIR_W, GATE_W, PNL_W, EXIT_W, DROPOUT, WEIGHT_DECAY, LR, FEATURE_NOISE_STD, etc.)
- Implement loss weight SCHEDULING (e.g., ramp gate_w from 0.5→1.0 over epochs using existing env values)
- Add sample weighting within existing loss computations (harder examples, time-of-day weights, VIX regime weights)
- Change learning rate schedules (warmup, cosine, OneCycle, cyclical — use existing LR value as base)
- Modify bias initialization for gate and direction heads
- Adjust feature noise patterns (targeted noise on specific feature groups)
- Change batch construction strategy (DAY_SEQ_RATIO, hard example mining)
- Add gradient accumulation steps
- Freeze early transformer layers during warm start (first N epochs)
- Use a lower learning rate multiplier for warm-started weights vs new/reset parameters
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
The model receives a 5-dimensional position state tensor at inference time:
- `[0]` is_holding: 1.0 if in a trade, 0.0 if flat
- `[1]` bars_held_norm: bars held / BARS_PER_DAY, clamped to [0, 1]
- `[2]` unrealized_pnl_norm: tanh(unrealized_pnl * 5.0), clamped to [-1, 1]
- `[3]` account_health: account_balance / starting_capital (1.0 = full, 0.0 = wiped)
- `[4]` loss_streak_frac: consecutive_losses / consec_loss_threshold, clamped to [0, 1]

During training:
- **Random batches** (30% of training via `1 - DAY_SEQ_RATIO`) receive **flat position state**: not_holding, account_health=1.0. This matches the starting state in evaluate_trades().
- **Day-sequential batches** (70%) track real position state through each simulated trading day.
- PositionStateGenerator was **removed** — it injected random noise uncorrelated with market data, causing train/eval mismatch.

During evaluation/replay, dims 3-4 use real tracked account state.

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
