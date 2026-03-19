# Autoresearch Trading Contract (Foundation Phase)

This file is the strict operating contract for autonomous loop experiments.
If anything else conflicts with this file, this file wins.

## Mission
Improve the training objective in `train.py` for SPX 0DTE option trading while preserving mechanical reliability.

## Scope Lock
- Foundation phase is locked to **70 features** (60 core + 10 extended: charm, vanna, Bollinger, momentum, VWAP crosses, RSI, ATR ratio).
- Model contract is locked to **two-head** outputs.
- Active trading semantics are locked to 6-direction entries + contextual exit.

## Model Contract (Required)
- Two-head architecture:
  - Gate head: `[NO_TRADE, TRADE]`
  - Direction head: `[CALL_ATM, CALL_OTM5, CALL_OTM10, PUT_ATM, PUT_OTM5, PUT_OTM10]`
- 8 effective actions:
  - `DO_NOTHING`
  - `BUY_CALL_ATM`, `BUY_CALL_OTM5`, `BUY_CALL_OTM10`
  - `BUY_PUT_ATM`, `BUY_PUT_OTM5`, `BUY_PUT_OTM10`
  - `EXIT` (gate=NO_TRADE while holding a position)

**Exit mechanics**: There is NO hardcoded profit target. The model's gate head is the PRIMARY exit mechanism — it must learn when to take profits and cut losses. A 30% stop loss exists as an emergency backstop only. Exits are: stop_loss (30%), model_exit (gate=NO_TRADE), end_of_day, max_hold.

## Data Contract (Required)
`data.pt` must include the two-head/OTM target fields and option price arrays required by training and replay:
- Targets:
  - `call_pnl`, `put_pnl`
  - `exit_call_label`, `exit_put_label` (hindsight-optimal exit timing signals)
  - `otm5_call_pnl`, `otm5_put_pnl`
  - `otm10_call_pnl`, `otm10_put_pnl`
- Prices:
  - `atm_call_prices`, `atm_put_prices`
  - `otm5_call_prices`, `otm5_put_prices`
  - `otm10_call_prices`, `otm10_put_prices`

Missing required fields must fail fast. No silent fallback behavior.

## Architecture Lock
Do not change tensor-shape-defining architecture values during this phase:
- `D_MODEL`
- `DEPTH`
- `N_HEADS`

Reason: warm-start compatibility with `best_model.pt` must be preserved.

## Safety + Runtime Constraints
- No `torch.compile`.
- No `DataParallel` / `DistributedDataParallel`.
- No `torch.jit.trace` / `torch.jit.script`.
- Keep memory/runtime within the configured budget.
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

### Score Tuning Environment Variables
These control the score formula and can be set per experiment:

| Variable | Default | Range | Effect |
|----------|---------|-------|--------|
| SCORE_WIN_RATE_BONUS | 0.0 | 0.0-1.0 | Multiplier bonus for win rates above 40% |
| SCORE_RR_BONUS | 0.0 | 0.0-2.0 | Multiplier bonus for avg_win/avg_loss > 1 |
| SCORE_DRAWDOWN_PENALTY | 0.0 | 0.0-1.0 | Multiplier penalty for deep drawdowns |
| SCORE_HOLD_BONUS | 0.0 | 0.0-1.0 | Multiplier bonus for hold times near 30 bars |
| SCORE_FREQ_CENTER | 3.0 | 1.0-8.0 | Center of trade frequency sweet spot |
| SCORE_FREQ_WIDTH | 3.0 | 1.0-6.0 | Width of frequency sweet spot band |
| SCORE_CONSEC_LOSS_THRESHOLD | 3 | 2-8 | Consecutive losses before penalty kicks in |
| SCORE_SHORT_HOLD_THRESHOLD | 0.30 | 0.10-0.60 | Short hold % that triggers penalty |
| SCORE_STOP_RATE_THRESHOLD | 0.30 | 0.10-0.60 | Stop loss rate that triggers penalty |

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

## Domain Knowledge (0DTE SPX Options)

### Exit Mechanics
- There is NO hardcoded profit target. The model must learn when to exit.
- The gate head predicting NO_TRADE while holding a position triggers an exit.
- 30% stop loss is an emergency backstop — the model should learn to cut losers BEFORE hitting the stop.
- EXIT labels in training data use hindsight-optimal timing: EXIT=1 at the bar nearest to peak P&L, or when P&L has dropped >50% from its high-water mark.
- The edge comes from exit TIMING, not just entry selection.

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
