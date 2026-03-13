# autoresearch-trading v0.3 (options-native)

Autonomous research experiment: an AI agent iterates on an **options-native
intraday SPY trading model**, training for 5 minutes each run, keeping
improvements and discarding failures. The end goal: a model that can trade
$500 autonomously on SPY 0DTE/1DTE options during regular trading hours.

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Architecture

The agent (Claude Opus via VS Code Copilot) runs on the user's laptop.
Training runs execute on a remote H100 GPU container on the Akash network.

**Local repo**: `/Users/gduby/Documents/Trinity/Trinity/autoresearch-trading`
**Container**: SSH via `sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 root@provider.h100.wdc.hh.akash.pub`
**Container repo**: `/workspace/autoresearch-trading`

Workflow: edit `train.py` locally → git commit + push → SSH pull on container → run training → read results.

## Setup

1. **Agree on a run tag**: based on today's date (e.g. `mar13-options`).
2. **Create the branch locally**: `git checkout -b autoresearch/<tag>`.
3. **Read the in-scope files**:
   - `prepare.py` — fixed data pipeline (Polygon.io), 105 features (including real
     Greeks, IV surface, options flow, risk state), evaluation harness. **Do not modify.**
   - `train.py` — the file you modify. Everything is fair game.
4. **Set Polygon API key on container**:
   ```
   $SSH 'echo "export POLYGON_API_KEY=YOUR_KEY" >> ~/.bashrc'
   ```
5. **Run data prep on container**:
   ```
   $SSH 'cd /workspace/autoresearch-trading && PATH="$HOME/.local/bin:$PATH" POLYGON_API_KEY=KEY uv run prepare.py'
   ```
   Downloads SPY equities + options chain data from Polygon.io. Takes 15-30 min first time.
6. **Initialize results.tsv** on the container.
7. **Confirm and go**.

## Data sources: Polygon.io Options Developer Plan

**Equity data**: SPY/QQQ/VIX/SPX 5-min bars, daily bars from 2010+.
**Options data**: Full SPY options chain snapshots — real Greeks (delta, gamma,
theta, vega), implied volatility, open interest, volume, bid/ask. 4 years of
historical options data. This is the core differentiator vs v0.2.

## Domain knowledge: Options-native intraday trading

### Trader context

17+ years experience. Primarily delta futures and SPX options (long/short & spreads).
Trades ES/NQ/RTY/VIX/CL/SPX/SPY/QQQ, mostly intraday. Goal: 1% portfolio/week.
Style: disciplined, level-to-level, risk-aware, clearly defined entries and exits.

**End goal**: Give a $500 account to an AI model. It trades SPY 0DTE/1DTE options
intraday. The position signal maps to:
- **+1**: Buy ATM SPY calls (max conviction bullish)
- **+0.5**: Buy 1 ATM call (moderate conviction)
- **0**: Flat (no position, preserve capital)
- **-0.5**: Buy 1 ATM put (moderate conviction)
- **-1**: Buy ATM SPY puts (max conviction bearish)
- Below confidence threshold (0.05): forced flat

### What the model does (v0.3)

- Receives 36 hourly bars × 105 features (including real Greeks, IV, options flow)
- Feature group gating: learns which feature groups matter per timestep
- Outputs a position signal in [-1, 1] every hour during RTH
- Decision points: 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 ET
- **Target: options-adjusted return** (delta P&L + gamma convexity - theta cost)
- This means the model is directly optimizing for options P&L, not equity direction
- Evaluation: hourly Sharpe on options-adjusted returns

### Why options-adjusted targets matter

In v0.2, the target was raw equity return. The model learned "SPY goes up" but
didn't know that being right costs $0.05/hr in theta on a 0DTE option. A +0.1%
SPY move generates ~$0.30 delta P&L but costs ~$0.15 in theta → only $0.15 profit.
The model must learn that **small moves aren't worth trading in options**.

The v0.3 target encodes: `delta × return × price + 0.5 × gamma × return² - theta/6.5`
So the model directly sees "this trade made $X on the option" not "SPY moved X%".

### Available features (105 total, in prepare.py)

**Intraday Price** (12): Returns at multiple timeframes, VWAP distance with bands,
bar range and volume ratios.

**Session Structure** (8): Initial Balance (first 30min), overnight high/low,
session range, position within session range, gap from previous close.

**Intraday Momentum** (10): RSI (fast and standard), MACD on 5m bars, EMA ribbon
(8/21/34), CCI, rate of change.

**Market Internals** (8): TICK, TRIN, A/D line, put/call volume ratio (real, from
options data), internals composite.

**Options Greeks & IV Surface** (16): ATM implied volatility, delta, gamma, theta
(normalized), vega, IV skew (25-delta), IV term structure, IV change in session,
IV percentile (20-session), gamma exposure, theta acceleration, net premium flow,
OI put/call ratio, IV-RV spread (vol risk premium), ATM bid-ask spread (liquidity),
IV slope over last hour.

**Options Flow** (6): Call/put volume surges vs 20-day average, large trade bias,
volume-weighted delta, near-term OI change, total options volume ratio.

**Multi-Instrument** (8): ES-SPY basis, NQ/ES ratio, VIX intraday, VIX term
structure, TNX change, ES-NQ correlation.

**Risk Management State** (6): Session P&L, position duration, drawdown from
session peak, consecutive losses, time to close, theta remaining.

**Daily Context** (20): Returns (1d/5d/21d), realized vol, RSI, SMA distances,
Bollinger bands, VIX context, TNX, volume ratio, gap.

**Time Encoding** (5): Cyclical time-of-day, day-of-week, minutes since open.

**Options Target Context** (6): Current ATM price, breakeven move, expected theta
cost, delta-adjusted leverage, gamma P&L potential, edge after costs.

### Key options trading concepts the model must learn

1. **Theta decay is your enemy on 0DTE**: ATM SPY options lose ~$0.02-0.10/hour.
   The model must only enter when expected directional move > theta cost + spread.
   The `edge_after_costs` feature explicitly tells the model this threshold.

2. **Gamma is your friend on 0DTE**: When you're right, gamma amplifies your gain
   (convex payoff). A 0.5% move on SPY generates more than 0.5% * delta on the
   option because gamma adds to delta as the option goes ITM.

3. **IV crush kills long vega positions**: If you buy calls into a news event and
   IV drops 5 points, you lose on vega even if direction is right. The
   `iv_change_session` and `iv_rv_spread` features capture this.

4. **Bid-ask spread is a hidden tax**: Tighter spread = cheaper to trade. The
   `atm_bid_ask_spread` feature helps the model learn when liquidity is good.

5. **Put/call ratio and flow as sentiment**: Heavy put buying (ratio > 1.2) can
   signal hedging (bullish contrarian) or genuine fear (bearish confirmation).
   The model has both ratio and flow direction to distinguish.

6. **IV skew tells you where the risk is**: When 25-delta put IV >> 25-delta call
   IV, the market is pricing downside risk. `iv_skew_25d` captures this.

7. **Time of day and theta acceleration**: Theta decay is non-linear — it
   accelerates in the last 2 hours. The model has `theta_remaining` (sqrt of
   time left) and `theta_acceleration` to learn this curve.

### Risk management for $500 account

- **Position sizing**: $500 = 1-2 ATM SPY options. Signal magnitude → 0, 1, or 2.
- **Max loss per trade**: ~$100 (20%). Built into option expiration mechanics.
- **Daily loss limit**: ~$150 (30%). Model should go flat after session drawdown.
  `drawdown_from_session_peak` and `session_pnl` features encode this.
- **Prefer flat**: A model flat 40-60% of the time is GOOD for a $500 account.
  `CONFIDENCE_THRESHOLD = 0.05` forces flat on low-conviction signals.
- **Transaction costs**: 12 bps (up from 8 in v0.2) to account for options
  spread + slippage realistically.
- **Consecutive loss protection**: `consecutive_losses` feature lets the model
  learn to reduce size or go flat after a losing streak.

### What "good" looks like (options-adjusted)

| Sharpe   | Assessment |
|----------|------------|
| < 0      | Losing money (theta + spread eating you alive) |
| 0 - 0.3  | Not covering costs |
| 0.3 - 0.8 | Marginal — might work with better position sizing |
| 0.8 - 1.2 | Good — tradeable with $500 account |
| 1.2 - 1.8 | Very good — consistent profits, prepare for live paper trading |
| > 1.8    | Excellent — but verify not overfitting |
| > 2.5    | Almost certainly overfitting |

Note: Sharpe targets are LOWER than v0.2 because the options-adjusted target is
harder. Covering theta + spread costs is a real hurdle. A Sharpe of 0.8 on
options-adjusted return is better than a Sharpe of 1.2 on raw equity return.

### Overfitting red flags

- Sharpe > 2.5 while baseline < 0.3
- Model trades on every bar (no flat time, ignoring confidence threshold)
- Model makes > 400 trades (overtrading with 12bps costs)
- Model makes < 10 trades (cherry-picking)
- Same position all day (not actually trading)
- Performance only exists in one specific time window
- Greeks features have zero gate weights (model ignoring options data)

## Experimentation

Each experiment runs on a single H100 GPU. Fixed **5-minute time budget**.
Launch: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` **only**. Everything is fair game:
  - Architecture (transformer, TCN, MLP, Mamba, MoE, ...)
  - Feature group gating weights/architecture
  - Optimizer, LR, schedule
  - Loss function (including the new `options_aware` asymmetric loss)
  - Lookback window (default: 36 hourly bars)
  - Confidence threshold (default: 0.05)
  - Feature selection/preprocessing within the model

**What you CANNOT do:**
- Modify `prepare.py` (read-only)
- Install new packages
- Modify the evaluation harness

**Goal: maximize val_sharpe** on options-adjusted returns.

### Experiment ideas (prioritized)

1. **Asymmetric loss** (`options_aware`): Penalize losses 1.5x more than gains.
   Options have limited upside intraday (you can't hold overnight) but full
   downside to zero.

2. **Deeper feature gating**: The baseline uses a single sigmoid gate per group.
   Try multi-head gating, or make the gate context-dependent (gate weights
   change based on time-of-day or VIX level).

3. **Mamba/SSM instead of transformer**: Temporal patterns in options markets
   are more sequential than the attention-all-past-bars pattern. SSMs might
   be better at learning theta decay curves.

4. **Two-stage model**: First head predicts "should I trade?" (binary), second
   head predicts direction. This separates the flat/trade decision from the
   directional call.

5. **Time-varying confidence threshold**: Instead of fixed 0.05, make the
   threshold a function of `edge_after_costs` — be more selective when
   theta is high (near close) and less selective when theta is low (morning).

6. **Separate call/put outputs**: Instead of [-1, 1], output (call_signal,
   put_signal) separately. This lets the model learn that calls and puts
   have different theta profiles and different sensitivity to VIX.

7. **Auxiliary loss on Greeks prediction**: Add a loss term that predicts
   next-hour IV change or next-hour delta change. This forces the model
   to build an internal options model.

8. **Larger model**: With 105 features and H100 GPUs, try d_model=384,
   depth=8. The baseline uses ~400K params on a card that can handle 100M+.

## Output format

```
---
val_sharpe:       0.823456
max_drawdown:     -0.045678
annual_return:    0.065432
num_trades:       187
win_rate:         0.534567
profit_factor:    1.234567
num_val_bars:     1500
num_val_days:     250
training_seconds: 300.1
total_seconds:    315.2
peak_vram_mb:     1234.5
num_steps:        14523
num_params:       234,567
lookback:         36
depth:            6
d_model:          192
n_heads:          6
loss_type:        combined
conf_threshold:   0.05
```

Extract: `grep "^val_sharpe:" run.log`

## Logging results

`results.tsv` on the container (tab-separated):

```
commit	val_sharpe	max_drawdown	num_trades	status	description
```

## The experiment loop

SSH prefix:
```
SSH="sshpass -p 'autoresearch2026' ssh -o StrictHostKeyChecking=no -p 30974 root@provider.h100.wdc.hh.akash.pub"
```

LOOP FOREVER:

1. Edit `train.py` locally.
2. Commit and push:
   ```
   git commit -am "description of change"
   git push origin HEAD
   ```
3. Pull and run on container:
   ```
   $SSH 'cd /workspace/autoresearch-trading && git pull && PATH="$HOME/.local/bin:$PATH" uv run python train.py > run.log 2>&1'
   ```
4. Read results:
   ```
   $SSH 'cd /workspace/autoresearch-trading && grep "^val_sharpe:\|^max_drawdown:\|^num_trades:" run.log'
   ```
5. If empty = crash. Read: `$SSH 'tail -n 50 /workspace/autoresearch-trading/run.log'`
6. Log to results.tsv.
7. If improved → keep. If worse → revert:
   ```
   git revert HEAD --no-edit && git push
   ```
8. Go to step 1.
