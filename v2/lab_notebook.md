# v2 Lab Notebook

## Data Audit (2026-04-03)

Full pipeline audit revealed 5 critical problems. See `v2/docs/data_audit_findings.md`.

Key findings:
- Oracle labels had 100% win rate (zero losers in training). PF=388 was an artifact.
- Moneyness drift: 84% of bars had >5pt drift from opening ATM. 58% of "OTM5 call" were actually ITM.
- Narrow grid: only 14 of 209 available strikes downloaded from Polygon.
- Spreads estimated incorrectly (old lookup table off by 4-20x).
- Simple momentum signals had no edge (PF=0.49-0.54).

## Signal Scan (2026-04-03)

Tested all 39 original features as direction signals against always-put baseline (PF=1.106).

Edge found in volatility-regime features:
- `option_spread_width`: PF=1.855 (+0.748 vs baseline)
- `session_range_pct`: PF=1.435 (+0.328)
- `rsi_7`: PF=1.219, `bollinger_position`: PF=1.216
- `atm_iv`: PF=1.182, `atm_gamma`: PF=1.176

**Pattern:** High vol/range/gamma = calls win (gamma convexity). Low vol = puts win (theta decay). Multiple features confirm independently.

Momentum (ret_6) has NO edge (PF=0.81).

## Data Rebuild (2026-04-03)

### Wide-grid download
- ATM +/- 100pt (82 contracts per day) from Polygon flat files
- 994 days, 2.2 GB, avg 41 strikes per day
- After 50pt intraday move, still 50pt OTM coverage

### Enriched features (16 new, total 71)
- `current_moneyness_pct`, `intraday_drift_pct`, `near_atm_moneyness_pct`
- `near_atm_call_volume`, `near_atm_put_volume`, `call_put_flow_ratio`
- `log_total_volume`, `chain_call_put_ratio`, `log_chain_volume`
- `volume_zero_flag`, `call_hl_range_pct`, `near_atm_call/put_price_norm`
- `theta_acceleration`, `near_atm_transactions`

All features are RELATIVE (moneyness %, normalized prices) so patterns at SPX 4300 transfer to SPX 6500.

### NaN handling
- 27% of bars had NaN (0DTE options stop trading near close)
- Fix: forward-fill within day (matches live IBKR behavior)
- Remaining NaN (start-of-day): filled with 0

### Cost model
- Commission: $1.30 round trip
- Bid-ask spread: adaptive by time-of-day and VIX regime
- Total cost per trade: varies, computed per-bar

## Phase 1: Classification Model (exp_001 -- exp_013, 2026-04-03)

Old approach: model classifies gate (trade/no-trade) and direction (call/put) as separate heads. Labels from pre-computed direction vote + oracle grid search.

**Problems identified:**
- Gate head stuck at majority-class baseline (79.8% accuracy = always predict True)
- Direction collapse to 100% puts during promote period
- CALL_BOOST inference hack needed to get any calls

**Best score: 5.59 (exp_004)** -- but achieved by tuning stop/target ranges, not by the model learning.

## Phase 2: Regime Conditioning (exp_014 -- exp_029, 2026-04-04)

### FiLM Regime Conditioning (exp_014) -- BREAKTHROUGH 1

Added a RegimeEncoder (55->32->16 MLP) that reads the last bar's raw features and produces a regime embedding. Each prediction head gets a FiLM layer (gamma * x + beta) that modulates the transformer output based on regime.

**Result:** Direction collapse solved without CALL_BOOST. 26% calls naturally. Score 5.59 -> 5.69.

**Why it worked:** The model can now produce different outputs for different volatility regimes. High vol -> calls, low vol -> puts, learned from data instead of hardcoded.

### Label Smoothing (exp_016) -- Score 5.80

Direction label smoothing 0.1 -> 0.25 reduced direction memorization. +day_rate jumped to 96.7% (only 2 losing days out of 60). Score hit 5.80.

### Score Ceiling Analysis (exp_017 -- exp_021)

7 consecutive reverts. Analysis showed:
- Score = min(sortino, 6.0) * positive_day_rate * dd_mult = 6.0 * 58/60 * 1.0 = 5.80
- The 2 losing days are the binding constraint
- Trade-level analysis: trades with 47-73% MFE giving it all back to stop loss
- Any change that reduces trade count kills calls first (model less confident about calls)

### Trailing Stops (exp_022) -- BREAKTHROUGH 2, PERFECT SCORE

Changed exit_policy from STOP_TP_TIME to TRAILING. The simulator's trailing stop tiers lock in profits at +30/+50/+80/+120% unrealized.

**Result:** Score 6.0 (perfect). All 60 promote days profitable. 0% drawdown.

**Why it worked:** The 2 losing days had trades that reached 47-73% profit then reversed to stop loss. Trailing stops locked in those profits.

### Focal Loss + MicroMoE (exp_023, exp_029)

- Focal loss for gate: down-weights easy examples, focuses on hard boundary cases. Achieved near-perfect 50/50 call/put direction balance.
- MicroMoE direction head: 2 expert MLPs routed by regime embedding. Best shadow generalization (PF=64 on shadow vs PF=52 without).

**Best classification model: exp_029** -- Score 6.0 on both promote and shadow. WR 80.1% promote, 72.2% shadow. PF 333 promote, 64 shadow.

## Phase 3: The Honesty Reckoning (2026-04-05)

### What the "perfect score" model was actually doing

The score was 6.0 but the model was NOT trading. It was:
- Firing 40+ trades per day (spray-and-pray)
- Gate head at 84% accuracy = majority class prediction, not selectivity
- Direction from pre-computed median split of volatility features, not learned
- Strike always ATM (never learned strike selection)
- Risk params curve-fit to oracle grid-searched values
- Trailing stops (hardcoded in simulator) doing all risk management

The 5x PF gap between promote (333) and shadow (64) confirmed overfitting.

### The Fundamental Redesign (exp_031)

**Old approach (classification):**
- 5 heads: gate, direction, strike, risk, confidence
- Gate: "should I trade?" (binary classification against oracle label)
- Direction: "call or put?" (classification against pre-computed vote)
- The model learned to predict labels, not to trade

**New approach (P&L prediction):**
- 2 regression heads: call_pnl_head, put_pnl_head
- Model predicts: "if I buy a call here, what P&L do I expect? And a put?"
- Trading decisions DERIVED from predictions at inference:
  - gate = max(pred_call, pred_put) > threshold
  - direction = argmax(pred_call, pred_put)
  - confidence = |pred_call - pred_put|

**Dataset changes:**
- For every eligible bar, simulate BOTH call AND put with fixed risk params (20% stop, 50% target, 30 bar hold)
- Store label_call_pnl and label_put_pnl (both outcomes)
- Direction = whichever had higher P&L
- Gate = True when best direction is profitable
- No pre-computed direction vote. The model learns direction from features.

**Training changes:**
- Loss = Huber regression on both call_pnl and put_pnl predictions (delta=0.3)
- No gate classification loss (gate is implicit from P&L prediction)
- No direction classification loss (direction is implicit from P&L comparison)
- FiLM regime conditioning retained
- Focal loss removed (not needed for regression)

### First Result: exp_031

Training metrics:
- gate_acc: 56.3% (model is selective -- not majority class)
- dir_acc: 62.8% (learning direction from features alone, above 50% random)
- val_loss: 0.082 (converging)

Replay:
- Score: 5.90 (1 losing day)
- PF: 314, WR: 80.1%
- Net P&L: $3.42M (highest ever)
- Direction: C=1288 P=1628 (44% calls, natural balance)
- Trades/day: 48.6

**Key difference from old model:** The gate_acc of 56% means the model is correctly identifying profitable setups more than half the time on a stochastic process. The old model had 84% "accuracy" but was just predicting the majority class. 56% on actual prediction is more meaningful than 84% on a lookup table.

## Architecture Summary (Current)

```
Input: (batch, 60, 71) -- 60 bars of 71 features (55 original + 16 enriched)

Backbone:
  Linear(71 -> 64) -> LayerNorm -> PositionalEncoding
  TransformerEncoder(3 layers, 4 heads, dim_ff=256, causal mask)
  -> last token: (batch, 64)

RegimeEncoder:
  Linear(71 -> 32) -> GELU -> Linear(32 -> 16)
  Reads last bar's raw features -> regime embedding

FiLM Conditioning:
  Per-head: regime -> (gamma, beta) -> modulate transformer output

Prediction Heads:
  call_pnl_head: FiLM(regime, last) -> Linear(64->32) -> GELU -> Linear(32->1)
  put_pnl_head:  FiLM(regime, last) -> Linear(64->32) -> GELU -> Linear(32->1)
  risk_head:     FiLM(regime, last) -> Linear(64->32) -> GELU -> Linear(32->3)

Inference:
  gate = max(call_pnl, put_pnl) * 5.0 > threshold
  direction = argmax(call_pnl, put_pnl)
  strike = always ATM
  risk = clamped to policy ranges (no sigmoid)
```

Parameters: ~170K. Trains in ~300s on H100 (19-24 epochs).

## Phase 4: Full Audit + Honest Baseline (2026-04-06)

### 8-Section Audit

Complete audit of all v2 code found 34 issues (3 critical, 6 high, 10 medium, 16 low).

**CRITICAL bugs invalidating all prior scores:**
1. Negative stop_pct creating phantom +3,734% avg STOP_LOSS profits (simulator.py:79)
2. Position overlap: in_position immediately set False (simulator.py:270, replay.py:299)
3. Random baseline score always 0.0 (replay.py:494-500)

**The score of 5.90 was fake.** Every STOP_LOSS "win" was a phantom fill at an imaginary price. Positions overlapped freely. The random baseline was trivially beaten.

### Three-Phase Fix

**Phase 1 (harness):** 11 fixes to simulator, replay, baselines. Added min entry price ($0.50), spread tick floor, TRAILING baseline, MIN_HOLD_BARS enforcement, daily loss cap enforcement.

**Phase 2 (labels/dataset):** Trailing stops in labeler. Commission $1.30 RT. Gate selectivity raised (best_pnl > 2%). Tick-floor spread. Duplicate feature append bug fixed. Dataset rebuilt.

**Phase 3 (model/training):** Risk head sigmoid removed (was producing 108% targets). Gate 5x scaling removed. gate_threshold raised to 0.5. Per-bar risk labels. Conditional metrics (dir_acc_gated, avg_pnl_gated). Dollar-weighted WR/PF. CVaR dead code removed. ATM hardcoded bias removed.

### First Honest Retrain: exp_001 (2026-04-06)

| Metric | Value |
|--------|-------|
| Score | 2.26 |
| Profit Factor | 1.66 |
| Win Rate | 52.4% |
| Sortino | 21.23 |
| Positive Day Rate | 71.7% |
| Max Drawdown | 13.7% |
| Net P&L | +$58,484 on $10K |
| Trades | 433 (7.2/day) |
| Direction | 239C / 194P (55/45%) |
| Beats Random | YES |
| Beats ATM-Always | YES |
| Beats Simple-Rules | YES |
| Beats ATM-Trailing | YES |

Training metrics:
- gate_acc: 58.9% (model is selective, not majority class)
- dir_acc: 64.6% (above 50% random)
- dir_acc_gated: 69.7% (gate selects bars where direction is more predictable)
- avg_pnl_gated: +0.095 vs avg_pnl_ungated: -0.030 (gate adds real value)

**This is the first honest score.** All prior scores were contaminated by phantom profits, overlapping positions, and broken baselines. The autoresearch loop can now improve from a truthful starting point.

## Session 3: Score Bottleneck Analysis (2026-04-06)

Score = min(sortino, 6.0) * positive_day_rate * dd_mult = 6.0 * (35/43) * 1.0 = 4.884. Sortino maxed at 6.0. Binding constraint: 8 losing days out of 43 traded (81.4% PDR).

Worst losing days: Dec 16 (-$1016, 6 puts on rally), Jan 6 (-$1079, 3 calls MAX_HOLD), Feb 11 (-$1287, 34.7% put loss at MAX_HOLD). Borderline: Dec 18 (-$70), Jan 2 (-$22), Jan 5 (-$111).

Exit analysis: STOP_LOSS 0% WR (avg -$372). MAX_HOLD 69.5% WR but 25-35% losses on losers. TAKE_PROFIT 100% WR avg +$1123.

### exp_018: gate_threshold 0.45 -> 0.50 -- KEEP (4.933)

PDR 82.2% (37/45). 2 more traded days, same 8 losing. Marginal improvement.

### exp_019: stop_range (0.10, 0.40) -> (0.10, 0.25)

**Hypothesis:** Worst losses are 33.5%, 34.7%, 31.3% at MAX_HOLD/STOP_LOSS. Max stop at 40% is too loose. Capping at 25% should reduce catastrophic single-trade losses.

**Result:** Score 4.468 (was 4.933). PDR dropped to 74.5% (12 losing days). Tighter stops = more stop-outs = more losing trades. REVERT.

### exp_020: cooldown_bars 5 -> 10

**Hypothesis:** Dec 16 had 6 puts in rapid succession, all lost. Jan 6 had 3 calls stacking losses. Wider cooldown prevents re-entering quickly after a loss, reducing loss stacking on worst days.

**Result:** Score 4.636 (was 4.933). PDR 77.3%. Fewer trades, same losing days. REVERT.

### exp_021: asymmetric loss 2x -> 3x -- KEEP (5.053)

Score 5.053 (was 4.933). PDR 84.2% (16/19 winning, 3 losing). Very selective: 44 trades in 19 traded days. Model barely above minimum thresholds (30 trades, 15 days). The 3x asymmetric penalty makes the model extremely conservative -- it only trades when very confident.
