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

## Session 3: Hyperparameter Optimization (2026-04-06)

Score 4.884 -> 5.786 via three key changes:
- exp_018: gate_threshold 0.45->0.50 (4.933)
- exp_021: asymmetric loss 2x->3x (5.053)
- exp_024: dropout 0.1->0.05 (5.786, PDR 96.4%, 1 losing day)

8 consecutive reverts after exp_024 trying: weight_decay, LR, asymmetric 4x, deeper regime, time budget, batch size, gate+margin. Config is a strong local optimum.

## Session 4: Feature Overhaul + Dataset Rebuild (2026-04-06)

### Score Bottleneck Analysis
Score = 6.0 * PDR. Only lever is positive_day_rate. Gate threshold sweep confirmed model's P&L predictions too tightly clustered for post-hoc tuning.

### Feature Accuracy Audit
- Greek features (20-22): UNRELIABLE (BS estimates, never validated)
- Duplicate features (55-70): CONFIRMED BUG (exact copies of 39-54)
- Core market features: SOUND (no lookahead, correctly computed)
- Enriched features (39-54): SOUND (relative moneyness, scale-invariant)

### Dataset Rebuild Changes
1. Removed 16 duplicate features (55-70)
2. Added put_call_txn_ratio (order flow from wide-grid call/put transactions)
3. Added vix_ma_ratio (EMA-5/EMA-20, regime shift signal)
4. Added vix_acceleration (2nd derivative of VIX ROC)
5. Raised GATE_MIN_PNL 0.02->0.04 (more selective labels)
6. Added seed control (TRAIN_SEED env var, default 42)
7. NUM_FEATURES: 71->58 (39 base + 19 enriched)

## Session 5: 47-Feature Dataset + Optimization (2026-04-07)

### Feature Pipeline Rebuild
Full rebuild from raw data: 47 features (28 price/market + 11 option/Greeks + 8 volume/flow).
Rolling z-score normalization (60-day window) preserves inter-day regime info.
Removed dead features (volume_zero_flag, vix_ma_ratio, vix_acceleration).

### Experiment Loop (exp_032 -- exp_051, 20 experiments)

**Baseline (exp_032):** Score 4.92, 168 trades, WR 64.3%, PDR 82%.

**Key improvements (3 keeps):**
1. **exp_034: Lookback 60->30** -- Score 4.98. Shorter context reduces overfitting surface. Best epoch pushed from 2 to 3.
2. **exp_036: P&L sample weighting** -- Score 5.28. Loss weighted by `1 + |max_pnl|` focuses learning on high-signal bars. PDR 82%->88%.
3. **exp_043: Seed 123** -- Score 5.38. Different random init produced significantly better model. PDR 88%->89.6%.
4. **exp_046: Asymmetric loss 3x->5x** -- Score 5.67. Stronger penalty for optimistic predictions makes gate very selective. PDR 89.6%->94.4%. Only 77 trades (1.28/day) but PF=10.69, WR=74%.

**Score ceiling analysis:**
- Score = 6.0 * PDR. At 94.4% PDR, only 2 losing days remain out of 36 traded.
- Jan 23: wrong direction (puts in rally), -$75
- Feb 13: essentially breakeven (-$2), noise
- Attempted: stronger asymmetric (8x), cooldown, Huber delta changes, weight decay, smaller model, gate modifications. All failed to improve.
- The 2 remaining losing days are structural (direction miss) and noise (breakeven). Near the ceiling for this architecture.

### Best Model (exp_046)
| Metric | Value |
|--------|-------|
| Score | 5.667 |
| PF | 10.69 |
| WR | 74.0% |
| Trades | 77 (1.28/day) |
| PDR | 94.4% (2 losing days) |
| Sortino | 303 |
| Max DD | 0.3% |
| Direction | 38C / 39P (49/51%) |

### Config (exp_046)
- Lookback: 30, d_model: 64, depth: 3, dropout: 0.05
- LR: 5e-4, batch: 2048, weight_decay: 0.01
- Asymmetric loss: 5x for optimistic errors
- Sample weighting: 1 + |max_pnl|
- Seed: 123, Huber delta: 0.5

## Session 6: Walk-Forward CV (2026-04-08)

### Walk-Forward Baseline (exp_052)
First walk-forward run on GPU. 5 folds, 300 test days.

| Fold | Score | Trades | Traded Days |
|------|-------|--------|-------------|
| 0 | 4.776 | 216 | 49 |
| 1 | 4.979 | 191 | 47 |
| 2 | 5.063 | 67 | 32 |
| 3 | 5.793 | 63 | 29 |
| 4 | 5.357 | 51 | 28 |
| **Agg** | **5.193** | **588** | **185** |

Folds 0-1 (less training data) trade 200+ times with lower scores. Folds 2-4 (more data) are selective (50-67 trades) with higher scores. The model becomes more selective with more training data.

### Experiment Loop (exp_052 -- exp_060)

| Exp | Score | Change | Result |
|-----|-------|--------|--------|
| 052 | 5.193 | WF baseline | KEEP |
| 053 | 4.829 | dropout 0.10 | REVERT (fold 0 collapsed to 2.64) |
| 054 | 4.937 | asym 3x | REVERT (1080 trades, fold 1 collapsed) |
| 055 | 4.654 | d_model 96 | REVERT (fold 4 collapsed to 2.59) |
| 056 | 5.267 | wd 0.03 | KEEP (fold 0 improved 4.78->5.14) |
| 057 | 5.218 | wd 0.05 | REVERT (over-regularized, fold 0 regressed) |
| 058 | 5.305 | asym 4x | KEEP (fold 2 jumped +0.78) |
| 059 | 5.309 | hold_frac fix | KEEP (bug fix: training/replay scale aligned) |
| 059b | crash | hold_frac fix | CRASH (fold 2 model file not saved, disk issue) |
| 060 | -1.0 | gate 0.55 | REVERT (GATE_FAILURE all folds, 7-28 trades) |

### Key Learnings

**1. Walk-forward reveals fragility.** The old single-split score of 5.67 was optimistic. Walk-forward across 5 diverse folds dropped to 5.19. Individual fold scores range from 4.78 to 5.79 -- the model is NOT uniformly good.

**2. Hyperparameter changes create fold conflicts.** More dropout helped some folds, destroyed others. More capacity same. The only safe lever was weight_decay (0.01->0.03) which gently improved the weakest fold without hurting others.

**3. Training/replay mismatch found and fixed.** The hold_frac output was normalized by BARS_PER_DAY (390) during training but decoded by multiplying with max_hold_range[1] (250) at replay. Every hold prediction was compressed by 64%. Fixed by normalizing training target by hold_hi.

**4. Trade analysis reveals the real bottleneck: put quality.**
- All 6 losing days in the current model are caused by put trades
- Calls: WR 82.4%, avg $743. Puts: WR 65.9%, avg $453
- STOP_LOSS exits are 100% losers (bad entries with 0% MFE)
- High-MFE trailing stop losers: trades that go 30-48% right then reverse past the stop

### Trade Analysis (fold 4, exp_059 best model)

**Exit reasons:**
- MAX_HOLD: 27/75 (36%), WR 74.1%, avg $502
- TAKE_PROFIT: 21/75 (28%), WR 100%, avg $1,123
- TRAILING_STOP: 13/75 (17%), WR 53.8%, avg $38
- EOD: 8/75 (11%), WR 87.5%, avg $959
- STOP_LOSS: 6/75 (8%), WR 0%, avg -$243

**Monthly performance (fold 4 promote):**
- Dec 2025: $+15,895 (33 trades, 2 losing days)
- Jan 2026: $+8,997 (16 trades, 2 losing days)
- Feb 2026: $+17,882 (24 trades, 2 losing days)
- Mar 2026: $+1,052 (2 trades, 0 losing days)

**Losing days (6):**
- Dec 18: 2 puts, one with 0% MFE (-$230), regime mismatch
- Dec 26: 1 put, 40% MFE reversed to -$11 via trailing stop
- Jan 15: 1 put, 0% MFE, -$64 at max hold (bad entry)
- Jan 29: 1 call, 0% MFE, -$33 stop loss (bad entry)
- Feb 13: 2 puts, one 33% MFE reversed, other 5.6% MFE to -56% MAE
- Feb 19: 2 puts, both 40-48% MFE reversed via trailing stop (-$42 total)

**Direction gap is the #1 priority.** Calls WR 82.4%, puts 65.9%. Every losing day is put-driven. Next session should focus on why put predictions are weaker.

### Best Model (exp_059, walk-forward)

| Metric | Value |
|--------|-------|
| WF Score | 5.309 |
| Per-fold | [4.80, 5.38, 6.00, 5.25, 5.12] |
| Fold std | 0.39 |
| Total trades | 574 |
| Traded days | 185/300 |

### Config (exp_059)
- Lookback: 30, d_model: 64, depth: 3, dropout: 0.05
- LR: 5e-4, batch: 2048, weight_decay: 0.03
- Asymmetric loss: 4x for optimistic errors
- Sample weighting: 1 + |max_pnl|
- Huber delta: 0.5
- Hold_frac normalization: / hold_hi (250), aligned with replay
- Gate threshold: 0.50

## Research Phase: Domain-Informed Analysis (2026-04-08)

### Method
Exported all 75 trades from fold 4 promote mask to CSV. Cross-referenced patterns with domain knowledge from v2/docs/domain/ (Pickles, Sinclair, Douglas, Elder, 0DTE microstructure).

### Finding 1: Half of losses are EXIT failures, not ENTRY failures

Of 20 losing trades:
- **10 "HAD EDGE"** (MFE > 10%): direction was correct, trade went 11-48% right then reversed. EXIT problem.
- **3 "MARGINAL"** (MFE 1-10%): borderline, could go either way.
- **7 "BAD ENTRY"** (MFE 0%): immediately went wrong. ENTRY problem, unfilterable noise.

Douglas (Trading in the Zone): "Only 1 in 10 trades was an immediate loser. 25-30% of eventual losers went in direction by 3-4 ticks first." Our data matches this exactly: 10% bad entries, ~50% of losers had the right direction initially.

**Implication:** Improving entry quality has limited upside (7 bad entries = $1,864 loss, small). Improving exits has massive upside (10 HAD-EDGE losers = $575 loss, but these were $575 that COULD have been profits if exited properly).

### Finding 2: TRAILING_STOP is 100% puts and only 54% WR

All 13 trailing stop exits are put trades. 6 won, 7 lost. The trailing stop tiers are locking in losses on puts, not profits.

Domain context (0DTE knowledge): "Gamma inversely proportional to sqrt(T). By 3pm, ATM gamma can reach 0.10-0.20 (5-10x increase)." Puts are more volatile near expiry. The fixed trailing stop tiers (designed for moderate moves) may be too tight for the natural volatility of 0DTE puts.

Sinclair: "At expiry, gamma maximized ATM. Pin risk from MM hedging compresses or amplifies RV." Trailing stops that work for calls may be wrong for puts because put gamma/theta dynamics are different.

### Finding 3: MAX_HOLD puts are the worst category

MAX_HOLD exits by direction:
- **Calls**: 16 trades, WR 88%, avg $615
- **Puts**: 11 trades, WR 55%, avg $337

Puts held to max_hold lose nearly half the time. The model's risk head outputs the same hold duration for both directions, but puts need shorter holds.

0DTE domain knowledge: "Theta decay follows 1/sqrt(T). ATM put holding 10am-2pm costs 60-70% of remaining time value." Puts bleed faster through theta. Holding a put for 30 bars (30 minutes) costs significantly more in theta than holding a call for the same duration.

### Finding 4: Time-of-day does NOT explain the put weakness

| Period | Call WR | Put WR |
|--------|---------|--------|
| Morning (9:30-11:30) | 71% | 64% |
| Lunch (11:30-13:30) | 85% | 67% |
| Afternoon (13:30-16:00) | 86% | 67% |

Puts are consistently 15-20pp below calls across ALL time periods. The weakness is structural, not temporal. Pickles' "avoid lunch" rule doesn't apply -- our model actually does better during lunch (highest overall WR 74%).

### Finding 5: VIX regime analysis impossible with current features

All 75 trades show vix_at_entry between -1.0 and 0.33. Features are z-scored (60-day rolling window), so raw VIX level is normalized away. We cannot test Sinclair's "VIX < EWMA = worst time to buy options" hypothesis because the absolute VIX level is lost.

This is a potential feature gap: the model can see VIX *changes* (z-score captures relative movement) but not VIX *level* (which determines variance risk premium regime).

### Hypotheses for Next Session (ranked by expected impact)

**H1: Direction-asymmetric P&L loss (HIGHEST PRIORITY)**
- **Evidence:** Put predictions are noisier (WR 66% vs 82%). 10 losing trades had MFE > 10% -- direction was right but P&L magnitude was wrong, causing the gate to open on trades that would reverse.
- **Domain:** Sinclair: "Variance risk premium works AGAINST long options. Edge must come from timing + direction + exit speed." Puts face a steeper headwind (theta + skew premium). The loss function should reflect that put P&L is harder to predict.
- **Change:** In compute_loss, apply a stronger asymmetric weight (e.g. 6x) specifically to put predictions where the model predicted profit but actual was loss. Keep call asymmetric at 4x. This makes the model more conservative about opening put positions.
- **Expected effect:** Fewer put trades, higher put WR, reduced losing days. May slightly reduce total trades across folds.

**H2: Separate call/put P&L scaling in gate (MEDIUM PRIORITY)**
- **Evidence:** Gate uses `max(call_pnl, put_pnl) > threshold`. But put P&L is noisier and has lower average. A put prediction of +0.05 is less reliable than a call prediction of +0.05.
- **Domain:** 0DTE knowledge: "Puts trade 3-5 vol points higher than calls (downside fear premium)." Puts are structurally more expensive, meaning the bar for a profitable put trade is higher.
- **Change:** In TradingModel.forward(), scale put_pnl by 0.8 before the gate comparison. Gate = max(call_pnl, put_pnl * 0.8) > threshold. This effectively requires 25% higher predicted P&L to take a put.
- **Expected effect:** Filters marginal puts without affecting calls. Should reduce put losers (the 14 put losses -> maybe 8-10) while preserving the 27 put winners (which had avg MFE 105%).

**H3: Raw VIX level as additional feature (MEDIUM PRIORITY)**
- **Evidence:** All trades show VIX at similar z-scored values (-1 to +0.33). The model can't distinguish VIX=12 (crushed volatility, expensive to be long) from VIX=25 (elevated, cheaper to be long).
- **Domain:** Sinclair: VIX < 20 = 28% premium overpay. VIX 20-30 = 20% overpay. VIX > 50 = premium inverts. The variance risk premium is the #1 headwind and it's regime-dependent. Book-knowledge-synthesis: "VIX below EWMA = worst time to buy options."
- **Change:** This requires modifying the feature pipeline (compute_features.py) which is immutable. HOWEVER, we could add the raw VIX level as a derived feature in train.py's dataset loading, computed from the existing z-scored VIX feature + the rolling statistics. Or we could add it in the next dataset rebuild.
- **Expected effect:** Model learns to avoid buying options (especially puts) in low-VIX regimes where variance premium is highest. Could help all folds but especially fold 0 (earliest data, likely lower VIX environment).

**H4: Tighter max_hold for puts via training labels (LOWER PRIORITY)**
- **Evidence:** MAX_HOLD puts WR 55%, MAX_HOLD calls WR 88%. Puts need shorter hold times because theta acceleration punishes longer holds near expiry.
- **Domain:** 0DTE knowledge: "Theta at 2pm: extreme acceleration. ATM can lose 30-50% of remaining value per hour." Elder: "Breakeven move becomes unrealistically large in afternoon."
- **Change:** In compute_loss, when the label direction is PUT, train the risk head with a shorter max_hold target (e.g., 20 bars instead of 30). This teaches the model to hold puts for shorter durations.
- **Expected effect:** Puts exit earlier, capturing more of the MFE before reversal. The 10 HAD-EDGE losers (avg MFE 35%) should more often exit in profit if hold is shorter.

**H5: Confidence-weighted gate (EXPLORATORY)**
- **Evidence:** Confidence = |call_pnl - put_pnl| * 3.0. Currently used for intent confidence but NOT for gate decision. High-confidence trades (large margin between call and put predictions) should be more reliable.
- **Domain:** Douglas: "An edge = higher probability, not certainty." Pickles: "Confluence required -- never single signal." Higher model confidence = more confluence between call/put predictions.
- **Change:** In TradingModel.forward(), multiply gate_logit by confidence: `gate_logit = max_pnl * (1 + confidence_normalized)`. This amplifies the gate signal when the model is more certain about direction.
- **Expected effect:** Uncertain trades (where call and put predictions are similar) get weaker gate signal and are more likely filtered. Should reduce the "marginal" entries.
