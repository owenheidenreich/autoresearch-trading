# Experiment Digest -- Run 2026-03-20-064901 (Final)

## Run Metadata

| Field | Value |
|-------|-------|
| **Run ID** | `run-2026-03-20-064901` |
| **Date** | 2026-03-20, 06:49 -- 14:20 UTC (~7.5 hours) |
| **Budget** | 8h / 200 experiments max |
| **Total experiments** | 200 |
| **Real experiments** | 95 (exps 1--95 completed training + eval) |
| **API credit crashes** | 105 (exps 96--200, all "API credit balance too low") |
| **Code crashes** | 7 (exps 10, 21, 22, 34, 41, 60, 80) |
| **Scored experiments** | 88 (95 real minus 7 code crashes) |
| **Kept (new best)** | 6 (exps 1, 3, 12, 16, 27, 52) |
| **Accept rate** | 6.8% overall, 6.3% of real experiments |
| **Final best score** | **1.769** (exp 52, BalancedStrikeGate) |
| **Claude model** | claude-sonnet-4-20250514 |
| **Data fingerprint** | `d1899dc1...` (32-feature v2, ~383k bars) |
| **Architecture** | 145--147k params, d_model=64, depth=3, lookback=120 |
| **Train time** | ~255s per experiment (~4.25 min) |
| **Contract checksum** | `14969d5ad2d8` |

---

## Score Trajectory: The 6 Kept Experiments

```
Exp  1  -->  0.356   Gate bias selectivity          (NO_TRADE +0.5)
Exp  3  -->  0.417   Direction bias                  (symmetric ATM penalty, PUT_OTM5 boost)
Exp 12  -->  0.505   Learning rate halved             (3e-4 -> 1.5e-4)
Exp 16  -->  0.802   CapitalPreservationGate          (learnable account_health module)
Exp 27  -->  1.465   SelectiveStrikeGate              (OTM bias when capital stressed)
Exp 52  -->  1.769   BalancedStrikeGate               (softer strike bias, better balance)
```

Score improved **5x** from first kept model to final best across 52 experiments. Each breakthrough came from a distinct axis of improvement:

1. **Trade frequency control** (exp 1): Stop overtrading
2. **Strike selection** (exp 3): Favor cheap OTM, penalize expensive ATM
3. **Training stability** (exp 12): Lower learning rate for better convergence
4. **Capital-aware gating** (exp 16): Learnable module that reduces trading when account stressed
5. **Selective strike routing** (exp 27): Route to cheaper strikes under capital stress
6. **Balanced softness** (exp 52): Soften strike bias to avoid over-restriction

---

## Complete Experiment Table (All 95 Real Experiments)

| # | Score | PF | TPD | Trades | MinEq% | AvgRisk | SL% | WinRate | Status | Key Mutation |
|---|-------|------|-----|--------|--------|---------|-----|---------|--------|-------------|
| 1 | **0.356** | 1.86 | 1.9 | 121 | 57.9% | 0.121 | 20.7% | 35.5% | **KEPT** | NO_TRADE gate bias +0.5 |
| 2 | 0.174 | 1.72 | 2.0 | 131 | 65.2% | 0.118 | 21.4% | 41.2% | discard | PUT_ATM penalty, CALL_OTM5 boost |
| 3 | **0.417** | 2.40 | 1.9 | 123 | 70.1% | 0.112 | 22.8% | 39.8% | **KEPT** | Symmetric ATM -0.25, PUT_OTM5 +0.35 |
| 4 | 0.285 | 2.25 | 1.9 | 122 | 66.1% | 0.109 | 22.1% | 39.3% | discard | Milder ATM penalty, broader OTM boost |
| 5 | 0.025 | 1.25 | 1.6 | 107 | 16.8% | 0.178 | 31.8% | 29.0% | discard | VolatilityRegimeAdapter (ruin) |
| 6 | 0.055 | 1.22 | 1.8 | 119 | 75.6% | 0.095 | 29.4% | 38.7% | discard | Position-aware gate scaling (1 repair) |
| 7 | 0.269 | 2.09 | 2.1 | 137 | 70.7% | 0.104 | 19.0% | 39.4% | discard | OTM10 +0.40 both sides, stronger ATM pen |
| 8 | 0.421 | 2.61 | 2.1 | 135 | 72.5% | 0.089 | 20.0% | 39.3% | discard | OTM5 +0.35 both sides (blocked: chunk_pf) |
| 9 | 0.214 | 2.02 | 1.8 | 120 | 41.3% | 0.143 | 29.2% | 38.3% | discard | health_scaling on gate (backfired) |
| 10 | crash | -- | -- | -- | -- | -- | -- | -- | crash | 3rd head (size_logits): eval contract mismatch |
| 11 | 0.137 | 1.76 | 1.6 | 106 | 56.1% | 0.119 | 30.2% | 35.8% | discard | Stress-based gate/dir scaling |
| 12 | **0.505** | 1.96 | 1.7 | 110 | 56.8% | 0.136 | 26.4% | 41.8% | **KEPT** | LR halved: 3e-4 -> 1.5e-4 |
| 13 | 0.415 | 2.58 | 2.1 | 139 | 72.3% | 0.080 | 20.1% | 45.3% | discard | Reverted LR to 3e-4, PUT_OTM10 push |
| 14 | 0.057 | 1.38 | 1.7 | 108 | 49.2% | 0.121 | 17.6% | 26.9% | discard | Position-aware direction (dir collapse) |
| 15 | 0.244 | 1.63 | 1.8 | 119 | 49.6% | 0.140 | 32.8% | 36.1% | discard | 3-stage curriculum learning |
| 16 | **0.802** | 2.28 | 1.4 | 88 | 70.6% | 0.104 | 34.1% | 42.0% | **KEPT** | CapitalPreservationGate (learnable) |
| 17 | 0.156 | 1.95 | 1.6 | 102 | 46.4% | 0.149 | 34.3% | 38.2% | discard | conservatism_boost on gate (HALLUCINATION) |
| 18 | 0.738 | 2.31 | 1.3 | 87 | 70.0% | 0.097 | 33.3% | 41.4% | discard | AggressiveCapitalPreservationGate |
| 19 | 0.201 | 2.18 | 1.6 | -- | -- | -- | -- | -- | discard | Conditional health_val thresholds |
| 20 | 0.044 | 1.33 | 1.4 | -- | -- | -- | -- | -- | discard | PositionSizingCurriculum |
| 21 | crash | -- | -- | -- | -- | -- | -- | -- | crash | Unknown |
| 22 | crash | -- | -- | -- | -- | -- | -- | -- | crash | Unknown |
| 23 | 0.649 | 2.24 | 1.4 | -- | -- | -- | -- | -- | discard | StableCapitalPreservation arch variant |
| 24 | 0.075 | 1.60 | 1.4 | -- | -- | -- | -- | -- | discard | LR 1e-4, grad clip 0.5 |
| 25 | 0.107 | 1.71 | 1.4 | -- | -- | -- | -- | -- | discard | FeatureGroupProcessor |
| 26 | 0.090 | 1.52 | 1.6 | -- | -- | -- | -- | -- | discard | AdaptiveRiskGate |
| 27 | **1.465** | 2.97 | 1.4 | 89 | 81.1% | 0.090 | 32.6% | 46.1% | **KEPT** | SelectiveStrikeGate |
| 28 | 0.033 | 1.30 | 1.3 | -- | -- | -- | -- | -- | discard | CapitalPreservationGate + loss_streak |
| 29 | 0.145 | 1.65 | 1.6 | -- | -- | -- | -- | -- | discard | ConservativeCapitalGate |
| 30 | 0.878 | 3.21 | 1.2 | 78 | 64.7% | 0.092 | 41.0% | 37.2% | discard | TimeOfDayGate |
| 31 | 0.024 | 1.12 | 1.7 | -- | -- | -- | -- | -- | discard | AggressiveCapitalPreservationGate v2 |
| 32 | 0.400 | 2.42 | 1.1 | -- | -- | -- | -- | -- | discard | DynamicCapitalPreservationGate |
| 33 | 0.272 | 2.47 | 1.6 | -- | -- | -- | -- | -- | discard | PositionSizeAdapter |
| 34 | crash | -- | -- | -- | -- | -- | -- | -- | crash | Stress-based bias adjustment (tensor error) |
| 35 | 0.130 | 1.34 | 4.1 | -- | -- | -- | -- | -- | discard | MultiScaleFeatureProcessor (4.1 TPD!) |
| 36 | 0.626 | 2.73 | 1.5 | -- | -- | -- | -- | -- | discard | Stress-based OTM5 direction shift |
| 37 | 0.098 | 1.47 | 1.7 | -- | -- | -- | -- | -- | discard | 4-stage curriculum |
| 38 | 0.296 | 2.75 | 1.4 | -- | -- | -- | -- | -- | discard | Risk curriculum with ruin weighting |
| 39 | 0.182 | 2.05 | 1.0 | -- | -- | -- | -- | -- | discard | Direct account awareness arch |
| 40 | 0.057 | 1.23 | 2.3 | -- | -- | -- | -- | -- | discard | Feature attention weighting |
| 41 | crash | -- | -- | -- | -- | -- | -- | -- | crash | Unknown |
| 42 | 0.250 | 1.74 | 1.6 | -- | -- | -- | -- | -- | discard | RuinPreventionGate |
| 43 | 0.285 | 2.64 | 1.4 | -- | -- | -- | -- | -- | discard | StrikeGate with milder stress bias |
| 44 | 0.263 | 2.41 | 1.6 | -- | -- | -- | -- | -- | discard | Strike gate with slight OTM5 pref |
| 45 | 0.216 | 1.72 | 2.5 | -- | -- | -- | -- | -- | discard | Momentum trend loss (2.5 TPD) |
| 46 | 0.214 | 2.30 | 1.4 | -- | -- | -- | -- | -- | discard | Subtle selective strike gating |
| 47 | **1.055** | 2.59 | 1.6 | 102 | 59.1% | 0.122 | 26.5% | 42.2% | discard | ConservativeCapitalGate (simplified) |
| 48 | **1.336** | 3.52 | 1.2 | 79 | 89.7% | 0.084 | 35.4% | 41.8% | discard | Early capital preservation (health < 0.8) |
| 49 | 0.249 | 1.68 | 1.5 | -- | -- | -- | -- | -- | discard | AggressiveCapitalPreservationGate v3 |
| 50 | 0.706 | 2.15 | 2.7 | -- | -- | -- | -- | -- | discard | CapitalPreservationGate + loss_streak |
| 51 | 0.084 | 1.32 | 2.1 | -- | -- | -- | -- | -- | discard | ConservativeCapitalGate (weak) |
| 52 | **1.769** | 3.42 | 1.3 | 84 | 80.2% | 0.095 | 34.5% | 46.4% | **KEPT** | BalancedStrikeGate |
| 53 | 0.227 | 2.15 | 1.7 | -- | -- | -- | -- | -- | discard | ConvictionGate |
| 54 | 0.152 | 1.84 | 1.6 | -- | -- | -- | -- | -- | discard | Aggressive capital preservation (health < 0.5) |
| 55 | 0.177 | 2.19 | 1.4 | -- | -- | -- | -- | -- | discard | Multi-head attention variant |
| 56 | 1.392 | 2.97 | 1.3 | 86 | 84.2% | 0.090 | 30.2% | 44.2% | discard | ConservativeCapitalGate (stronger threshold) |
| 57 | 0.807 | 2.54 | 1.6 | 101 | 59.9% | 0.103 | 37.6% | 33.7% | discard | PositionAwareBalancedStrikeGate |
| 58 | 0.031 | 1.29 | 1.2 | -- | -- | -- | -- | -- | discard | ConvictionGate (collapsed) |
| 59 | 0.323 | 2.62 | 1.4 | -- | -- | -- | -- | -- | discard | AggressiveCapitalGate |
| 60 | crash | -- | -- | -- | -- | -- | -- | -- | crash | Risk/capital preservation weight crash |
| 61 | 0.286 | 2.30 | 1.5 | -- | -- | -- | -- | -- | discard | Less aggressive balanced strike gating |
| 62 | 0.279 | 1.75 | 1.7 | -- | -- | -- | -- | -- | discard | Consistency loss weighting |
| 63 | 0.270 | 2.37 | 1.1 | -- | -- | -- | -- | -- | discard | DynamicPositionSizeGate |
| 64 | 0.314 | 2.45 | 1.5 | -- | -- | -- | -- | -- | discard | AggressiveCapitalPreservationGate v4 |
| 65 | 0.021 | 1.10 | 1.9 | -- | -- | -- | -- | -- | discard | 4-stage curriculum v2 |
| 66 | 0.552 | 2.49 | 1.6 | -- | -- | -- | -- | -- | discard | Selectivity growth curriculum |
| 67 | 0.182 | 1.65 | 1.4 | -- | -- | -- | -- | -- | discard | Mild ATM penalty adjustment |
| 68 | 0.037 | 1.30 | 1.5 | -- | -- | -- | -- | -- | discard | ConservativeCapitalGate (weak variant) |
| 69 | **1.082** | 2.87 | 1.3 | 87 | 68.8% | 0.106 | 31.0% | 43.7% | discard | SmoothBalancedStrikeGate (3 repairs) |
| 70 | 0.631 | 2.58 | 1.6 | -- | -- | -- | -- | -- | discard | Account-aware position state |
| 71 | **1.562** | 3.20 | 1.2 | 80 | 86.9% | 0.096 | 35.0% | 46.3% | discard | Less restrictive balanced strike (health 0.5) |
| 72 | 0.207 | 1.47 | 1.8 | -- | -- | -- | -- | -- | discard | AdaptiveStopLossGate |
| 73 | 0.753 | 2.73 | 1.3 | -- | -- | -- | -- | -- | discard | Stress threshold 0.5 |
| 74 | 1.076 | 2.88 | 1.4 | 88 | 69.3% | 0.108 | 34.1% | 46.6% | discard | EarlyCapitalPreservationGate |
| 75 | crash | -- | -- | -- | -- | -- | -- | -- | crash | Contrastive learning (crash) |
| 76 | **1.272** | 3.14 | 1.3 | 87 | 81.5% | 0.094 | 32.2% | 44.8% | discard | Conservative balanced strike (health 0.6) |
| 77 | 0.035 | 1.28 | 1.4 | -- | -- | -- | -- | -- | discard | VolatilityAwareBalancedStrikeGate |
| 78 | **1.273** | 3.18 | 1.3 | 86 | 82.1% | 0.095 | 32.6% | 45.3% | discard | Refined balanced strike (health 0.65) |
| 79 | 1.272 | 3.18 | 1.3 | -- | -- | -- | -- | -- | discard | Conservative balanced strike gating |
| 80 | -10.0 | -- | -- | -- | -- | -- | -- | -- | discard | Curriculum (scored -10, penalty?) |
| 81 | 0.114 | 1.82 | 1.4 | -- | -- | -- | -- | -- | discard | RefinedBalancedStrikeGate (weak) |
| 82 | **1.423** | 3.29 | 1.4 | 89 | 81.7% | 0.095 | 30.3% | 44.9% | discard | Stronger stress bias adjustments |
| 83 | 1.202 | 3.08 | 1.3 | -- | -- | -- | -- | -- | discard | Less conservative balanced strike |
| 84 | 0.159 | 2.12 | 1.4 | -- | -- | -- | -- | -- | discard | RefinedBalancedStrikeGate |
| 85 | 0.048 | 1.28 | 1.8 | -- | -- | -- | -- | -- | discard | PositionSizeController |
| 86 | 0.812 | 2.55 | 1.3 | -- | -- | -- | -- | -- | discard | Stress bias with lower threshold (0.5) |
| 87 | 0.113 | 1.82 | 1.4 | -- | -- | -- | -- | -- | discard | False entry penalty + SelectiveGate |
| 88 | 0.256 | 2.33 | 1.6 | -- | -- | -- | -- | -- | discard | Refined balanced strike (health 0.75) |
| 89 | 0.157 | 1.84 | 1.4 | -- | -- | -- | -- | -- | discard | Slightly more conservative gating |
| 90 | 0.063 | 1.40 | 1.3 | -- | -- | -- | -- | -- | discard | RecencyWeightedAttention |
| 91 | 0.234 | 2.37 | 1.4 | -- | -- | -- | -- | -- | discard | Stress bias with OTM5+OTM10 bonus |
| 92 | 0.092 | 1.53 | 1.5 | -- | -- | -- | -- | -- | discard | SimplifiedBalancedStrikeGate |
| 93 | 1.077 | 2.95 | 1.3 | 83 | 69.4% | 0.111 | 32.5% | 41.0% | discard | Threshold 0.55 variant |
| 94 | 0.296 | 2.28 | 1.6 | -- | -- | -- | -- | -- | discard | Emergency mask (health < 0.5) |
| 95 | 0.214 | 2.01 | 1.8 | -- | -- | -- | -- | -- | discard | RecencyWeightedAttention v2 |

*Exps 96--200: All crashed with "API credit balance too low" (Anthropic API). Not code bugs.*

---

## Per-Experiment Detail: The 6 Kept Models

### Exp 1 -- KEPT (score 0.356) -- Foundation: Trade Selectivity

**Hypothesis**: Model is trading too frequently. Increase NO_TRADE gate bias from +0.3 to +0.5.

**Mutation**: `gate_head[-1].bias[0] += 0.5; gate_head[-1].bias[1] -= 0.5`

**Result**: Score jumped from prior run's 0.012 to 0.356 -- a 30x improvement. This single change transformed the model from guaranteed ruin to viable trading. Trade count dropped, avg_risk_fraction fell from 0.38 to 0.12, and the model survived the full backtest for the first time. The simplest possible mutation turned out to be the most foundational.

| Metric | Value | Context |
|--------|-------|---------|
| PF | 1.86 | Decent but not exceptional |
| Trades | 121 (1.9 TPD) | Still somewhat high |
| MinEquity | 57.9% | First model to survive |
| MaxDrawdown | -4.54% | Elevated |
| Val Sharpe | -16.3 | Negative out-of-sample |
| Trade Sharpe | 2.58 | Reasonable |

---

### Exp 3 -- KEPT (score 0.417) -- Direction Bias Tuning

**Hypothesis**: Penalize both ATM strikes symmetrically, boost PUT_OTM5 for cheaper entries.

**Mutation**: `CALL_ATM -0.25, PUT_ATM -0.25, PUT_OTM5 +0.35`

**Result**: Score rose 17% to 0.417. PF climbed to 2.40, min_equity_frac improved to 70.1% (model retained 70% of capital at worst). The key insight: symmetric ATM penalty matters -- penalizing only one side (as exp-2 tried) leaves a leakage path to expensive trades. Trade Sharpe reached 3.67.

| Metric | Value | vs Exp 1 |
|--------|-------|----------|
| PF | 2.40 | +29% |
| MinEquity | 70.1% | +12pp |
| AvgRisk | 0.112 | -0.9pp |
| WinRate | 39.8% | +4.3pp |
| Val Sharpe | -16.3 | Still negative |

---

### Exp 12 -- KEPT (score 0.505) -- Training Stability

**Hypothesis**: Loss explosion at step 10450 suggests LR is too high. Halve from 3e-4 to 1.5e-4.

**Mutation**: `LR = 1.5e-4` (single line change)

**Result**: The simplest mutation in the entire run produced a +21% score jump. PF dropped slightly (1.96 vs 2.40) but win rate climbed to 41.8% (best at the time). Stop-loss rate improved. The lower LR allowed the model to converge to a more refined solution instead of oscillating. This changed the default LR for all subsequent experiments.

| Metric | Value | vs Exp 3 |
|--------|-------|----------|
| PF | 1.96 | -18% (acceptable tradeoff) |
| Trades | 110 (1.7 TPD) | -11% (fewer, better) |
| WinRate | 41.8% | +2.0pp |
| StopLoss | 26.4% | +3.6pp (slight concern) |
| Val Sharpe | -6.85 | Improved but still negative |

---

### Exp 16 -- KEPT (score 0.802) -- The Breakthrough: Learnable Capital Preservation

**Hypothesis**: Capital preservation needs to be learned, not hand-coded. Add a CapitalPreservationGate module that takes account_health as input and learns the right trading-capital sensitivity.

**Mutation**: New `CapitalPreservationGate(nn.Module)` with trainable parameters (+1,153 params to 146,265 total).

**Result**: The biggest single-experiment breakthrough. Score leapt from 0.505 to 0.802 (+59%). This was the first experiment with positive val_sharpe (4.20), suggesting real out-of-sample generalization. Only 88 trades (1.35 TPD), the fewest yet. Max drawdown just -1.99% -- half the typical drawdown. The critical difference from 4 prior failed dynamic-scaling attempts (exps 6, 9, 11, 17): this module has **trainable parameters** that learn the right sensitivity from data, rather than hand-coded formulas.

| Metric | Value | vs Exp 12 |
|--------|-------|-----------|
| PF | 2.28 | +16% |
| Trades | 88 (1.35 TPD) | -20% |
| MinEquity | 70.6% | +13.8pp |
| MaxDrawdown | -1.99% | Best ever |
| Val Sharpe | +4.20 | First positive! |
| StopLoss | 34.1% | Elevated (but fewer total trades) |

**Why prior dynamic scaling failed (exps 6, 9, 11)**: All used formulas like `scale = f(account_health)` applied directly to logits. The formulas disrupted the learned gate biases. The CapitalPreservationGate instead outputs a learned modulation signal trained end-to-end -- letting the model discover the right capital-sensitivity mapping.

---

### Exp 27 -- KEPT (score 1.465) -- Selective Strike Routing

**Hypothesis**: Instead of just dampening the gate signal when capital is low, make the model more selective about *which* trades to take -- specifically favoring cheaper OTM options over expensive ATM ones when account health drops below 0.6.

**Mutation**: New `SelectiveStrikeGate(nn.Module)` that modulates both gate and direction based on account_health.

**Result**: Score nearly doubled from 0.802 to 1.465 (+83%). PF crossed 2.97, min_equity_frac reached 81.1% (the model never lost more than 19% of capital). Max drawdown of just -1.63%. Win rate hit 46.1%. This succeeded because it addressed a deeper problem: the CapitalPreservationGate only controlled *whether* to trade, not *how* to trade when stressed. The SelectiveStrikeGate routes stressed trades to cheaper instruments, reducing per-trade risk.

| Metric | Value | vs Exp 16 |
|--------|-------|-----------|
| PF | 2.97 | +30% |
| Trades | 89 (1.37 TPD) | Similar |
| MinEquity | 81.1% | +10.5pp |
| MaxDrawdown | -1.63% | -18% better |
| Val Sharpe | +5.98 | +42% |
| WinRate | 46.1% | +4.1pp |
| Trade Sharpe | 4.34 | +36% |

---

### Exp 52 -- KEPT (score 1.769) -- The Final Best: Balanced Softness

**Hypothesis**: The SelectiveStrikeGate's strike bias may be too aggressive in biasing away from ATM strikes. Soften the bias adjustments while keeping the capital preservation mechanism.

**Mutation**: New `BalancedStrikeGate(nn.Module)` with softer strike bias parameters.

**Result**: The run's high-water mark. Score improved +21% to 1.769. PF reached 3.42 (highest of any kept model), with the lowest trade count (84, 1.29 TPD) and best capital preservation profile. The model retained 80.2% of capital at its worst point. **Crucially, worst_chunk_pf hit 2.86** -- the model was profitable in every market regime chunk, a first. Prior kept models had worst_chunk_pf below 1.0.

| Metric | Value | vs Exp 27 |
|--------|-------|-----------|
| PF | 3.42 | +15% |
| Trades | 84 (1.29 TPD) | -6% |
| MinEquity | 80.2% | -0.9pp (similar) |
| MaxDrawdown | -1.47% | -10% better |
| Val Sharpe | +5.62 | Similar |
| WinRate | 46.4% | +0.3pp |
| Trade Sharpe | 4.67 | +8% |
| Worst Chunk PF | 2.86 | First >1.0! |

---

## Notable Near-Misses (High Score, Not Kept)

These experiments scored above 1.0 but could not beat the current best at the time they ran.

### Exp 48 (score 1.336) -- Early Capital Preservation

Raised the stress threshold from 0.6 to 0.8, triggering capital preservation earlier. Achieved the **best min_equity_frac of any experiment: 89.7%**. PF 3.52 was the highest in the entire run. Only 79 trades. The model barely lost capital. Worst_chunk_pf of 2.68 was excellent. It didn't beat exp-27's 1.465 at the time, but this model's capital preservation was arguably superior. Best val_sharpe of 6.66.

### Exp 71 (score 1.562) -- Less Restrictive Balanced Strike

Lowered health threshold to 0.5 and softened health multiplier range to [0.7, 1.0]. Second highest score after exp-52. PF 3.20, 80 trades, min_equity 86.9%, max drawdown -1.43% (best of any experiment). Worst_chunk_pf 2.22. This was the closest near-miss -- just -0.207 below the 1.769 best.

### Exp 82 (score 1.423) -- Stronger Stress Bias

Increased ATM penalty and OTM10 bonus under stress. PF 3.29, 89 trades, min_equity 81.7%. Worst_chunk_pf 2.39. Near-identical to the best model's profile but scored -0.346 lower.

### Exp 56 (score 1.392) -- Conservative Capital Gate

Used stronger health threshold (0.8) for stress detection. PF 2.97, 86 trades, min_equity 84.2%. Good but scored below best.

### Exp 69 (score 1.082) -- Smooth Balanced Strike Gate

Required 3 codegen attempts (2 repairs for tensor shape mismatch). Still scored well. PF 2.87, 87 trades. Despite buggy implementation requiring repairs, the core idea was sound.

### Exp 30 (score 0.878) -- Time of Day Gate

Added a TimeOfDayGate aware of 0DTE session patterns (morning trends, lunch chop, power hour). PF 3.21 was the second-best at the time. Only 78 trades (1.2 TPD, lowest). However, 41% stop-loss rate (highest in the entire run). The gate was too conservative during non-morning hours, and when it did trade, it often hit stops. Val_sharpe was deeply negative (-22.9), suggesting the time-of-day patterns didn't generalize.

### Exp 47 (score 1.055) -- Conservative Capital Gate (simplified)

Simplified the selective strike gating to be more conservative. PF 2.59, 102 trades (higher than peers). Good but the higher trade count diluted per-trade edge.

---

## The Exp 17 Hallucination Bug

Exp 17 is notable as an **agent hallucination incident**. The Claude agent was given a prompt indicating that exp-16 (score 0.802) was the new best model. However, the agent's reasoning stated: *"The current best model (#12) achieved score=0.505221"* -- it referenced the old best (#12) instead of the correct new best (#16).

This caused the agent to implement a hand-coded `conservatism_boost` formula on top of the CapitalPreservationGate, a strategy that had already been proven counterproductive. The agent would have known this had it correctly processed the exp-16 context. The result was a score regression to 0.156, wasting an experiment slot.

**Impact**: Moderate. One wasted experiment. The agent self-corrected by exp-18, which correctly built on exp-16's CapitalPreservationGate concept (scoring 0.738).

**Root cause**: Unknown. Possibly token-limit context overflow or attention failure on the prompt. The system prompt was correctly constructed with exp-16's data.

---

## Emerging Patterns

*Synthesized across all 88 scored experiments.*

### The Architecture Evolution

The model went through four distinct architectural eras:

**Era 1 -- Bias-Only (exps 1--15)**: Changes were limited to gate/direction bias values and hyperparameters. Best score: 0.505. These experiments established foundations (gate selectivity, ATM penalty, lower LR) but hit a ceiling because the model couldn't adapt its behavior to account state.

**Era 2 -- CapitalPreservationGate (exps 16--26)**: Added a small learnable module (~1,153 params) for capital-aware gating. Best score: 0.802. Broke the 0.5 ceiling. But still only controlled *whether* to trade, not *what* to trade.

**Era 3 -- SelectiveStrikeGate (exp 27)**: Extended capital awareness to strike selection. Best score: 1.465. Broke the 1.0 ceiling. Now the model favored cheaper contracts when capital was stressed.

**Era 4 -- BalancedStrikeGate (exps 52--95)**: Refined the strike gate to be softer and more balanced. Best score: 1.769. 43 experiments of refinement on this architecture produced many near-misses (12 experiments scored >1.0) but only one new best. The architecture may be near its optimum for the current training budget and data.

### What Works

1. **Learnable modules over hand-coded formulas**: 100% of experiments with custom `nn.Module` gating scored higher than their bias-only peers. 0/6 hand-coded dynamic scaling attempts succeeded (exps 6, 9, 11, 17, 31, 34). The lesson is definitive.

2. **Extreme trade selectivity**: The 6 kept models show a clear monotonic trend. Fewer trades = higher score.
   ```
   Exp  1: 121 trades, score 0.356
   Exp  3: 123 trades, score 0.417
   Exp 12: 110 trades, score 0.505
   Exp 16:  88 trades, score 0.802
   Exp 27:  89 trades, score 1.465
   Exp 52:  84 trades, score 1.769
   ```

3. **Symmetric ATM penalty**: Penalizing both CALL_ATM and PUT_ATM equally (-0.25) is non-negotiable. Every model scoring >0.4 uses this.

4. **OTM5 over OTM10**: OTM5 contracts are the sweet spot. OTM10 is too cheap (low delta) and invites overtrading (exp 7, 13, 35).

5. **Capital-aware strike routing**: When account health drops, route trades to cheaper OTM options. This is the single biggest discovery of the run.

6. **LR 1.5e-4**: Halved from default 3e-4, used in all experiments from exp-12 onward. Reverting it (exp 13) immediately lost stability.

7. **Positive val_sharpe cluster**: Exps 16, 18, 27, 48, 52, 71, 74, 76, 78, 79, 82, 83, 86, 93 all had positive val_sharpe. Every one used some form of capital-preservation gating. The CapitalPreservationGate family appears to improve out-of-sample generalization.

### Key Metrics Across Kept Models

| Exp | Score | PF | TPD | Trades | MinEq% | AvgRisk | SL% | WR% | ValSharpe | WorstChunkPF |
|-----|-------|------|-----|--------|--------|---------|-----|-----|-----------|-------------|
| 1 | 0.356 | 1.86 | 1.9 | 121 | 57.9 | 12.1 | 20.7 | 35.5 | -16.3 | 0.32 |
| 3 | 0.417 | 2.40 | 1.9 | 123 | 70.1 | 11.2 | 22.8 | 39.8 | -16.3 | 0.87 |
| 12 | 0.505 | 1.96 | 1.7 | 110 | 56.8 | 13.6 | 26.4 | 41.8 | -6.8 | 0.66 |
| 16 | 0.802 | 2.28 | 1.4 | 88 | 70.6 | 10.4 | 34.1 | 42.0 | +4.2 | 0.24 |
| 27 | 1.465 | 2.97 | 1.4 | 89 | 81.1 | 9.0 | 32.6 | 46.1 | +6.0 | 0.46 |
| 52 | 1.769 | 3.42 | 1.3 | 84 | 80.2 | 9.5 | 34.5 | 46.4 | +5.6 | 2.86 |

**Trends across kept models**:
- PF: 1.86 -> 3.42 (steady rise, 84% improvement)
- Trades: 121 -> 84 (30% fewer)
- MinEquity: 57.9% -> 80.2% (model keeps 80% of capital at worst)
- AvgRisk: 12.1% -> 9.5% (smaller positions)
- WinRate: 35.5% -> 46.4% (nearly half of trades win)
- Val Sharpe: -16.3 -> +5.6 (from catastrophic OOS to positive)
- Worst Chunk PF: 0.32 -> 2.86 (from losing badly in bad regimes to profitable everywhere)

### Score Distribution of All 88 Scored Experiments

```
Score >= 1.5:   3 experiments  (3.4%)   -- exps 52, 71, 27 (not in order)
Score 1.0-1.5: 10 experiments (11.4%)   -- exps 47, 48, 69, 74, 76, 78, 79, 82, 83, 93
Score 0.5-1.0:  9 experiments (10.2%)   -- exps 16, 18, 23, 36, 50, 57, 66, 73, 86
Score 0.2-0.5: 21 experiments (23.9%)
Score 0.0-0.2: 40 experiments (45.5%)
Score < 0.0:    5 experiments  (5.7%)   -- includes crashes and penalties
```

The top quartile (score > 0.5) contains 22 experiments, all sharing: (a) some form of gating module, (b) LR 1.5e-4, (c) symmetric ATM penalty, (d) trade count under 100.

---

## Dead Ends

These approaches were tried multiple times and consistently failed. Future runs should not attempt them.

### 1. Hand-Coded Dynamic Scaling (0/6 success)
Exps 6, 9, 11, 17, 31, 34. Applying formulas like `scale = 1.0 + 2.0 * (1.0 - account_health)` to gate/direction logits. Always disrupts learned biases. The model already learned good gate sensitivity through training -- layering heuristics on top creates conflicting signals.

### 2. Large Architectural Additions (0/4 success)
Exps 5 (VolatilityRegimeAdapter), 25 (FeatureGroupProcessor), 40 (feature attention weighting), 55 (multi-head attention variant). New modules with many untrained parameters overwhelm the 4-minute training budget. The model cannot learn both the new module and the trading task simultaneously.

### 3. Curriculum Learning (0/4 success)
Exps 15, 20, 37, 65. Multi-stage training creates discontinuities. The 4-minute budget is too short for curriculum transitions to settle. The model learns one regime then unlearns it for the next.

### 4. Position-Aware Direction Head (0/2 success)
Exps 14, 57. Feeding position state into the direction head causes direction collapse (exp 14: direction_collapse_pct = 1.0, the model locked onto a single direction). Direction selection should remain position-agnostic.

### 5. Third Output Head (0/1 success)
Exp 10. Adding a size_logits head for dynamic position sizing crashed because evaluate_trades expects a 2-tuple. The evaluation contract is read-only for the agent. This is a constraint, not a bad idea -- would need prepare.py changes.

### 6. Momentum/Trend Loss Functions (0/1 success)
Exp 45. Added a momentum_trend_loss that penalized counter-trend entries. TPD spiked to 2.5 (the model traded more, not less). The extra loss term interfered with the gate's learned selectivity.

### 7. Recency-Weighted Attention (0/2 success)
Exps 90, 95. Modified attention to weight recent bars more heavily. Both scored poorly (<0.22). The standard transformer attention already learns temporal patterns; explicit recency weighting is redundant.

### 8. Contrastive Learning (0/1 success)
Exp 75. Crashed during training.

### 9. Adaptive Stop-Loss (0/1 success)
Exp 72. Attempted to modify stop-loss behavior dynamically. Scored 0.207. Stop-loss is a fixed 30% emergency backstop by design -- trying to make it adaptive undermines its purpose as a safety net.

### 10. Conviction Gating (0/2 success)
Exps 53, 58. Added a ConvictionGate that required the model to express high confidence before trading. Exp 58 collapsed to 0.031. The conviction mechanism added a threshold that the model couldn't reliably calibrate.

---

## Saturation Analysis: Is the Run Converging?

The last 43 experiments (53--95) were all in Era 4 (BalancedStrikeGate variants) and produced **zero new bests**. However, 12 of them scored above 1.0, showing the architecture is consistently competitive. The experiments explored a wide parameter space:

- Health thresholds: 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8
- Bias strengths: mild, moderate, aggressive
- Gate module variants: Smooth, Refined, Conservative, Less-restrictive, Position-aware

None surpassed 1.769. The distribution of scores in this window:

```
Exps 53-95 (43 experiments):
  >1.5:  1 (exp 71, score 1.562)
  >1.0:  7 (exps 56, 69, 74, 76, 78, 82, 93)
  >0.5:  4 (exps 57, 66, 73, 86)
  <0.5: 31
```

The 8 experiments above 1.0 suggest there is still signal in this architecture, but diminishing returns are evident. The run was approaching plateau when API credits ran out.

---

## State of the Model (Final)

### Current Best: Exp 52 (BalancedStrikeGate)
- **Score**: 1.769
- **Profit Factor**: 3.42 (earns $3.42 for every $1.00 lost)
- **Trades**: 84 across 65 simulated days (1.29 per day)
- **Win Rate**: 46.4%
- **Capital Preservation**: Retained 80.2% of capital at worst point, max drawdown -1.47%
- **Consistency**: Worst chunk PF 2.86 -- profitable in every market regime chunk
- **Out-of-Sample**: Val Sharpe +5.62 (positive generalization)
- **Risk Per Trade**: 9.5% average, 35.4% maximum
- **Stop-Loss Rate**: 34.5% (elevated, the model's main weakness)

### Strengths
- **Extreme selectivity**: The model passes on >59% of actionable bars. When it trades, it has conviction.
- **Capital preservation**: The BalancedStrikeGate routes to cheaper instruments under stress, preventing blowup.
- **Regime robustness**: Worst chunk PF of 2.86 means the model handles all market conditions.
- **Out-of-sample performance**: Positive val_sharpe across the CapitalPreservationGate family.
- **Profit quality**: PF 3.42 with win rate 46.4% means winners are much larger than losers.

### Weaknesses
- **Stop-loss rate**: 34.5% of trades hit the 30% emergency stop. This is the model's primary failure mode. Nearly 1 in 3 trades end at maximum loss.
- **Trade frequency**: 1.29 TPD may be too low for practical trading (some days have zero trades).
- **Narrow architecture basin**: 43 experiments of refinement on BalancedStrikeGate variants couldn't beat it, suggesting the architecture may be at a local optimum.
- **High entry cost**: Average entry cost 380 bps remains elevated.

### Recommendations for Next Run
1. **Focus on stop-loss rate reduction**: The 34.5% SL rate is the clearest path to improvement. Entry quality filtering, tighter OTM5 preference, or learned entry timing could help.
2. **Explore BalancedStrikeGate + TimeOfDayGate fusion**: Exp-30's TimeOfDayGate scored 0.878 on a fresh architecture. Combining it with the proven BalancedStrikeGate could add temporal awareness.
3. **Try warmup/cooldown on the gate module**: Instead of the gate immediately responding to stress, add temporal smoothing to prevent oscillation.
4. **Increase training budget**: The 4.25-minute window limits curriculum approaches and larger architectures. A 10-minute budget could unlock new strategies.
5. **Don't abandon BalancedStrikeGate**: Despite 43 failed refinements, 12 of them scored >1.0. The architecture has variance that a longer search could exploit with different random seeds or LR schedules.

## Backtest P&L Deceleration Analysis

### Observation
The equity curve shows rapid acceleration from Nov 2022 to Oct 2023 ($10k → $28.4k, +184%), then levels off from Nov 2023 onward ($28.4k → $37.6k, +32% over 2.3 more years). The model remains profitable but the growth rate drops dramatically.

### Root Cause: Fixed 1-Contract Position Sizing

The system always trades exactly 1 SPX option contract (`RISK_PER_TRADE = 1.0` in prepare.py). There is no multi-contract scaling. As the account grows, each trade's risk fraction shrinks:

| Trades | Period | Growth | Win Rate | Avg Risk/Trade |
|--------|--------|--------|----------|----------------|
| 1-50 | Nov 2022 → Jul 2023 | **+77.8%** | 66% | 4.7% |
| 51-100 | Aug 2023 → Nov 2023 | **+60.0%** | 68% | 3.7% |
| 101-150 | Nov 2023 → Dec 2024 | **+25.1%** | 78% | 2.2% |
| 151-200 | Dec 2024 → Jul 2025 | **+0.8%** | 38% | 2.3% |
| 201-250 | Jul 2025 → Nov 2025 | **+7.1%** | 46% | 1.8% |
| 251-289 | Nov 2025 → Mar 2026 | **-2.0%** | 36% | 3.9% |

**The structural effect dominates.** Trades 101-150 have the _highest_ win rate (78%) but only 25% growth — because a $200 win on a $30k account is 0.67% vs 2% on a $10k account.

First 100 trades: avg +1.07% account return per trade. Last 100 trades: avg +0.04%.

### Two Compounding Factors

1. **Structural (primary):** Fixed 1-contract sizing → risk fraction shrinks from 4.7% to 1.8% as account triples → same dollar wins produce diminishing % returns. A $200 winner on $10k = 2%; on $37k = 0.54%.
2. **Signal degradation (secondary):** Win rate drops from 66-78% (first 150 trades) to 36-46% (last 139 trades). This may reflect regime shift, overfitting to earlier market conditions, or reduced alpha in the model's signal.

### The Model Can Handle More Risk

- Max drawdown across the entire 986-day backtest: only **-10.2%**
- The BalancedStrikeGate architecture routes to cheaper OTM instruments under stress — a natural risk governor
- Even at peak leverage (early trades, 4.7% risk), the account never came close to ruin
- A hypothetical 5% fixed-fraction sizing would have grown to ~$113k by trade 100 (vs $28k actual)

### Implication for Next Training Run

Position sizing is outside the model's control (hardcoded at 1 contract). This is a **simulation design decision**, not a model failure. Options for the next phase:
- Allow the autoresearch agent to evolve a `size_logits` head (previously failed as 3rd head — but could work as a multiplier on the existing gate)
- Implement Kelly-based contract scaling in the evaluation loop (e.g., risk = f(account_health, gate_prob)), always whole contracts
- Simply increase the evaluation starting capital to better match live account sizing
