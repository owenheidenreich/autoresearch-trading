# Deep Analysis: How to Teach a Neural Network to Trade 0DTE Options

**Date**: 2026-04-13
**Context**: After 20 official experiments, the model is profitable in 2/5 market regimes and fails in 3/5. This document diagnoses WHY and proposes HOW to fix it.

---

## The Core Problem

The model is a **contract ranker trained on historical PnL**, not a **trader that understands why trades work**. It learns "contracts with these features were profitable" but never learns "SPX is about to move up because momentum + volume + dealer positioning align."

### Evidence: Fold Performance Across 20 Experiments

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| **Period** | Dec24-Mar25 | Mar-Jun25 | Jun-Sep25 | Sep-Dec25 | Dec25-Mar26 |
| **Pass rate** | 5/20 (25%) | **0/20 (0%)** | 1/20 (5%) | 1/20 (5%) | 5/20 (25%) |
| **Best score** | 0.910 | 0.000 | 1.930 | 1.720 | 3.720 |
| **At DD floor** | 8/20 | 11/20 | 13/20 | 13/20 | 9/20 |
| **Avg VIX regime** | -0.19 | **+0.16** | -0.39 | -0.21 | -0.34 |
| **Avg ATM IV** | 0.119 | **0.166** | 0.070 | 0.088 | 0.081 |
| **SPX range** | 642 pts | **1,218 pts** | 607 pts | 393 pts | 290 pts |

**Fold 1 has NEVER passed.** It's the highest-vol period (VIX +0.16, ATM IV 0.166, SPX range 1,218 pts). The model simply cannot trade in volatile markets.

---

## What The Model Actually Learns

### The Training Signal (what we teach it)

1. **Gate loss**: "Should I trade this bar?" Binary decision: `best_contract_score > no_trade_score`
2. **Selection loss**: "Which contract is best?" KL-divergence toward `softmax(realized_pnl / 0.10)`

The selection target is a near-one-hot distribution over contracts, ranked by their **realized PnL under a fixed exit policy** (stop=-35%, TP=+50%, trailing stops). The model learns to predict which contract **happened to make money** given exact future price paths it can't see.

### What's Missing (what a trader actually does)

A skilled 0DTE trader makes decisions in layers:

```
1. REGIME ASSESSMENT
   "What kind of market is this? Low vol (theta wins) or high vol (gamma wins)?"
   → Model has no regime-conditional behavior

2. DIRECTIONAL THESIS
   "Where is SPX going in the next 30-60 minutes? Why?"
   → Model has NO directional prediction head

3. CONTRACT SELECTION
   "Given my thesis, which contract expresses it best?"
   → Model does this, but without thesis context

4. RISK SIZING
   "How much should I risk given my confidence and the regime?"
   → Model has fixed sizing (1 contract, fixed stop/TP)

5. ACTIVE MANAGEMENT
   "Is my thesis still valid? Should I take profit or cut?"
   → Model relies on fixed trailing stops
```

The model only does step 3, and does it without context from steps 1-2. This is why it overfits to the regime it was trained on.

---

## Why It Fails Across Regimes

### The Regime-Dependency Trap

The oracle labels encode regime-specific outcomes:
- In low-vol (fold 4): calls at ATM with tight trailing stops consistently profit → model learns "pick ATM calls"
- In high-vol (fold 1): those same ATM calls get stopped out at -35% because moves are bigger → model loses

The model learns the WHAT (which contracts) but not the WHY (market conditions that make them work). When conditions change, the learned patterns break.

### The Training Data Problem

- Walk-forward fold structure means earlier folds train on LESS data
- Best epoch is 1-2 in most experiments → the model barely learns before overfitting
- `SOFT_TEMP=0.10` creates near-one-hot targets → model hyper-focuses on exact oracle contract instead of learning "this REGION of the chain is good"
- `NOISE_MARGIN=0.01` discards ambiguous bars — exactly the bars where generalization matters

### The Label Problem

Labels are computed under a FIXED exit policy. This means:
- A contract that's great for a 10-bar scalp but terrible for 120-bar hold gets labeled "bad"
- A contract that's good in low-vol but terrible in high-vol gets a single label averaging both outcomes
- The model can't learn "this contract is good IF you exit quickly" or "this contract is good IF vol is low"

---

## How to Fix It: Restructure What The Model Learns

### Change 1: Directional Prediction Head (Highest Priority)

**What**: Add a head that predicts SPX price direction and magnitude over the next 30 bars.

**Why**: A trader forms a thesis BEFORE picking a contract. The model currently picks contracts without any directional thesis — it's like picking a tool without knowing what you're building.

**Implementation**:
```python
# New head in TradingModel
self.direction_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 3),  # up / flat / down
)

self.magnitude_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 1),  # predicted |SPX move| in %
)
```

**Label computation**: At each bar, compute actual SPX return over next 30 bars. Classify as up (>+0.1%), down (<-0.1%), flat. Store |return| as magnitude target.

**How it changes the model**: The direction embedding feeds into contract scoring. Instead of `[context || contract_emb]`, use `[context || direction_embedding || contract_emb]`. The scoring heads now know "model thinks SPX is going up" when ranking calls vs puts.

**Loss**: `DIR_W * cross_entropy(direction_logits, direction_label) + MAG_W * mse(magnitude_pred, magnitude_label)`

### Change 2: Softer Selection Targets

**What**: Raise `SOFT_TEMP` from 0.10 to 0.50 or 1.0.

**Why**: At T=0.10, a +5% contract vs +4% contract gets 22,000:1 probability ratio. The model is taught "THIS exact contract" instead of "contracts in this region of the chain." Higher temperature teaches the model that multiple nearby contracts are acceptable — which is the reality.

**Implementation**: Change `SOFT_TEMP = 0.10` to `SOFT_TEMP = 0.50`.

### Change 3: Multi-Horizon Labels

**What**: Compute labels at multiple hold durations (15, 30, 60, 120 bars) instead of only 120.

**Why**: The current labels bake in one trading style. A contract that's perfect for a scalp (hold 15 bars) gets labeled "bad" if the 120-bar simulation hits the stop. The model should learn which contracts are good at EACH horizon.

**Implementation options**:
- Store per-horizon PnL in sidecars: `row_labels_15`, `row_labels_30`, `row_labels_60`, `row_labels_120`
- Use the maximum across horizons as the label (MFE-style): "what's the best this contract could have done?"
- Add a horizon prediction head: model predicts how long to hold

### Change 4: Greeks-Derived Features (Replace Raw With Interpretable)

**What**: Add derived features that encode what the Greeks MEAN for trading, not just their values.

| Feature | Formula | What it tells the model |
|---------|---------|------------------------|
| **gamma_scalp_viability** | `expected_move / sqrt(2*|theta|/gamma)` | >1.0 = market moves enough for long gamma to profit. <0.5 = theta wins |
| **charm_pressure_direction** | `sign(aggregate_charm) * volume_weighted_magnitude` | Positive = dealer buying pressure ahead. Negative = selling pressure |
| **theta_to_premium_ratio** | `|theta_per_bar| / mid_price` | How fast this contract decays relative to its price. High = expensive to hold |
| **breakeven_bars** | `mid_price / |theta_per_bar|` | How many bars before theta eats the entire premium |

**Why**: Instead of "gamma=0.05, theta=-0.02" (meaningless numbers the model must figure out), the model sees "gamma_scalp_viability=1.3" (clear signal: long gamma should work here).

### Change 5: Regime-Aware Training

**What**: Train the model to classify the market regime and condition its decisions on it.

**Why**: The same contract can be a great trade in low-vol and a terrible trade in high-vol. The model needs to learn different decision boundaries for different regimes.

**Implementation approaches (pick one)**:

**Option A — Regime classification auxiliary loss**:
```python
# Classify each bar's regime from VIX, realized vol, ATM IV, etc.
# 4 regimes: low_vol, normal, elevated, crisis
# Add cross-entropy loss on regime prediction

self.regime_head = nn.Linear(d_model, 4)
regime_loss = F.cross_entropy(regime_logits, regime_labels)
```

Labels can be computed from VIX level at each bar (no lookahead needed):
- VIX < 15 → low_vol
- 15 ≤ VIX < 25 → normal
- 25 ≤ VIX < 35 → elevated
- VIX ≥ 35 → crisis

**Option B — Mixture of experts**:
```python
# 4 separate contract scoring networks, one per regime
# Gated by regime classifier output
# Each expert specializes in its regime
```

**Option C — Regime-weighted loss** (simplest):
```python
# Weight the selection loss by regime
# Upweight regimes where the model struggles
# Forces the model to not ignore hard regimes
```

### Change 6: Contract Cross-Attention

**What**: Let contracts see each other before being scored.

**Why**: A trader looks at the ENTIRE options chain — relative pricing matters. "This call is cheap relative to the one 5 strikes closer to ATM" is a signal. Currently each contract is scored independently.

```python
self.contract_attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)

# In forward():
contract_emb = self.contract_proj(c)  # [batch, contracts, d]
# Contracts attend to each other
contract_emb, _ = self.contract_attn(contract_emb, contract_emb, contract_emb,
                                      key_padding_mask=~valid_mask)
```

### Change 7: Longer Lookback (30 → 60 bars)

**What**: Increase the transformer's context window from 30 bars (30 min) to 60 bars (1 hour).

**Why**: At bar 65 (10:35am), the model currently sees back to bar 35 — it misses the opening range, initial balance, and overnight gap fill. These are the most important structural signals for morning trades.

---

## Implementation Priority

### Phase 1: Quick wins (no sidecar rebuild)
1. **SOFT_TEMP sweep**: 0.10 → 0.50 (one line change)
2. **Directional prediction head**: Add auxiliary loss, labels computed from spot prices already in data.pt
3. **Regime-weighted loss**: Weight selection loss by VIX regime, upweight hard regimes

### Phase 2: Feature engineering (sidecar rebuild required)
4. **Greeks-derived features**: gamma_scalp_viability, theta_to_premium_ratio, breakeven_bars
5. **Multi-horizon labels**: Compute PnL at 15/30/60/120 bars
6. **Contract cross-attention**: Architecture change in train.py

### Phase 3: Architecture (after Phase 1-2 results)
7. **LOOKBACK 30 → 60**
8. **Regime mixture-of-experts** (if regime classification shows promise)
9. **Confluence gating** (multi-signal entry requirements)

---

## What Success Looks Like

The model should:
1. **Pass 4+ folds out of 5** — proving it generalizes across market regimes
2. **Predict direction above 55%** on the directional head — proving it understands price movement
3. **Trade LESS in hostile regimes** — proving it knows when NOT to trade
4. **Have different contract preferences by regime** — proving it learned regime-conditional strategies
5. **Show PF > 1.2 across all passing folds** — proving the trades have genuine edge, not just lucky execution

The current model passes 2/5 folds with PF barely above 1.0 in its best regime. The target is a model that thinks like a trader: assess the regime, form a directional thesis, pick the right contract to express it, and manage risk accordingly.

---

## Data Limitations (permanent, do not re-investigate)

- **No open interest**: Polygon minute_aggs flat files don't include OI. GEX/gamma flip level cannot be computed.
- **No bid/ask**: Only OHLC + volume + transactions. Spread is estimated.
- These limit what features we can compute but do NOT prevent the model from learning to trade with the available data.
