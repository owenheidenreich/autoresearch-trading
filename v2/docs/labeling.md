# v2 Labeling Contract

## Purpose

Defines how training labels are generated from historical data.
This is the core of whether the model learns to trade or learns hindsight.

---

## v1 Label Problem

v1 used 5 label heads, 4 of which were proxies:

| Head | v1 Label Source | Problem |
|------|----------------|---------|
| Market | SPX returns at 15/30/60 bars | Direction signal, not trade outcome |
| Entry gate | sigmoid(MFE - MAE / ATR) | Proxy for "is this a good spot" |
| Risk | MAE/ATR, MFE/ATR, MFE/(MFE+MAE) | ATR-normalized proxies, not executable risk levels |
| Exit | Remaining MFE vs MAE comparison | Proxy for "should I exit", not tied to a position |
| Direction | argmax of 6 stopped option P&L | Only real trade-outcome label |

The direction label was the only one using actual option P&L with stops.
The model could optimize the proxy heads without improving trade quality.

---

## v2 Label Design: Oracle Labeler

At each decision bar, the oracle labeler asks: "What is the best TradeIntent
that could have been emitted here, given the evaluator's rules?"

### Process

For each bar `t` in the training set:

1. **Generate candidates.** Same candidate universe as evaluator.md:
   ATM +/- 30 points in 5-point steps, calls and puts = up to 26 candidates.
   Plus a no-trade option.

2. **For each candidate, simulate the trade forward.**
   Using the exact same simulator rules as the evaluator:
   - Entry at next bar (bar t+1)
   - Stop loss, take profit, trailing stops per evaluator rules
   - Max hold enforced
   - EOD flatten at bar 389
   - Spread costs applied

3. **Search over risk parameters.**
   For each candidate, try a bounded grid of stop/TP combinations:
   - Stop: [15%, 20%, 25%, 30%, 40%, 50%] of premium
   - Target: [20%, 30%, 50%, 80%, 120%] of premium
   - Max hold: [30, 60, 120, 240, 390] bars

   This is 6 x 5 x 5 = 150 configurations per candidate.
   26 candidates x 150 configs = 3,900 simulations per bar.

4. **Select the best trade** by P&L after costs.
   - If the best trade has positive P&L: label = that TradeIntent
   - If no trade has positive P&L: label = no-trade (trade=False)

5. **Store the oracle TradeIntent** as the label for bar `t`.

### What This Gives the Model

The model is not learning "which direction will SPX move." It is learning
"which specific option trade, with which specific risk parameters, would
have made money here." This is a fundamentally different learning signal.

---

## Label Fields

The oracle TradeIntent label provides supervision for:

| Field | Supervision Signal |
|-------|-------------------|
| trade (bool) | Should the model enter here at all? |
| strike / right | Which contract? |
| stop_price | How tight should the stop be? |
| take_profit_price | Where to take profit? |
| max_hold_bars | How long to hold? |
| confidence | Oracle P&L magnitude as proxy for conviction |

The model does NOT need to predict every field of TradeIntent. Some fields
are set by policy (order_style, tif, exit_policy). The model's job is the
subset above.

---

## Computational Budget

At 390 bars/day, 252 days/year, 4 years of data:
- ~393,000 bars total
- 3,900 simulations per bar (worst case)
- ~1.5 billion forward simulations

This is expensive but parallelizable:
- Each bar is independent (embarrassingly parallel)
- Each candidate x config is independent
- Simulation is pure arithmetic (no GPU needed)
- Estimated: ~2 hours on 8-core CPU, or ~15 minutes on 64-core

Labels are computed once and cached. Recomputed only when:
- Evaluator rules change (new spread model, new stop rules)
- New data is added
- Feature schema changes

---

## Fallback: Simplified Oracle

If full oracle search is too slow for iteration:

**Tier 1 (fast):** Only search ATM call and ATM put, fixed stop/TP.
2 candidates x 1 config = 2 simulations per bar.
Gives: direction label + entry/no-entry label.

**Tier 2 (medium):** ATM + OTM5 + OTM10, calls and puts, 3 stop levels.
6 candidates x 3 configs = 18 simulations per bar.
Gives: direction + strike selection + rough risk levels.

**Tier 3 (full):** Complete search as described above.

Start with Tier 1 to validate the pipeline. Graduate to Tier 3 for
production labeling.

---

## Label Quality Checks

After oracle labeling, verify:

1. **No-trade rate:** Should be 70-90% of bars. If > 95%, labels are too conservative.
   If < 50%, labels are too aggressive (or the spread model is too lenient).

2. **Direction balance:** Oracle should find both call and put opportunities.
   If > 85% one direction, the spot-return signal is leaking into the label.

3. **Average holding period:** Should be 5-60 bars (5-60 minutes).
   If < 5, the oracle is scalping with unrealistic costs.
   If > 120, the oracle is holding too long for 0DTE.

4. **Win rate:** Oracle should win > 55% of trades it takes (by construction,
   since it only trades when positive P&L is available). If < 55%, the
   forward simulation has a bug.

5. **Consistency:** Re-running oracle labeling with same parameters should
   produce identical labels (determinism check).

---

## Relationship to Evaluator

The oracle labeler uses the **exact same** simulator and cost model as the
evaluator (evaluator.md). This is critical:

- If the oracle uses different spread assumptions than replay, the model
  learns trades that look good under one cost model but fail under another.
- If the oracle uses different stop rules than replay, the model's risk
  parameters won't match the execution environment.

The oracle IS the evaluator running in "what's the best trade?" mode instead
of "how did this model's trade do?" mode.
