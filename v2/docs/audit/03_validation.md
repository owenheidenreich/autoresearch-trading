# Section 3: Validation

## Scope
Replay simulation, trade scoring, baseline comparison, walk-forward cross-validation, post-trade analysis, and visualization. This is the immutable evaluation harness that determines whether a model is promoted or reverted.

## Critical Files

| File | Role | Lines | Mutable? |
|------|------|------:|----------|
| `v2/replay.py` | Replay engine. Loads trained model, generates TradeIntents from predictions, simulates trades on val/promote/shadow masks. Computes 4 baselines (random, ATM-always, simple-rules, ATM-trailing). Applies score formula. | 903 | No (harness) |
| `v2/core/simulator.py` | Trade simulation engine. Executes TradeIntent against historical option data. Implements trailing stop tiers, adaptive spread cost (BPS + min-tick floor), entry/exit fill logic. Outputs SimulatedTrade with exit reason, bars_held, P&L. | 317 | No (harness) |
| `v2/core/metrics.py` | ReplayMetrics dataclass. Computes profit_factor, win_rate, total_trades, daily_sortino, max_account_drawdown, direction balance, exit reason breakdown, MAE/MFE. Score formula: `min(daily_sortino, 6.0) * positive_day_rate * dd_mult`. | 338 | No (harness) |
| `v2/core/walkforward.py` | Walk-forward CV harness. Generates 5 folds with expanding training windows and 60-day test windows. Trains fresh model per fold. Aggregates scores. | 324 | No (harness) |
| `v2/ops/run_experiment_wf.py` | Walk-forward experiment runner. Drop-in replacement for `run_experiment.py`. Runs 5 folds internally, reports per-fold breakdown. | 180 | No (infra) |
| `v2/analysis/analyze_losses.py` | Losing day post-mortem. Groups trades by date, computes daily P&L, analyzes exit reasons, direction balance, confidence distributions on losing days. | 79 | No (analysis) |
| `v2/analysis/analyze_whipsaw.py` | Whipsaw detection. Identifies consecutive opposite trades, measures reversal patterns and cost impact. | 138 | No (analysis) |
| `v2/plot_trades.py` | Interactive HTML visualization. Generates `trades.html` (trade markers on SPX chart) and `equity.html` (equity curve) via Plotly. Also exports `trades.csv`. | 485 | No (viz) |
| `v2/plot_progress.py` | Experiment progress chart. Reads `results.tsv`, generates `progress.png` (Karpathy-style score over experiments). | 238 | No (viz) |

## Data Flow

```
model_candidate.pt (from Training Runs)
    +
data.pt (from Market Data)
    |
    v
replay.py
    |  loads: model + data.pt
    |  for each bar in mask (val/promote/shadow):
    |    model forward pass -> (call_pnl, put_pnl) predictions
    |    model_to_intent() -> TradeIntent
    |    simulator.simulate_trade() -> SimulatedTrade
    |  collects: all SimulatedTrade records
    |
    v
core/metrics.py
    |  aggregates: trades -> daily returns -> ReplayMetrics
    |  applies: score formula
    |  applies: hard gates (>= 30 trades, >= 15 days, >= 15% minority direction, <= 20% drawdown)
    |
    v
baselines (computed in replay.py)
    |  random: random gate decisions
    |  atm_always: always trade ATM
    |  simple_rules: rule-based signals
    |  atm_trailing: ATM with trailing stops
    |  model must beat ALL four
    |
    v
score -> stdout (consumed by ART2 keep/revert)
    |
    v
plot_trades.py -> trades.html, equity.html, trades.csv
plot_progress.py -> progress.png
analyze_losses.py -> console output (losing day patterns)
```

## Walk-Forward CV Flow

```
300 test days total (Dec 2024 -- Mar 2026)
    |
    v
walkforward.py generates 5 folds:
    Fold 1: train [earliest..T1] | val [T1..T1+40] | test [T1+40..T1+100]
    Fold 2: train [earliest..T2] | val [T2..T2+40] | test [T2+40..T2+100]
    ...expanding train window, sliding test window
    |
    v
Each fold: fresh model trained from scratch
    |
    v
Per-fold scores aggregated -> WF composite score
```

## Key Interfaces

**Inputs:**
- `model_candidate.pt` -- trained model checkpoint
- `data.pt` -- features, option prices, masks
- `core/policy.py` -- DecisionPolicy for gate threshold, risk ranges

**Outputs:**
- Score (float) printed to stdout
- `v2/output/trades.html` -- interactive trade chart
- `v2/output/equity.html` -- equity curve
- `v2/output/trades.csv` -- trade log
- `v2/output/progress.png` -- experiment history chart
- `v2/results.tsv` -- appended with experiment result

**Score formula:**
```
score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult
```
Where `dd_mult = 1.0` when `max_drawdown <= 8%`, linear decay to `0.0` at `20%`.

**Hard gates (model fails if any violated):**
- >= 30 total trades
- >= 15 traded days
- >= 15% minority direction (calls vs puts)
- max_drawdown <= 20%

## Dependencies on Other Sections

| Section | Dependency |
|---------|------------|
| Market Data | Consumes `data.pt` for simulation prices and features |
| Training Runs | Consumes `model_candidate.pt`; `run_experiment.py` calls replay as its scoring step |
| ART2 Pipeline | Score output drives the keep/revert decision; visual artifacts are regenerated after every experiment |

## Audit Surface Area

- Score formula: is `min(sortino, 6.0) * positive_day_rate * dd_mult` the right objective?
- Hard gates: are the thresholds (30 trades, 15 days, 15% direction, 20% DD) appropriate?
- Simulator fidelity: does `simulator.py` accurately model real execution (fills, stops, spread)?
- Spread cost model: adaptive BPS + min-tick -- does it match IBKR paper trading costs?
- Baseline comparison: are all 4 baselines fair comparisons?
- Walk-forward: is there data leakage between folds? Is the expanding window appropriate?
- Replay determinism: given the same model and data, does replay produce identical results?
- Exit reason tracking: are all exit paths (stop, target, trailing, model_exit, EOD, max_hold) correctly implemented?

---

## Audit Questions -- Direct Improvements

**1. Random baseline pools trades across 5 seeds instead of averaging scores -- distorts trade count and equity curve.**
`replay.py:457-520` accumulates all trades from all 5 seeds into `all_trades_combined`, then divides `num_days` by `n_seeds`. The random baseline gets 5x normal trades but 1x days. Its Sortino and drawdown are computed on a synthetic equity curve no single random policy would ever produce. The docs say "5 seeds averaged" but the code does not average scores.

**2. Direction balance gate uses ratio-of-percentages, not the "at least 15% minority" the docs describe.**
`metrics.py:288-293` computes `dir_balance = dir_minority / dir_majority` (ratio). 10% calls / 90% puts gives `0.10/0.90 = 0.11` (fails). But `program.md` reads as `min(call_pct, put_pct) >= 0.15`. A model with 13% calls and 87% puts passes the code gate (`0.13/0.87 = 0.149`) but fails the documented gate. Ambiguity in what "direction balance" means.

**3. `daily_loss_cap` in `simulator.py:simulate_day()` is never called from the main replay path -- dead code.**
`replay.py:replay_validation()` calls `simulate_trade()` directly, never `simulate_day()`. The daily loss cap logic at `simulator.py:284-286` only exists in `simulate_day()`. The `daily_loss_cap_pct=0.05` from `policy.py:46` is dead code during replay scoring. A model can lose unlimited amounts on a single day.

**4. `replay_validation` uses `option_mid = max(atm_call_px, atm_put_px)` regardless of direction -- wrong reference for risk.**
`replay.py:287-289` uses the max of call/put price for stop/TP computation. If the model picks a put when `atm_call_px > atm_put_px`, risk percentages are computed against the wrong reference price.

**5. Simple-rules baseline momentum threshold (0.005) operates on z-scored values, not raw returns as docs specify.**
`replay.py:639`: `abs(momentum) < 0.005` where `momentum` is a z-scored feature. `baselines.md` says "5-bar momentum > +0.15%". Z-scored 0.005 is not equivalent to 0.15% raw return. Baseline sensitivity is different than intended.

**6. MFE/MAE tracked during MIN_HOLD_BARS when exits are suppressed -- misleading trade quality view.**
`simulator.py:143-149` updates MFE/MAE every bar including the no-exit window. High MAE during this period was structurally unavoidable. Analysis tools using MFE/MAE attribute adverse excursion that the model could never have prevented.

**7. Commission formula understates friction for cheap options.**
`simulator.py:203`: `commission_frac = (2 * 0.65) / (entry_px * 100)`. At min entry ($0.50), commission is 2.6% of notional. Combined with spread cost floor, cheap options face 5-7% round-trip friction. Surface aggregate cost per trade so experiments can detect if the model systematically trades options where friction dominates P&L.

## Audit Questions -- Deeper Planning

**8. The fill model uses mid-price for entry, not the ask -- single largest source of simulation optimism.**
`evaluator.md:36` specifies "Entry fill: at the ask price." But `simulator.py:88` uses mid-price. For 0DTE SPX options, the ask is typically 5-20% above mid. Every trade in replay is better than it would be on IBKR by the half-spread on entry. When the model goes live, every trade will be worse than simulation.

**9. Walk-forward CV: val days carved from end of train, potential leakage via global normalization.**
`walkforward.py:102-106` includes val within train, then `walkforward.py:192` excludes val from training mask. But if train.py uses any global statistics (batch norm, feature scaling) computed on the full data before masking, val data leaks into training. Needs explicit audit of train.py's data handling.

**10. OTM detection uses a 2.5-point cliff threshold that causes cost to jump 2x on sub-penny spot movements.**
`simulator.py:107-109`: `abs(intent.strike - underlying_price) > 2.5` determines OTM, doubling spread cost. On a 5-point grid, spot at 5972.50 vs 5972.49 flips a strike between ATM and OTM. A smoother transition or actual bid-ask from data would be more realistic.

**11. ATM-always and ATM-trailing baselines only trade calls -- direction balance not tested.**
`replay.py:573` and `replay.py:737` both use `right="C"`. A model excellent at puts but mediocre at calls beats these baselines easily. No baseline tests whether the model's put-trading adds value. The "isolates neural net value from exit strategy" claim is only testing half the story.

**12. No MODEL_EXIT implementation -- one of three documented exit policies is silently non-functional.**
`schema.py:23` defines `EXIT_POLICIES = ("STOP_TP_TIME", "TRAILING", "MODEL_EXIT")` and `evaluator.md:128-133` describes MODEL_EXIT. But `simulator.py` has no code path for it. If policy or model sets `exit_policy="MODEL_EXIT"`, the position runs to max_hold or EOD with no model-driven exit.

**13. Replay determinism broken by UUID intent_id generation.**
`replay.py:141`: `intent_id=str(uuid.uuid4())`. Each run produces different UUIDs. While this may not affect trade simulation currently, `evaluator.md:249` requires deterministic replay. If intent_id is ever used in comparison or hash, determinism breaks silently.

**14. Score formula rewards reducing losing days over improving trade quality -- misaligned with live deployment needs.**
`min(sortino, 6.0) * positive_day_rate * dd_mult` makes PDR a direct multiplier. A model trading 15 days winning 14 (93% PDR) scores much higher than one trading 40 days winning 30 (75% PDR), even if the latter makes more total money. With minimum 15 traded days, this drives extreme selectivity. A live 0DTE bot needs to trade every market day -- the optimization target and deployment goal are misaligned.

**15. `_intent_to_price_key` bins strikes into coarse offset buckets, losing model precision.**
`replay.py:340-364` rounds strike offsets so that offset 12 and offset 8 both map to `otm10_*_prices`. Two different model strike choices produce identical simulated P&L. The model learns that strike selection doesn't matter for many offsets because the simulation can't distinguish them.

## Related Documentation

- `v2/docs/evaluator.md` -- complete simulation rules and scoring spec
- `v2/docs/baselines.md` -- baseline model definitions
