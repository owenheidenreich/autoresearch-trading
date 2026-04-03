# ART² Principles (v1)

Subordinate to program.md on model contract, data contract, architecture.
This file governs the bot's purpose, integrity, and evolution.

---

## 1. What This Bot Is

A long-options SPX 0DTE trading bot. It sees greeks, price action, indicators, volume -- 39 features embedded in its weights through a transformer backbone. It makes complete trading decisions:

- **No trade?** (most bars)
- **Call or put, which strike?** (ATM, OTM5, OTM10)
- **Where to stop?**
- **When to exit?**

It learns from the market. The goal is profitability. It trades rarely. It follows the principles in `docs/domain/`.

It is NOT a direction predictor with trading rules bolted on. It IS a trader.

---

## 2. Current State (What Is True in Code Today)

### Architecture (grounded)
- 5-head TradingModel: market prediction, entry gate, risk params, exit signal, 6-class direction
- Direction labels already use real stopped option P&L (argmax of 6-leg P&L in prepare.py)
- Replay system simulates real trades with real option prices, dynamic stops, spread costs
- Validation is date-based with purge day (no leakage)

### Scoring (prediction-based, not P&L)
- `score = dir_accuracy * (1 + max(0, rank_corr))` (train.py:584) -- rewards predicting SPX direction, not trading profitability
- inner_loop.py promotes models when this score improves (inner_loop.py:585)
- Replay P&L metrics exist (PF, win rate, drawdown) but are NOT used in promotion decisions
- `_score_config` dict (train.py:120) is cryptographically locked by inner_loop.py:88, but this lock guards metadata that the active score formula does not actually use -- it is locking dead config

### Label gaps (4 of 5 heads use proxies, not trade outcomes)
- **Entry gate**: sigmoid of `(MFE - |MAE|) / ATR` -- path quality, not "did this entry make money"
- **Exit**: `1.0` when 5-bar forward risk exceeds reward 2x -- SPX movement, not option P&L
- **Risk params**: ATR-normalized price extrema -- not option-greek-aware
- **Direction**: argmax of stopped option P&L -- THIS ONE IS REAL
- **Market prediction**: SPX return targets at 15/30/60 bars -- auxiliary, fine as-is

### Policy parity gap (replay != live)
- Replay: model outputs direction from 6-class head, then `should_enter()` overwrites it with ATM call/put based on `pred_return_30` sign (replay.py:1865, trading_rules.py:146). The model's 6-class direction choice is never tested in replay.
- Live: uses direction head directly (`_V18_DIR_CLS_TO_ACTION[dir_cls]` in decision.py:203)
- `select_strike()` and `should_exit()` are defined in trading_rules.py but have no live callers -- dead policy code
- The repo currently has more than one effective policy

### ATR stop-scaling mismatch (bug)
- Training labels compute `risk_stop_distance` in ATR units from raw ATR (prepare.py:3706)
- Replay loads pre-normalized data.pt features and converts stop using normalized `atr_14` (replay.py:1937)
- Live uses raw latest-row ATR (service.py:628 into decision.py:277)
- Replay stops and live stops operate on different scales. MUST be fixed before replay P&L becomes the promotion signal.

### Time-of-day inconsistency (3 different policies)
- prepare.py: zeros gate/exit labels during lunch and power hour
- trading_rules.py: blocks power hour, doubles confidence during lunch
- live decision.py: only blocks pre-10am
- Three incompatible implementations of the same concept

### Risk head partially connected
- `risk_stop_distance` drives actual stop placement in both replay and live
- `risk_target_distance` is unused -- take profit hardcoded to `entry * 6.0` (effectively disabled)

### Cost realism is proxy-based
- Spread estimated from `action_cost_bps` in data.pt or adaptive spread fallback
- Not actual bid/ask execution prices -- a reasonable proxy but not ground truth

### Observability drift
- `test_live_decision.py` passes
- `test_observability_upgrade.py` fails to collect (missing `tools.ingest_evidence`) -- anti-cheating layer is partially broken
- Metrics path is fragmented: run_loop.py prefers METRICS_JSON, inner_loop.py falls back to metrics.json, train.py prints plain text only. No single structured metrics contract.
- Most P&L anomaly thresholds in run_loop.py are data-starved because train.py only emits prediction metrics. Replay metrics are not in the output path.

---

## 3. Target State (What We're Building Toward)

### Profitability as the reward signal
- The model's decisions flow through replay. Replay computes real P&L. That P&L becomes the primary promotion signal.
- Score formula evolves: replay profit factor and risk-adjusted return replace (or dominate over) direction accuracy.
- The model learns patience, time-of-day, strike selection BECAUSE profitable behavior is rewarded.

### Policy parity
- Replay executes the SAME policy as live -- model's direction head choice respected, not overwritten
- One policy, two execution environments (simulation and real)

### Labels derived from trade outcomes
- Entry gate: "did entries at this bar lead to profitable trades?" (from replay, not MFE/MAE)
- Exit: "would exiting here maximize option P&L?" (not SPX path quality)
- Risk params: optimal stop/target from realized trade outcomes
- Direction: keep as-is (already P&L-based)

### Time-of-day learned, not hardcoded
- Remove label zeroing during lunch/power hour
- Remove hardcoded time blocks
- The model discovers these patterns through P&L

### Risk head fully connected
- `risk_target_distance` drives actual profit targets
- Model learns when to take profit

### What a good model looks like (paper-trade ready)
- Profit factor > 1.2 in replay across >= 200 validation days
- Trades 0.5-5 times per day on average
- Win rate > 50% with realistic spread costs
- Survives high-VIX and low-VIX regimes
- Uses both calls and puts (direction collapse < 85%)
- Stop loss rate < 40%
- Maximum drawdown < 25% of peak equity

### What a good model looks like (live-ready)
- All of the above, plus:
- Profit factor > 1.5
- Positive P&L in at least 60% of trading weeks
- Edge persists in most recent 30 trading days

---

## 4. Trading Principles (From Domain Docs)

These are the principles the model should learn organically through the P&L reward signal. They serve as sanity checks -- if the model violates them, the training signal is broken.

1. **Patience pays.** Most bars = no trade. Overtrading bleeds theta and spread costs. "There is always another trade."
2. **Time is dominant.** Morning has strongest trends. Lunch is low-edge. Power hour is high-gamma danger. The model discovers this from P&L data.
3. **Risk is pre-defined.** Stop loss fires before any other exit. Position size is a function of account and risk, not conviction.
4. **Always take profits.** First test of target = exit. Trailing stops lock in gains at tiers. Holding through profit into loss destroys P&L.
5. **Gamma awareness.** ATM gamma concentrates near expiry. Put skew gives structural edge. The model learns from option P&L outcomes across strikes.
6. **VIX regime matters.** Strategy must adapt to volatility regime. A model that only works in low-VIX is useless.
7. **Confluence over single signals.** 39 features provide confluence. The model learns which combinations have edge.
8. **Losses are data.** A stop loss is the model learning where its edge doesn't exist. Optimize the win/loss ratio, not the absence of losses.
9. **Operate like a casino.** Take every valid signal. Measure over 20+ trade windows, not individual trades. Accept the random distribution of wins and losses for any given edge.
10. **Pre-define risk, then accept it.** Douglas's core: objectively identify edges, predefine risk, completely accept it or skip the trade. No second-guessing after entry.

---

## 5. Integrity Rules (Anti-Cheating)

1. **No future information.** Features are causal only. Labels use forward data as targets, never features.
2. **Real costs.** Every trade pays bid-ask spread. Currently proxy-based (action_cost_bps or adaptive model). Target: actual bid/ask from option data.
3. **No single-day specialists.** Must profit across 3+ distinct dates.
4. **No direction collapse.** Must use multiple direction classes (< 85% one direction).
5. **Anomaly flags auto-revert.** Critical anomalies revert even if score improves (run_loop.py OBSERVABILITY_CONFIG).
6. **Validation integrity.** Train/val split is date-based with purge day. No leakage.

---

## 6. Training Governance

### Session limits
- Max 50 experiments or 6 hours per session (hard ceiling, prevents GPU waste)
- One change per experiment
- Every experiment has a written hypothesis BEFORE GPU spend

### Stop rules

| Rule | Trigger | Action |
|------|---------|--------|
| S1 Goal | Model meets paper-trading criteria (Section 3) | STOP, run full backtest, celebrate |
| S2 Plateau | No >= 2% score improvement in 8 experiments | STOP, change research axis |
| S3 Budget | 50 experiments or 6 hours | STOP, hard ceiling |
| S4 Diminishing | Last 3 KEEPs each < 1% improvement | STOP, current approach exhausted |
| S5 Stuck | 5 consecutive reverts | STOP, read replay data, new hypothesis |
| S6 Crashes | 3 of last 5 experiments crashed | STOP, infra/code problem |

### When stopped
Log findings to lab_notebook.md, summarize what worked, propose next directions, wait for human.

---

## 7. Migration Path (Current -> Target)

Each phase is a distinct class of change with its own verification requirements.

### Phase 0: Fix execution-parity bugs
Before anything else.
- Fix ATR stop-scaling mismatch: replay uses normalized atr_14, live uses raw ATR. Align both.
- Establish a replay baseline with current best model so PF comparisons are meaningful.
- Files: replay.py, decision.py, prepare.py
- Verification: replay stop distances match live stop distances for same inputs

### Phase 1: Policy parity (unify on CURRENT hardcoded policy)
- Remove `should_enter()` override in replay -- let model's direction head drive action choice
- Decide fate of `select_strike()` and `should_exit()` in trading_rules.py (dead code in live path)
- Unify time-of-day handling on the CURRENT hardcoded policy: make replay and live apply the same time blocks. Do NOT remove them yet -- that is Phase 4.
- Connect `risk_target_distance` to actual profit targets (replace `entry * 6.0`)
- Files: replay.py, trading_rules.py, decision.py
- Verification: replay a sample date with old and new policy, compare trade logs

### Phase 1.5: Metrics contract / observability repair
- Establish a single structured metrics output path (METRICS_JSON)
- Add METRICS_JSON output to train.py with placeholder keys for replay metrics (PF, win_rate, TPD, drawdown, account_balance) as null/0 until Phase 2 populates them
- Fix or remove test_observability_upgrade.py (missing tools.ingest_evidence dependency)
- Ensure inner_loop.py parses structured METRICS_JSON as primary path, plain text as fallback
- Files: train.py, inner_loop.py, run_loop.py, tests/
- Verification: inner_loop.py correctly ingests structured metrics from a test training run

### Phase 2: Wire replay P&L into promotion scoring
- Add replay run to train.py evaluation (after supervised metrics)
- Populate the replay metric keys added in Phase 1.5 with actual values
- Add replay PF to score formula (weighted blend with prediction score)
- Requires: best-score reset (new formula = incompatible scores)
- Files: train.py, inner_loop.py
- Verification: run 3 experiments, confirm keep/revert reflects P&L quality. Confirm anomaly thresholds in run_loop.py are now data-fed.

### Phase 3: Trade-outcome labels
- Replace MFE/MAE entry gate labels with "did this entry produce a profitable trade?"
- Replace SPX-based exit labels with "would exiting now maximize option P&L?"
- Replace ATR-normalized risk targets with optimal stop/target from realized outcomes
- Requires: prepare.py rebuild, fresh start (new labels = incompatible semantics)
- Files: prepare.py, train.py
- Verification: label distribution sanity checks, fresh training run, replay P&L comparison

### Phase 4: Model-learned time-of-day (remove training wheels)
- Remove hardcoded time-of-day label zeroing in prepare.py
- Remove hardcoded time blocks in trading_rules.py and decision.py
- The model now has full freedom to trade any bar; P&L reward teaches it where edge exists
- Score formula shifts to P&L-dominant (PF primary, direction accuracy secondary or removed)
- Requires: fresh start, prepare.py rebuild
- Files: prepare.py, trading_rules.py, decision.py, train.py
- Verification: model organically learns time-of-day patterns visible in trade logs

Each phase requires a fresh replay baseline before trusting PF comparisons. Phases 3 and 4 require fresh training starts.
