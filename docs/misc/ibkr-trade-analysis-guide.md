# IBKR Trade Analysis Guide

Reference for an AI analyzing live IBKR paper trading sessions. Read this first, then the files it points to.

## The Core Problem (2026-03-25)

The model takes 10-20+ trades per day. It should take ~1. Two root causes:

1. **Gate head fires on 100% of bars.** Gate P(trade) ranges 0.81-0.92 across all 76 inferences today. The `min_trade_prob` threshold is 0.55. The gate never says "don't trade." Every bar that has no open position generates an entry.

2. **Zero cooldown after position close.** When a stop/TP fills, `current_position_id` resets to `None` immediately (service.py:559). The very next bar generates a new entry. Evidence: in `v6_bugfix4_session.jsonl`, five entries fire in 13 minutes (close→re-enter in 3 seconds each time).

3. **Session restarts reset position tracking.** Each session starts with `current_position_id = None`. If a previous session left a position open in IBKR, the new session doesn't know about it and opens another. `session4` placed 6 entries in 8 minutes this way.

## File Map for Trade Analysis

### Audit Data (read these to see what happened)

| File | What it contains |
|------|-----------------|
| `results/live/audit.jsonl` | Only `context_refresh` events (pre-session). Not useful alone. |
| `results/live/v6_*.jsonl` | Per-session audit logs. **These have all the real data.** |
| `results/live/trades.jsonl` | Subset of trade records written by `_emit_position_closed`. |

**To load all sessions**, read every `*.jsonl` in `results/live/`. Each line is JSON with `{ts, event, payload}` (service.py) or `{ts, event, session_id, payload}` (execution.py).

### Event Types (what each audit event means)

| Event | Source | Meaning |
|-------|--------|---------|
| `session_start` | service.py:459 | New session began. Payload: `session_id`, `seed_spx`, `dry_run`, feature counts. |
| `session_end` | service.py:809 | Session ended. Payload: `processed_minutes`, `signals_generated`, `entries_applied`, `exits_applied`. |
| `model_inference` | service.py:591 | Model ran on a bar. Payload: `action`, `gate_trade_prob`, `confidence`, `direction_probs`, `reason_codes`. |
| `entry_intent` | service.py:617 | Gate said TRADE, intent was created. Payload: `contract`, `stop_price`, `take_profit_price`, `action`, `qty`. |
| `entry_intent_applied` | service.py:650 | Intent was sent to execution engine. Payload: `position_id`, `intent_id`. |
| `entry_live` | execution.py:429 | OCO bracket order placed on IBKR. Payload: order details, prices. |
| `entry_dry_run` | execution.py:332 | Simulated fill (dry_run mode). Payload: `fill_price`, `fill_status`, `contract`. |
| `ib_order_status` | execution.py:169 | IBKR order state change. Payload: `status` (PreSubmitted, Submitted, Filled, Cancelled). |
| `ib_exec_details` | execution.py:210 | IBKR fill confirmed. Payload: `exec_id`, `price`, `shares`, `side`. |
| `position_closed` | service.py:277 | Position fully closed. Payload: `pnl_pct`, `pnl_dollar`, `exit_reason`, `bars_held`, entry/exit prices. |
| `pnl_update` | execution.py:561 | P&L snapshot after close. Payload: `trade_pnl_pct`, `trade_pnl_dollars`, `session_pnl_dollars`. |
| `bar_snapshot` | service.py:507 | Feature completeness for a bar. Payload: `completeness`, `missing_feature_names`, `staleness_seconds`. |
| `context_refresh` | service.py:344 | Historical context bundle loaded. Payload: `bundle_path`, `bars`, `as_of_date`. |

### Signal-to-Trade Pipeline (the code path)

Read these files in order to understand how a bar becomes a trade:

#### 1. `training/live/features.py` — Bar Assembly
- `FiveSecondMinuteAggregator` builds 1-min bars from 5-sec IBKR data
- `LiveFeatureEngine.compute_snapshot()` returns normalized feature window + completeness score
- Bars with completeness < 0.65 are skipped (no inference)

#### 2. `training/live/decision.py` — Model Inference & Intent
- **Lines 240-307: `infer()`** — Runs gate head + direction head
  - Gate head outputs 2 logits: [NO_TRADE, TRADE]. Argmax decides.
  - If TRADE: direction head picks best of 6 actions (CALL_ATM, CALL_OTM5, ..., PUT_OTM10)
  - Position state tensor (7-dim) is passed to gate: `[in_trade, bars_held, unrealized_pnl, account_health, loss_streak, best_pnl, bars_since_high]`
  - **When position just closed**: in_trade=0, bars_held=0, unrealized_pnl=0 -- gate has NO memory of prior trade
- **Lines 314-385: `build_entry_intent()`** — Filters and builds order
  - Rejects if `gate_trade_prob < min_trade_prob` (default 0.55)
  - Rejects if `bar_of_day < NO_TRADE_BEFORE_BAR` (first 30 bars, 9:30-9:59 ET)
  - Stop: dynamic based on IV/VIX or risk head
  - Take profit: `6.0x` entry mid (extremely wide -- effectively no TP)
  - Entry: LMT at `1.05x` mid (5% above mid)

#### 3. `training/live/service.py` — Main Loop (the orchestrator)
- **Line 605: `if current_position_id is None:`** — Only checks this variable. No cooldown, no trade count limit, no time-since-last-trade check.
- **Line 635: `current_position_id = state.position_id`** — Blocks further entries while position is open.
- **Line 559: `current_position_id = None`** — Immediately allows new entry after close.
- **NO reference to `STOP_COOLDOWN_BARS`** — This constant (=5) exists in prepare.py for training but is never imported or checked in the live service.

#### 4. `training/live/execution.py` — Order Placement
- **Lines 285-293:** Only blocks entries on kill switch or daily loss limit. No cooldown check.
- **Lines 295-445:** Places OCO bracket (parent + stop + TP) on IBKR.
- **Lines 193-255:** Fill callbacks update position state, trigger closure detection.

#### 5. `training/prepare.py` — Constants
- `STOP_COOLDOWN_BARS = 5` (line 64) — **training only, not enforced live**
- `NO_TRADE_BEFORE_BAR = 30` (line 65) — enforced in decision.py
- `ACTION_DO_NOTHING = 0`, `ACTION_EXIT = 7`

### What to Look For When Analyzing Trades

#### Trade Frequency
```python
# Count entries per session — should be 0-2, not 5-20
for each session_start → session_end window:
    count entry_intent_applied events
```

#### Gate Behavior
```python
# Gate should say NO_TRADE (action=0) on most bars
for each model_inference:
    check gate_trade_prob — should be < 0.5 on most bars
    check action — should be 0 (DO_NOTHING) most of the time
```
**Today's data**: gate_trade_prob is 0.81-0.92 on ALL 76 bars. Never once says don't trade.

#### Close-to-Reentry Timing
```python
# Time between position_closed and next entry_intent should be >> 1 bar
for each position_closed:
    find next entry_intent in same session
    compute time delta — should be > 5 minutes
```
**Today's data**: reentry happens 3 seconds after close, consistently.

#### Session Boundary Orphans
```python
# Entries applied in session N with no matching position_closed
# These are "orphan positions" — IBKR may still hold them
for each entry_intent_applied:
    check if position_id appears in any position_closed event across ALL files
```
**Today's data**: 16 entries have no matching close event.

#### Stop Distance
```python
# Check if stops are too tight (causing rapid stop-outs → re-entries)
for each entry_intent:
    stop_pct = (entry_price - stop_price) / entry_price
    # If stop_pct < 5%, stops will fire constantly on 0DTE options
```

### Key Metrics to Compute

| Metric | How | Healthy Range |
|--------|-----|--------------|
| Trades per session | count `entry_intent_applied` per session | 0-2 |
| Gate selectivity | % of inferences with `gate_trade_prob < 0.5` | > 70% |
| Avg gate_trade_prob | mean across all `model_inference` events | 0.3-0.5 |
| Close-to-reentry gap | time between `position_closed` and next `entry_intent` | > 5 min |
| Stop-out rate | `position_closed` with `exit_reason` containing "stop" / total closes | < 50% |
| Win rate | closes with `pnl_pct > 0` / total closes | > 40% |
| Orphan positions | entries with no matching close | 0 |

### Root Cause Checklist

When investigating over-trading, check these in order:

1. **Is the gate always firing?** → Read `model_inference` events, check `gate_trade_prob` distribution. If always > 0.5, the gate head is not selective. This is a model/training problem.

2. **Is there cooldown after close?** → Check service.py for any cooldown logic after `current_position_id = None`. Currently there is none.

3. **Are sessions restarting and losing position state?** → Count `session_start` events. If > 1 per day, each restart resets position tracking. Check if IBKR positions from prior sessions are still open.

4. **Are stops too tight?** → Compute stop distance from `entry_intent` payloads. If stops are < 5% on 0DTE options, they'll fire within minutes due to normal volatility.

5. **Is the TP unreachable?** → TP is set at `6.0x` entry mid. On 0DTE options, this is essentially never reached. All exits come from stops or model exits.

## Files Quick Reference

| Purpose | File |
|---------|------|
| Main loop / orchestrator | `training/live/service.py` |
| Gate + direction inference | `training/live/decision.py` |
| Order execution (IBKR) | `training/live/execution.py` |
| Feature assembly | `training/live/features.py` |
| Contract resolution | `training/live/resolver.py` |
| Training constants | `training/prepare.py` |
| Model architecture | `training/train.py` (or `training/best_train.py`) |
| Domain knowledge | `docs/domain/0dte-domain-knowledge.md` |
| Trading rules | `docs/domain/pickles-trading-knowledge.md` |
| Dashboard | `tools/live_dash.py` |
| Session audit logs | `results/live/*.jsonl` |
