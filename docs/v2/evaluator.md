# v2 Evaluator Contract

## Purpose

This document defines exactly how TradeIntents are scored during replay.
The evaluator is the authority for model promotion. If this spec is ambiguous,
the loop will optimize the wrong thing.

All rules in this document are implemented in `v2/core/simulator.py` and
`v2/core/metrics.py`. Changes to these rules require a new evaluator version
and reset the best score.

---

## Candidate Universe

At each decision bar, the candidate set is:
- SPX spot rounded to nearest 5 = ATM strike
- ATM +/- 30 points in 5-point increments (13 strikes)
- Both calls and puts = 26 candidates per bar

**Filters (a candidate is excluded if any of these fail):**
- Option bid <= 0 (no market)
- Bid-ask spread > 300 bps of mid (illiquid)
- Option price < $0.05 (penny options, unreliable Greeks)
- Remaining DTE < 0 (expired)

The model may also emit `trade=False` (no-trade), which is always a valid choice.

---

## Fill Model

**Market orders (order_style="MKT"):**
- Entry fill: at the ask price of the next bar after decision
- Exit fill: at the bid price of the bar where exit triggers

**Limit orders (order_style="LMT"):**
- Entry fill: filled if next bar's low <= limit_price (buy side)
- If not filled within 1 bar, order is cancelled (no chase)

**Partial fills:** Not modeled. 0DTE SPX options are liquid enough that
partial fills are not a material concern for single-contract positions.

**Fill timing:** Decision at bar N, fill attempted at bar N+1 open.
The model cannot act on the current bar's close and fill at the same bar.

---

## Spread and Slippage

Spread cost is applied as a round-trip deduction from P&L, not on fill price.
This matches v1 behavior and avoids distorting the price series.

### Base Spread (ATM, normal VIX)

| Time Window | Minutes Remaining | Base BPS |
|-------------|-------------------|----------|
| 9:30-10:00 (open chaos) | 330-390 | 40 |
| 10:00-11:00 (morning) | 270-330 | 30 |
| 11:00-12:00 | 210-270 | 50 |
| 12:00-13:00 (lunch core) | 150-210 | 80 |
| 13:00-14:30 | 90-150 | 50 |
| 14:30-15:30 | 30-90 | 60 |
| 15:30-16:00 (power hour) | 0-30 | 150 |

### Multipliers

- **OTM:** 2.0x base (wider spreads away from ATM)
- **VIX regime:** `1.0 + max(0, vix_regime - 0.3) * 2.0` (up to 2.4x in crisis)
- **Cap:** 500 BPS maximum regardless of multipliers

### Round-Trip Cost

`cost = (entry_spread_bps + exit_spread_bps) / 10000`

Applied to final P&L: `pnl_after_cost = raw_pnl - cost`

---

## Stop Loss Rules

**Check frequency:** Every bar while position is open (intra-bar).

**Trigger condition:** `unrealized_pnl_pct <= -stop_distance`

Where `stop_distance` comes from the TradeIntent's `stop_price` converted to a
percentage of entry premium.

**Precedence:** Stop loss is checked BEFORE take profit on the same bar.
If both would trigger on the same bar, the stop wins (conservative assumption).

**Fill on stop:** Exit at the stop price (assumes stop order was live in market).
No additional slippage beyond the round-trip spread already accounted for.

---

## Take Profit Rules

**Trigger condition:** `unrealized_pnl_pct >= +target_distance`

Where `target_distance` comes from the TradeIntent's `take_profit_price`
converted to a percentage of entry premium.

**Fill on TP:** Exit at the take-profit price.

---

## Trailing Stop Tiers

When exit_policy is "TRAILING", these trailing stop tiers activate:

| Unrealized P&L Reaches | Lock In |
|------------------------|---------|
| +120% | +80% (lock 80% of premium as floor) |
| +80% | +50% |
| +50% | +25% |
| +30% | breakeven (lock entry price) |

Tiers are checked in descending order. Once a tier activates, the stop
ratchets up and never falls back. If price then drops to the locked level,
the position exits at the locked price.

---

## Model Exit Signal

When exit_policy is "MODEL_EXIT":
- Exit signal checked only after `min_hold_bars` (default: 2 bars)
- Model must produce exit_signal > EXIT_GATE_THRESHOLD (default: 0.60)
- If threshold met, position exits at current bar's price

---

## Max Hold and EOD

- `max_hold_bars` from TradeIntent enforced. If the position has been held
  for max_hold_bars without any other exit, it closes at market.
- Hard EOD flatten at bar 389 (15:59 ET). All positions close regardless
  of P&L, stop, or model signal. This is non-negotiable for 0DTE.

---

## Trade Filters

**Cooldown:** 5-bar (5-minute) cooldown after a stop-loss exit before the
next entry is allowed. Prevents revenge trading after losses.

**Max concurrent positions:** 1. No pyramiding, no scaling in. One position
must close before the next can open.

**Time blocks:**
- No entry before bar 30 (9:30-9:59 ET). Opening chaos.
- No entry after bar 330 (15:00+ ET). Power hour, too dangerous for 0DTE longs.
- Lunch suppression (bars 60-240): entries allowed but require higher confidence.

---

## Promotion Score Formula

The promotion score is a single scalar that determines keep/revert decisions.
Based on a **dollar equity curve** starting at $50,000 with $100 SPX multiplier.

```python
def compute_score(metrics: ReplayMetrics) -> float:
    """
    score = min(daily_sortino, 6.0) * positive_day_rate * dd_mult

    Hard gates return negative scores on failure.
    """
    # Hard gates
    if metrics.total_trades < 30: return -1.0
    if metrics.traded_days < 15: return -0.5
    if dir_balance < 0.15: return -0.3
    if metrics.max_account_drawdown > 0.20: return -0.2

    # Ranking
    sortino = min(metrics.daily_sortino, 6.0)
    pdr = metrics.positive_day_rate

    # DD multiplier: 1.0 at <= 8%, linear to 0.0 at 20%
    dd = metrics.max_account_drawdown
    dd_mult = 1.0 if dd <= 0.08 else max(0.0, 1.0 - (dd - 0.08) / 0.12)

    return sortino * pdr * dd_mult
```

### Promotion Rule

`KEEP` if ALL of:
1. `new_score > best_score`
2. All hard gates pass (min trades, min days, direction balance, max DD)
3. Model beats all three baselines (see baselines.md)
4. Evaluator version matches (score_config fingerprint unchanged)
5. Deterministic replay (two runs produce identical scores)

Otherwise: `REVERT`.

### Score Config Fingerprint

The score formula parameters are SHA-256 fingerprinted. Any change to the
formula resets best_score to -5.0 and requires a fresh baseline comparison.

---

## Data Split (4-way)

| Split | Days | Purpose |
|-------|------|---------|
| train_mask | 859 | Model training |
| val_mask | 60 | Checkpoint selection (best val_loss during training) |
| promote_mask | 60 | Keep/revert scoring (model never sees during training) |
| shadow_mask | 20 | Live-readiness eval only (never used for promotion) |

- All splits are date-only (no bar-level mixing)
- Score is computed on `promote_mask` only
- Model trains on `train_mask`, checkpoints on `val_mask`
- Model never sees `promote_mask` or `shadow_mask` during training

---

## Regime Diagnostics (Not Scored)

These are computed and logged but do NOT affect the promotion score:

| Slice | Definition |
|-------|-----------|
| Low VIX | VIX < 15 |
| Medium VIX | 15 <= VIX < 25 |
| High VIX | VIX >= 25 |
| Morning | bars 0-60 |
| Midday | bars 60-240 |
| Afternoon | bars 240-390 |

Purpose: diagnose whether the model is only profitable in one regime
and failing in others. A model that only wins in low VIX is fragile.

---

## Determinism Requirement

Same model + same data + same evaluator version = identical score.

- Random seeds fixed (torch, numpy, python random)
- No stochastic fills
- No random dropout during eval
- Evaluator version = SHA-256 of: fill model code + spread function code + stop/TP logic + score formula

Two replay runs on the same model must produce byte-identical trade logs
and identical scores. If they don't, there is a bug.
