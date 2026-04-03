# v2 Baselines and GPU Spend Gates

## Hard Rule

No GPU run starts unless ALL of the following pass locally first.
This is not a guideline. It is a gate enforced by the experiment orchestrator.

---

## Pre-GPU Checklist

### 1. Smoke Tests Pass

```bash
pytest tests/ -x --timeout=60
```

All existing tests must pass. A failing test means the pipeline is broken
and GPU time would be wasted.

### 2. Feature Contract Tests Pass

- Feature array shape is (N, 39)
- Feature names match FEATURE_NAMES exactly
- Normalized features are in [-5.0, 5.0] range (clipped)
- No NaN values in mandatory features (minutes_to_close, vix_regime)

### 3. Replay Determinism Test Passes

Run the same model on the same data twice. The scores must be identical.

```python
score_1 = replay(model, data, seed=42)
score_2 = replay(model, data, seed=42)
assert score_1 == score_2, f"Non-deterministic replay: {score_1} vs {score_2}"
```

If this fails, there is a bug in the simulator (random state leak, floating
point non-determinism, or stochastic model behavior during eval).

### 4. Model Beats Random Baseline

**Random baseline:** At each bar with a valid candidate set, flip a coin
(50% trade, 50% no-trade). If trading, select a random candidate from the
universe. Use fixed stop=30%, target=50%, max_hold=120 bars.

The model's replay PF must exceed the random baseline's PF on the same
validation days. The random baseline is computed once (averaged over 100
random seeds) and cached.

### 5. Model Beats ATM-Always Baseline

**ATM-always baseline:** At bar 30 every day, buy 1 ATM call. Fixed stop=30%,
target=50%, max_hold=120 bars. No model involved.

This baseline tests whether the model adds value beyond "buy ATM at open
and hope." If the model can't beat this, it has learned nothing useful.

### 6. Model Beats Simple-Rules Baseline

**Simple-rules baseline:**
- If 5-bar momentum > +0.15%: buy ATM call
- If 5-bar momentum < -0.15%: buy ATM put
- Otherwise: no trade
- Fixed stop=25%, target=40%, max_hold=60 bars
- Cooldown: 10 bars after any exit
- No entry before bar 30 or after bar 300

This baseline tests whether a learned model beats a hand-coded heuristic.
It is deliberately simple. If the model can't beat this, the learning
objective is wrong.

---

## Baseline Caching

Baselines are computed once on the current validation set and stored as:

```
results/baselines/
  random_baseline.json       # avg PF, WR, trades over 100 seeds
  atm_always_baseline.json   # PF, WR, trades
  simple_rules_baseline.json # PF, WR, trades
  baseline_meta.json         # data fingerprint, evaluator version, date computed
```

Baselines are recomputed when:
- The validation set changes (new data added, split date moves)
- The evaluator version changes (spread model, stop rules, etc.)
- Manually requested

---

## GPU Budget Gates

Once the pre-GPU checklist passes and a GPU session starts:

| Gate | Limit | Action |
|------|-------|--------|
| Session time | 6 hours max | Auto-stop |
| Experiment count | 50 experiments max | Auto-stop |
| No-improve streak | 8 consecutive reverts | Auto-stop |
| Stale plateau | Best score unchanged for 3 hours | Auto-stop |
| Crash storm | 3 consecutive crashes | Auto-stop, log for human review |

These gates are enforced by `v2/ops/inner_loop.py`. The agent cannot
override them.

---

## Cheap Local Research Loop

Before any full historical run or GPU deployment, run this sequence locally:

1. **Tiny debug dataset:** 5 days of data, 1 epoch, verify loss decreases
2. **One-day replay smoke:** Run replay on a single day, verify trade log makes sense
3. **Five-day replay smoke:** Run on 5 days, verify PF/WR/trades are reasonable
4. **Baseline comparison:** Compare against cached baselines on same 5 days
5. **Deterministic rerun:** Run step 3 twice, verify identical scores

If any step fails, do not proceed to full training. Fix the issue first.

Total local time: under 10 minutes on CPU. Cost: zero.
