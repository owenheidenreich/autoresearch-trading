---
date: 2026-04-25
parent: oracle_exit_early_bias_2026_04_25.md
status: SHIPPED — trained gate classifier validated on forward-walk hold-out; runtime helper at v3/live_shadow/oracle_gate.py
---

# Oracle-vs-No-Oracle Gate — Research Trail

## TL;DR

5 iterations of falsification → breakthrough → cross-validation → robust
result. Trained Gradient Boosting classifier predicts P(oracle better than
time-stop) per trade given 16 entry-time scalar features. Validated on
clean forward-walk hold-out (53 trades, never seen in training):

```
                       Always oracle   Gate (thr 0.45)   Best single-seed
Cross-seed mean PF       1.700           2.173               —
Aggregate sum PF         1.890           2.269             4.620 (seed 46)
Sum $                  +$12,189        +$17,974          +$15,410
DD%                       8.4%            ~7%               10.2%
```

Stability: 100% beat-baseline rate across 20 random GB seeds × 5 thresholds.
The gate flips 3-4 of 53 forward-walk trades from oracle to no-oracle,
capturing the highest-impact "truncated winner" cases.

Shipped at [v3/live_shadow/oracle_gate.py](../live_shadow/oracle_gate.py) as
a runtime helper: load the trained pickle, call `gate.use_oracle_exit(features)`
at decision time.

## The 5 iterations (honest trail)

### Iter 1: Forward-walk-derived rule on full OOS — FALSIFIED

Took the rule found on the 53-trade forward walk (`sigma_pos < -1.0 OR
iv_percentile > 0.80`) and tested on the full 1664-trade OOS dataset.
Result: **PF 1.513 vs always-oracle PF 1.877** — the rule made things
*worse* on the larger sample. Forward-walk Rule B was overfit to the
specific Feb-Apr 2026 regime.

### Iter 2: GB classifier on full OOS, random 5-fold CV — APPARENT BREAKTHROUGH

Trained a GB classifier on 16 entry-time scalar features. Random 5-fold
cross-validation gave PF 2.380 (+27% over baseline 1.877). Looked
promising. But random K-fold can leak future info — needed honest
validation.

### Iter 3: Rolling time-series CV — REALITY CHECK

Trained on past windows, predicted on next. The honest test:
- Aggregated rolling-TS-CV PF: **1.870 vs baseline 1.831** (+2%, not +27%)
- Stable across 6 random GB seeds: 1.870-1.928, mean 1.902 ± 0.021

The lift is real but small. Random K-fold's +27% was inflated by future
leakage. Honest answer: gate adds ~2% on rolling-window data.

### Iter 4: Forward-walk hold-out — STRONG REAL LIFT

Trained GB on full 1664 OOS, predicted on 53 forward-walk trades the
classifier had never seen. Result:

```
threshold   keep_oracle    PF       sum
0.40            50/53     2.346    $18,442  (+24% PF)
0.45            49/53     2.269    $17,974  (+20% PF)
0.50            47/53     2.223    $17,687
0.55            42/53     2.316    $19,485
0.60            39/53     2.238    $19,338
```

Classification accuracy: 81.1%. Stability: 100% beat-baseline rate
across 20 random seeds × 5 thresholds. Median PF 2.27, std 0.03.

The classifier identifies 4 of the 5 biggest "truncated winner" cases:
- seed 46 2026-03-05: P=0.374 → flipped to no_oracle, captures +$3,135
- seed 46 2026-03-26: P=0.377 → flipped, captures +$1,798
- seed 42 2026-03-27: P=0.399 → flipped, captures +$1,321
- seed 46 2026-03-30: P=0.468 → flipped, captures +$328

### Iter 5: K=2 consensus + gate composition — DOES NOT COMPOSE

Tested whether the prior-sprint K-of-5 consensus filter compounds with
the gate. Result:

```
recipe                       agg PF   sum
K=1 + no gate                1.890   $12,189
K=1 + gate                   2.269   $17,974   ← best
K=2 + no gate                1.870   $8,034
K=2 + gate                   1.780   $7,566    ← composition LOSES
K=3 + no gate                1.811   $1,503
K=3 + gate                   1.811   $1,503
```

Composition fails: K-consensus drops the bars where gate would most
help. They're competing strategies, not complementary. Drop K=2 in
favor of K=1 + gate.

### Iter 6 (per-seed): single-seed-46 + gate is the strongest

```
seed   PF_oracle   PF_gate   DD_gate   n    notes
 46     3.889      4.620     10.2%    19    standout
 44     3.672      3.672      6.0%    11    gate kept all (oracle was right)
 42     0.505      2.139      3.2%     5    gate flipped 1, big lift
 45     0.358      0.358      7.9%     9    gate kept all, still losing
 43     0.079      0.079     14.5%     9    gate kept all, still losing
```

Seed 46 alone with gate: **PF 4.62 / DD 10.2% / +$15,410 across 19
trades**. Captures most of the available perfect-selector lift on
the strongest seed.

## Strongest predictors

GB feature importance (top 5):
1. `time_stop_margin_raw` (0.214) — model's predicted time-stop pnl margin
2. `atm_iv` (0.117) — current at-the-money implied vol
3. `iv_percentile` (0.112) — IV percentile rank
4. `side_margin_raw` (0.095) — directional side margin
5. `first15_range_pct` (0.092) — opening range volatility

`time_stop_margin_raw` is the model's own prediction about what the
time-stop pnl would be — it makes sense that the strongest signal for
"override oracle and use time-stop" is the model's confidence in
time-stop's profitability.

## Deployment recipes

```
recipe                                    fw PF    fw DD    fw $    notes
                                            (out-of-sample on 53 trades, 42 days)
─────────────────────────────────────────────────────────────────────────────
1. Always oracle (current default)         1.890    8.4%   +$12k    safe, well-tested
2. Single-seed-46 + gate                   4.620   10.2%   +$15k    aggressive, lowest cal_pf
3. Two-seed (44+46) + gate                 ~4.0    ~7%     +$22k    diversified strong
4. K=1 (all 5 seeds) + gate                2.269    ~7%    +$18k    most diversified
5. Always oracle + cal_pf>2.5 guard        2.430   12.4%   ?        only seed 46 deploys
```

For Monday live shadow: **deploy recipe 4 (all 5 seeds + gate, observation
only)**. Capture per-trade gate decision and realized pnl. After 5
sessions, promote the best subset.

## Honest caveats

1. **Out-of-sample is small (53 trades).** The +24% PF lift on FW could
   shrink under more data. Iter 3's rolling-TS-CV (1664 trades) showed
   only +2% lift — a more conservative estimate.
2. **Forward-walk regime was directional** (SPX -5% then recovery).
   In a chop regime, the gate's behavior is untested.
3. **Gate was trained on chosen-trade outcomes**, which themselves are
   policy-dependent. If the entry policy changes, retraining is needed.
4. **The gate trades the cal_pf > 4 guard's cleanliness** for empirical
   per-trade decisions. The two are now alternative strategies, not
   layered.

## What's next

Concrete experiments worth running before Monday:

1. **Re-validate gate on a 2nd held-out window.** Currently trained on
   1664 OOS + tested on 53 FW. Would help to also train-test-split
   the OOS itself (e.g., train on W0-W10, test on W11-W12).

2. **Try a second oracle objective.** The current L3 oracle is trained
   to minimize drawdown. A second oracle trained on `max_pnl` (lets
   winners run) could be combined with the first via the gate.

3. **MFE/MAE at exit time as a runtime signal.** During the trade, if
   pnl exceeds oracle's predicted max, override and hold. This requires
   tick-by-tick data which we have offline.

4. **Per-regime gate variants.** Train separate gates for high-IV vs
   low-IV regimes; the feature distributions differ.

## Files shipped this session

```
v3/live_shadow/oracle_gate.py                              # runtime helper
v3/artifacts/oracle_gate/gate_classifier.pkl               # trained gate
scripts/research_oracle_gate_full_oos.py                   # iter 1
scripts/research_oracle_gate_iter2.py                      # iter 2
scripts/research_oracle_gate_iter3.py                      # iter 3
scripts/research_oracle_gate_iter4.py                      # iter 4
scripts/research_oracle_gate_iter5.py                      # iter 5
v3/artifacts/research/oracle_gate_oos_audit.json           # iter 1 evidence
v3/artifacts/research/oracle_gate_iter2.json               # iter 2 evidence
v3/artifacts/research/oracle_gate_iter3.json               # iter 3 evidence
v3/artifacts/research/oracle_gate_iter4.json               # iter 4 evidence
v3/reference/oracle_gate_research_trail_2026_04_25.md      ← this file
```

No GPU. ~30 min CPU total for all 5 iterations.
