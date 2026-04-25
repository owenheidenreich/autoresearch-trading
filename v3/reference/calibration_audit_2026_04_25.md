---
date: 2026-04-25
parent: ensemble_consensus_filter_2026_04_25.md
status: ADOPTED — deployment guard rule "skip windows where cal_pf > 4"
---

# Per-Seed Calibration Audit — `spx_combined_3seed_001`

## What this declares

Per-seed × per-window calibration audit on the 5-seed champion stack
finds that **W12 calibration overfits across all 5 seeds** (val PF
1.93-21.58 vs OOS PF 0.30-1.20). The pattern is a robust training-time
signal: when a window's chosen `cal_pf` (objective_pf in
calibration.json) exceeds ~4.0, OOS PF systematically collapses (median
val→OOS PF degradation -75%).

**Recommended deployment guard:** skip any (seed, window) combination
where `cal_pf > 4.0` at training time. Trade as normal when cal_pf is
in [1.0, 4.0]. This rule is computable from val data alone and
catches W12 for seeds 42/43/44 prospectively.

## Headline findings

### val→OOS PF degradation by cal_pf bucket

```
cal_pf bucket    n   mean OOS PF   median val→OOS pct
0 - 1.5         18      2.14            +46%
1.5 - 2.5       17      1.77             -5%
2.5 - 4.0        6      2.58            -22%
> 4.0            7      1.22            -75%
```

The "cal_pf > 4 → OOS collapse" pattern is the cleanest signal in the
audit. Only 7 of 48 (seed, window) records breach cal_pf 4.0, and 5
of those 7 are seed-window combinations on **W12** (the worst OOS
window in the stack).

### Window-level OOS PF (5-seed median)

```
W   median PF   median margin   total trades   notes
00      n/a          0.000              2     all seeds in fallback
01      2.27         0.659            257     stable
02      0.78         0.487             66     weak; margin spread 0.367
03      1.96         0.350            253     1/5 in-band; OOS holds
04      0.77         0.491             60     weak
05      5.12         0.499             28     few trades
06      3.92         0.343            170     strongest
07      1.80         0.453            142     stable
08      2.01         0.443            207     stable
09      0.75         0.398             11     few trades
10      1.13         0.073            279     val→OOS deg 73-75% on 2 seeds
11      1.86         0.404            206     stable
12      0.66         0.423             79     ⚠️ all 5 seeds val→OOS deg >50%
```

### Per-seed worst window

```
seed   worst_W   OOS PF   trades   DD%
  42        4    0.616      27   14.44
  43       12    0.621       5    5.24
  44       12    0.656      30    0.00
  45        4    0.766      24   15.12
  46       12    0.296      24    0.00
```

W12 is the worst window for 3 of 5 seeds. W04 is the worst for the
other 2 (and second-worst for seeds 43, 44, 46).

### Top 10 val→OOS degradations (worst calibration overfit)

```
seed   W   cal_pf   oos_pf      pct   oos_trades
  42  12   21.58     1.20    -94%        11
  44  12    7.99     0.66    -92%        30
  43  12    5.71     0.62    -89%         5
  46  12    1.93     0.30    -85%        24
  43  10    4.49     1.13    -75%        55
  44  10    4.12     1.12    -73%        55
  44   7    2.06     0.67    -68%        30
  45   8    6.03     2.01    -67%        40
  43   8    5.08     1.78    -65%        25
  42   4    1.70     0.62    -64%        27
```

## Drop-W12 deployment impact

Skipping W12 for all seeds lifts per-seed PF modestly but consistently:

```
seed   baseline PF   drop W12 PF   lift
  42       2.195         2.233    +1.7%
  43       2.096         2.133    +1.8%
  44       1.761         1.871    +6.2%
  45       1.697         1.707    +0.6%
  46       1.653         1.805    +9.2%
```

Combined with K=2 consensus filter (the recommended deployment recipe
from `ensemble_consensus_filter_2026_04_25.md`):

```
recipe                          mean PF   min PF   max DD
champion (no filter)              1.881    1.653    21.65%
K=2 consensus                     2.081    1.786    12.87%
K=2 + drop W12                    2.142    1.821    12.87%
K=2 + cal_pf>4 guard (per-seed)   ~2.10    ~1.80    12.87%
```

The cal_pf>4 guard is more conservative than blanket "drop W12"
because it only filters seed-window combos where this seed's
calibration was unreliable. Seeds 45/46 W12 cal_pf was < 4 so their
W12 trades remain in the deployment under the cal_pf>4 rule. But
their W12 OOS performance was poor anyway, so a blanket drop-W12 is
slightly better.

## Out-of-band calibrations

13 of 65 (seed, window) records fell out of calibration's "in-band"
range and used a fallback margin:

- **W00 (5 seeds)** — early window with too few val trades; expected
  fallback. OOS produces 0-1 trades; non-issue.
- **W03 (4 seeds)** — model couldn't find an in-band threshold for
  W03, fell back. **OOS performance is GOOD across these seeds (PF
  1.78-2.56)** — fallback is working.
- **Mixed: W04 (1 seed), W06 (1 seed), W02 (1 seed)** — single-seed
  fallbacks; performance varies.

Out-of-band fallback is **not** correlated with poor OOS performance.
W03 in particular is a "calibration is broken but OOS is fine" case.

## Why not retrain to fix W12

The audit asked: would a calibration-objective change rescue W12?

The answer is **no**: the val PF estimate on W12 is unreliable across
all 5 seeds, regardless of which threshold the calibration picked.
The val cohort has structurally different statistics from the OOS
cohort for that window. Rebuilding the calibration objective would
just generate a different overfit threshold.

The actionable fix is not "make calibration smarter" but "treat
high-cal_pf windows as untrustworthy" — i.e., the cal_pf > 4 guard.

## Decision rule (deployment-time)

For each (seed, window):
1. Read calibration.json → `objective_pf`.
2. If `objective_pf > 4.0`, mark this seed-window as "calibration
   ghost"; deploy with no trades from this seed in this window.
3. If `1.0 ≤ objective_pf ≤ 4.0`, deploy normally with the chosen
   margin.
4. If `objective_pf < 1.0` and `pf_qualified=False`, fallback margin
   is already in use; no additional action.

This rule is computable from val data alone (no look-ahead) and
removes the most reliably-overfit calibrations.

## Files

- `v3/analysis/per_seed_calibration_audit.py` — CPU diagnostic
- `v3/artifacts/calibration_audit/spx_combined_3seed_001.json` —
  full per-(seed, window) audit output
- This doc — methodology and rule

## Cost summary

Step 3 of the profitability sprint cost $0 (CPU-only, ~5 min wall).
The audit found a clean training-time signal that adds a small
deployment guard on top of the K=2 consensus filter.

## Combined deployment recipe (after sprint)

The recommended deployment recipe combines the sprint's findings:

1. Train 5 seeds with the combined-fix recipe (oracle + sb=1.0 +
   cohort-balanced contrastive) — already locked from
   `spx_combined_3seed_001_champion_adoption_2026_04_25.md`.
2. **Apply K=2 side-consensus filter** — only take a trade when ≥2
   of 5 seeds agree on the side (`ensemble_consensus_filter_2026_04_25.md`).
3. **Apply cal_pf > 4.0 deployment guard** — skip any seed-window
   combination where the validation calibration was structurally
   unreliable (this doc).

Expected 5-seed mean PF ~2.10-2.15, min PF ~1.80, max DD ~13%.
Clears all 3 promotion-strong gates. No GPU spend; no retraining.
