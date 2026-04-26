---
date: 2026-04-25
parent: forward_walk_with_oracle_2026_04_25.md
status: structural property of the L3 oracle — exits early; cuts losses but truncates winners
---

# L3 Oracle's "Exit Early" Bias — Trade-Level Analysis

## TL;DR

The forward-walk-with-oracle audit raised a question: why do seeds 42 and
43 perform *worse* with oracle restored, while seeds 44 and 46 perform
better? Trade-level inspection finds a clean structural answer: **the
L3 oracle systematically exits early**, which truncates winners and
mitigates losers. Whether it's net helpful per seed depends on whether
that seed's entry picks skew toward winners (oracle hurts) or losers
(oracle helps).

This is not random per-seed variance — it's a uniform property of the
oracle. It has direct implications for live execution.

## Counts (forward walk, 42 days, 5 seeds)

```
seed   no_oracle_wins   oracle_wins   truncated_winners   rescued_losers
  42         3              2               3                  1
  43         2              1               2                  5
  44         3              5               0                  4
  45         1              5               1                  6
  46        10             12               3                  5
```

Definitions:
- *truncated winner*: a no_oracle-winning trade where oracle's exit
  pnl is < 50% of no_oracle pnl. (Oracle cut the win short.)
- *rescued loser*: a no_oracle-losing trade where oracle reduced the
  loss by >50%. (Oracle cut early, saving the trade.)

## Per-seed pattern

**Seed 42 (oracle hurts: 1.62 → 0.51):**
- 3 winners truncated, only 1 loser rescued.
- Concrete: 2026-03-13 put won +$1235 with time-stop; oracle cut at
  +$350. 2026-04-13 call won +$1394 time-stop; oracle TURNED IT INTO
  -$19 LOSS by exiting before the move.
- Net: oracle sums -$400 vs no-oracle +$1,668.

**Seed 43 (oracle hurts: 0.29 → 0.08):**
- 2 winners truncated AND 5 losers rescued.
- Concrete: 2026-03-13 put +$1326 → oracle +$308 (truncated).
  2026-03-27 put +$924 → oracle -$594 (turned into loss).
- Even with 5 losers rescued, the truncated winners dominate.

**Seed 44 (oracle helps: 1.21 → 3.67):**
- Zero winners truncated, 4 losers rescued.
- Concrete: 2026-04-16 call -$282 (no_oracle) → +$1431 (oracle). The
  oracle's early exit found a profitable cutoff that the time-stop
  couldn't.
- Net: oracle sums +$6,925 vs no-oracle +$1,491.

**Seed 46 (oracle helps: 2.43 → 3.89):**
- 3 winners truncated, 5 losers rescued.
- Concrete: 2026-03-05 put +$3,490 (no_oracle) → +$72 (oracle).
  Massive winner truncated. But many other rescued losers compensate.
- Net: oracle sums +$10,945 vs no-oracle +$10,656.

## Why this matters for live execution

The oracle's early-exit pattern is **structural**, not random. It comes
from the way `simulated_l3_oracle` is trained: a HistGradientBoosting
classifier on prior chosen trades, with the loss function pushing
toward "minimize drawdown" rather than "maximize peak pnl."

In live execution:
1. **Expect oracle exits to be conservative.** Big winners will likely
   be truncated; bad trades will likely be cut early.
2. **The oracle is most valuable on seeds whose entries are
   loss-heavy.** Seeds with many small losses (like 44 and 46 on
   forward-walk) get net positive lift from oracle.
3. **The oracle is harmful on seeds whose entries hit big winners
   often.** Seeds 42 and 43 had a few extreme winners that got cut.

## Implications for Monday's live shadow

Three concrete recommendations refining the prior plan:

**1. Capture both oracle and time-stop exit pnl per shadow trade.**
Per `DecisionSnapshot`, also record what the time-stop exit would have
delivered. This lets us A/B compare oracle vs time-stop without
deploying both in production.

**2. Watch for the "winner-cut" pattern in shadow.** If shadow
sessions show several big winners truncated by oracle (similar to
seeds 42/43 forward-walk), consider testing a *blended exit*: use
oracle's stop-loss prediction but extend hold beyond oracle's
predicted exit_bar if pnl is rising.

**3. Per-seed oracle suitability is empirical.** Don't assume the
"seeds 44 + 46 are the strong ones" finding generalizes — check each
seed's no_oracle vs oracle pnl distribution post-shadow. The right
deployment may be heterogeneous: some seeds with oracle, some without.

## A cleaner deployment heuristic (research, not yet locked)

Rather than "use oracle uniformly" or "drop oracle uniformly", a
better runtime rule might be:

```
For each shadow trade, capture:
  - entry features (model's chosen action)
  - oracle's predicted exit_bar and exit_pnl
  - time-stop reference exit_pnl (held to bar 120)
  - actual realized pnl

After 5 sessions, fit a simple model that predicts:
  use_oracle ∈ {True, False}
  given (entry features, oracle confidence proxy, day regime)
```

This is research, not a tonight task. But it's the empirically-grounded
direction once shadow data exists.

## Files

```
v3/reference/oracle_exit_early_bias_2026_04_25.md  ← this file
v3/artifacts/forward_walk/forward_walk_chosen_seed{42..46}.pkl
v3/artifacts/forward_walk/spx_combined_3seed_001_with_oracle.json
```

## Cost summary

CPU only, ~10 min for trade-level audit.
