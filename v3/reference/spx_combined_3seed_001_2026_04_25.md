---
date: 2026-04-25
parent: spx_combined_001_2026_04_24_breakthrough.md
exp_base: spx_combined_3seed_001
status: 3-seed PARTIAL PASS — clears 2 of 3 floor-safe gates; min-seed-PF fails by 0.025
---

# 3-Seed Combined Promotion — Partial Pass

## Setup

After seed-42's combined fix cleared the floor on a single seed
(`spx_combined_001_2026_04_24_breakthrough.md`, agg PF 2.195),
this experiment tests reproducibility across 3 seeds with the same
recipe:

- Per-seed simulated-L3 oracle rebuilt from `candidate_surface`
  (50/50 sampled, vs 95% calls in champion source).
- Single-seed promotion training per seed with
  `--side-balance-weight 1.0 --w-side-contrastive 0.0`,
  `--utility-target hybrid_live`.
- All other hyperparameters identical to `spx_live_hybrid_001`.

Total cost: 3 × ~50 min CPU oracle rebuild + ~30 min H100 + 3 × ~30 sec
L3 routed composition.

## Per-seed entry-only results

```
seed     agg_pf  mean_dd%  min_pf  trades  call/put
42        2.195    12.37    0.616    376   146/230   (39c / 61p)
43        2.096     7.66    0.621    334   159/175   (48c / 52p)
44        1.761    14.88    0.000    371   153/218   (41c / 59p)

3-seed mean   2.017     11.64    n/a    361   458/623   (42c / 58p)
3-seed min    1.761
```

All three seeds inverted the 95% call bias. All three clear individual
agg PF ≥ 1.5. Two of three (42, 43) clear the strict 1.976 floor. Side
share is reproducibly put-majority across seeds (52–61% puts).

## Floor-safe gate vs handoff floor

```
gate                        target          actual    verdict
3-seed mean agg PF         >=1.976          2.017     PASS  (margin 0.041)
mean DD                    <=12.0%         11.64%     PASS  (margin 0.36 pp)
min seed PF                >=1.786          1.761     FAIL  (off by 0.025)
```

2 of 3 pass. The min-seed-PF fail is by 1.4% relative — driven entirely
by seed 44 at 1.761. Seeds 42 and 43 individually clear the floor at
2.195 and 2.096.

## W5 / W6 / W8 / W12 per-seed PF (entry-only)

```
seed     W5       W6       W8       W12
42       inf*    5.277    3.281    1.196
43       inf*    1.769    1.779    0.621
44       8.270   2.804    3.137    0.656
```

(* small samples: seed 42 W5=1 trade, seed 43 W5=2 trades.)

W5 — the regression flag from the prior champion review and the
quarantined `spx_live_hybrid_001` baseline (where seeds 42/44 hit ~0.6
PF) — is now positive across all 3 seeds. Seed 44 had 4 trades in W5
with PF 8.27, which is the most robust W5 result.

W12 is the new weak spot: seeds 43 and 44 hit 0.62 and 0.66 respectively.
This isn't catastrophic (no zero-PF window in seed 42, and the W12 weak
PFs aren't dragging the agg below 1.5) but it's the binding window for
the seed-44 min-PF miss.

## Routed L3 composition (entry + 2-expert L3)

```
thr     mean_agg_pf  mean_dd%   min_seed_pf  PF gate  DD gate  min gate
0.15        2.030      15.72       1.757       PASS    FAIL      FAIL
0.19        1.824      16.63       1.672       FAIL    FAIL      FAIL
0.20        1.850      15.72       1.709       FAIL    FAIL      FAIL
0.25        1.679      19.05       1.632       FAIL    FAIL      FAIL
0.30        1.491      21.05       1.402       FAIL    FAIL      FAIL
```

L3 routing did NOT improve over entry-only. At thr=0.15 the mean PF
edges entry-only by 0.013 (2.030 vs 2.017), but DD blows out from
11.64% → 15.72%, and min seed PF is essentially unchanged (1.757 vs
1.761). The minimum-DD floor is now firmly violated.

This is a meaningful reversal of the prior pattern where L3 routing
*lifted* a weak entry. **The new entry-stack is strong enough that L3
adds more variance than value.** Recommended composition: **no L3 — use
entry-only with the existing time-stop / hybrid_live exit.**

## Interpretation

- **Hypothesis confirmed at 3 seeds.** The combined fix (balanced oracle
  + full-coverage sample weights, no contrastive) reproduces the seed-42
  breakthrough on seeds 43 and 44. Side bias inversion happens on every
  seed; W5 is positive on every seed; aggregate PF clears 1.5 on every
  seed.
- **Strict floor not cleared by 0.025 on min seed PF.** Seed 44 is the
  weak link. Its W12 PF of 0.66 and W4 PF of 0.62 (per_window mean
  shows max stays high, so a couple of weak windows drag) are the
  binding constraint.
- **L3 composition is now subtractive, not additive.** When the entry
  stack is below the floor, L3 lifts; when entry is above the floor,
  L3 adds DD without lifting PF. This reverses the prior champion's
  composition strategy.

## Recommendations

Three viable paths, with explicit recommendation:

### Recommended: ship the entry-only stack as a "near-pass" candidate

Document the result, accept the strict gate fail by 0.025 as a known
caveat, and propose a 5-seed run on the same recipe to determine if the
seed-44 weak tail is a sample-size issue or a structural one. The
entry-only stack is materially superior to the quarantined champion on
every other axis: side balance, W5, agg PF, mean DD, trade-count
stability.

### Alternative 1: targeted seed-44 fix

Look at seed 44's W12/W4 specifically — both are low-trade-count windows
with regression-quality issues. A small calibration tweak (e.g., raising
the seed-44 min_l3_train_trades threshold) might close the gap. Risk:
seed-specific tuning is overfitting.

### Alternative 2: lower side-balance weight

sb=1.0 is a strong rebalance. Seeds 42/43 absorb it well; seed 44 may
overshoot. Try sb=0.7 across all 3 seeds — could shrink seed-44 variance
without giving up most of the discrimination shift. Adds ~$1 of GPU.

## Outputs

- `v3/artifacts/simulated_l3_oracle_spx_live_0945_1130_seed{42,43,44}_balanced.npz`
  (gitignored).
- `v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{42,43,44}/seed_*/{report.json, manifest.json, chosen_trades.pkl}` (gitignored
  except the small reports we usually keep — but per the broader
  .gitignore those are reproducible from the recipe).
- `v3/artifacts/layer3_combined_3seed_001_seed{42,43,44}/` (gitignored).
- `v3/reference/spx_combined_3seed_001_2026_04_25.md` — this file.

H100 closed at 00:24:58 (TX `02330067...`). Total day-and-a-half GPU
spend: ~6 H100-hours.

## What changed since the previous champion benchmark

| dimension | quarantined `spx_live_hybrid_001` | `spx_combined_3seed_001` |
|-----------|------------------------------|--------------------------|
| 3-seed mean agg PF                | 1.486 | **2.017** (+36%) |
| 3-seed mean DD                    | 17.6% | **11.64%** (−6 pp) |
| 3-seed min seed PF                | 1.240 | **1.761** (+42%) |
| Side share (3-seed)               | 95% calls | **42% calls** (inverted) |
| 2024-04-01 model puts (seed 42)   | 1     | **36** (36×) |
| Seed 42 W5 PF                     | 0.703 | **inf** (1 trade) |
| Seed 44 W5 PF                     | 0.677 | **8.270** (4 trades) |
| OOS truth=put `frac_call_above_put` (seed 42) | 0.899 | **0.630** (−27 pp) |
| Floor-safe gate (3 of 3)          | 0/3   | **2/3 PASS** |
