---
date: 2026-04-24
parent: w_side_sweep_2026_04_24_falsified.md
exp_base: spx_live_hybrid_001 / seed_42 / window_05
status: real put opportunities confirmed in W5; side-aware calibration justified
---

# W5 Day-Level Decomposition + Side-Substitution Counterfactual

## Setup

After the `spx_w_side_sweep_001` falsification of H1 (re-enabling the
existing side-contrastive loss made discrimination worse, not better, due
to 2.83:1 train class imbalance), the next-plan called for a W5 day-level
diagnostic with a side-substitution counterfactual to decide between
three branches:

1. W5 has no put opportunities → build regime abstention, don't side-balance.
2. W5 has put opportunities the model under-scores → side-aware repair.
3. Labels are 90%+ call-dominant → repair the utility target.

Script: `v3/analysis/w5_day_diagnostic.py`. Computes hybrid_live target
utility per row across the W5 OOS slice, joins the production `chosen_trades`
to the routed L3 exits at thr=0.20, and runs a swap counterfactual at
multiple label-edge margins.

## Result: branch 2 (real put opportunities, model under-scores them)

### W5 OOS bar-level label distribution (5,349 tradeable bars)

```
n_bars truth=call: 2710  truth=put: 2639
label_put_minus_call: mean=-76.97 median=-8.16
n_bars label_put beats call by:
   $50:  2453 (45.9% of all tradeable W5 bars)
  $100:  2322 (43.4%)
  $200:  2022 (37.8%)
  $500:  1109 (20.7%)
 $1000:   509 ( 9.5%)
```

Roughly half of W5's tradeable bars favor puts, including 9.5% with at
least a $1,000 edge. Branch 1 ("no put opportunities") is decisively
falsified.

### Chosen-trade summary

```
n_chosen = 14    call=13   put=1
production L3 routed PnL (thr=0.20): -$1479.96 total, -$105.71/trade  (PF 0.61)
oracle-best PnL (sim-L3 exits):       +$8506.15 total, +$607.58/trade
```

The oracle-best ceiling is +$10,000 above what the production stack
delivered on the same 14 chosen days. The model isn't just losing money;
it's actively walking past large put opportunities.

### Per-day decomposition (chosen=14)

```
day          side strike   l3$       lbl_call  lbl_put   lbl_p-c   swap_l3$  oracle$  oracle_side  sigma_pos  omar
2024-03-26   call 5225        6.31    -200      199       +399      179       179      put          +1.49      +1.49
2024-04-01   call 5250    -1074.01    -286     1699      +1986     1522     1522       put          +0.33      +0.86
2024-04-03   call 5220     -232.50    -182      230       +412      214       214      put          +1.37      +6.25
2024-04-04   call 5240     -468.70    -290      135       +425      127       127      put          -0.64      -0.12
2024-05-13   call 5220     -868.92    -189      362       +551      335       335      put          -0.73      -1.84
2024-05-16   call 5310     +368.00    +400     -178       -578      -63       368      call         +1.61      +2.00
2024-05-17   call 5295     -394.96    -188      398       +586      347       347      put          +1.45      -0.44
2024-05-20   call 5315     -244.13    -194       74       +268       71        71      put          +1.28     +10.00
2024-05-21   call 5300      +15.70    +518     -251       -769      -97       417      call         +1.91      +2.76
2024-05-22   call 5310      +74.20    +139     1126       +987      950       950      put          +0.11      +0.48
2024-05-28   call 5295     +226.40    -461      369       +830      343       343      put          -0.76      -2.06
2024-05-30   put  5250     -484.61    +452     -229       -681      -83       420      call         -0.20      -1.75
2024-06-10   call 5340     +458.11    +489     -100       -589      -46       458      call         +2.35      +2.03
2024-06-17   call 5435    +1139.14   +3040     -190      -3230     -101      2754      call         +2.33      +2.13
```

Signal-rich observations:

- **9 of 14 chosen-bars favored puts by ≥$50 label edge.** The model
  picked call on every one of them (including 2024-04-01 with a +$1,986
  put-call edge — the golden-day case).
- **The single time the model picked put (2024-05-30), the label said call.**
  So the model isn't a balanced discriminator that happens to bias toward
  calls; it's an unreliable side discriminator that overwhelmingly defaults
  to call.
- **Sigma_pos and OMAR position don't track side well.** Days the model
  got right (2024-06-17 with sigma_pos +2.33, +$1,139) and days it got
  wrong (2024-04-01 with sigma_pos +0.33, −$1,074) span the same regime
  features. The model isn't using these to discriminate; the call prior
  dominates.

### Counterfactual: swap to best-put when label_put_minus_call ≥ margin

```
margin($)  n_swap  delta_total$  scenario_total$
       50       9      +7065.09         +5585.13
      100       9      +7065.09         +5585.13
      200       9      +7065.09         +5585.13
      500       5      +5534.55         +4054.58
     1000       1      +2596.11         +1116.14
```

(`delta_total` = sum over swapped trades of `swap_l3 − l3_routed`;
`scenario_total` = W5 PnL if you'd kept the model's choice on non-swap
days and substituted best-put on swap days.)

At margin $50, the simple side swap on 9 days flips W5 from a $1,480
losing window (PF 0.61) to a $5,585 winning window (~PF 2.5). At
margin $200 the same 9 days swap; W5's put edges are *that* clear in the
labels.

## Caveat

The simulated-L3 oracle was trained on the seed-42 champion's chosen-trade
set, which was itself ~95% calls. Put-side simulated exits are
extrapolation, not validated holdouts. Treat the dollar magnitudes as
upper-bound estimates — but the **direction** of the result (real put
opportunities exist, model misses them) is robust because:

1. The label utilities themselves (label_best_put, label_best_call)
   come from the same `hybrid_live` target the model trained against.
2. 5 of the 9 swap days have label put-call edges of ≥$500, well outside
   any plausible L3 extrapolation error.
3. The 1 day at margin $1,000 (2024-04-01) has been independently
   confirmed by Codex's golden-day overfit canary — the architecture
   *can* learn the put on that day if isolated.

## Decision

Per the next-plan branch tree:

> If W5 has positive put opportunities that the model under-scores or
> calibration filters out, proceed to side-aware selection repair.

Proceeding with side-aware calibration as the next experiment. It's the
cheapest valid intervention (no retraining required), and the diagnostic
shows the model's underlying scores still rank puts below calls even on
high-edge bars, so per-side margin/win/stopout thresholds may not be
sufficient on their own — but they cost ~$0 to test and will tell us
whether the gap is purely score-magnitude (calibration claws back) or
score-discrimination (calibration cannot fix; need sample reweighting).

## Outputs

- `v3/analysis/w5_day_diagnostic.py` — diagnostic harness.
- `v3/artifacts/w5_diagnostic/seed42_w5.json` — full per-day JSON
  including all per-row regime context, hybrid_live label utilities per
  side, and counterfactual sums.
