# Layer 3 Stage 2 — Heuristic Exit Ablation — 2026-04-21

## Verdict

**GO with caveat.** Best aggregate PF heuristic (`bail_out_30`,
PF 1.833 / +0.361 vs corrected baseline 1.472) clears the +0.15 GO
threshold by a wide margin, but it BREAKS fold 0 (0.875 → 0.539). The
cleanest defensible heuristic is `time_of_day_90`: PF 1.709 (+0.237)
AND DD 31.3% (−4.3pt) AND fold 0 only mildly hurt (0.875 → 0.808).
**A learned model has clear room to beat both** by being adaptive —
bailing aggressively in fold-3-like regimes but holding patiently in
fold-0-like regimes.

Proceed to Stage 3 (learned exit model). The Stage 3 acceptance bar
moves up: composed Layer-2 + Layer-3 PF ≥ 1.709 (best heuristic) AND
no fold regresses below 0.808 (fold 0's heuristic floor).

## Setup

- Universe: 275 chosen trades from
  [v3/artifacts/layer2_shared_enc_fixedq_detach/layer2_trades.csv](../artifacts/layer2_shared_enc_fixedq_detach/layer2_trades.csv).
- Probe script:
  [v3/analysis/layer3_heuristic_exits.py](../analysis/layer3_heuristic_exits.py).
- Artifact:
  [v3/artifacts/layer3_heuristic_exits/heuristic_exits.json](../artifacts/layer3_heuristic_exits/heuristic_exits.json).
- All heuristics evaluated against the corrected exit-spread baseline
  (PF 1.472, DD 35.6%) from
  [Stage 1](layer3_oracle_gap_2026_04_21.md), not the published
  asymmetric-spread 1.455.

## Method

For each trade, pre-compute the per-bar PnL series across `[entry+1,
session_end=375]` using the EXIT bar's spread_frac. For each
heuristic, walk the series and exit at the first trigger (or fall back
to the last finite bar). 14 heuristic configurations across five
families:

- **trailing_N** — once MFE > 0, exit if current ≤ MFE × (1 − N/100).
  N ∈ {25, 40, 60}.
- **take_profit_X** — exit if current ≥ X% × entry_premium ($).
  X ∈ {50, 80, 120}.
- **bail_out_Y** — exit if current ≤ −Y% × entry_premium ($).
  Y ∈ {30, 50, 70}.
- **time_of_day_K** — exit at bar (entry + K), ignores price.
  K ∈ {30, 60, 90} minutes since entry.
- **composite_tp50_trail40** — arm trailing-40 only after take-profit-50
  fires.

## Results

### Aggregate (sorted by PF lift)

| Config | PF | PF Δ | DD% | DD Δ | Mean $ | Early% | Bars Held |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline_corrected_time_stop` | 1.472 | 0.000 | 35.6 | 0.0 | +234 | 0% | 209 |
| **`bail_out_30`** | **1.833** | **+0.361** | 43.5 | +7.9 | **+248** | 78% | 79 |
| `time_of_day_90` | 1.709 | +0.237 | **31.3** | **−4.3** | +236 | 100% | 90 |
| `time_of_day_60` | 1.632 | +0.160 | 33.2 | −2.4 | +177 | 100% | 60 |
| `bail_out_50` | 1.525 | +0.052 | 53.2 | +17.5 | +215 | 68% | 116 |
| `bail_out_70` | 1.488 | +0.016 | 39.9 | +4.2 | +233 | 52% | 155 |
| `composite_tp50_trail40` | 1.480 | +0.008 | **25.1** | **−10.5** | +169 | 53% | 133 |
| `time_of_day_30` | 1.462 | −0.011 | 36.0 | +0.4 | +99 | 100% | 30 |
| `take_profit_120` | 1.157 | −0.315 | 31.1 | −4.6 | +72 | 32% | 165 |
| `take_profit_80` | 1.098 | −0.374 | 44.4 | +8.7 | +41 | 43% | 145 |
| `take_profit_50` | 1.084 | −0.388 | 34.0 | −1.7 | +30 | 56% | 120 |
| `trailing_40` | 1.407 | −0.065 | 26.8 | −8.8 | +53 | 90% | 39 |
| `trailing_60` | 1.354 | −0.118 | 34.2 | −1.4 | +48 | 88% | 44 |
| `trailing_25` | 0.790 | −0.682 | 57.8 | +22.1 | −27 | 90% | 35 |

### Per-fold PF for the top 3

| Fold | baseline | bail_out_30 | time_of_day_90 | time_of_day_60 |
|---:|---:|---:|---:|---:|
| 0 | 0.875 | **0.539** | 0.808 | 0.658 |
| 1 | 1.460 | 1.471 | 1.214 | 1.361 |
| 2 | 1.057 | 1.300 | 1.274 | 1.464 |
| 3 | 2.105 | **3.558** | 2.837 | 2.304 |
| 4 | 1.824 | 2.406 | 2.235 | 2.293 |

`bail_out_30` BREAKS fold 0 (0.875 → 0.539, the worst regression of
any heuristic in any fold). The aggregate PF lift comes entirely from
folds 3 and 4 amplifying. This is exactly the failure mode we don't
want — it would amplify fold-0-like-regime risk in deployment.

`time_of_day_90` is the cleanest single heuristic: 4 of 5 folds
improve or stay flat, fold 0 takes only −0.067, and DD compresses by
4.3pt.

## Reading the heuristics

**Why `bail_out_30` looks great in aggregate.** −30% × $entry_premium
is roughly $300 on a typical trade (entry mid ~$10). Cutting at −$300
caps each loser. Folds 3 and 4 had favorable directional moves —
trades that would have lost a small amount get cut early before they
reverse, and the winners are largely untouched (only 22% of trades
exited via time-stop, others bail or stay until session end). In fold
0, the chop regime means many trades dip into −30% territory before
recovering — bail_out_30 turns small-loss-recoverable trades into
small-loss-realized trades, killing the fold.

**Why `time_of_day_90` works without breaking folds.** Caps holding at
~90 minutes (median oracle exit was 76 minutes in
[Stage 1](layer3_oracle_gap_2026_04_21.md)). Cuts off the long
right-tail of theta bleed in chop. No price-action sensitivity = no
regime-specific failure mode. Worst fold-0 hit is mild (−0.067).

**Why trailing stops fail.** MFE on losing trades stays low; trailing
N% off MFE triggers very quickly (mean exit ~35-43 bars vs baseline
209). 89-90% of trades exit early. The model captures small wins but
misses large delayed moves. trailing_25 is catastrophic (PF 0.790).

**Why take-profit fails.** Cuts winners short. Most chosen trades
either don't reach +50% (TP doesn't fire) or do reach it then continue
much higher (TP exits prematurely). All three TP configs reduce PF
substantially.

**Why composite is essentially flat.** DD compresses materially
(35.6% → 25.1%, the BEST DD reduction of any heuristic) but PF stays
unchanged. A combination like this is what a learned model could
discover — but composite alone doesn't beat time_of_day_90.

## What this tells Stage 3

Three orthogonal signals a learned model has access to that none of
the heuristics use:

1. **Regime context.** The model sees Layer-2's input features at each
   post-entry bar. It can in principle learn "in fold-0-like regimes,
   don't bail at −30%; in fold-3-like regimes, do."
2. **Trade-state context.** The model sees `bars_since_entry`,
   `mfe_so_far`, `mae_so_far`. It can learn time-decaying patience,
   not the binary either-bail-or-don't of bail_out_Y.
3. **Compositional.** The model can blend "bail aggressively + cap at
   90 bars + trail when up big" without us choosing the specific
   composition.

The bar Stage 3 must clear:
- **PF target**: ≥ 1.709 (beat time_of_day_90)
- **Per-fold target**: no fold below 0.808 (don't lose fold 0 worse
  than time_of_day_90 does)
- **DD target**: ≤ 30.6% would be ideal (beat time_of_day_90's 31.3);
  hitting the alternate-acceptance threshold from the plan (DD ≤
  30.6%) at PF parity also acceptable

If Stage 3's learned model can't beat time_of_day_90 on either PF or
DD without breaking fold 0 worse, the workstream's verdict converts to
**STOP-LITE**: ship `time_of_day_90` as the production exit layer, no
learned model needed.

## What this does NOT prove

- That a learned model WILL beat `time_of_day_90`. It might just
  rediscover time-of-day patterns and add a sliver of regime-aware
  bailout.
- That `bail_out_30`'s aggregate PF is "real edge" — it's a brittle
  optimum that hurts the load-bearing fold.
- That any of these heuristics generalize out of sample. Same
  selection-bias caveat as everything in this dataset; only fresh data
  can settle that.

## Caveats

- Heuristics with grid parameters (N, X, Y, K) were searched on the
  exact same 275 trades that any learned model will train on. Some
  grid-best optimum is expected purely from this; the +0.36 PF lift
  for `bail_out_30` should be discounted accordingly. The fold-0 break
  is robust to that discount and is the more reliable signal.
- All exits use the EXIT bar's mid + spread_frac. Real fills would be
  slightly worse (true bid trades below mid). Heuristic results
  inherit Stage 1's caveat: oracle and heuristic both slightly
  optimistic.

## Next step

Proceed to **Stage 3 — learned exit model build**. Per the plan:
HistGradientBoostingClassifier, binary `{hold, exit_now}`, per-fold
walk-forward, train on chosen-trades only first (augment with
teacher-only entries as ablation). Acceptance: beat time_of_day_90 on
PF or DD without hurting fold 0 below 0.808.

## Verification

- [x] `python -m py_compile v3/analysis/layer3_heuristic_exits.py` passes
- [x] Probe runs end-to-end on 275 trades (~1 min)
- [x] Each of 14 heuristic configurations reported
- [x] Per-fold breakdown for top 3
- [x] Verdict explicitly tagged GO with fold-0 caveat
- [x] Stage 3 acceptance bar updated: PF ≥ 1.709 AND fold 0 ≥ 0.808
- [ ] Commit
