# Layer 3 Stage 1 — Oracle-Gap Probe — 2026-04-21

## Verdict

**GO — exit timing is a substantial lever.** Hindsight-optimal exits on
the detach-side baseline's 275 chosen trades produce PF 133.965 vs
time-stop's 1.472, a per-trade gap of +$1192 (median +$1064). The
ceiling is unrealistically high (no model approaches oracle), but the
upside is so large that even capturing 5–10% of the gap would
transform the system. Proceed to Stage 2 (heuristic exit ablation).

Secondary finding: **the suspected "asymmetric spread bug" in the
current `_time_stop_pnl` is essentially benign.** Corrected baseline
PF (1.472) is +0.017 above the published asymmetric baseline (1.455).
The honest anchor for Layer 3 is therefore PF 1.472 / DD 35.6%, not a
substantially-worse number as I had warned.

## Setup

- Universe: 275 chosen trades from
  [v3/artifacts/layer2_shared_enc_fixedq_detach/layer2_trades.csv](../artifacts/layer2_shared_enc_fixedq_detach/layer2_trades.csv).
- Probe script:
  [v3/analysis/layer3_oracle_gap_probe.py](../analysis/layer3_oracle_gap_probe.py).
- Artifact:
  [v3/artifacts/layer3_oracle_gap_probe/oracle_gap_probe.json](../artifacts/layer3_oracle_gap_probe/oracle_gap_probe.json)
  + per-trade CSV.

## Method

For each chosen trade, recover the contract (`select_contract`), pull
its full mid + spread_frac trajectory (`_build_contract_paths` +
`_contract_idx_for_record`), and compute three PnLs:

1. **`asym_pnl`** — re-runs the existing pipeline (`_time_stop_pnl`),
   which uses the entry bar's `spread_fraction` for both legs. Sanity
   check: must match `layer2_trades.csv` `pnl` column exactly.
2. **`corrected_pnl`** — same hold-to-end exit, but applies the EXIT
   bar's `spread_fraction` to the exit leg. Honest baseline.
3. **`oracle_pnl`** — scan every valid post-entry bar `t ∈ [entry+1,
   session_end=375]`, compute PnL using `mids[t]` and `spread_fracs[t]`,
   take the maximum. Hindsight ceiling.

Aggregation: trades sorted chronologically by `(day, bar_index)` so
DD reflects the actual equity curve.

## Sanity check

- `asym_pnl − csv_pnl` max diff: **$0.0000**, mean diff: **$0.0000**
  across all 275 trades. The probe reproduces the published numbers
  bit-for-bit; the pipeline is faithful.

## Results

### Aggregate

| Policy | PF | DD% | Mean $/trade | Trades |
|---|---:|---:|---:|---:|
| asym_pnl (csv repro) | 1.455 | 36.9 | +225.4 | 275 |
| corrected_time_stop | **1.472** | **35.6** | +234.3 | 275 |
| oracle_exit | 133.965 | 0.7 | +1426.5 | 275 |

The asym → corrected delta is +0.017 PF and −1.3 DD. The published
detach-side number is essentially honest under exit-bar spreads, which
is a relief — no anchor revision needed for the rest of Layer 3.

### Per-fold

| Fold | asym | corrected | oracle | gap (corr→oracle) | n_trades |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.860 | 0.875 | 57.648 | **+56.773** | 55 |
| 1 | 1.439 | 1.460 | 182.739 | +181.279 | 40 |
| 2 | 1.038 | 1.057 | 155.254 | +154.197 | 60 |
| 3 | 2.095 | 2.105 | 116.974 | +114.869 | 60 |
| 4 | 1.801 | 1.824 | 487.951 | +486.127 | 60 |

Every fold has substantial oracle headroom. **Fold 0 (the loser fold)
shows oracle PF 57.6** — the biggest absolute mismatch with current
performance. If a learned model could capture even 1% of fold 0's
oracle gap, fold 0 would lift from 0.875 to ~1.45, which would close
the load-bearing failure mode.

### Per-trade gap distribution (oracle − corrected, $)

| Stat | Value |
|---|---:|
| mean | +$1192.2 |
| median | +$1063.6 |
| std | $850.1 |
| p5 | +$109.3 |
| p25 | +$618.0 |
| p50 | +$1063.6 |
| p75 | +$1477.1 |
| p95 | +$3106.3 |
| n trades with gap = 0 | 6 |
| n trades with gap > 0 | 269 |

**269 of 275 trades have a positive oracle gap.** Only 6 trades had
their time-stop exit coincide with the hindsight optimum. Even the
worst 5% of the gap distribution is +$109/trade.

### Oracle exit timing (bars held)

| Stat | Oracle | Time-Stop |
|---|---:|---:|
| mean bars held | 89.6 | 209.1 |
| median | 76 | — |
| p5 | 1 | — |
| p25 | 16 | — |
| p75 | 154 | — |
| p95 | 226 | — |

**Oracle holds for ~76 minutes (median) vs ~209 minutes for time-stop.**
The system is leaving trades on for ~2.7× longer than optimal. 25% of
trades have an oracle exit within the first 16 minutes of entry.

This is consistent with the long-premium decay structure — once a
trade has reached its peak in the early-to-mid session, holding longer
is theta-bleeding. A learned exit only needs to detect "this is
roughly the peak" with reasonable accuracy to capture meaningful lift.

## What this tells us

**The lever is real, large, and structurally consistent across folds.**
Hindsight is unrealistically generous, but the gap is so wide
(time-stop captures less than 1% of oracle PF on aggregate) that a
modest learned model has substantial margin.

The two cleanest signals from this probe:

1. **Time-stop holds too long.** Oracle exits at ~36% of the bars
   time-stop holds. Even a heuristic "exit at MFE − N%" trailing stop
   should capture some of this — see Stage 2.
2. **Fold 0 has the biggest relative headroom.** This is the load-
   bearing failure mode (chop regime, put-bleed). If learned exits can
   trim losing trades early in fold 0, the regime fragility may
   compress without needing a regime detector.

## What this does NOT prove

- That any model can capture meaningful fraction of oracle. Hindsight
  is the upper bound; the realistic ceiling is much lower.
- That heuristic exits will work. Stage 2 tests that.
- That learned exits won't have selection-bias problems analogous to
  atm_iv K=2. Stage 4's reality checks address that — but only if we
  get to Stage 3.
- That fold 0's regime risk goes away with better exits. Earlier exits
  may compress losses but don't characterize the regime in real time.

## Caveats

- The published 1.455 PF used the asymmetric spread model. The honest
  anchor is **1.472 / 35.6%**. The Layer 3 acceptance threshold (PF ≥
  baseline + 0.10 OR DD ≥ −5pt) should be measured against the
  corrected anchor, not the published one. So acceptance is **PF ≥
  1.572 OR DD ≤ 30.6%** at composed Layer-2 + Layer-3.
- Oracle uses every bar including the bar that contains a single
  outlier mid quote. There's no microstructure smoothing. Real fills
  would be slightly worse than the per-bar mid (bid actually trades
  below mid). The oracle therefore overstates the realistic ceiling
  by some amount; the gap is still vastly larger than any plausible
  noise.
- Session_end_bar is 375 (15:45 ET), avoiding last-15-minute chaos.
  Oracle and time-stop both respect this cutoff.

## Next step

Proceed to **Stage 2 — heuristic exit ablation**. Test trailing stops,
take-profits, bail-outs, time-of-day stops, and a composite. Anchor
what a learned model has to beat. Per-fold results required.

## Verification

- [x] `python -m py_compile v3/analysis/layer3_oracle_gap_probe.py` passes
- [x] Probe runs end-to-end on existing artifacts (~1 min)
- [x] Sanity check: 275/275 asym_pnl == csv_pnl exactly
- [x] Per-fold gap table reported
- [x] Per-trade distribution + oracle exit-bar timing reported
- [x] Verdict explicitly tagged GO
- [ ] Commit
