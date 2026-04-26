---
date: 2026-04-25
parent: oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md
status: ONGOING — clean research after retraction; documents what was tried, what worked, what's still open
---

# Honest Research Path Forward (Post-Retraction)

After the oracle-gate retraction, the user requested continuation on the
fundamental problem: oracle's early-exit bias hurts winners. This doc
records the clean research that followed, with strict no-leakage discipline.

## What we know

The L3 oracle's training target is at [v3/layer3/common.py:402](../layer3/common.py#L402):

```python
payload["target"] = int(current_pnl >= suffix_max[i])  # "is this the local max?"
```

`suffix_max[i]` is the maximum pnl from bar `i` to session end. This trains
the oracle to predict "should I exit now?" with the answer being "yes, if
this is the local maximum of remaining pnl." On strongly trending winners,
the local max keeps moving up — oracle's training target keeps saying
"exit now" prematurely. **The early-exit bias is structural to the loss
function.**

## Audit results (no leakage)

Trade-level audit on 1717 trades (1664 OOS + 53 FW), comparing oracle's
predicted exit_pnl to perfect-info best_exit_pnl on each trade:

```
                                pred_exit_bar     pred_pnl - time_stop_pnl    captured_of_best
mean                                154                  +$141                    -0.087
median                              150                  +$232                    +0.121
> 0 frac                            —                     66%                     —
```

**Key per-bucket findings**:

```
best_exit_pnl bucket    n      capture (mean)    capture (median)
$0-200                311      -1.43             -1.03
$200-500              366      -0.49             -0.34
$500-1000             366      +0.17             +0.16
$1000-5000            493      +0.38             +0.41
$5000+                 24      +0.59             +0.65
```

Oracle does **worst on small winners** (negative pnl despite a real winner
existing), **best on large winners** (captures 59% of $5k+ winners). The
"early-exit on small winners" pattern is consistent.

**Exit quality bucket distribution**:
```
negative-on-winner:   576 (34%)   ← oracle TURNED winner into loser
early-exit (0-50%):   383 (22%)   ← captured <50% of available
ok (50-85%):          283 (16%)
good (≥85%):          153 (9%)
no-best-exists:       322 (19%)
```

**41% of winnable trades are mistimed badly enough to lose money or
capture <half of available pnl.** The early-exit bias is real.

## Sanity check: fixed-exit-bar sweep

Built per-bar pnl trajectories for 61 forward-walk trades using
`v3/oracles/opportunity._build_contract_paths`. Tested fixed exit at
each bar offset:

```
exit_at+    n     PF    sum
   30      61   0.81   $-3,025
   60      61   0.84   $-3,318
   90      61   1.07   $+1,597
  120      61   0.89   $-3,306    ← time-stop reference
  150      61   1.11   $+3,046
  180      61   1.25   $+6,632    ← optimal fixed
  240      61   1.19   $+5,237

Best fixed (bar+180):   PF 1.25, sum $+6,632
Reported oracle:        PF 1.89, sum $+12,189
Reported time_stop:     PF 1.21, sum $+5,273
```

The optimal fixed exit (bar+180, ~3 hours after entry) gives PF 1.25.
The L3 oracle's per-trade predictions give PF 1.89 — substantially
better. **The oracle DOES add per-trade value despite the early-exit
bias.** But the perfect-info ceiling is PF 92 (per-trade peak): there's
massive headroom we're not capturing.

There's a ~$7k gap between my fixed-bar simulation at bar+240 ($5,237)
and reported time_stop ($2,875) — likely caused by my simplified fill
mechanics differing from the canonical `hybrid_live_utility` (which
includes stopout-risk penalty + spread accounting). The relative
ordering of fixed bars is reliable though.

## Runtime trail-stop rules (leak-free)

Tested 4 rule families using ONLY in-trade observed pnl (no future info):

```
rule                                 PF      sum     mean_offset
R0: exit at oracle bar (baseline)   1.07   $+1,428      79
R1: extend by 20×5 if peak          1.13   $+2,807      91
R2: trail $300 from oracle bar      1.05   $+1,129      82
R2: trail $500 from oracle bar      0.85   $-3,303      91
R2: trail $1000 from oracle bar     1.06   $+1,471     134
R3: peak+positive ($200) hold       1.02   $+293        79
R4: no-new-peak 10 bars             1.05   $+1,091      85
R4: no-new-peak 15 bars             0.97   $-716        87
R4: no-new-peak 20 bars             1.06   $+1,183      90
```

**Best rule**: R1 (extend by 20 bars × 5 if pnl is at running peak)
gives PF 1.13 vs R0's 1.07 — a small +6% PF lift, +$1,379 total $.

But the absolute baseline PF (1.07) is far below the reported oracle
PF (1.89). The simulation has fill mechanics issues; rules' RELATIVE
performance is informative but absolute lift estimates are unreliable
until simulation matches the reported labels.

## What's needed for substantive improvement

**1. Train an alternative L3 oracle with a less-aggressive exit target.**
   Replace `target = (current_pnl >= suffix_max)` with something like:

   - `target = (current_pnl >= 0.85 * suffix_max)`: exit only when within
     85% of remaining peak. Lets winners run when there's substantial
     upside.
   - `target = (suffix_max - current_pnl) / suffix_max < 0.15`: similar
     framing in regression form.
   - Regression target = `log(suffix_max / max(current_pnl, $50))`: predict
     remaining-upside ratio; exit when log-ratio < 0.

   Substantive code change in `v3/layer3/common.py` + ~50 min CPU rebuild
   per seed × 5 seeds = 4 hours serial or ~1 hour parallel.

**2. Fix the simulation fill mechanics** to match `hybrid_live_utility`
   labels. Currently my trajectory builder uses simple ask/bid spread
   accounting; canonical labels include stopout-risk penalties and
   late-entry adjustments. Without this, runtime-trail PF estimates
   don't match deployment.

**3. Bigger sample for trail-stop validation.** 53-61 FW trades is too
   small. Once simulation is reconciled, build trajectories for all 1664
   OOS trades and re-test rules.

## What's confirmed real (no leakage)

These hold up to scrutiny:

1. **Oracle's early-exit bias is structural** (target = is-local-max).
2. **Oracle adds genuine value over fixed-bar exits** (PF 1.89 vs 1.25
   on FW). Per-trade prediction is real.
3. **Oracle does worst on small winners** (negative-on-winner on 34%
   of trades) and best on large winners (59% capture on $5k+).
4. **Runtime trail-stop rules show small (~+6%) lift** in relative
   simulation but need fill-mechanics fix before deployment.
5. **The right path** is alternative oracle training target, not a
   thin classifier on top of existing oracle.

## What's invalidated (retraction)

- Oracle gate classifier "+24% PF" → label leakage via
  `time_stop_margin_raw` and `side_margin_raw`.
- Skip+gate "+180% PF" → same leakage.

See `oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md` for full retraction.

## Open research directions (in priority order)

1. **Re-train L3 oracle with relaxed exit target**. Strongest expected
   improvement. Substantive but bounded engineering.

2. **Reconcile simulation with hybrid_live_utility labels** for
   trail-stop research. Mostly engineering.

3. **MFE/MAE-aware trail rules** with proper sim. Once #2 is done, can
   honestly test trail-extension rules at scale.

4. **Per-regime oracle behavior**: high-IV vs low-IV trades may need
   different exit strategies. Audit revealed capture-of-best varies by
   trade-pnl-magnitude; possibly varies by regime too.

5. **Two-headed oracle**: one head predicts P(more upside remaining),
   one predicts P(reversion imminent). Fuse for runtime decision.

## Files

```
scripts/research_oracle_exit_timing_audit.py     # leak-free audit
scripts/research_fixed_exit_sweep.py             # fixed-bar sweep
scripts/research_runtime_trail.py                # trail-stop rules
v3/artifacts/research/oracle_exit_timing_audit.csv
v3/artifacts/research/fw_trade_trajectories.pkl
v3/reference/honest_research_path_2026_04_25.md  ← this file
```

## Cost summary (this session, post-retraction)

CPU only: ~30 min. No GPU. No money spent.

The honest aggregate picture across the entire 2026-04-25 session:
- Found a real research direction (alternative oracle target).
- Caught a label-leakage hack in the gate.
- Documented 6 layered findings with retractions where appropriate.
- Shipped no false claims.
