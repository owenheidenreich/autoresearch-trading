# Fork A1 Stage 1 results — SPX-only proxy of Pickles Row 1 (falsified)

> **Scope-of-claim header.** This is a test of the SPX-only proxy of Row 1.
> A positive result would show edge *on this dataset, under this cost model,
> with these specific substitutions* (SPX VWAP for ES/NQ VWAP; no A/D ratio;
> no A/D volume). The negative result below falsifies the proxy, not Row 1
> as originally stated. The cheapest next step is not pre-determined —
> Fork C, Row 3, and ES/NQ/AD data acquisition are all live options whose
> relative priority depends on the specific shape of the Stage-1 result.

## Verdict

**Stage 1 falsifies the SPX-only proxy of Row 1.**

Stage 2 was not run (gated out per plan's rule: Stage 1 failed at all deltas
with no cell anywhere near the pass bar). A random-entry control was added
in its place to strengthen the interpretation.

## Numbers

Dataset: `v2/data.pt` (fingerprint `6162cf3d83db3586`, sidecar schema
`v5_exact_chain_v2_slice`, 986 RTH sessions from 2022-04-11 through 2026-04-01).
Delta grid: `{0.40, 0.50}`. Window: bars 15-150 (09:45-12:00 ET). Custom
runner (no `simulate_day`, no `DEFAULT_POLICY` inherited).

**Stage 1 pass bar (from plan):** ≥ 50% of candidates reach +30 bps SPX MFE
within 30 min AND option-end-return median positive at 20 min.

**Actual — at the 30-minute horizon:**

| Cell | Δ | n | MFE ≥ +30 bps | MAE ≤ −30 bps | opt_end+ | opt_end median |
|---|---|---|---|---|---|---|
| P-open | 0.40 | 609 | **9.0%** | 13.6% | 41.2% | **−8.33%** |
| P-open | 0.50 | 624 | 8.8% | 13.5% | 43.9% | −4.98% |
| P-15m  | 0.40 | 381 | 8.4% | 13.9% | 38.1% | −10.20% |
| P-15m  | 0.50 | 394 | 8.1% | 13.7% | 41.9% | −5.97% |
| P-cooldown | 0.40 | 298 | 7.0% | 15.8% | 35.6% | −12.82% |
| P-cooldown | 0.50 | 306 | 6.9% | 15.7% | 39.2% | −9.31% |
| P-all  | 0.40 | 155 | 7.1% | **18.1%** | 32.3% | **−18.84%** |
| P-all  | 0.50 | 162 | 6.8% | 17.9% | 35.2% | −15.39% |
| **RANDOM** (1 random bar / session) | 0.40 | 853 | **10.2%** | 10.9% | 41.7% | −8.33% |
| **RANDOM** (1 random bar / session) | 0.50 | 850 | 10.2% | 10.9% | 43.5% | −4.74% |

Pass bar on spot MFE ≥ +30 bps: 50%. Best actual: 10.2% (random). Best Pickles cell: 9.0% (P-open × 0.40). Worst Pickles cell: 6.8% (P-all × 0.50).

## What the numbers actually say

1. **Random beats every Pickles cell** on both directions. RANDOM × 0.50 at
   30m: 10.2% favorable hit rate, 10.9% adverse hit rate. P-open × 0.50:
   8.8% favorable, 13.5% adverse. The Pickles-qualifier layer, operationalized
   on SPX-only inputs, carries no information and slightly worsens the
   favorable/adverse ratio.

2. **The real killer is not the qualifiers; P-open is already awful.**
   *"Pickles qualifiers are anti-predictive"* would overclaim. The honest
   framing: **the SPX-only proxy of Pickles' Row 1 qualifiers does not help**,
   and on thinner cells hurts, but the base layer it's built on (VWAP-support
   entry on SPX alone) is already below random for this horizon / cost / instrument.

3. **Directional 0DTE long calls in the morning window are a losing mode on
   this data, period.** RANDOM × 0.50 at 30m has 43.5% positive option-end
   rate and a −4.74% median option-end return. That is the pool from which
   every operationalization of Row 1 on SPX is drawing. Theta burn and
   typical intraday move sizes make +30 bps in 30 min a ~10% event; a
   0.50-delta call needs closer to +60 bps to break even on spread +
   commission; the distribution does not deliver that often enough.

4. **The 2023-12-14 template day is not reproducible on SPX VWAP alone.**
   At bar 38 (Pickles' 7:08 AM PT entry), SPX was −11.64 bps *below* its
   own VWAP, `bar_delta` was −0.69 (bearish), and `volume_ratio` was 0.71.
   No Stage-1 operationalization fires at that bar. Pickles was tracking
   *ES* VWAP, which was at a different level. This is a direct falsification
   of the proxy substitution "SPX VWAP ≈ ES VWAP for Row-1 purposes."

5. **Ablation cells do not order as the plan predicted.** Plan's expected
   ordering was `P-all ≥ P-cooldown ≥ P-15m ≥ P-open` if the qualifiers
   help. Actual ordering is the reverse. The pool the strict cells draw
   from is smaller (155 vs 609) but the degradation is consistent across
   every metric.

## What we actually ran

- Runner: [v2/strategies/fork_a1_stage1.py](../strategies/fork_a1_stage1.py).
  Custom bar-walk over `v2/data.pt`, session-local state (running VWAP,
  first-15m stats at bar 14, OVN-proxy direction from open-vs-prior-close,
  rolling 30-bar vwap-rally buffer for the trigger, σ-band precomputed
  from prior 30 sessions).
- Strategy: [v2/strategies/pickles_row1.py](../strategies/pickles_row1.py).
  **Trigger revised 2026-04-19** after a smoke test showed the original
  strict form ("prior bar > +10 bps above VWAP") fired zero candidates over
  575 sessions. Revised form: "session saw SPX > +10 bps above VWAP in the
  prior 30 bars" (see module docstring for full explanation).
- Control: [v2/strategies/fork_a1_random_control.py](../strategies/fork_a1_random_control.py).
  One random bar per session in the 09:45–12:00 ET window, same delta-grid
  contract pick, same forward-MFE/MAE measurement. Seed = 7 (in provenance).
- Outputs: `v2/artifacts/fork_a1_stage1/candidates.csv`,
  `forward_metrics.csv`, `summary.json`, `provenance.json`,
  `random_control_*.{csv,json}`.

## What this proves

The SPX-only proxy of Row 1 has no tradeable entry signal on this dataset,
under this cost model, with these specific substitutions. Adding the
Pickles qualifier layer (15m strength, OVN proxy, half-hour cool-down,
σ-band target) makes the proxy worse, not better, across every cell and
every horizon.

## What this does not prove

- **Row 1 as originally stated** (ES + NQ + A/D ratio + A/D volume + VWAP
  σ bands). That requires data we do not have. A negative proxy result
  is *compatible with* a profitable real Row 1.
- **Pickles' broader trading method.** Row 1 is one of six canonical
  setups and a minority of his journaled activity.
- **0DTE long options as an instrument class.** It proves only that
  *morning-window directional long entries on SPX*, operationalized this
  specific way, are unprofitable after costs. Other instruments (Row 3
  confluence, theta-gang spreads, delta-hedged positions) are out of
  scope.
- **That a tighter operationalization couldn't work.** It could. But
  iterating the trigger inside a tract is close to p-hacking — one
  revision was already needed to get any candidates at all. Further
  iteration without new data is not a discipline-healthy move.

## Next steps (for the user's decision)

1. **Fork C — behavioral cloning on Pickles' journal.** R3 feasibility
   already shows Tier-1 95% High, Tier-3 93% Medium+. Imitates Pickles'
   actual decisions rather than reverse-engineering one rule. The Stage-1
   finding that the qualifier layer *hurts* on SPX-only inputs strongly
   suggests Pickles uses context (ES dynamics, internals, regime cues)
   that SPX-only inputs cannot capture — which is exactly what imitation
   learning encodes. My recommendation.
2. **Row 3 — supply-zone break with confluence.** Different operational
   shape (level break + multi-asset confluence) so shares fewer failure
   modes with Row 1. Partially blocked by the same data gap (confluence
   requires DXY + 10YR + sector ETFs).
3. **Acquire ES/NQ/AD data and retest real Row 1.** Cleanest on principle
   but the most expensive and least leveraged given what we've seen.

Per the plan's refined guidance, the choice depends on the shape of this
Stage-1 result, not on a pre-committed escalation path. The shape suggests
Fork C.
