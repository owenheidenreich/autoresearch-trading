# v3 Context Bundle for v4 Planning

**Purpose.** This is a curated bundle of the 14 most relevant v3 artifacts to
attach to a ChatGPT planning conversation about the v4 0DTE SPX rebuild.
v3 has been pronounced dead; v4 starts clean. The files here exist to
inform v4 planning — as priors to re-validate, as anti-patterns to avoid,
and as concrete examples of how v3 implemented things v4 may want to
re-implement differently.

The v4 plan being solidified is in
`~/.claude/plans/ok-well-this-just-declarative-puppy.md`. **Do not modify
that plan from this bundle.** This bundle is research input only.

---

## What v3 was

A monolithic neural ranker for 0DTE SPX option entries with a 3-layer stack:

- **L1** — action-surface dataset (per-bar, per-contract candidate features
  + utility labels under a fixed exit policy).
- **L2** — unified policy: shared encoder + 6 prediction heads (utility,
  dollar, return, win, clean, stopout) + ranking + calibration losses,
  picks daily-best contract per bar via argmax over utility logits.
- **L3** — exit-timing oracle: HistGradientBoosting / TCN classifier
  predicting "is this the peak?" per bar; learned exit on top of the L2
  selection.

Trained on ~4 years of SPX 0DTE data (2022-04 → 2026-04, 91k rows after
9:45-11:30 ET window restriction), 5 seeds, 13-window rolling
walk-forward.

## What v3 produced (durable findings — TIER 1, treat as priors to re-validate)

Several signals validated across the 986-day pipeline:

- **VWAP direction rule** (Pickles' "long below VWAP, put above"):
  `vwap_bands_2026_04_20.md`. Modest directional asymmetry on 986 days.
- **OMAR + VWAP confluence**: 1.92× enrichment on abstention oracle bars
  (`combined_confluence_2026_04_20.md`). VWAP+OMAR is the strongest
  2-filter stack v3 found.
- **Late-session abstention is high-utility**: `time_of_day_cap` — see
  `stage1_research_findings_2026_04_20.md` for the full Stage-1 summary.
- **Side asymmetry / put-greek inversion**: VWAP σ-position discriminates
  side error (`side_error_deep_dive_2026_04_20.md`). Calls and puts
  behave asymmetrically wrt distance to VWAP.
- **0DTE rail calibration**: `decay_analysis_2026_04_20.md` derives
  Layer-0 risk rails from 5194 0DTE decay paths (per-trade cap, daily
  brake, full-premium-loser limits).
- **5-seed K=2 consensus ensembling**: `ensemble_consensus_filter_2026_04_25.md`
  — taking trades only when ≥2 of 5 seeds agree on side lifts mean PF
  1.881 → 2.081, max DD 21.65% → 12.87% in-sample. Cheap CPU-only filter,
  no retrain.

## What v3 failed at (TIER 2, treat as anti-patterns)

The model itself never cleared a deployable forward-walk gate. Five
intertwined failure modes:

1. **Simulator/fill realism** — `forward_walk_failure_2026_04_25.md`:
   in-sample PF 2.14 vs forward-walk PF 0.72 (-66%) on 26 unseen days.
   Re-extending to 42 days only recovered to PF 1.14 vs offline 1.88.
   This is a fill-realism failure, not a learning failure. **Single most
   important v3 finding for v4 planning** — it's why v4's "Hard Truth
   #2" is "the simulator is the bottleneck, not the model."

2. **Label leakage discipline failure** — `oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md`:
   a +24% FW PF lift from a "trained classifier oracle gate" was
   reward-hacking. Two features (`time_stop_margin_raw`, `side_margin_raw`)
   were oracle labels computed from realized future PnL. With clean
   features, the gate was net negative. **Lesson v4 must keep**:
   mutate-future audit before any feature ships; massive offline
   improvements deserve immediate skepticism.

3. **L2 architectural pathologies** — `l2_architectural_review_2026_04_26.md`:
   8 hypotheses audited against 5 documented symptoms; the 5 symptoms
   collapse to **3 reinforcing mechanisms**:
   - Loss-weight imbalance (ranking 2.5× vs calibration 1.45× vs
     regression 0.5×) × 2.83:1 call/put gradient skew with side-balance
     defaults at 0.0
   - Single 160-dim shared bottleneck + flat-anchor pinning + arcsinh
     compression
   - L3 oracle trained only on L2's top-1 picks (sparse alternate
     coverage; any post-L2 intervention measures worse on the oracle
     metric than its actual realised PnL)

4. **L2 wrong-side picks** — `phase_c_failure_modes_2026_04_26.md`:
   in 4 of 9 sigma_pos × iv_percentile cells the L2 model systematically
   enters the wrong side. Hundreds of trades where the alternate side's
   PF was meaningfully higher. Side selection is broken upstream of any
   exit policy. **Implication for v4**: abstention must be a labeled
   action with non-zero utility, not a "no signal" default; per-cell
   side preference must be learned or imposed, not assumed.

5. **Methodology drift / single-period overfit** —
   `methodology_overhaul_summary_2026_04_22.md`: V1+L3 was "champion"
   on a 20-day artifact; rolling 13-window OOS REVERSED the claim
   (V0 wins 10/13 windows, agg PF 1.132). **The 13-window walk-forward
   is the v3 methodology v4 inherits.**

## What v3 produced for methodology (still useful for v4)

- **13-window rolling OOS framework** —
  `methodology_overhaul_summary_2026_04_22.md` + the code in
  `rolling_windows.py` and `forward_walk.py`. v4 plan section 5 phases
  reference this directly.
- **5-seed K=2 consensus** — `ensemble_consensus_filter_2026_04_25.md`.
  v4 plan section 2 lists this as a salvageable pattern.
- **Causal-feature audit pattern** — see the leakage retraction doc;
  every v3 feature shipped only after a mutate-future test.

## File index (alphabetical; tier annotations in brackets)

| File | Tier | What it is |
|---|---|---|
| `action_surface_dataset.py` | code | v3's per-bar action-surface label builder + `hybrid_live_utility` cost model (PnL + 125×return − 0.35×spread − 60×stopout − late-hold penalty). Useful as a concrete example of v3's cost composition; v4's simulator should be more sophisticated. |
| `combined_confluence_2026_04_20.md` | prior | OMAR + VWAP confluence (1.92× enrichment). |
| `decay_analysis_2026_04_20.md` | prior | 0DTE decay characteristics on 5194 paths; basis for Layer-0 rails. |
| `ensemble_consensus_filter_2026_04_25.md` | methodology | 5-seed K=2 consensus filter. |
| `forward_walk.py` | code | The walk-forward evaluator that compares offline vs OOS; carries the +`--apply-veto`+ flag from the recent post-L2 veto experiment. |
| `forward_walk_failure_2026_04_25.md` | failure | The 2.14 → 0.72 forward-walk collapse. The simulator-realism finding that motivates v4's "build simulator first" thesis. |
| `l2_architectural_review_2026_04_26.md` | failure | 8 pathologies → 3 reinforcing mechanisms synthesis. The most thorough v3 architectural post-mortem. |
| `methodology_overhaul_summary_2026_04_22.md` | methodology | The rolling 13-window OOS rigor; reverses prior champion claims. |
| `oracle_gate_LEAKAGE_RETRACTION_2026_04_25.md` | failure | Label-leakage discipline retraction; +24% lift was reward-hacking. |
| `phase_c_failure_modes_2026_04_26.md` | failure | L2 wrong-side picks per (sigma_pos × iv_percentile) cell. |
| `rolling_windows.py` | code | The 13-window walk-forward generator. |
| `side_error_deep_dive_2026_04_20.md` | prior | VWAP σ-position discriminates side error. |
| `stage1_research_findings_2026_04_20.md` | prior | Stage-1 summary: four sharp findings on the 986-day pipeline. |
| `vwap_bands_2026_04_20.md` | prior | VWAP direction rule validation. |

## Mapping to v4 plan sections

| v4 plan section | Bundle files most relevant |
|---|---|
| §1 "Hard Truth #2 (simulator is bottleneck)" | `forward_walk_failure_*` |
| §2 "What to Salvage from v3" | All TIER-1 priors + `ensemble_consensus_filter_*` |
| §4.1 (simulator + fill model) | `action_surface_dataset.py` (cost-model reference, NOT to copy as-is) |
| §4.3 (oracle, no leak) | `oracle_gate_LEAKAGE_RETRACTION_*` |
| §5 (phases / walk-forward gates) | `methodology_overhaul_*`, `rolling_windows.py`, `forward_walk.py` |
| §7 anti-patterns 1, 7, 10 | `oracle_gate_LEAKAGE_RETRACTION_*`, `l2_architectural_review_*`, `phase_c_failure_modes_*` |

## Honest caveats

- v3 priors were measured on data that includes the pre-2022-04 regime
  (before SPXW Tuesday/Thursday daily expirations). Per the v4 plan,
  edge claims must be validated on the modern 2023+ daily-0DTE regime,
  not averaged across the regime change. v3 did not cleanly stratify
  by this break.
- v3's labels and exits were entangled (the L2 ranking objective and
  the L3 exit policy were trained against each other), which is the
  root of the FW oracle-coupling pathology described in
  `l2_architectural_review_*`. v4 plan §4.3 explicitly forbids this
  by separating the labeling exit from the learned exit.
- The PF numbers in v3 docs use v3's own simulator, which v4's "Hard
  Truth #2" identifies as not-realistic-enough. Treat all v3 PF
  numbers as "this is what the model thought," not "this is what
  IBKR would have produced."

## What is NOT in this bundle (intentionally)

- The dozen+ falsified-experiment docs (H3a/H3c/H3e/H3f, sideblind,
  side-balance sweep, etc.) — interesting but not load-bearing for v4
  planning. The architectural review summarises them.
- Recent L2 retrain attempt artifacts (Phase 0/1/2/3 contracts and
  scripts from the now-dead L2_combined_v1 experiment). v4 starts
  clean.
- v3 code modules beyond the three included (the rest is the failed
  L2/L3 architecture — v4 §4 explicitly rebuilds this).
- Memory entries from the auto-memory system. The doc-trail above is
  authoritative.
