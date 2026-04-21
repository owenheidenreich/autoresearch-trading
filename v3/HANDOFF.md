# v3 Handoff — End of Research, Start of Implementation

**Date:** 2026-04-20
**Branch:** `codex/v3-orc-xsp-10k` (name is historical; project is now SPX/$25k)
**Project framing:** SPX 0DTE long premium, $25k account, 1 contract, Moderate-tier Layer 0 rails

## Purpose

This document hands off the v3 project at the boundary between **research (complete)** and **teacher-layer implementation (next)**. A fresh agent should be able to read this doc + the files it references and resume work without replaying the prior conversation.

## TL;DR — where we are

The v3 Stage 1 pipeline (guardrails, teachers, logger, dual oracles, feasibility + attribution reporters) is operational and has been run over the full 986-day SPX cache. The baseline teachers (ORC, FailedBreak) catch ~12% of oracle-best opportunities. The two gaps are diagnosed:

- **402 side_error sessions**: ORC fires in the wrong direction
- **467 abstention sessions**: no teacher fires but a real oracle opportunity exists in the late session

Eleven research investigations have been completed and documented. The exits are now specified:

- **ORC direction filter** (fixes side_error): add `sigma_pos ≤ 0.0` for calls / `≥ 0.0` for puts
- **NarrowRangeBreakout (NR10) teacher** (fixes abstention): new teacher with a B+C filter stack (VWAP σ-position + OMAR retest), window [40, 120], squeeze gate

Both are evidence-locked. Implementation is the next phase.

## Hard assumptions (DO NOT revisit without cause)

These were locked during research and should not be relitigated:

- Symbol = SPX, account = $25k, 1 contract, long premium only
- Moderate-tier Layer 0 rails in [v3/config.py](config.py): premium cap $1200/4.8%, daily brake $1500/6%, 2 full-premium losers. Calibrated from 5194-path 0DTE decay analysis. See [decay_analysis_2026_04_20.md](reference/decay_analysis_2026_04_20.md).
- **No fixed stop-loss fraction in Layer 0.** Exit discipline belongs to Layer 2/3. 5-9% of -60% drawdown contracts recover.
- Time window for NR10 teacher: `[40, 120]` (10:10-11:30 ET). Locked in [time_of_day_cap_2026_04_20.md](reference/time_of_day_cap_2026_04_20.md).
- Teacher playbooks are baselines, not truth sources. They generate supervised labels for Layer 2.
- HVN/LVN, basic S/R confluence, prior-day VP: tested, all negative. Do not re-test.

## Next phases

### Phase A — implement & validate teachers (START HERE)

**A1. ORC direction fix** — smallest change, do this first.

- Add SPY-derived VWAP σ-position computation as a BarContext field (if not already present; see [side_error_dive.py](analysis/side_error_dive.py) for a reference implementation of `_vwap_sigma_position`)
- Add two lines to [v3/teachers/orc.py](teachers/orc.py): reject BUY_CALL if `sigma_pos > 0`, reject BUY_PUT if `sigma_pos < 0`
- Re-run [run_full_attribution.py](analysis/run_full_attribution.py) on the 986-day cache
- Compare side_error count vs 402 baseline. Expected drop based on 52 correct / 171 wrong call ratio in the research: the filter should remove majority of the 330 wrong fires while keeping most of the 104 correct ones. If the drop is substantial and entered_right doesn't collapse, the side-error research is validated.

Research anchor: [side_error_deep_dive_2026_04_20.md](reference/side_error_deep_dive_2026_04_20.md)

**A2. NarrowRangeBreakout (NR10) teacher** — bigger lift, do after A1 passes.

New teacher with this exact spec (copied verbatim from [INDEX.md](reference/INDEX.md)):

```
NarrowRangeBreakout teacher (NR10):
  Window:          minutes [40, 120] (10:10-11:30 ET, post-MAGIC TIME)
  Eligibility:     SPX close inside first15 range
  Trigger:         current close breaks last-10-bar high (BUY_CALL)
                   or last-10-bar low (BUY_PUT)
  Squeeze gate:    last-10-bar SPX range ≤ 1.0× OMAR range
  Direction gate:  BUY_CALL requires SPX ≤ VWAP + 0.5σ (SPY-derived)
                   BUY_PUT  requires SPX ≥ VWAP − 0.5σ
  Retest gate:     SPX close within 0.5× OMAR of OMAR high, low, or mid
```

Implementation requires:
- New BarContext fields: `omar_high`, `omar_low`, `omar_mid`, `omar_range`, `last10_high`, `last10_low`, `sigma_pos`
- Preprocessing module (or extension of [v3/harness/v2_adapter.py](harness/v2_adapter.py)) to compute these per-day/per-bar
- New [v3/teachers/nr10.py](teachers/nr10.py) with a 5-gate evaluate function
- Registration in the logger runner so it's one of the teachers evaluated each bar

Re-run attribution. Expected: abstention count drops meaningfully. B+C captures 39.6% of 467 abstention bars at 1.92× enrichment, so ~185 new catches if the enrichment translates to teacher-level capture.

Research anchor: [combined_confluence_2026_04_20.md](reference/combined_confluence_2026_04_20.md)

### Phase B — decision point (after A's numbers land)

Three possible outcomes:

1. **Both counts drop as predicted.** Teacher layer is done. Proceed to Phase C.
2. **Counts drop, but less than predicted.** Diagnose. Likely causes: feature-computation mismatch (research vs implementation), teacher overlap, or cohort-to-trigger translation loss. Do not write more teachers before diagnosing.
3. **Counts don't drop at all.** Evidence-to-implementation gap. Re-read the research and the feature code side-by-side. This is the honest-failure path — accept it and investigate before pushing forward.

### Phase C — realistic replay

The teacher layer tells us *when* to enter. Phase C adds *what happens next*:

- Route teacher entries through Layer 0 guardrails in [v3/guardrails.py](guardrails.py)
- Select contracts under the cap/floor/delta/spread envelope
- Simulate fills and PnL across a held position using [v3/logger/](logger/) + [v3/oracles/exit_headroom.py](oracles/exit_headroom.py)
- Re-compute the four feasibility metrics + attribution under teacher behavior (vs oracle-only)
- This dataset is the supervised training set for Layer 2

### Phase D — supervised Layer 2 (deferred)

Only after Phase C PnL numbers are stable. Simple interpretable first: calibrated logistic regression or GBT predicting `p_continuation`, `p_reversal`, expected hold-vs-exit value. See the master plan for the full architecture.

## File inventory

### Research reference docs (read in order — see [INDEX.md](reference/INDEX.md))

All in `v3/reference/`:

1. [decay_analysis_2026_04_20.md](reference/decay_analysis_2026_04_20.md) — Layer 0 rail calibration
2. [attribution_full_2026-04-20.txt](reference/attribution_full_2026-04-20.txt) — 986-day pipeline output
3. [stage1_research_findings_2026_04_20.md](reference/stage1_research_findings_2026_04_20.md) — the four sharp findings
4. [omar_findings_2026_04_20.md](reference/omar_findings_2026_04_20.md) — OMAR as retest confluence + display unit
5. [time_of_day_cap_2026_04_20.md](reference/time_of_day_cap_2026_04_20.md) — [40, 120] window locked; raw NR10 weak alone
6. [vwap_bands_2026_04_20.md](reference/vwap_bands_2026_04_20.md) — VWAP σ direction rule
7. [sr_confluence_2026_04_20.md](reference/sr_confluence_2026_04_20.md) — basic S/R: null result
8. [volume_profile_2026_04_20.md](reference/volume_profile_2026_04_20.md) — intraday VP 1.24× (later deprioritized)
9. [side_error_deep_dive_2026_04_20.md](reference/side_error_deep_dive_2026_04_20.md) — **A1 source**
10. [combined_confluence_2026_04_20.md](reference/combined_confluence_2026_04_20.md) — **A2 source** (B+C stack)
11. [hvn_lvn_2026_04_20.md](reference/hvn_lvn_2026_04_20.md) — HVN/LVN: null result

### Analysis scripts (all reproducible from 986-day SPX + SPY cache)

All in `v3/analysis/`:

- [decay_analysis.py](analysis/decay_analysis.py) — rail calibration
- [run_full_attribution.py](analysis/run_full_attribution.py) — main pipeline runner; attribution output goes to `v3/reference/attribution_full_*.txt`
- [miss_analysis.py](analysis/miss_analysis.py) — first pass on abstention vs side_error split
- [abstention_deep_dive.py](analysis/abstention_deep_dive.py) — late-session pattern discovery
- [omar_and_magic_time.py](analysis/omar_and_magic_time.py) — OMAR + 09:55-10:10 window
- [omar_retest.py](analysis/omar_retest.py) — OMAR confluence enrichment
- [time_of_day_cap.py](analysis/time_of_day_cap.py) — window + raw NR10 forward-move quality
- [vwap_bands.py](analysis/vwap_bands.py) — VWAP σ-position direction rule
- [sr_confluence.py](analysis/sr_confluence.py) — basic S/R (null)
- [volume_profile.py](analysis/volume_profile.py) — POC/VAH/VAL
- [side_error_dive.py](analysis/side_error_dive.py) — **A1 reference**: `_vwap_sigma_position`, `_orc_would_fire`, feature-at-trigger analysis
- [combined_confluence.py](analysis/combined_confluence.py) — **A2 reference**: A/B/C independence + stack enrichment
- [hvn_lvn.py](analysis/hvn_lvn.py) — HVN/LVN (null)

### Pipeline code (operational, ready to extend)

- [v3/config.py](config.py) — `SymbolConfig`, `GuardrailConfig` (Moderate-tier, locked)
- [v3/guardrails.py](guardrails.py) — `filter_contract`, `DailyBudget` (per-gate flags, counterfactual properties)
- [v3/teachers/base.py](teachers/base.py) — `TeacherAction`, `BarContext`, abstract `Teacher`
- [v3/teachers/orc.py](teachers/orc.py) — ORC teacher (A1 modifies this)
- [v3/teachers/failed_break.py](teachers/failed_break.py) — FailedBreak teacher
- [v3/logger/schema.py](logger/schema.py) — `BarRecord`, `ContractRecord`, `OracleLabels`
- [v3/logger/builder.py](logger/builder.py) — composition layer
- [v3/logger/run.py](logger/run.py) — streaming runner
- [v3/harness/v2_adapter.py](harness/v2_adapter.py) — reads v2 data + sidecars; **extend here for new BarContext fields**
- [v3/oracles/opportunity.py](oracles/opportunity.py) — primary oracle (session-best hindsight)
- [v3/oracles/exit_headroom.py](oracles/exit_headroom.py) — secondary oracle (exit-policy ceiling)
- [v3/reporter/feasibility.py](reporter/feasibility.py) — four-diagnostic reporter (coverage, cap_bind_severity, suppression, low_delta_forcing)
- [v3/reporter/attribution.py](reporter/attribution.py) — guardrail_suppression / abstention / side_error / entered_right counts
- [v3/reporter/selection_quality.py](reporter/selection_quality.py) — contract-selection gap across 44,877 entered bars

### v2 feature extensions (done, integrated)

- [v2/pipeline/compute_features.py](../v2/pipeline/compute_features.py) — added `bars_since_break_above_first15`, `bars_since_break_below_first15`. Feature indices downstream shifted +2.

### Master plan

- `~/.claude/plans/new-clean-context-for-spicy-kay.md` — the source-of-truth architectural plan. Refers to SPX/$25k framing, multi-playbook teacher library, dual oracles, error attribution taxonomy, supervised Layer 2 before RL.

### Memory entries (auto-recalled on new sessions)

All in `~/.claude/projects/-Users-gduby-Documents-autoresearch-trading/memory/`:

- `project_v3_xsp_tract.md` — project scope and SPX pivot
- `project_v3_decay_evidence.md` — Layer 0 rail calibration
- `project_v3_stage1_complete.md` — pipeline operational milestone
- `project_v3_stage1_findings.md` — 986-day attribution findings
- `project_v3_omar_findings.md` — OMAR as retest + display unit
- `project_v3_vwap_findings.md` — VWAP direction rule (validated)
- `project_v3_volume_profile_findings.md` — VP as filter A
- `project_v3_side_error_findings.md` — **A1 source of truth**
- `project_v3_combined_confluence_findings.md` — **A2 source of truth**
- `project_v3_hvn_lvn_findings.md` — HVN/LVN null
- `project_error_attribution_taxonomy.md` — four-diagnostic taxonomy

## Critical reading order for a fresh agent

If you have 30 minutes:

1. This file ([v3/HANDOFF.md](HANDOFF.md))
2. [v3/reference/INDEX.md](reference/INDEX.md) — especially the final teacher spec section
3. [v3/reference/side_error_deep_dive_2026_04_20.md](reference/side_error_deep_dive_2026_04_20.md) (for A1)
4. [v3/reference/combined_confluence_2026_04_20.md](reference/combined_confluence_2026_04_20.md) (for A2)
5. [v3/teachers/orc.py](teachers/orc.py) (A1 target) and [v3/analysis/side_error_dive.py](analysis/side_error_dive.py) (A1 reference code)
6. [v3/config.py](config.py) (Layer 0 rails)

If you have 10 minutes: just this file + INDEX.md's teacher-spec section + [v3/teachers/orc.py](teachers/orc.py).

## Known gotchas

- **`oracle_direction` is in the attribution JSON; `orc_direction` is NOT.** To get ORC's fired direction per session, you must re-simulate the teacher trigger from bar state (see `_orc_would_fire` in [side_error_dive.py](analysis/side_error_dive.py)).
- **SPY-derived VWAP, not SPX.** Research uses `spy_1min.pkl` for volume aggregation because SPX has no direct volume. All σ-position numbers are SPY VWAP translated into SPX space. Implementation must follow this exact convention.
- **OMAR = first-minute H/L of the SPX session.** Not first-15, not IB. Single-bar range from 09:30-09:31.
- **V3 attribution outputs go to `v3/reference/attribution_full_YYYY-MM-DD.txt`.** The analysis scripts read the latest via glob.
- **Feature indices in v2 data shifted +2** after the `bars_since_break_*` additions. Any code reading v2 features by numeric index needs to account for this.
- **Branch name is misleading.** `codex/v3-orc-xsp-10k` suggests XSP + $10k; project is actually SPX + $25k. Ignore the branch name as a spec.

## Immediate next concrete task

Phase A1:

1. Verify current BarContext has `sigma_pos` or equivalent. If not, add it to [v3/teachers/base.py](teachers/base.py) BarContext and compute it in the v2 adapter using `_vwap_sigma_position` logic from [side_error_dive.py](analysis/side_error_dive.py).
2. Edit [v3/teachers/orc.py](teachers/orc.py): reject BUY_CALL when `bar.sigma_pos > 0.0`, reject BUY_PUT when `bar.sigma_pos < 0.0`.
3. Run `python -m py_compile v3/teachers/orc.py`.
4. Re-run attribution: `python -m v3.analysis.run_full_attribution` (or whatever the canonical invocation is per that script's `__main__`).
5. Compare new side_error count vs 402 baseline. Document in a new reference doc `v3/reference/orc_direction_fix_YYYY_MM_DD.md` mirroring the style of the existing research docs.

If the side_error count drops substantially without collapsing entered_right, proceed to A2. If not, diagnose before writing more code.
