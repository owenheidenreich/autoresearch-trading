# Findings — `COMPOSITE_CALIBRATION_TERMINAL_RULE` (pre-decision evidence)

**STATUS: research findings (Claude-written, read-only).** Prepared to inform the one unresolved
owner decision that blocks the machinery/stability seal and any fit. This note gathers the evidence;
it does NOT pick the rule. Next step: adversarial Codex investigation (see the paired goal), then an
owner decision backed by both analyses.

Source of record: `protocol101_pathd_entry_exit_model_research_corrected_2026_08_01/preregistration.json`
(+ `session_assignments.json`). The open decision is defined at
`/foundation_correction/prefit_pause/unresolved_owner_decision`:
> a calibration-valid session-count block can still produce an `INVALID_TARGET_COVERAGE` or
> `INSUFFICIENT_EVIDENCE` composite correction; choosing its durable terminal/minimum-power rule
> changes scientific topology and is outside P1/P2/P3. Required before: machinery/stability seal or
> any model fit.

## What the decision is (plain terms)
Calibration = the honesty check on the model's stated confidence bands, done on a held-aside slice of
each fold's training days (last `ceil(20%)` of primary training sessions, `calibration_embargo`
`2025-10-17`, minimum 10 sessions). Sometimes the composite correction can't be formed honestly even
with ≥10 days: either the math can't reach nominal coverage (`INVALID_TARGET_COVERAGE`) or, after
dropping label-invalid rows, too little trustworthy data remains (`INSUFFICIENT_EVIDENCE`). The
decision is the **durable behavior when that happens in a fold**, and it is forward-looking (it
governs future data-era runs, not only this one).

## Finding 1 — the exit half already abstains this run, by design
`/combined_evaluation/known_run_consequence`: *"the frozen exit model-fit session counts are all <60,
so this run must stop `insufficient_evidence` before exit weights and will not lawfully instantiate
C/D or claim a combined pass unless a new data-era plan changes the evidence geometry."* So this is
effectively an **entry-only feasibility run** that stands down at the exit boundary; the exit model
needs a larger data era. ⇒ For THIS run, the terminal rule bites on the **entry composite**; for
future runs it governs all targets.

## Finding 2 — the trade-count power rules are already frozen (NOT this decision)
- `entry/acceptance/minimum_power`: 40 completed trades/fold, 200 pooled.
- `entry/acceptance/power_denominator`: <40/fold ⇒ `insufficient_evidence`, no lifecycle substitution.
- `metrics_and_gates/minimum_power`: 500 eligible exit trajectories / model-fit fold; 40 entry
  executed trades/fold; `entry/control_exit_selection` requires ≥25 valid model-fit sessions.
- `verdict_vocabulary.entry_failure_mapping`: `insufficient_entry_evidence` (power miss) vs
  `no_genuine_entry_signal` (adequately powered but fails).
This decision is specifically about the **composite calibration correction itself**, distinct from
trade-count power.

## Finding 3 — what the composite calibration spans (to be confirmed by Codex)
- `entry/composer/mean_composites`: `LCB_mean_upside_$` and `LCB_mean_upside_return`, each the
  arithmetic mean of two calibrated conditional-mean LCB components (MFE + profit-area).
- `metrics_and_gates/action_calibration/actions/ENTER/composite_conformal`: ENTER/WAIT conformal on
  the disjoint earlier calibration intents.
- Calibration heads (`mean_lcb`, `monotone_q10_q50_q90`, `q10_conformal`) each carry
  `minimum_sessions: 10`.
Codex should enumerate every target/head the composite depends on and each one's fold-level failure
surface (which can independently trip `INVALID_TARGET_COVERAGE`).

## Finding 4 — the decision is LIVE because of fold 1 (knife-edge margin)
Calibration needs ≥10 sessions. Per-fold `calibration_last_20_percent` sizes:

| fold | model_fit | calibration days | margin over 10 |
|---:|---:|---:|---:|
| 1 | 54 | **14** | **+4** ⚠️ |
| 2 | 75 | 20 | +10 |
| 3 | 97 | 25 | +15 |
| 4 | 119 | 31 | +21 |
| 5 | 142 | 36 | +26 |

Fold 1 has only 4 days of slack: a few label-incomplete calibration days could tip its composite into
insufficient — the exact scenario the terminal rule governs. Folds 2–5 are comfortable. So the choice
of rule genuinely changes fold-1's disposition and therefore the ≥4/5 gate.

## Finding 5 — unresolved discrepancy for Codex to reconcile
`known_run_consequence` says exit model-fit counts are "all <60," but outer `model_fit` is
54/75/97/119/142 — four of five are ≥60. The "<60" is likely measured on the **nested inner exit-fit
blocks**, not the outer train window. This must be pinned down: it determines whether the exit truly
must abstain in every fold or only fold 1, which in turn affects how much weight the entry-side
terminal rule actually carries this run.

## The candidate rules (for the investigation to evaluate against evidence — not yet chosen)
- **A. Fold-scoped abstain + power floor:** an un-calibratable required target ⇒ that fold returns
  `insufficient_evidence`; acceptance still needs pooled-pass AND ≥4/5 *fully valid* folds (an
  insufficient fold cannot fill a slot); if <4 folds are ever valid, the run terminates
  `insufficient_evidence`; never drop a target to rescue coverage. (Matches the owner's
  abstention-first, no-reward-hacking stance.)
- **B. Whole-run hard stop:** any target failing calibration in any fold terminates the run
  `insufficient_evidence`. Strictest; brittle to one weak fold.
- **C. Drop the offending target, continue:** changes the composite fold-to-fold; biases toward
  easy-to-calibrate targets. Flagged as reward-hacking-adjacent.

## Open questions for Codex (adversarial, evidence-based)
1. Reconcile Finding 5 (nested vs outer counts) with a precise citation.
2. Enumerate the full composite target/head set and each one's per-fold (esp. fold-1) failure surface.
3. Quantify, from label completeness in the frozen calibration partitions, the actual probability
   that fold-1's composite trips insufficient (is this a real risk or theoretical?).
4. Recommend a durable terminal + minimum-power rule from the evidence, and state which existing frozen
   contracts it must bind to (acceptance ≥4/5, verdict_vocabulary, minimum_power) — read-only; no fit,
   no holdout, no seal.
